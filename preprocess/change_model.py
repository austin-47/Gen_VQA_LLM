#!/usr/bin/env python3
"""
Fine-tune the vision encoder of a VLM (Qwen) with contrastive image<->report loss.
Now supports Qwen2.5-VL (requires grid_thw input).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    Qwen2_5_VLForConditionalGeneration,
    logging,
    set_seed
)

logging.set_verbosity_error()


# -------------------------
# Dataset
# -------------------------
class MedicalVQADataset(Dataset):
    def __init__(self, json_files: List[str], image_roots: List[str], transform=None):
        assert len(json_files) == len(image_roots)
        self.items = []
        for jf, root in zip(json_files, image_roots):
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for entry in data:
                img_field = None
                for k in ('images', 'image', 'images_path', 'images_file'):
                    if k in entry:
                        img_field = k
                        break
                if img_field is None:
                    continue
                imgname = entry[img_field]
                if not imgname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    imgname += '.jpg'
                imgpath = os.path.join(root, imgname)
                report = entry.get('report', '') or entry.get('new_report', '')
                if not report:
                    q = entry.get('new_q', entry.get('question', ''))
                    a = entry.get('new_a', entry.get('answer', ''))
                    report = f"Question: {q}\nAnswer: {a}"
                self.items.append({'image_path': imgpath, 'report': report})
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        path = it['image_path']
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, it['report'], path


# -------------------------
# Helper Functions
# -------------------------
def get_vision_module(model):
    for name, module in model.named_children():
        if 'visual' in name or 'image' in name:
            return module, name
    return None, None


def mean_pool_last_hidden(hidden_states, attention_mask=None):
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


# -------------------------
# Training function
# -------------------------
def train_contrastive(
    model_dir,
    json_files,
    image_roots,
    output_dir,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    device='cuda',
    seed=42,
    max_report_length=256,
    image_size=224,
    temperature_init=0.07,
):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] Loading Qwen2.5-VL model from {model_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, trust_remote_code=True)
    vision_module, _ = get_vision_module(model)

    if vision_module is None:
        raise RuntimeError("Could not find vision encoder inside model")

    text_id = "emilyalsentzer/Bio_ClinicalBERT"
    text_module = AutoModel.from_pretrained(text_id)
    tokenizer = AutoTokenizer.from_pretrained(text_id)

    model.to(device)
    text_module.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in vision_module.parameters():
        p.requires_grad = True
    for p in text_module.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = MedicalVQADataset(json_files, image_roots, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Get text embedding dim
    with torch.no_grad():
        dummy = tokenizer("normal lungs", return_tensors='pt').to(device)
        t_out = text_module(**dummy)
        text_dim = t_out.last_hidden_state.size(-1)

    print(f"[INFO] Text embedding dim = {text_dim}")

    # Get image embedding dim
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, image_size, image_size).to(device)
        grid_thw = torch.tensor([[14, 14, 1]], device=device)  # approximate for 224/16 patch size
        try:
            img_out = vision_module(dummy_img, grid_thw=grid_thw, return_dict=True)
            img_dim = img_out.pooler_output.size(-1)
        except Exception as e:
            print("[WARN] Vision forward failed:", e)
            img_dim = 1024
    print(f"[INFO] Image embedding dim = {img_dim}")

    # Projection heads
    proj_dim = 256
    class ProjectionHead(nn.Module):
        def __init__(self, in_dim, out_dim=proj_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim)
            )
        def forward(self, x): return self.net(x)

    image_proj = ProjectionHead(img_dim, proj_dim).to(device)
    text_proj = ProjectionHead(text_dim, proj_dim).to(device)
    for p in text_proj.parameters():
        p.requires_grad = False

    logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))
    optimizer = torch.optim.AdamW(
        list(vision_module.parameters()) + list(image_proj.parameters()) + [logit_scale],
        lr=lr, weight_decay=0.01
    )

    def info_nce_loss(img_emb, txt_emb, scale):
        img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)
        logits = torch.matmul(img_emb, txt_emb.t()) * scale.exp()
        labels = torch.arange(len(logits), device=logits.device)
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    # Training loop
    for epoch in range(epochs):
        vision_module.train()
        image_proj.train()
        total_loss = 0
        for images, reports, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            toks = tokenizer(
                reports, padding=True, truncation=True,
                max_length=max_report_length, return_tensors='pt'
            ).to(device)

            # text embedding
            with torch.no_grad():
                t_out = text_module(**toks)
                t_emb = mean_pool_last_hidden(t_out.last_hidden_state, toks['attention_mask'])
                t_emb = text_proj(t_emb)

            # vision embedding
            grid_thw = torch.tensor([[14, 14, 1]] * images.size(0), device=device)
            # try:
            #     v_out = vision_module(images, grid_thw=grid_thw, return_dict=True)
            #     v_feat = v_out.pooler_output
            # except TypeError:
            #     v_out = vision_module(pixel_values=images, grid_thw=grid_thw, return_dict=True)
            #     v_feat = v_out.pooler_output
            print(images.shape)
            v_out = vision_module(images, grid_thw=grid_thw)

            v_feat = v_out.pooler_output
            v_emb = image_proj(v_feat)
            loss = info_nce_loss(v_emb, t_emb, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vision_module.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[EPOCH {epoch+1}] avg_loss = {total_loss / len(dataloader):.4f}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'vision': vision_module.state_dict(),
            'image_proj': image_proj.state_dict(),
            'logit_scale': logit_scale.detach().cpu(),
        }, os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pt"))

    print("[DONE] Training complete.")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', type=str, required=True)
    ap.add_argument('--jsons', nargs='+', required=True)
    ap.add_argument('--image_roots', nargs='+', required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--max_report_length', type=int, default=256)
    ap.add_argument('--image_size', type=int, default=224)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_contrastive(
        args.model_dir, args.jsons, args.image_roots,
        args.output_dir, args.epochs, args.batch_size,
        args.lr, args.device, max_report_length=args.max_report_length,
        image_size=args.image_size
    )

#!/usr/bin/env python3
"""
Convert a JSON file with reports into a Verl-compatible parquet dataset.
Usage:
    python convert_to_parquet.py --input with_reports.json --output_dir parquet_dataset

- Resizes images to 224x224 and caches them under RESIZED_ROOT.
- Adds prompt using INSTRUCTION_TEMPLATE.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
import re

# -------------------------
IMAGE_SIZE = (224, 224)
RESIZED_ROOT = "resized_224"  # can override via CLI
INSTRUCTION_TEMPLATE = (
    "<image>"
    "{Question} "
    "You are tasked with analyzing an image to generate a detailed description to help you answer the question. "
    "First analyze the image and produce a self-contained description—detailed enough that can lead to the correct answer. Wrap the entire description in <report></report> tags. "
    "Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Provide your detailed, step-by-step reasoning based on the description and image, and enclose this part within <think></think> tags. "
    "Finally, base on your reasoning, provide the final answer in <answer> </answer> tags. "
    "The output format should be as follows: <report> description here </report> <think> reasoning process here </think><answer> answer here (Do not provide any explanation)</answer>. "
    "Please strictly follow the format."
)
# -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_text(x: Optional[str]) -> str:
    return re.sub(r'\s+', ' ', str(x).strip()) if x is not None else ""

def resize_and_cache_image(src_path: str) -> Optional[str]:
    """Resize image to IMAGE_SIZE and cache under RESIZED_ROOT."""
    if not src_path or not os.path.isfile(src_path):
        return None
    rel_name = os.path.basename(src_path)
    dst = os.path.join(RESIZED_ROOT, rel_name)
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return os.path.abspath(dst)
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = im.resize(IMAGE_SIZE, Image.LANCZOS)
            im.save(dst, quality=95)
        return os.path.abspath(dst)
    except Exception as e:
        print(f"[ERROR] resizing {src_path} -> {dst}: {e}")
        return None

def build_record(item: dict, idx: int) -> dict:
    prompt_q = normalize_text(item.get("question") or item.get("new_q") or "")
    answer = normalize_text(item.get("answer") or item.get("new_a") or "")
    report = normalize_text(item.get("report") or "")
    img_paths = item.get("image_name") or item.get("images") or item.get("image")
    if isinstance(img_paths, str):
        img_paths = [img_paths]
    resized_imgs = [resize_and_cache_image(p) for p in img_paths if p]
    prompt_text = INSTRUCTION_TEMPLATE.replace("{Question}", prompt_q)
    rec = {
        "data_source": "custom_dataset",
        "prompt": [{"role": "user", "content": prompt_text}],
        "images": resized_imgs,
        "ability": "vqa",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": idx,
            "question": prompt_q,
            "answer": answer,
            "report_label": report
        }
    }
    return rec

def main(args):
    ensure_dir(RESIZED_ROOT)
    ensure_dir(args.output_dir)

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    records = []
    skipped = 0
    for idx, item in enumerate(tqdm(items, desc="Processing JSON")):
        rec = build_record(item, idx)
        if not rec["images"]:
            skipped += 1
            continue
        records.append(rec)

    print(f"Processed {len(items)} items → kept {len(records)}, skipped {skipped}")

    if records:
        ds = Dataset.from_list(records)
        out_path = os.path.join(args.output_dir, "dataset.parquet")
        ds.to_parquet(out_path)
        print(f"Saved parquet: {out_path} ({len(records)} rows)")
    else:
        print("No valid records to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file with reports")
    parser.add_argument("--output_dir", required=True, help="Output directory for parquet dataset")
    parser.add_argument("--resized_root", default=RESIZED_ROOT, help="Folder to cache resized images")
    args = parser.parse_args()
    RESIZED_ROOT = args.resized_root
    main(args)

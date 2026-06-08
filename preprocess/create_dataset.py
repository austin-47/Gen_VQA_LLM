#!/usr/bin/env python3
"""
Create Verl-compatible parquet datasets (train/test) from refined JSONs:
- VQA-RAD (refined)
- SLAKE  (refined)
- PathVQA (refined)

Resizes images to 224x224 and caches them under RESIZED_ROOT/<dataset>/...
Outputs:
  <out_dir>/train.parquet
  <out_dir>/test.parquet  (if any test inputs provided)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
import re

# -------------------------
# Default paths you provided (change via CLI if needed)
# -------------------------
# DEFAULTS = {
#     "vqarad_train": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_report_describe_nofurther.json",
#     "vqarad_test": None,  # set if you have a test file
#     "vqarad_img_root": "/home/work/austin/Dataset/vqa-rad/VQA_RAD Image Folder",

#     "slake_train": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_report_describe_nofurther.json",
#     "slake_test": None,
#     "slake_img_root": "/home/work/austin/Dataset/SLAKE/imgs",

#     "pathvqa_train": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine_report_describe_nofurther.json",
#     "pathvqa_test": None,
#     "pathvqa_img_root_train": "/home/work/austin/Dataset/path-vqa/pvqa/images/train",
#     "pathvqa_img_root_val": "/home/work/austin/Dataset/path-vqa/pvqa/images/val",
#     "pathvqa_img_root_test": "/home/work/austin/Dataset/path-vqa/pvqa/images/test",
# }

# DEFAULTS = {
#     "vqarad_train": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_v2.json",
#     "vqarad_test": None,  # set if you have a test file
#     "vqarad_img_root": "/home/work/austin/Dataset/vqa-rad/VQA_RAD Image Folder",

#     "slake_train": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_v2_en.json",
#     "slake_test": None,
#     "slake_img_root": "/home/work/austin/Dataset/SLAKE/imgs",

#     "pathvqa_train": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine.json",
#     "pathvqa_test": None,
#     "pathvqa_img_root_train": "/home/work/austin/Dataset/path-vqa/pvqa/images/train",
#     "pathvqa_img_root_val": "/home/work/austin/Dataset/path-vqa/pvqa/images/val",
#     "pathvqa_img_root_test": "/home/work/austin/Dataset/path-vqa/pvqa/images/test",
# }

DEFAULTS = {
    "vqarad_train": "/home/work/austin/MedCCO/preprocess/VQARAD_report_wqa_roi_new.json",
    "vqarad_test": None,  # set if you have a test file
    "vqarad_img_root": "/home/work/austin/Dataset/vqa-rad/VQA_RAD Image Folder",

    "slake_train": "/home/work/austin/MedCCO/preprocess/SLAKE_report_wqa_roi.json",
    "slake_test": None,
    "slake_img_root": "/home/work/austin/Dataset/SLAKE/imgs",

    "pathvqa_train": "/home/work/austin/MedCCO/preprocess/PathVQA_report_wqa_roi.json",
    "pathvqa_test": None,
    "pathvqa_img_root_train": "/home/work/austin/Dataset/path-vqa/pvqa/images/train",
    "pathvqa_img_root_val": "/home/work/austin/Dataset/path-vqa/pvqa/images/val",
    "pathvqa_img_root_test": "/home/work/austin/Dataset/path-vqa/pvqa/images/test",
}

# -------------------------
# Resize / output config
# -------------------------
IMAGE_SIZE = (448, 448)
RESIZED_ROOT = "/home/work/austin/Dataset/resized_224"  # where resized images will be stored
OUT_DIR = "verl_parquet"  # defaults; can be overridden by args

# INSTRUCTION_TEMPLATE = (
#      "<image>"
#      "{Question} " 
#      "Carefully analyze the image and produce a detailed description in <report> </report> tags. "
#      "Next, provide your reasoning process based on the given image and generated description in <think> </think> tags "
#      "Finally, base on your reasoning, provide the final answer in <answer> </answer> tags. "
#      "The output answer format should be as follows: <report> description here </report> <think> reasoning process here </think><answer> answer here (Do not provide any explanation)</answer>. "
#      "Please strictly follow the format."
# )


# INSTRUCTION_TEMPLATE = (
#     "<image>"
#     "{Question} " 
#     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows: <think> reasoning process here </think><answer> answer here (Do not provide any explanation) </answer> Please strictly follow the format."
# )

# INSTRUCTION_TEMPLATE = (
#    "<image>"
#    "{Question} " 
#    "You are tasked with analyzing an image to generate a detailed description to help you answer the question. "
#    "First, analyze the image and produce a detailed roi-focused description in <report> </report> tags. "
#    "Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Provide your detailed, step-by-step reasoning based on the image and description, wrap this in <think> </think> tags. "
#    "Finally, provide the answer to the question in <answer> </answer> tags. "
#    "The output answer format should be as follows: <report> description here </report> <think> reasoning process here </think><answer> answer here (Do not provide any explanation)</answer>. "
#    "Please strictly follow the format."
# )

###2901
INSTRUCTION_TEMPLATE = (
   "<image>"
   "{Question} " 
   "You are tasked with analyzing an image to generate a detailed description to help you answer the question. "
   "First, analyze the image and produce a detailed roi-focused description (including anatomical structures, appearance, abnormalities,...). Wrap this in <report> </report> tags. "
   "Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Reference relevant visual findings from the report. Use medical knowledge to connect findings to possible interpretations. Output the detailed, step-by-step reasoning based on the image and description, wrap this in <think> </think> tags. "
   "Finally, based ONLY on the report and reasoning, provide the answer to the question in <answer> </answer> tags. "
   "The output answer format should be as follows: <report> description here </report> <think> reasoning process here </think><answer> answer here (Do not provide any explanation)</answer>. "
   "Please strictly follow the format."
)

# INSTRUCTION_TEMPLATE = (
#    "<image>"
#    "{Question} " 
#    "You are tasked with analyzing an image to generate a detailed description to help you answer the question. "   
#    "First, analyze the image and produce a detailed roi-focused description (including anatomical structures, shapes, densities, colors, textures, notable findings, abnormalities,...). Wrap this in <report> </report> tags. "
#    "Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Output the detailed, step-by-step clinical reasoning that explicitly references findings stated in the description, wrap this in <think> </think> tags. "
#    "Finally, based ONLY on the description and reasoning, provide the answer to the question in <answer> </answer> tags. "
#    "The output answer format should be as follows: <report> description here </report> <think> reasoning process here </think><answer> answer here (Do not provide any explanation)</answer>. "
#    "Please strictly follow the format."
# )


# -------------------------
# Utilities
# -------------------------
def normalize_text(x: Optional[str]) -> str:
    return re.sub(r'\s+', ' ', str(x).strip()) if x is not None else ""

def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def resize_and_cache_image(src_path: str, dataset_name: str) -> Optional[str]:
    """
    Resize src_path to IMAGE_SIZE and save under RESIZED_ROOT/<dataset_name>/<relative_path>.
    Returns absolute path of resized image or None if source missing/error.
    """
    if not src_path:
        return None
    if not os.path.isfile(src_path):
        return None

    # Attempt to create a friendly preserved relative path:
    # Use basename or relative path if it already contains '/xmlab.../...'
    rel = os.path.basename(src_path)
    # If original src contains subdirs (like slake xmlabX/source.jpg), preserve them
    # by using everything after last known dataset root slash
    # We'll try to keep folder structure if src_path contains dataset_name as part of path
    try:
        # if src contains dataset_name, attempt to keep path after that
        idx = src_path.lower().find(dataset_name.lower())
        if idx != -1:
            rel = src_path[idx+len(dataset_name)+1:].lstrip("/\\")
    except Exception:
        pass

    dst = os.path.join(RESIZED_ROOT, dataset_name, rel)
    dst_dir = os.path.dirname(dst)
    ensure_dir(dst_dir)

    # Ensure dst ends with .jpg
    base, ext = os.path.splitext(dst)
    dst = base + ".jpg"

    if os.path.exists(dst):
        return os.path.abspath(dst)

    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = im.resize(IMAGE_SIZE, resample=Image.LANCZOS)
            im.save(dst, quality=95)
        return os.path.abspath(dst)
    except Exception as e:
        print(f"[ERROR] resizing {src_path} -> {dst}: {e}")
        return None

def extract_q_a(item: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Prefer new_q/new_a if present and non-empty.
    Otherwise, try conversations (supports 'role'/'value' and 'from'/'value').
    Otherwise fallback to 'question'/'answer' fields.
    """
    # Prefer new_q/new_a
    nq = item.get("new_q")
    na = item.get("new_a")
    if nq and str(nq).strip() and na and str(na).strip():
        return normalize_text(nq), normalize_text(na)

    # Conversations
    conv = item.get("conversations") or item.get("conversation") or item.get("convos")
    if isinstance(conv, list) and len(conv) >= 2:
        # handle objects with keys 'role' or 'from' and 'value' or 'content'
        def get_val(o):
            if isinstance(o, dict):
                if "value" in o:
                    return o["value"]
                if "content" in o:
                    return o["content"]
            return None
        try:
            q = get_val(conv[0]) or item.get("question")
            a = get_val(conv[-1]) or item.get("answer")
            if q is not None and a is not None:
                return normalize_text(q), normalize_text(a)
        except Exception:
            pass

    # fallback to question/answer fields
    q = item.get("question") or item.get("ori_q") or item.get("ori_question")
    a = item.get("answer") or item.get("ori_a")
    if q is not None and a is not None:
        return normalize_text(q), normalize_text(a)

    return None, None

def resolve_vqarad_image(item: dict, img_root: str) -> Optional[str]:
    # VQA-RAD uses "image_name"
    img_name = item.get("images") or item.get("image")
    if not img_name:
        return None
    path = os.path.join(img_root, img_name)
    return path if os.path.isfile(path) else None

def resolve_slake_image(item: dict, img_root: str) -> Optional[str]:
    # SLAKE uses 'img_name' and may include subfolders like xmlab1/source.jpg
    img_name = item.get("images") or item.get("image")
    if not img_name:
        return None
    path = os.path.join(img_root, img_name)
    return path if os.path.isfile(path) else None

def resolve_pathvqa_image(item: dict, roots: List[str]) -> Optional[str]:
    # PathVQA uses 'image' w/out extension, e.g., train_0422
    # We will try the provided roots (train/val/test) for existence.
    img_base = item.get("images") or item.get("image")
    if not img_base:
        return None
    # If item contains full filename already (with .jpg), handle it
    if img_base.lower().endswith(".jpg") or img_base.lower().endswith(".jpeg") or img_base.lower().endswith(".png"):
        candidate = img_base
        for r in roots:
            p = os.path.join(r, os.path.basename(candidate))
            if os.path.isfile(p):
                return p
        # absolute path?
        if os.path.isfile(img_base):
            return img_base
        return None

    # try with suffixes
    for r in roots:
        p = os.path.join(r, f"{img_base}.jpg")
        if os.path.isfile(p):
            return p
        p2 = os.path.join(r, f"{img_base}.jpeg")
        if os.path.isfile(p2):
            return p2
        p3 = os.path.join(r, img_base)
        if os.path.isfile(p3):
            return p3
    return None

def build_record(dataset_name: str, prompt_q: str, answer: str, resized_image_path: str,
                 split: str, idx: int, answer_type: Optional[str] = None, report: Optional[str] = None) -> dict:
    prompt_text = INSTRUCTION_TEMPLATE.replace("{Question}", prompt_q)
    rec = {
        "data_source": dataset_name,
        "prompt": [
            {"role": "user", "content": prompt_text}
        ],
        "images": [resized_image_path],
        "ability": "vqa",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer,
            "question": prompt_q,
            "answer_type": answer_type if answer_type else "UNKNOWN",
            "report_label": report if report else ""
        }
    }
    return rec


# -------------------------
# Processing functions for each dataset
# -------------------------
def process_dataset_file(json_path: str,
                         dataset_name: str,
                         img_roots: List[str],
                         split_tag: str) -> List[dict]:
    """
    json_path: path to refined json (list of dict items)
    dataset_name: identifier string, e.g., 'vqarad', 'slake', 'pathvqa'
    img_roots: list of candidate image roots to resolve images
    split_tag: 'train' or 'test' (used in extra_info)
    Returns list of final records (already resized images).
    """
    items = load_json(json_path)
    records = []
    skipped = 0
    for idx, item in enumerate(tqdm(items, desc=f"{dataset_name}:{split_tag}")):
        # Skip drop status
        if item.get("status") == "drop":
            skipped += 1
            continue

        q, a = extract_q_a(item)
        if q is None or a is None:
            skipped += 1
            continue

        # resolve source image path depending on dataset
        src = None
        if dataset_name.lower() == "vqarad":
            src = resolve_vqarad_image(item, img_roots[0])
        elif dataset_name.lower() == "slake":
            src = resolve_slake_image(item, img_roots[0])
        elif dataset_name.lower() == "pathvqa":
            src = resolve_pathvqa_image(item, img_roots)
        else:
            # generic: try first root + image field
            img_field = item.get("image") or item.get("image_name") or item.get("img_name")
            if img_field:
                src = os.path.join(img_roots[0], img_field)

        if not src or not os.path.isfile(src):
            # try alternative approach for SLAKE: sometimes img_name contains "xmlabX/source.jpg" but user root differs
            # or for VQA-RAD try alt roots
            found = None
            for r in img_roots:
                cand = os.path.join(r, item.get("image") or item.get("image_name") or item.get("img_name") or "")
                if os.path.isfile(cand):
                    found = cand
                    break
            if found:
                src = found
            else:
                skipped += 1
                continue

        # resize and cache
        resized = resize_and_cache_image(src, dataset_name)
        if not resized:
            skipped += 1
            continue

        answer_type = item.get("answer_type")
        report = item.get("report")
        rec = build_record(dataset_name, q, a, resized, split_tag, idx, answer_type, report)
        records.append(rec)

    print(f"[{dataset_name}:{split_tag}] processed {len(items)} rows -> kept {len(records)}, skipped {skipped}")
    return records

# -------------------------
# Main
# -------------------------
def main(args):
    ensure_dir(RESIZED_ROOT)
    ensure_dir(args.out_dir)

    all_train_records = []
    all_test_records = []

    # ----- VQA-RAD -----
    if args.vqarad_train:
        vq_roots = [args.vqarad_img_root]
        tr = process_dataset_file(args.vqarad_train, "vqarad", vq_roots, "train")
        all_train_records.extend(tr)
    if args.vqarad_test:
        vq_roots = [args.vqarad_img_root]
        te = process_dataset_file(args.vqarad_test, "vqarad", vq_roots, "test")
        all_test_records.extend(te)

    # ----- SLAKE -----
    if args.slake_train:
        sl_roots = [args.slake_img_root]
        tr = process_dataset_file(args.slake_train, "slake", sl_roots, "train")
        all_train_records.extend(tr)
    if args.slake_test:
        sl_roots = [args.slake_img_root]
        te = process_dataset_file(args.slake_test, "slake", sl_roots, "test")
        all_test_records.extend(te)

    # ----- PathVQA -----
    path_roots = [args.pathvqa_img_root_train, args.pathvqa_img_root_val, args.pathvqa_img_root_test]
    if args.pathvqa_train:
        tr = process_dataset_file(args.pathvqa_train, "pathvqa", path_roots, "train")
        all_train_records.extend(tr)
    if args.pathvqa_test:
        te = process_dataset_file(args.pathvqa_test, "pathvqa", path_roots, "test")
        all_test_records.extend(te)

    # Save parquet outputs
    ensure_dir(args.out_dir)
    if all_train_records:
        ds_train = Dataset.from_list(all_train_records)
        train_parquet = os.path.join(args.out_dir, "train_medreport_gpt2901.parquet")
        ds_train.to_parquet(train_parquet)
        print(f"Saved train parquet: {train_parquet} ({len(all_train_records)} rows)")
    else:
        print("No train records produced.")

    if all_test_records:
        ds_test = Dataset.from_list(all_test_records)
        test_parquet = os.path.join(args.out_dir, "test.parquet")
        ds_test.to_parquet(test_parquet)
        print(f"Saved test parquet: {test_parquet} ({len(all_test_records)} rows)")
    else:
        print("No test records produced.")

    print("Done.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build Verl parquet from refined JSONs")
    p.add_argument("--vqarad-train", default=DEFAULTS["vqarad_train"], help="VQA-RAD train JSON (refined)")
    p.add_argument("--vqarad-test", default=DEFAULTS["vqarad_test"], help="VQA-RAD test JSON (optional)")
    p.add_argument("--vqarad-img-root", default=DEFAULTS["vqarad_img_root"], help="VQA-RAD image folder")

    p.add_argument("--slake-train", default=DEFAULTS["slake_train"], help="SLAKE train JSON (refined)")
    p.add_argument("--slake-test", default=DEFAULTS["slake_test"], help="SLAKE test JSON (optional)")
    p.add_argument("--slake-img-root", default=DEFAULTS["slake_img_root"], help="SLAKE image root")

    p.add_argument("--pathvqa-train", default=DEFAULTS["pathvqa_train"], help="PathVQA train JSON (refined)")
    p.add_argument("--pathvqa-test", default=DEFAULTS["pathvqa_test"], help="PathVQA test JSON (optional)")
    p.add_argument("--pathvqa-img-root-train", default=DEFAULTS["pathvqa_img_root_train"], help="PathVQA train images root")
    p.add_argument("--pathvqa-img-root-val", default=DEFAULTS["pathvqa_img_root_val"], help="PathVQA val images root")
    p.add_argument("--pathvqa-img-root-test", default=DEFAULTS["pathvqa_img_root_test"], help="PathVQA test images root")

    p.add_argument("--resized-root", default=RESIZED_ROOT, help="Where to put resized images")
    p.add_argument("--out-dir", default=OUT_DIR, help="Output dir for parquet files")

    args = p.parse_args()

    # map args names
    RESIZED_ROOT = args.resized_root
    DEFAULTS["vqarad_img_root"] = args.vqarad_img_root
    # call main with constructed roots
    # Build args object containing unified names used in main()
    class A: pass
    A.vqarad_train = args.vqarad_train
    A.vqarad_test = args.vqarad_test
    A.vqarad_img_root = args.vqarad_img_root

    A.slake_train = args.slake_train
    A.slake_test = args.slake_test
    A.slake_img_root = args.slake_img_root

    A.pathvqa_train = args.pathvqa_train
    A.pathvqa_test = args.pathvqa_test
    A.pathvqa_img_root_train = args.pathvqa_img_root_train
    A.pathvqa_img_root_val = args.pathvqa_img_root_val
    A.pathvqa_img_root_test = args.pathvqa_img_root_test

    A.resized_root = args.resized_root
    A.out_dir = args.out_dir

    # set globals used in functions
    RESIZED_ROOT = args.resized_root

    # run
    main(A)

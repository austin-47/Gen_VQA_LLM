#!/usr/bin/env python3
"""
Generate medical image descriptions for a JSON created by convert_json_format.py.

Usage:
  python generate_description.py --input formatted.json --output with_reports.json

- Adds a "report" field to every JSON entry.
- Uses a cache to allow resuming.
"""

import os
import json
import time
from tqdm import tqdm
import PIL
import argparse

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# ---------------- CONFIG ----------------
MODEL_ID = "lingshu-medical-mllm/Lingshu-32B"
SLEEP_BETWEEN = 0.05
MAX_NEW_TOKENS = 512
# -----------------------------------------

print("Loading processor and vLLM model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

llm = LLM(
    model=MODEL_ID,
    limit_mm_per_prompt={"image": 4},
    tensor_parallel_size=2,
    enforce_eager=True,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=1.0,
    repetition_penalty=1.0,
    max_tokens=MAX_NEW_TOKENS,
)
print("Model loaded successfully.")


def generate_report_vllm(img_path: str) -> str:
    """Generate a description for the image using vLLM."""
    try:
        image = PIL.Image.open(img_path).convert("RGB")

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        mm_data = {"image": image_inputs}

        processed_input = {"prompt": prompt, "multi_modal_data": mm_data}
        outputs = llm.generate([processed_input], sampling_params=sampling_params)
        report = outputs[0].outputs[0].text.strip()
        return report

    except Exception as e:
        print(f"[ERROR] Report generation failed for {img_path}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file with reports")
    args = parser.parse_args()

    # Load input JSON
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    cache_file = args.output.replace(".json", "_cache.json")
    report_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            report_cache = json.load(f)
        print(f"Loaded cache with {len(report_cache)} reports from {cache_file}")

    # Collect unique images
    unique_images = {item["image_name"] for item in data if "image_name" in item}
    print(f"Found {len(unique_images)} unique images.")

    for img in tqdm(unique_images, desc="Generating reports"):
        if img in report_cache and report_cache[img]:
            continue

        if not os.path.isfile(img):
            print(f"[WARN] Image file does not exist: {img}")
            report_cache[img] = ""
            continue

        report_cache[img] = generate_report_vllm(img)
        time.sleep(SLEEP_BETWEEN)

        # Periodic cache save
        if len(report_cache) % 10 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(report_cache, f, ensure_ascii=False, indent=2)

    # Final cache save
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(report_cache, f, ensure_ascii=False, indent=2)
    print(f"Cache saved with {len(report_cache)} reports → {cache_file}")

    # Attach reports to JSON
    for item in data:
        img = item.get("image_name")
        item["report"] = report_cache.get(img, "")

    # Save output JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items with reports → {args.output}")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Generate medical reports for each unique image in VQARAD, SLAKE, and PathVQA,
# and add a "report" field to every JSON entry.

# - Saves new dataset JSONs with "report" added.
# - Uses a per-dataset cache file to allow resuming.

# Outputs:
#   - VQARAD_train_refine_report.json
#   - SLAKE_train_refine_report.json
#   - PathVQA_train_refine_report.json
# """

# import os
# import json
# import time
# from pathlib import Path
# from tqdm import tqdm
# import torch

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

# # ---------------- CONFIG ----------------
# DATASETS = [
#     {
#         "name": "VQARAD",
#         "input_json": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_v2.json",
#         "output_json": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_report_describe.json",
#         "image_roots": ["/home/work/austin/Dataset/resized_224/vqarad"],
#     },
#     {
#         "name": "SLAKE",
#         "input_json": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_v2_en.json",
#         "output_json": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_report_describe.json",
#         "image_roots": ["/home/work/austin/Dataset/resized_224/slake/imgs"],
#     },
#     {
#         "name": "PathVQA",
#         "input_json": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine.json",
#         "output_json": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine_report_dsecribe.json",
#         "image_roots": ["/home/work/austin/Dataset/resized_224/pathvqa"],
#     },
# ]

# MODEL_ID = "lingshu-medical-mllm/Lingshu-32B"
# MAX_NEW_TOKENS = 512
# SLEEP_BETWEEN = 0.1  # pause between generations
# # -----------------------------------------

# # Load model and processor
# print("Loading model...")
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(MODEL_ID)
# device = model.device
# print("Model loaded on", device)


# def resolve_image_path(img_name: str, roots: list) -> str | None:
#     """
#     Try to resolve an image filename/relative path against dataset roots.
#     """
#     if not img_name:
#         return None

#     if os.path.isabs(img_name) and os.path.isfile(img_name):
#         return img_name

#     for root in roots:
#         cand = os.path.join(root, img_name)
#         if os.path.isfile(cand):
#             return cand

#         cand2 = os.path.join(root, os.path.basename(img_name))
#         if os.path.isfile(cand2):
#             return cand2

#         for ext in [".jpg", ".jpeg", ".png"]:
#             cand3 = os.path.join(root, os.path.splitext(img_name)[0] + ext)
#             if os.path.isfile(cand3):
#                 return cand3

#     return None


# def generate_report(img_path: str) -> str:
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": img_path},
#                 {
#                     "type": "text",
#                     "text": "Describe this image in detail.",
#                 },
#             ],
#         }
#     ]

#     try:
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         ).to(device)

#         with torch.no_grad():
#             generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

#         trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
#         decoded = processor.batch_decode(
#             trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         return decoded[0] if decoded else ""
#     except Exception as e:
#         print(f"[ERROR] Report generation failed for {img_path}: {e}")
#         return ""


# def process_dataset(cfg: dict):
#     print(f"\n=== Processing {cfg['name']} ===")
#     with open(cfg["input_json"], "r", encoding="utf-8") as f:
#         data = json.load(f)

#     cache_file = cfg["output_json"].replace(".json", "_cache.json")

#     # Load cache if exists
#     if os.path.exists(cache_file):
#         with open(cache_file, "r", encoding="utf-8") as f:
#             report_cache = json.load(f)
#         print(f"Loaded cache with {len(report_cache)} reports from {cache_file}")
#     else:
#         report_cache = {}

#     unique_images = {item["images"] for item in data if "images" in item}
#     print(f"Found {len(unique_images)} unique images.")

#     for img in tqdm(unique_images, desc="Generating reports"):
#         if img in report_cache and report_cache[img]:
#             continue  # already cached

#         resolved = resolve_image_path(img, cfg["image_roots"])
#         if not resolved:
#             print(f"[WARN] Could not resolve: {img}")
#             report_cache[img] = ""
#             continue

#         report = generate_report(resolved)
#         report_cache[img] = report
#         time.sleep(SLEEP_BETWEEN)

#         # Save cache periodically
#         if len(report_cache) % 10 == 0:
#             with open(cache_file, "w", encoding="utf-8") as f:
#                 json.dump(report_cache, f, ensure_ascii=False, indent=2)

#     # Final cache save
#     with open(cache_file, "w", encoding="utf-8") as f:
#         json.dump(report_cache, f, ensure_ascii=False, indent=2)
#     print(f"Cache saved with {len(report_cache)} reports → {cache_file}")

#     # Attach reports to dataset
#     for item in data:
#         img = item.get("images")
#         item["report"] = report_cache.get(img, "")

#     # Save final dataset
#     with open(cfg["output_json"], "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     print(f"Saved {len(data)} items with reports → {cfg['output_json']}")


# def main():
#     for cfg in DATASETS:
#         process_dataset(cfg)


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# Generate medical reports for each unique image in VQARAD, SLAKE, and PathVQA using vLLM.

# - Adds a "report" field to every JSON entry.
# - Uses a per-dataset cache file to allow resuming.

# Outputs:
#   - *_train_refine_report_describe.json
# """

# import os
# import json
# import time
# from pathlib import Path
# from tqdm import tqdm
# import PIL

# from transformers import AutoProcessor
# from qwen_vl_utils import process_vision_info
# from vllm import LLM, SamplingParams

# # ---------------- CONFIG ----------------
# DATASETS = [
#     {
#         "name": "VQARAD",
#         "input_json": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_v2.json",
#         "output_json": "/home/work/austin/MedCCO/preprocess/VQARAD_grok_report.json",
#         "image_roots": ["/home/work/austin/Dataset/vqa-rad/VQA_RAD Image Folder"],
#     },
#     {
#         "name": "SLAKE",
#         "input_json": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_v2_en.json",
#         "output_json": "/home/work/austin/MedCCO/preprocess/SLAKE_grok_report.json",
#         "image_roots": ["/home/work/austin/Dataset/SLAKE/imgs"],
#     },
#     {
#         "name": "PathVQA",
#         "input_json": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine.json",
#         "output_json": "/home/work/austin/MedCCO/preprocess/PathVQA_grok_report.json",
#         "image_roots": ["/home/work/austin/Dataset/path-vqa/pvqa/images"],
#     },
# ]

# MODEL_ID = "lingshu-medical-mllm/Lingshu-32B"
# SLEEP_BETWEEN = 0.05  # small pause between generations
# MAX_NEW_TOKENS = 512
# # -----------------------------------------

# print("Loading processor and vLLM model...")
# processor = AutoProcessor.from_pretrained(MODEL_ID)

# llm = LLM(
#     model=MODEL_ID,
#     limit_mm_per_prompt={"image": 4},
#     tensor_parallel_size=2,
#     enforce_eager=True,
#     trust_remote_code=True,
#     gpu_memory_utilization=1.0
# )

# sampling_params = SamplingParams(
#     temperature=0.7,
#     top_p=1.0,
#     repetition_penalty=1.0,
#     max_tokens=MAX_NEW_TOKENS,
# )
# print("Model loaded successfully.")


# def resolve_image_path(img_name: str, roots: list) -> str | None:
#     """Try to resolve an image filename/relative path against dataset roots."""
#     if not img_name:
#         return None

#     if os.path.isabs(img_name) and os.path.isfile(img_name):
#         return img_name

#     for root in roots:
#         cand = os.path.join(root, img_name)
#         if os.path.isfile(cand):
#             return cand

#         cand2 = os.path.join(root, os.path.basename(img_name))
#         if os.path.isfile(cand2):
#             return cand2

#         for ext in [".jpg", ".jpeg", ".png"]:
#             cand3 = os.path.join(root, os.path.splitext(img_name)[0] + ext)
#             if os.path.isfile(cand3):
#                 return cand3

#     return None


# def generate_report_vllm(img_path: str) -> str:
#     """Generate report for a given image using vLLM."""
#     try:
#         image = PIL.Image.open(img_path).convert("RGB")

#         # message = [
#         #     {
#         #         "role": "user",
#         #         "content": [
#         #             {"type": "image", "image": image},
#         #             {"type": "text", "text": "Describe this image in detail."},
#         #         ],
#         #     }
#         # ]

#         message = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {"type": "text", "text": "Carefully examine this medical image. Focus on anatomical structures, abnormalities, textures, contrasts, measurements, and any artifacts. Use precise medical terminology and describe only what is visibly present, avoiding assumptions or unsubstantiated details. Provide a detailed description."},
#                 ],
#             }
#         ]

#         prompt = processor.apply_chat_template(
#             message, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(message)
#         mm_data = {"image": image_inputs}

#         processed_input = {
#             "prompt": prompt,
#             "multi_modal_data": mm_data,
#         }

#         outputs = llm.generate([processed_input], sampling_params=sampling_params)
#         report = outputs[0].outputs[0].text.strip()
#         return report

#     except Exception as e:
#         print(f"[ERROR] Report generation failed for {img_path}: {e}")
#         return ""


# def process_dataset(cfg: dict):
#     print(f"\n=== Processing {cfg['name']} ===")
#     with open(cfg["input_json"], "r", encoding="utf-8") as f:
#         data = json.load(f)

#     cache_file = cfg["output_json"].replace(".json", "_cache.json")

#     # Load or initialize cache
#     if os.path.exists(cache_file):
#         with open(cache_file, "r", encoding="utf-8") as f:
#             report_cache = json.load(f)
#         print(f"Loaded cache with {len(report_cache)} reports from {cache_file}")
#     else:
#         report_cache = {}

#     # Get unique images
#     unique_images = {item["images"] for item in data if "images" in item}
#     print(f"Found {len(unique_images)} unique images.")

#     # Generate reports
#     for img in tqdm(unique_images, desc="Generating reports"):
#         if img in report_cache and report_cache[img]:
#             continue  # already cached

#         resolved = resolve_image_path(img, cfg["image_roots"])
#         if not resolved:
#             print(f"[WARN] Could not resolve: {img}")
#             report_cache[img] = ""
#             continue

#         report = generate_report_vllm(resolved)
#         report_cache[img] = report
#         time.sleep(SLEEP_BETWEEN)

#         # Periodic cache save
#         if len(report_cache) % 10 == 0:
#             with open(cache_file, "w", encoding="utf-8") as f:
#                 json.dump(report_cache, f, ensure_ascii=False, indent=2)

#     # Final cache save
#     with open(cache_file, "w", encoding="utf-8") as f:
#         json.dump(report_cache, f, ensure_ascii=False, indent=2)
#     print(f"Cache saved with {len(report_cache)} reports → {cache_file}")

#     # Attach reports to dataset
#     for item in data:
#         img = item.get("images")
#         item["report"] = report_cache.get(img, "")

#     # Save final dataset
#     with open(cfg["output_json"], "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     print(f"Saved {len(data)} items with reports → {cfg['output_json']}")


# def main():
#     for cfg in DATASETS:
#         process_dataset(cfg)


# if __name__ == "__main__":
#     main()


###################################GENERATE FOR ALL###########################################

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import PIL

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# ---------------- CONFIG ----------------
DATASETS = [
    {
        "name": "VQARAD",
        "input_json": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_v2.json",
        "output_json": "/home/work/austin/MedCCO/preprocess/VQARAD_report_all.json",
        "image_roots": ["/home/work/austin/Dataset/vqa-rad/VQA_RAD Image Folder"],
    },
    {
        "name": "SLAKE",
        "input_json": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_v2_en.json",
        "output_json": "/home/work/austin/MedCCO/preprocess/SLAKE_report_all.json",
        "image_roots": ["/home/work/austin/Dataset/SLAKE/imgs"],
    },
    {
        "name": "PathVQA",
        "input_json": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine.json",
        "output_json": "/home/work/austin/MedCCO/preprocess/PathVQA_report_all.json",
        "image_roots": ["/home/work/austin/Dataset/path-vqa/pvqa/images"],
    },
]

MODEL_ID = "lingshu-medical-mllm/Lingshu-32B"
SLEEP_BETWEEN = 0.05  # small pause between generations
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
    gpu_memory_utilization=1.0
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=1.0,
    repetition_penalty=1.0,
    max_tokens=MAX_NEW_TOKENS,
)
print("Model loaded successfully.")


def resolve_image_path(img_name: str, roots: list) -> str | None:
    """Try to resolve an image filename/relative path against dataset roots."""
    if not img_name:
        return None

    if os.path.isabs(img_name) and os.path.isfile(img_name):
        return img_name

    for root in roots:
        cand = os.path.join(root, img_name)
        if os.path.isfile(cand):
            return cand

        cand2 = os.path.join(root, os.path.basename(img_name))
        if os.path.isfile(cand2):
            return cand2

        for ext in [".jpg", ".jpeg", ".png"]:
            cand3 = os.path.join(root, os.path.splitext(img_name)[0] + ext)
            if os.path.isfile(cand3):
                return cand3

    return None

###########for PATHVQA#################
# def resolve_image_path(img_name: str, roots: list) -> str | None:
#     """Try to resolve an image filename/relative path against dataset roots."""
#     if not img_name:
#         return None

#     # If absolute path works, return it
#     if os.path.isabs(img_name) and os.path.isfile(img_name):
#         return img_name

#     # Try exact match and common extensions
#     possible_names = [img_name]
#     base_name = os.path.splitext(os.path.basename(img_name))[0]
#     for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
#         possible_names.append(base_name + ext)

#     # Search through roots
#     for root in roots:
#         # 1️⃣ Direct path check (root/img_name)
#         for name in possible_names:
#             cand = os.path.join(root, name)
#             if os.path.isfile(cand):
#                 return cand

#         # 2️⃣ Recursively search all subdirectories
#         for dirpath, _, filenames in os.walk(root):
#             for fname in filenames:
#                 if fname.lower() in [n.lower() for n in possible_names]:
#                     return os.path.join(dirpath, fname)

#     return None


def generate_report_vllm(img_path: str) -> str:
    """Generate report for a given image using vLLM."""
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

        processed_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = llm.generate([processed_input], sampling_params=sampling_params)
        report = outputs[0].outputs[0].text.strip()
        return report

    except Exception as e:
        print(f"[ERROR] Report generation failed for {img_path}: {e}")
        return ""


def process_dataset(cfg: dict):
    print(f"\n=== Processing {cfg['name']} ===")
    with open(cfg["input_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    cache_file = cfg["output_json"].replace(".json", "_cache.json")

    # Load or initialize cache
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            report_cache = json.load(f)
        print(f"Loaded cache with {len(report_cache)} reports from {cache_file}")
    else:
        report_cache = {}

    print(f"Found {len(data)} total items.")

    # Generate reports for every item (even if images repeat)
    for idx, item in enumerate(tqdm(data, desc="Generating reports")):
        img = item.get("images")
        if not img:
            continue

        resolved = resolve_image_path(img, cfg["image_roots"])
        if not resolved:
            print(f"[WARN] Could not resolve: {img}")
            item["report"] = ""
            continue

        # Optional: use cache to skip already processed images
        if img in report_cache and report_cache[img]:
            item["report"] = report_cache[img]
            continue

        # Generate a new report for this item (even if the image repeats)
        report = generate_report_vllm(resolved)
        item["report"] = report
        report_cache[img] = report  # still cache in case you want to reuse

        time.sleep(SLEEP_BETWEEN)

        # Periodically save cache
        if idx % 10 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(report_cache, f, ensure_ascii=False, indent=2)

    # Final cache save
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(report_cache, f, ensure_ascii=False, indent=2)
    print(f"Cache saved with {len(report_cache)} reports → {cache_file}")

    # Save final dataset
    with open(cfg["output_json"], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items with reports → {cfg['output_json']}")



def main():
    for cfg in DATASETS:
        process_dataset(cfg)


if __name__ == "__main__":
    main()
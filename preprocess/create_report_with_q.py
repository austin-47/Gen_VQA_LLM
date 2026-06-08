import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import PIL
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import re
# ---------------- CONFIG ----------------
DATASETS = [
    {
        "name": "VQARAD",
        "input_json": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_v2.json",
        "output_json": "/home/work/austin/MedCCO/preprocess/VQARAD_report_wq_roi.json",
        "image_roots": ["/home/work/austin/Dataset/vqa-rad/VQA_RAD Image Folder"],
    },
    {
        "name": "SLAKE",
        "input_json": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_v2_en.json",
        "output_json": "/home/work/austin/MedCCO/preprocess/SLAKE_report_wq_roi.json",
        "image_roots": ["/home/work/austin/Dataset/SLAKE/imgs"],
    },
    {
        "name": "PathVQA",
        "input_json": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine.json",
        "output_json": "/home/work/austin/MedCCO/preprocess/PathVQA_report_wq_roi.json",
        "image_roots": ["/home/work/austin/Dataset/path-vqa/pvqa/images"],
    },
]

MODEL_ID = "lingshu-medical-mllm/Lingshu-32B"
SLEEP_BETWEEN = 0.05
MAX_NEW_TOKENS = 512

# ---------------- PROMPT TEMPLATE ----------------
# INSTRUCTION_TEMPLATE = """

# 1. First, carefully analyze the image, provide the detailed description of the image.
# 2. Then identify **key regions or structures** relevant to the question. Further provide a detailed **ROI-focused description** in a way that supports reasoning for answering the question. Do not provie the answer.

# Question:
# {question}
# """
INSTRUCTION_TEMPLATE = """
Carefully analyze the image and the question, identify **key regions or structures** relevant to the question. Provide a detailed **ROI-focused description** in a way that supports reasoning for answering the question. Do not provie the answer.
Do NOT mention “clinical correlation,” “follow-up,” “further,” or recommendations.


Question:
{question}
"""
# ---------------- LOAD MODEL ----------------
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

# ---------------- HELPERS ----------------
# def resolve_image_path(img_name: str, roots: list) -> str | None:
#     """Try to resolve an image filename/relative path against dataset roots."""
#     if not img_name:
#         return None
#     if os.path.isabs(img_name) and os.path.isfile(img_name):
#         return img_name
#     for root in roots:
#         # Exact file
#         cand = os.path.join(root, img_name)
#         if os.path.isfile(cand):
#             return cand
#         # Basename with extensions
#         base_name = os.path.splitext(os.path.basename(img_name))[0]
#         for ext in [".jpg", ".jpeg", ".png"]:
#             cand2 = os.path.join(root, base_name + ext)
#             if os.path.isfile(cand2):
#                 return cand2
#     return None

###########for PATHVQA#################
def resolve_image_path(img_name: str, roots: list) -> str | None:
    """Try to resolve an image filename/relative path against dataset roots."""
    if not img_name:
        return None

    # If absolute path works, return it
    if os.path.isabs(img_name) and os.path.isfile(img_name):
        return img_name

    # Try exact match and common extensions
    possible_names = [img_name]
    base_name = os.path.splitext(os.path.basename(img_name))[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        possible_names.append(base_name + ext)

    # Search through roots
    for root in roots:
        # 1️⃣ Direct path check (root/img_name)
        for name in possible_names:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand

        # 2️⃣ Recursively search all subdirectories
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower() in [n.lower() for n in possible_names]:
                    return os.path.join(dirpath, fname)

    return None


def normalize_question(q: str) -> str:
    if q is None:
        return ""
    # collapse multiple spaces and trim
    s = re.sub(r'\s+', ' ', str(q)).strip()
    # optionally truncate to avoid extremely long cache keys, but keep enough text:
    return s[:1000]

def extract_question_from_item(item: dict) -> str:
    """Prefer new_q, otherwise find first user question from conversations (supports different keys)."""
    nq = item.get("new_q")
    if nq and str(nq).strip():
        return normalize_question(nq)

    convs = item.get("conversations") or item.get("conversation") or item.get("convos")
    if isinstance(convs, list) and len(convs) > 0:
        # pick first element that looks like user
        first = convs[0]
        if isinstance(first, dict):
            # many variants: 'value' field, 'content' field, possibly 'from'/'role' indicates speaker
            val = None
            for k in ("value", "content", "text"):
                if k in first and first[k]:
                    val = first[k]
                    break
            if val:
                return normalize_question(val)
        # fallback: try to join values
        for entry in convs:
            if isinstance(entry, dict):
                for k in ("value", "content", "text"):
                    if k in entry and entry[k]:
                        return normalize_question(entry[k])
    # ultimate fallback to 'question' field or generic prompt
    q = item.get("question") or item.get("ori_q") or item.get("prompt") or ""
    return normalize_question(q)

def generate_report_vllm(img_path: str, question: str) -> str:
    """Generate report for a given image using vLLM and question."""
    try:
        image = PIL.Image.open(img_path).convert("RGB")
        prompt_text = INSTRUCTION_TEMPLATE.format(question=question or "Describe the image in detail.")
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
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

# ---------------- PROCESS DATASET ----------------
def process_dataset(cfg: dict):
    print(f"\n=== Processing {cfg['name']} ===")
    with open(cfg["input_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    cache_file = cfg["output_json"].replace(".json", "_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            report_cache = json.load(f)
        print(f"Loaded cache with {len(report_cache)} reports from {cache_file}")
    else:
        report_cache = {}

    print(f"Found {len(data)} total items.")

    for idx, item in enumerate(tqdm(data, desc="Generating reports")):
        # accept either 'images' (list) or 'image' (single)
        img_list = item.get("images") or item.get("image") or ""
        if not img_list:
            # nothing to do
            item["report"] = ""
            continue
        if isinstance(img_list, list):
            img_name = img_list[0]
        else:
            img_name = img_list

        resolved = resolve_image_path(img_name, cfg["image_roots"])
        if not resolved:
            print(f"[WARN] Could not resolve: {img_name}")
            item["report"] = ""
            continue

        # Determine question robustly
        question = extract_question_from_item(item)
        # build cache key by image name + question (so each distinct QA pair is cached separately)
        key = f"{img_name}||{question}"

        # Use cached report if available for this (image, question) pair
        if key in report_cache and report_cache[key]:
            item["report"] = report_cache[key]
            continue

        # Generate new report for this (image, question)
        report = generate_report_vllm(resolved, question)
        item["report"] = report
        report_cache[key] = report

        time.sleep(SLEEP_BETWEEN)

        # Periodically save cache
        if idx % 10 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(report_cache, f, ensure_ascii=False, indent=2)

    # Final cache save
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(report_cache, f, ensure_ascii=False, indent=2)
    print(f"Cache saved with {len(report_cache)} reports → {cache_file}")

    # Save final dataset (original items updated with "report")
    with open(cfg["output_json"], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items with reports → {cfg['output_json']}")

# ---------------- MAIN ----------------
def main():
    for cfg in DATASETS:
        process_dataset(cfg)

if __name__ == "__main__":
    main()
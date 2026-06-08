import os
import json
import time
import re
from tqdm import tqdm
import PIL

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# ===================== CONFIG =====================
DATASETS = [
    {
        "name": "VQARAD",
        "input_json": "/home/work/austin/MedCCO/preprocess/VQARAD_train_refine_v2.json",
        "output_json": "/home/work/austin/MedCCO/preprocess/VQARAD_report_wqa_roi_new.json",
        "image_roots": ["/home/work/austin/Dataset/vqa-rad/VQA_RAD Image Folder"],
    },
    # {
    #     "name": "SLAKE",
    #     "input_json": "/home/work/austin/MedCCO/preprocess/SLAKE_train_refine_v2_en.json",
    #     "output_json": "/home/work/austin/MedCCO/preprocess/SLAKE_report_wqa_roi.json",
    #     "image_roots": ["/home/work/austin/Dataset/SLAKE/imgs"],
    # },
    # {
    #     "name": "PathVQA",
    #     "input_json": "/home/work/austin/MedCCO/preprocess/PathVQA_train_refine.json",
    #     "output_json": "/home/work/austin/MedCCO/preprocess/PathVQA_report_wqa_roi.json",
    #     "image_roots": ["/home/work/austin/Dataset/path-vqa/pvqa/images"],
    # },
]

# MODEL_ID = "lingshu-medical-mllm/Lingshu-32B"
MODEL_ID = "/home/work/austin/MedCCO/llms/Lingshu-32B"
SLEEP_BETWEEN = 0.05
MAX_NEW_TOKENS = 512

# ===================== PROMPT =====================
INSTRUCTION_TEMPLATE = """
Carefully analyze the image and describe it.
Identify **key regions or structures** relevant to the question–answer pair, provide a detailed **ROI-focused description** that supports reasoning for the given answer.
Do NOT provide the answer itself.
Do NOT mention “clinical correlation,” “follow-up,” “further,” or recommendations.

Question:
{question}

Answer:
{answer}
"""

# ===================== LOAD MODEL =====================
print("Loading processor and vLLM model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

llm = LLM(
    model=MODEL_ID,
    limit_mm_per_prompt={"image": 4},
    tensor_parallel_size=2,
    enforce_eager=True,
    trust_remote_code=True,
    gpu_memory_utilization=1.0,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=1.0,
    repetition_penalty=1.0,
    max_tokens=MAX_NEW_TOKENS,
)

print("Model loaded successfully.")

# ===================== HELPERS =====================
def resolve_image_path(img_name: str, roots: list) -> str | None:
    if not img_name:
        return None

    if os.path.isabs(img_name) and os.path.isfile(img_name):
        return img_name

    base = os.path.splitext(os.path.basename(img_name))[0]
    possible = [img_name] + [base + ext for ext in [".jpg", ".jpeg", ".png", ".bmp"]]

    for root in roots:
        for name in possible:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower() in [n.lower() for n in possible]:
                    return os.path.join(dirpath, fname)

    return None


def normalize_text(x: str, max_len=1000) -> str:
    if not x:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()[:max_len]


def extract_question_from_item(item: dict) -> str:
    if item.get("new_q"):
        return normalize_text(item["new_q"])

    convs = item.get("conversations") or item.get("conversation") or []
    for entry in convs:
        if isinstance(entry, dict):
            if entry.get("from") in ("user", "human"):
                return normalize_text(entry.get("value") or entry.get("content"))

    return normalize_text(item.get("question") or item.get("ori_q") or "")


def extract_answer_from_item(item: dict) -> str:
    if item.get("new_a"):
        return normalize_text(item["new_a"], 500)

   # 2️⃣ Direct answer field (VQARAD)
    if item.get("answer"):
        return normalize_text(item["answer"], 500)

    # 3️⃣ Conversation-based extraction (robust)
    convs = item.get("conversations") or item.get("conversation") or []
    for entry in convs:
        if not isinstance(entry, dict):
            continue

        role = entry.get("role") or entry.get("from")
        if role in ("assistant", "gpt", "model"):
            val = entry.get("value") or entry.get("content") or entry.get("text")
            if val:
                return normalize_text(val, 500)

    return ""


def generate_report_vllm(img_path: str, question: str, answer: str) -> str:
    try:
        image = PIL.Image.open(img_path).convert("RGB")
        prompt_text = INSTRUCTION_TEMPLATE.format(
            question=question or "Describe the image.",
            answer=answer or "Not provided."
        )

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

        image_inputs, _ = process_vision_info(message)

        outputs = llm.generate(
            [{
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            }],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text.strip()

    except Exception as e:
        print(f"[ERROR] Generation failed for {img_path}: {e}")
        return ""

# ===================== DATASET PROCESS =====================
def process_dataset(cfg: dict):
    print(f"\n=== Processing {cfg['name']} ===")

    with open(cfg["input_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    cache_file = cfg["output_json"].replace(".json", "_cache.json")
    report_cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}

    for idx, item in enumerate(tqdm(data, desc="Generating reports")):

        # 🚫 Skip dropped samples
        if item.get("status") == "drop":
            item["report"] = ""
            continue

        img = item.get("images") or item.get("image")
        img_name = img[0] if isinstance(img, list) else img

        if not img_name:
            item["report"] = ""
            continue

        resolved = resolve_image_path(img_name, cfg["image_roots"])
        if not resolved:
            print(f"[WARN] Could not resolve: {img_name}")
            item["report"] = ""
            continue

        question = extract_question_from_item(item)
        answer = extract_answer_from_item(item)

        key = f"{img_name}||{question}||{answer}"
        if key in report_cache:
            item["report"] = report_cache[key]
            continue

        report = generate_report_vllm(resolved, question, answer)
        item["report"] = report
        report_cache[key] = report

        time.sleep(SLEEP_BETWEEN)

        if idx % 10 == 0:
            json.dump(report_cache, open(cache_file, "w"), ensure_ascii=False, indent=2)

    json.dump(report_cache, open(cache_file, "w"), ensure_ascii=False, indent=2)
    json.dump(data, open(cfg["output_json"], "w"), ensure_ascii=False, indent=2)

    print(f"Saved → {cfg['output_json']}")

# ===================== MAIN =====================
def main():
    for cfg in DATASETS:
        process_dataset(cfg)

if __name__ == "__main__":
    main()

# from vllm import LLM, SamplingParams
# from qwen_vl_utils import process_vision_info
# import PIL
# from transformers import AutoProcessor
# llm = LLM(
#     model=MODEL_ID,
#     limit_mm_per_prompt={"image": 4},
#     tensor_parallel_size=2,
#     enforce_eager=True,
#     trust_remote_code=True,
#     gpu_memory_utilization=1.0,
# )

# sampling_params = SamplingParams(
#     temperature=0.7,
#     top_p=1.0,
#     repetition_penalty=1.0,
#     max_tokens=MAX_NEW_TOKENS,
# )

# processor = AutoProcessor.from_pretrained("/home/work/austin/MedCCO/llms/Lingshu-32B")
# llm = LLM(model="/home/work/austin/MedCCO/llms/Lingshu-32B", limit_mm_per_prompt = {"image": 4}, tensor_parallel_size=2, enforce_eager=True, trust_remote_code=True,gpu_memory_utilization=1.0)
# sampling_params = SamplingParams(
#             temperature=0.7,
#             top_p=1,
#             repetition_penalty=1,
#             max_tokens=512,
#             stop_token_ids=[],
#         )

# text = "Which lobe is normal? Carefully analyze the image and the question, produce a detailed roi-focused description in <report> </report> tags. .Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Output your detailed, step-by-step reasoning based on the image and description, wrap this in <think> </think> tags. Finally, base on your reasoning, provide the answer to the question in <answer> </answer> tags. The output answer format should be as follows: <report> description here </report> <think> reasoning process here </think><answer> answer here </answer>. Please strictly follow the format."
# image_path = "/home/work/austin/MedCCO/MedicalDataset/Medical_Multimodal_Evaluation_Data/images/xmlab483_source.jpg"
# image = PIL.Image.open(image_path)

# message = [
#     {
#         "role":"user",
#         "content":[
#             {"type":"image","image":image},
#             {"type":"text","text":text}
#             ]
#             }
# ]
# prompt = processor.apply_chat_template(
#     message,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# image_inputs, video_inputs = process_vision_info(message)
# mm_data = {}
# mm_data["image"] = image_inputs
# processed_input = {
#   "prompt": prompt,
#   "multi_modal_data": mm_data,
# }

# outputs = llm.generate([processed_input], sampling_params=sampling_params)
# print(outputs[0].outputs[0].text)

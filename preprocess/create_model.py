import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# Config: change paths here
# -------------------------------
STAGE2_CKPT_DIR = "/home/work/austin/MedCCO/verl/checkpoints/austin_grpo_medVQA/medreport_stage1_Qwen2.5-VL-3B-Instruct/global_step_100/actor"
HF_OUTPUT_DIR = "/home/work/austin/MedCCO/verl/llms/Qwen2.5-VL-3B-Instruct-Stage2"

BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # original pretrained model

# -------------------------------
# Step 1: Load pretrained base model
# -------------------------------
print("[1/4] Loading base Hugging Face model...")
config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, config=config)

# -------------------------------
# Step 2: Load stage 2 checkpoint
# -------------------------------
print("[2/4] Loading stage 2 checkpoint...")
def load_stage2_weights(model, ckpt_dir):
    """
    Loads the Verl checkpoint weights into HF model.
    Assumes tensor_model_parallel_size=1 for simplicity.
    """
    shard_file = os.path.join(ckpt_dir, "model_world_size_1_rank_0.pt")
    if not os.path.exists(shard_file):
        # fallback: maybe rank_1
        shard_file = os.path.join(ckpt_dir, "model_world_size_2_rank_1.pt")
    state_dict = torch.load(shard_file, map_location="cpu")
    
    # If wrapped in 'module' or 'model', extract
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "actor" in state_dict:
        state_dict = state_dict["actor"]
    
    # Clean keys if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        # remove prefix like "actor.model." or "module."
        for prefix in ["actor.model.", "module."]:
            if k.startswith(prefix):
                new_k = k[len(prefix):]
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

model = load_stage2_weights(model, STAGE2_CKPT_DIR)

# -------------------------------
# Step 3: Save Hugging Face model
# -------------------------------
print("[3/4] Saving HF model to", HF_OUTPUT_DIR)
os.makedirs(HF_OUTPUT_DIR, exist_ok=True)
model.save_pretrained(HF_OUTPUT_DIR)
tokenizer.save_pretrained(HF_OUTPUT_DIR)

print("[4/4] Done! You can now serve with vLLM:")
print(f"python -m vllm.entrypoints.openai.api_server --model {HF_OUTPUT_DIR} --served-model-name qwen2vl_stage2")

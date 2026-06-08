from collections import defaultdict
import torch
import re
from rouge_score import rouge_scorer
import nltk

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
# from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import json
import os

# def format_reward(predict_str: str) -> float:
#     """Check if the solution_str matches the <think>...</think> <answer>...</answer> pattern."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     return 1.0 if re.fullmatch(pattern, predict_str, re.DOTALL) else 0.0

def format_reward(predict_str: str) -> float:
    """Check if the solution_str matches the <think>...</think> <answer>...</answer> pattern."""
    pattern = r"<report>.*?</report>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    return 1.0 if re.fullmatch(pattern, predict_str, re.DOTALL) else 0.0

def accuracy_reward_close(pred: str, gt: str) -> float:
    """Exact-match reward for CLOSED answers."""
    try:
        gt_norm = gt.replace(" ", "").replace("_", "").lower()
        m = re.search(r"<answer>(.*?)</answer>", pred)
        pred_norm = m.group(1).strip().replace(" ", "").replace("_", "").lower() if m else ""
        return 1.0 if gt_norm == pred_norm else 0.0
    except Exception:
        return 0.0
    
def preprocess(text: str):
    return str(text).lower().replace(".", " .").split(" ")

# def preprocess(input_str):
#     return word_tokenize(str(input_str).lower())

def compute_nlp_metrics(preds, gts):
    """
    Compute BLEU-1 and ROUGE-1 metrics using preprocess for tokenization.
    - If preds and gts are lists, returns two lists: bleu1s and rouge1s.
    - Uses sentence_bleu for per-sample BLEU-1.
    - RougeScorer only supports single-sentence scoring, so we loop internally for ROUGE-1.
    - If preds and gts are strings, returns a tuple (bleu1, rouge1).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    # Batch mode
    if isinstance(preds, list) and isinstance(gts, list):
        assert len(preds) == len(gts), "Predictions and ground truths lists must match length"
        # Per-sample BLEU using sentence_bleu
        bleu1s = []
        rouge1s = []
        for p, g in zip(preds, gts):
            tokens_p = preprocess(p)
            tokens_g = preprocess(g)
            b1 = sentence_bleu([tokens_g], tokens_p, weights=(1, 0, 0, 0))
            r1 = scorer.score(g, p)['rouge1'].fmeasure
            bleu1s.append(round(b1, 4))
            rouge1s.append(round(r1, 4))
        return bleu1s, rouge1s

    # Single sample mode
    tokens_pred = preprocess(preds)
    tokens_gt = preprocess(gts)
    bleu1 = sentence_bleu([tokens_gt], tokens_pred, weights=(1, 0, 0, 0))
    rouge1 = scorer.score(gts, preds)['rouge1'].fmeasure
    return round(bleu1, 4), round(rouge1, 4)

def accuracy_reward_open_batch(
    bert_model,
    preds: list[str],
    gts: list[str],
    alpha: float = 0.3
) -> list[float]:
    """
    Batched open-answer reward: combine BERTScore and NLP metrics.
    Calls bert_model.score once for all preds/gts, uses corpus BLEU-1 and ROUGE-1 list.
    """
    n = len(preds)
    if n == 0 or len(gts) != n:
        return {
            "open_reward": [0.0] * n,
            "open_reward_bert_score": [0.0] * n,
            "open_reward_bleu1": [0.0] * n,
            "open_reward_rouge1": [0.0] * n,
        }

    # Clean whitespace
    pred_list = []
    gt_list = []
    for i in range(len(preds)):
        cur_pred = re.search(r"<answer>(.*?)</answer>", preds[i])
        cur_pred = cur_pred.group(1).strip() if cur_pred else preds[i].strip()
        pred_list.append(cur_pred)
        gt_list.append(gts[i])
    preds_clean = [re.sub(r' +', ' ', p.strip()) for p in pred_list]
    gts_clean = [re.sub(r' +', ' ', g.strip()) for g in gt_list]
    # 1) BERTScore f1
    # try:
    with torch.no_grad():
        # bert_model._model.to("cuda") ## no gpu still
        _, _, f1_tensor = bert_model.score(preds_clean, gts_clean)
    bert_f1 = [float(x) for x in f1_tensor]
    # print("*"*50)
    # print(preds_clean[:5])
    # print(gts_clean[:5])
    # print(bert_f1[:5])
    # print("*"*50)
    # except Exception:
    #     bert_f1 = [0.0] * n

    # 2) NLP metrics in batch: returns bleu1s, rouge1s
    bleu1s, rouge1s = compute_nlp_metrics(preds_clean, gts_clean)

    # 3) Combine
    combined = []
    for i in range(n):
        combined.append(alpha * bert_f1[i] + (1 - alpha) * 0.5 * (bleu1s[i] + rouge1s[i]))
        
    return {
        "open_reward": combined,
        "open_reward_bert_score": bert_f1,
        "open_reward_bleu1": bleu1s,
        "open_reward_rouge1": rouge1s,
    }

def accuracy_reward_report(
    bert_model,
    solution_strs: list[str],
    extra_list: list[dict],
    alpha: float = 0.3
) -> dict[str, list[float]]:
    """
    Batched report reward: extract <report> from solution_strs and compare to report_label in extra_list.
    Combines BERTScore and NLP metrics.
    """
    report_preds = []
    report_gts = []


    for idx, s in enumerate(solution_strs):
        m = re.search(r"<report>(.*?)</report>", s, re.DOTALL)
        report_preds.append(m.group(1).strip() if m else "")
        report_gts.append(extra_list[idx].get("report_label", "").strip())

    n = len(report_preds)
    if n == 0 or len(report_gts) != n:
        return {
            "report_reward": [0.0] * n,
            "report_reward_bert_score": [0.0] * n,
            "report_reward_bleu1": [0.0] * n,
            "report_reward_rouge1": [0.0] * n,
        }

    # Clean whitespace
    preds_clean = [re.sub(r' +', ' ', p.strip()) for p in report_preds]
    gts_clean = [re.sub(r' +', ' ', g.strip()) for g in report_gts]


    # 1) BERTScore f1
    with torch.no_grad():
        _, _, f1_tensor = bert_model.score(preds_clean, gts_clean)
    bert_f1 = [float(x) for x in f1_tensor]

    # 2) NLP metrics
    bleu1s, rouge1s = compute_nlp_metrics(preds_clean, gts_clean)

    # 3) Combine
    combined = [alpha * bert_f1[i] + (1 - alpha) * 0.5 * (bleu1s[i] + rouge1s[i]) for i in range(n)]

    return {
        "report_reward": combined,
        "report_reward_bert_score": bert_f1,
        "report_reward_bleu1": bleu1s,
        "report_reward_rouge1": rouge1s,
    }

# def batch_compute_score(
#     bert_model,
#     data_sources: list,
#     solution_strs: list[str],
#     ground_truths: list[str],
#     extra_infos: list[dict] | None = None,
#     alpha: float = 0.3,
# ) -> list[float]:
#     """
#     Batch compute rewards, grouping CLOSED and OPEN samples.
#     Safely handles homogeneous types and uses batched NLP metrics.
#     Returns rewards list in original order.
#     """
#     n = len(solution_strs)
#     extra_list = [{}] * n if extra_infos is None else [info if isinstance(info, dict) else {} for info in extra_infos]
#     fmt_rewards = [format_reward(s) for s in solution_strs]

#     # Partition
#     closed_idx, open_idx = [], []
#     closed_preds, closed_gts = [], []
#     open_preds, open_gts = [], []
#     for i, info in enumerate(extra_list):
#         if info.get('answer_type', 'CLOSED').upper() == 'CLOSED':
#             closed_idx.append(i)
#             # print("CLOSED")
#             closed_preds.append(solution_strs[i]); closed_gts.append(ground_truths[i])
#         else:
#             # print("OPEN")
#             open_idx.append(i)
#             open_preds.append(solution_strs[i]); open_gts.append(ground_truths[i])

#     # Compute rewards
#     closed_rews = [accuracy_reward_close(p, g) for p, g in zip(closed_preds, closed_gts)] if closed_idx else []
#     open_reward_dict = accuracy_reward_open_batch(bert_model, open_preds, open_gts, alpha) if open_idx else {
#         "open_reward": [],
#         "open_reward_bert_score": [],
#         "open_reward_bleu1": [],
#         "open_reward_rouge1": [],
#     }
#     rewards = [0.0] * n
#     open_reward = [0.0] * n
#     open_bert = [0.0] * n
#     open_bleu = [0.0] * n
#     open_rouge = [0.0] * n
#     close_reward = [0.0] * n
#     # Merge
#     for idx, r in zip(closed_idx, closed_rews):
#         rewards[idx] = 0.8 * r + 0.2 * fmt_rewards[idx]
#         close_reward[idx] = r
        
#     for i, idx in enumerate(open_idx):
#         r = open_reward_dict["open_reward"][i]
#         b = open_reward_dict["open_reward_bert_score"][i]
#         bl = open_reward_dict["open_reward_bleu1"][i]
#         ro = open_reward_dict["open_reward_rouge1"][i]
#         rewards[idx] = 0.8 * r + 0.2 * fmt_rewards[idx]
#         open_reward[idx] = r
#         open_bert[idx] = b
#         open_bleu[idx] = bl
#         open_rouge[idx] = ro
        
#     reward_extra_info = {
#         "format_reward": fmt_rewards,
#         "close_reward": close_reward,
#         "open_reward": open_reward,
#         "open_reward_bert_score": open_bert,
#         "open_reward_bleu1": open_bleu,
#         "open_reward_rouge1": open_rouge,
#     }

#     # log_file = "/home/work/austin/MedCCO/verl/checkpoints/verl_grpo_medVQA/medcco_stage1_Qwen2.5-VL-7B-Instruct-forgraph/metrics.jsonl"
#     # with open(log_file, "a", encoding="utf-8") as f:
#     #     for i in range(n):
#     #         log_entry = {
#     #             "source": data_sources[i],
#     #             # "solution": solution_strs[i],
#     #             # "ground_truth": ground_truths[i],
#     #             "format_reward": fmt_rewards[i],
#     #             "close_reward": close_reward[i],
#     #             "open_reward": open_reward[i],
#     #             "open_bert": open_bert[i],
#     #             "open_bleu": open_bleu[i],
#     #             "open_rouge": open_rouge[i],
#     #             # "report_reward": report_reward[i],
#     #             "total_reward": rewards[i],
#     #         }
#     #         f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

#     return rewards, reward_extra_info

def batch_compute_score(
    bert_model,
    data_sources: list,
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict] | None = None,
    alpha: float = 0.3,
) -> list[float]:

    n = len(solution_strs)
    extra_list = [{}] * n if extra_infos is None else [info if isinstance(info, dict) else {} for info in extra_infos]

    fmt_rewards = [format_reward(s) for s in solution_strs]

    # 2. Partition 
    closed_idx, open_idx = [], []
    closed_preds, closed_gts = [], []
    open_preds, open_gts = [], []
    for i, info in enumerate(extra_list):
        if info.get('answer_type', 'CLOSED').upper() == 'CLOSED':
            closed_idx.append(i)
            closed_preds.append(solution_strs[i]); closed_gts.append(ground_truths[i])
        else:
            open_idx.append(i)
            open_preds.append(solution_strs[i]); open_gts.append(ground_truths[i])

    # 3. Compute rewards
    closed_rews = [accuracy_reward_close(p, g) for p, g in zip(closed_preds, closed_gts)] if closed_idx else []
    open_reward_dict = accuracy_reward_open_batch(bert_model, open_preds, open_gts, alpha) if open_idx else {
        "open_reward": [], "open_reward_bert_score": [], "open_reward_bleu1": [], "open_reward_rouge1": [],
    }

    # 4. Compute REPORT reward (BLEU+ROUGE on <report>)
    report_reward_dict = accuracy_reward_report(bert_model, solution_strs, extra_list, alpha)
    report_rewards = report_reward_dict["report_reward"]

    rewards = [0.0] * n
    close_reward = [0.0] * n
    open_reward = [0.0] * n
    open_bert = [0.0] * n
    open_bleu = [0.0] * n
    open_rouge = [0.0] * n
    report_reward = [0.0] * n

    # CLOSED samples
    for idx, r in zip(closed_idx, closed_rews):
        # if r < 0.1:
        #     report_rewards[idx] = 0.0
        # rewards[idx] = 0.8 * r + 0.1 * fmt_rewards[idx] + 0.1 * report_rewards[idx]
        rewards[idx] = 0.8 * r + 0.2 * fmt_rewards[idx]  
        close_reward[idx] = r
        report_reward[idx] = report_rewards[idx]

    # OPEN samples      
    for i, idx in enumerate(open_idx):
        r = open_reward_dict["open_reward"][i]
        b = open_reward_dict["open_reward_bert_score"][i]
        bl = open_reward_dict["open_reward_bleu1"][i]
        ro = open_reward_dict["open_reward_rouge1"][i]
        rewards[idx] = 0.8 * r + 0.2 * fmt_rewards[idx]
        open_reward[idx] = r
        open_bert[idx] = b
        open_bleu[idx] = bl
        open_rouge[idx] = ro
        report_reward[idx] = report_rewards[idx]

    # Reward extra info
    reward_extra_info = {
        "format_reward": fmt_rewards,
        "close_reward": close_reward,
        "open_reward": open_reward,
        "open_reward_bert_score": open_bert,
        "open_reward_bleu1": open_bleu,
        "open_reward_rouge1": open_rouge,
        "report_reward": report_reward,
        "report_reward_bert_score": report_reward_dict["report_reward_bert_score"],
        "report_reward_bleu1": report_reward_dict["report_reward_bleu1"],
        "report_reward_rouge1": report_reward_dict["report_reward_rouge1"],
    }


    # log_file = "/home/work/austin/MedCCO/verl/checkpoints/verl_grpo_medVQA/medcco_report0.1_roi_qa_stage1_Qwen2.5-VL-3B-Instruct/metrics.jsonl"
    # with open(log_file, "a", encoding="utf-8") as f:
    #     for i in range(n):
    #         log_entry = {
    #             "source": data_sources[i],
    #             # "solution": solution_strs[i],
    #             # "ground_truth": ground_truths[i],
    #             "format_reward": fmt_rewards[i],
    #             "close_reward": close_reward[i],
    #             "open_reward": open_reward[i],
    #             "open_bert": open_bert[i],
    #             "open_bleu": open_bleu[i],
    #             "open_rouge": open_rouge[i],
    #             "report_reward": report_reward[i],
    #             "total_reward": rewards[i],
    #         }
    #         f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return rewards, reward_extra_info

# 文件路径：your_project/reward_functions/accuracy_reward.py

import re
from datetime import datetime
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
# from math_verify import parse, verify  # 假设你有一个符号验证模块
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#def format_reward(predict_str):
#    """Reward function that checks if the completion has a specific format."""
#    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#    match = re.fullmatch(pattern, predict_str, re.DOTALL) 
#    return 1.0 if match else 0.0 

def format_reward(predict_str: str) -> float:
    """Check if the solution_str matches the <think>...</think> <answer>...</answer> pattern."""
    pattern = r"<report>.*?</report>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    print("using open_close")
    return 1.0 if re.fullmatch(pattern, predict_str, re.DOTALL) else 0.0

def preprocess(text: str):
    return str(text).lower().replace(".", " .").split(" ")


def compute_metrics(pred, gt):
    pred_tokens = preprocess(pred)
    gt_tokens = preprocess(gt)

    # BLEU-1（unigram precision）
    bleu1 = sentence_bleu(
        [gt_tokens], pred_tokens, weights=(1, 0, 0, 0)
    )

    # ROUGE-1 和 ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt, pred)

    rouge1_f = scores["rouge1"].fmeasure
    rougeL_f = scores["rougeL"].fmeasure

    return {
        "BLEU-1": round(bleu1, 4),
        "ROUGE-1": round(rouge1_f, 4),
        "ROUGE-L": round(rougeL_f, 4),
    }


def accuracy_reward_close(solution_str, ground_truth):

    content = solution_str
    reward = 0.0

    if reward == 0.0:
        try:
            # 提取 <answer> 标签内的答案
            gt = ground_truth.replace(" ", "").replace("_", "").lower()

            pred = re.search(r"<answer>(.*?)</answer>", content)
            pred = pred.group(1).strip()
            pred = pred.replace(" ", "").replace("_", "").lower()

            if gt == pred:
                reward = 1.0
        except Exception:
            pass
    return reward


def accuracy_reward_open(bert_model, solution_str, ground_truth):
    # 解包
    alpha = 0.3
    beta = 0.5
    
    gt = ground_truth.strip()

    pred = re.search(r"<answer>(.*?)</answer>", solution_str)
    pred = pred.group(1).strip() if pred else solution_str.strip()

    # 1) BERTScore
    pred_list = [pred]
    gt_list = [gt]
    pred_list = [re.sub(r' +', ' ', str(test)) for test in pred_list]
    gt_list = [re.sub(r' +', ' ', str(test)) for test in gt_list]
    
    # bert_f1 = bert_model.score(pred_list, gt_list)[2]
    # bert_f1 = [x.item() for x in bert_f1][0]
    bert_f1 = 0.0

    # 2) NLP 传统指标
    metrics = compute_metrics(pred_list[0], gt_list[0])
    bleu1 = metrics["BLEU-1"]
    rouge1 = metrics["ROUGE-1"]
    rougeL = metrics["ROUGE-L"]

    # 加权组合
    # combined = alpha * bert_f1 + (1-alpha) * (bleu1 + rouge1)
    combined = alpha * float(bert_f1) + (1 - alpha) * (float(bleu1) + float(rouge1))

    # 缩放到 [0, 2]
    return combined

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
    combined = [alpha * bert_f1[i] + (1 - alpha) * (bleu1s[i] + rouge1s[i]) for i in range(n)]

    return {
        "report_reward": combined,
        "report_reward_bert_score": bert_f1,
        "report_reward_bleu1": bleu1s,
        "report_reward_rouge1": rouge1s,
    }

# 为 RewardManager 提供统一接口
def my_compute_score(bert_model,data_source, solution_str, ground_truth, extra_info=None):
    """
    根据 extra_info 中的 answer_type 决定调用 open/close
    """
    extra_list = [{}] * n if extra_info is None else [info if isinstance(info, dict) else {} for info in extra_info]

    # atype = extra_info.get("answer_type", ["CLOSED"]).upper()
    atype = extra_list.get("answer_type", ["CLOSED"]).upper()
    # print(atype)
    if atype == "CLOSED":
        acc_reward =  accuracy_reward_close(solution_str, ground_truth)
    else:
        acc_reward = accuracy_reward_open(bert_model, solution_str, ground_truth)

    # 4. Compute REPORT reward (BLEU+ROUGE on <report>)

    report_reward_dict = accuracy_reward_report(bert_model, solution_str, extra_list)
    report_rewards = report_reward_dict["report_reward"]

    return acc_reward + 0.2 * format_reward(solution_str) + report_rewards
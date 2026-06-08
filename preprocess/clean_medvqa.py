import json
from vllm import LLM, SamplingParams
import torch
from transformers import AutoProcessor
import re

def load_dataset(input_path):
    """
    Load the JSON dataset from `input_path`.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_dataset(data, output_path):
    """
    Save the reformatted dataset to `output_path` in JSON format with indentation.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(question: str, answer: str) -> str:
    """
    Construct the prompt for the QA-Consistency Auditor.
    """
    return f'''ori_q: {question}
ori_a: {answer}
You are **“QA-Consistency Auditor”** – an expert data-curator. 
Your task is to refine **open-ended visual-question-answering (VQA)** pairs so that the revised
question and answer remain logically and granularly consistent. These are **open-end VQA pairs**,
*not* closed-end: do **not** embed answer choices in the question.
Process
1. Read the original question (*ori_q*).
2. Ignore the visual content; focus only on the wording of the question and the expected
   form of the answer.
3. Internally simulate an expert’s likely free-form answer (*Expert_Guess*).
4. Compare *Expert_Guess* to the original answer (*ori_a*) to spot missing components
   or granularity gaps.
5. Decide on a status:
   • **consistent** – *ori_q* already elicits exactly the information found in *ori_a*.
     Copy *ori_q* → *new_q* and *ori_a* → *new_a*.
   • **needs_fix** – *ori_q* is too broad, ambiguous, or does not explicitly request every
     element found in *ori_a*. Rewrite the question and/or minimally adjust the answer.
   • **drop** – The pair is unusable (contradictory, nonsensical, etc.). Leave *new_q*
     and *new_a* empty.
6. If the status is **needs_fix**, craft *new_q* that:
   • Starts with a precise **action verb** ("Identify", "Describe", "Explain", …).  
   • Explicitly requests **every component** required by *ori_a*.  
   • Maintains an **open-end** format (no yes/no phrasing, no embedded choices).  
   • Provides a **1-to-1 mapping**: each phrase in *ori_a* must correspond to a clearly
     stated element in *new_q* (e.g., *MRI* ↔ *modality*, *T2* ↔ *sequence*).  
   • Matches the granularity of *ori_a* exactly—no more, no less.  
   • Ensures *new_a* presents components in the **same order** that *new_q* requests them.
7. Adjust *new_a* only if wording changes are necessary for brevity or clarity;
   never change the meaning.
Key requirements
• **Open-ended**: Questions must allow free-form expert responses; never embed answer choices.  
• **Multi-component precision**: If the answer contains multiple elements, the question must
  explicitly ask for each, preserving clear element-to-element correspondence.  
• **Action-verb prompts**: Begin revised questions with verbs like “Identify”, “Describe”, “Explain”.  
• **Granularity match**: Question scope must match answer specificity exactly.  
• **Order consistency**: Arrange components in *new_a* in the **same sequence** they are requested
  in *new_q*.  
• **Answer conciseness**: Keep *new_a* as short as possible while fully capturing the meaning.
Output format
Return **one** JSON object—nothing else—using this template:
{{
  "status": "consistent | needs_fix | drop",
  "ori_q": "<string>",
  "ori_a": "<string>",
  "new_q": "<string>",
  "new_a": "<string>",
  "notes": "rationale for fixing original QA-pair"
}}
Follow these rules strictly. Always output exactly the JSON object, and nothing else.
'''  # end of prompt]


def parse_json_response(response_text: str) -> dict:
    """
    Strips markdown fences and extracts the first JSON object from the response text.
    Raises ValueError if parsing fails.
    """
    # Remove markdown code fences
    text = re.sub(r"^```[a-zA-Z]*|```$", "", response_text).strip()
    # Extract the JSON block
    match = re.search(r"\{(?:.|\s)*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {response_text}")
    json_str = match.group(0)
    return json.loads(json_str)


def reformat_batch(llm: LLM, processor, items: list):
    """
    Reformat a batch of QA items using the QA-Consistency Auditor.
    Returns a tuple of (parsed_results, error_cases), where:
      - parsed_results: list of parsed JSON dicts or None for failures
      - error_cases: list of dicts {"item": item, "response": raw_text}
    """
    prompts = []
    for item in items:
        q = str(item['conversations'][0]['value']).strip()
        a = str(item['conversations'][-1]['value']).strip()
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",    "content": build_prompt(q, a)}
        ]
        prompts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

    llm_inputs = [{"prompt": p} for p in prompts]
    responses = llm.generate(
        llm_inputs,
        sampling_params=SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=1024
        )
    )

    parsed_results = []
    error_cases = []
    for idx, resp in enumerate(responses):
        raw = resp.outputs[0].text
        try:
            parsed = parse_json_response(raw)
            parsed_results.append(parsed)
        except Exception:
            parsed_results.append(None)
            error_cases.append({"item": items[idx], "response": raw})
    return parsed_results, error_cases


def reformat_and_save(config, llm,processor):
    print(f"Processing dataset: {config['name']}")
    data = load_dataset(config['input_path'])
    

    open_items = [item for item in data if item.get('answer_type') == 'OPEN']
    other_items = [item for item in data if item.get('answer_type') != 'OPEN']

    reformatted = []
    all_errors = []
    batch_size = 1

    # First pass: batch processing
    for i in range(0, len(open_items), batch_size):
        batch = open_items[i:i + batch_size]
        results, errors = reformat_batch(llm, processor, batch)
        for item, parsed in zip(batch, results):
            if parsed:
                item.update({
                    'status': parsed.get('status'),
                    'new_q': parsed.get('new_q'),
                    'new_a': parsed.get('new_a'),
                    'notes': parsed.get('notes')
                })
                reformatted.append(item)
        all_errors.extend(errors)

    # Retry failures one by one
    for err in all_errors:
        single = [err['item']]
        results, errors = reformat_batch(llm, processor, single)
        if results[0]:
            parsed = results[0]
            item = err['item']
            item.update({
                'status': parsed.get('status'),
                'new_q': parsed.get('new_q'),
                'new_a': parsed.get('new_a'),
                'notes': parsed.get('notes')
            })
        else:
            item = err['item']
            item.update({'status': 'drop', 'new_q': '', 'new_a': '', 'notes': 'Parsing failed'})
        reformatted.append(item)

    # Add non-open items
    reformatted.extend(other_items)

    # Save output
    save_dataset(reformatted, config['output_path'])
    print(f"Saved reformatted dataset to {config['output_path']}\n")


def main():
    # model_path = '../llms/Qwen2.5-72B-Instruct'
    model_path = '/home/work/austin/MedCCO/llms/Qwen2.5-72B-Instruct'
    gpu_num = torch.cuda.device_count()

    # Define all dataset configurations
    dataset_configs = [
        {
            'name': 'VQA-RAD',
            'input_path': 'VQARAD_train_new.json',
            'output_path': 'VQARAD_train_refine_72.json'
        },
        # {
        #     'name': 'VQA-RAD',
        #     'input_path': 'VQARAD_test_new.json',
        #     'output_path': 'VQARAD_test_refine_v2.json'
        # },
        # {
        #     'name': 'SLAKE',
        #     'input_path': '../dataset/SLAKE_train_full_qa.json',
        #     'output_path': 'SLAKE_train_refine_v2.json'
        # },
        #  {
        #     'name': 'SLAKE',
        #     'input_path': '../dataset/SLAKE_validation_full_qa.json',
        #     'output_path': 'SLAKE_val_refine_v2.json'
        # },
        #   {
        #     'name': 'SLAKE',
        #     'input_path': '../dataset/SLAKE_test_full_qa.json',
        #     'output_path': 'SLAKE_test_refine_v2.json'
        # },
        # {
        #     'name': 'PathVQA',
        #     'input_path': '../dataset/PathVQA_train.json',
        #     'output_path': 'PathVQA_train_refine.json'
        # },
        # {
        #     'name': 'PathVQA',
        #     'input_path': '../dataset/PathVQA_val.json',
        #     'output_path': 'PathVQA_val_refine.json'
        # },
        # {
        #     'name': 'PathVQA',
        #     'input_path': '../dataset/PathVQA_test.json',
        #     'output_path': 'PathVQA_test_refine.json'
        # },
    ]

    # Loop through each
    llm = LLM(model=model_path, dtype="bfloat16", tensor_parallel_size=gpu_num, gpu_memory_utilization=1.0)
    processor = AutoProcessor.from_pretrained(model_path)
    for cfg in dataset_configs:
        reformat_and_save(cfg,llm,processor)


if __name__ == '__main__':
    main()

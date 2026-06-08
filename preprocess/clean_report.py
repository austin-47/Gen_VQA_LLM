import json
import re
import os

# List your JSON files here
json_files = ["/home/work/austin/MedCCO/preprocess/PathVQA_report_wq.json"]
# def clean_report_text(text):
#     """
#     Removes any sentence that starts with 'Further' (case-insensitive)
#     and everything that follows it.
#     """
#     # Use regex to remove the sentence starting with 'Further' until the end
#     cleaned = re.sub(r'(\s*Further[^.]*\.)', '', text, flags=re.IGNORECASE)
#     return cleaned.strip()

# for file_path in json_files:
#     print(f"Processing {file_path}...")
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Handle both list of dicts and single dict formats
#     if isinstance(data, list):
#         for item in data:
#             if "report" in item and isinstance(item["report"], str):
#                 item["report"] = clean_report_text(item["report"])
#     elif isinstance(data, dict) and "report" in data:
#         data["report"] = clean_report_text(data["report"])

#     # Save cleaned file (you can change to overwrite if you prefer)
#     base, ext = os.path.splitext(file_path)
#     output_path = f"{base}_nofurther{ext}"

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

#     print(f"✅ Cleaned file saved to: {output_path}")

import json
import re

def clean_report_text(text):
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Keep only sentences that do NOT start with "Further"
    cleaned_sentences = [
        s for s in sentences
        if not re.match(r'^\s*Further\b', s, flags=re.IGNORECASE)
    ]

    # Rejoin sentences
    return " ".join(cleaned_sentences).strip()


for file_path in json_files:
    print(f"Processing {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if "report" in item and isinstance(item["report"], str):
                item["report"] = clean_report_text(item["report"])
    elif isinstance(data, dict) and "report" in data:
        data["report"] = clean_report_text(data["report"])

    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_nofurther{ext}"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned file saved to: {output_path}")

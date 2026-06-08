#!/usr/bin/env python3
"""
Convert dataset JSON format to unified structure.

Usage:
  python convert_json_format.py --input annotations.json --output formatted.json --image_root /path/to/images
"""

import json
import argparse
from pathlib import Path


def make_item(item, question, phrase_type, image_root=None, qid=None):
    # Get image name(s)
    image_field = item.get("image_name") or item.get("images")
    if isinstance(image_field, list):
        images = [str(Path(image_root) / img) if image_root else img for img in image_field]
    else:
        images = str(Path(image_root) / image_field) if image_root else image_field

    return {
        "qid": qid if qid is not None else item.get("qid"),
        "phrase_type": phrase_type,
        "image_name": images,
        "image_organ": item.get("image_organ"),
        "question": question,
        "answer": item.get("answer"),
        "answer_type": item.get("answer_type"),
        "conversations": [
            {"role": "user", "value": str(question)},
            {"role": "assistant", "value": str(item.get("answer", ""))}
        ]
    }


def convert_dataset(input_path: str, output_path: str, image_root: str = None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = []

    max_id = max(int(d["qid"]) for d in data if str(d.get("qid")).isdigit())
    new_id = max_id + 1

    for d in data:
        q = d.get("question", "")
        # All entries go to train_data
        train_data.append(make_item(d, q, d.get("phrase_type", ""), image_root=image_root))

        # add ALL frame questions to train (if exists)
        q_frame = d.get("question_frame", "")
        if q_frame and q_frame != "NULL":
            train_data.append(make_item(d, q_frame, "frame", image_root=image_root, qid=new_id))
            new_id += 1

    print(f"Converted {len(train_data)} samples → {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output formatted JSON path")
    parser.add_argument("--image_root", default=None, help="Root folder for images")
    args = parser.parse_args()

    convert_dataset(args.input, args.output, image_root=args.image_root)


if __name__ == "__main__":
    main()

################################################VQA-RAD################################################
# import json

# # Input and output paths
# input_path = "/home/work/austin/Dataset/vqa-rad/VQA_RAD Dataset Public.json"
# train_out = "VQARAD_train_new.json"
# test_out = "VQARAD_test_new.json"

# # Load dataset
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# def reformat_item(item):
#     """Convert question/answer into conversations format."""
#     return {
#         "qid": item.get("qid"),
#         "phrase_type": item.get("phrase_type"),
#         "image_name": item.get("image_name"),
#         "image_organ": item.get("image_organ"),
#         "question": item.get("question"),
#         "question_rephrase": item.get("question_rephrase"),
#         "question_type": item.get("question_type"),
#         "answer": item.get("answer"),
#         "answer_type": item.get("answer_type"),
#         "conversations": [
#             {"role": "user", "value": str(item.get("question", ""))},
#             {"role": "assistant", "value": str(item.get("answer", ""))}
#         ]
#     }


# # Split into train/test
# train_data = [
#     reformat_item(d)
#     for d in data
#     if d.get("phrase_type") in ["freeform", "para"]
# ]

# # Test includes test_freeform + test_para
# test_data = [
#     reformat_item(d)
#     for d in data
#     if d.get("phrase_type") in ["test_freeform", "test_para"]
# ]



# # Save outputs
# with open(train_out, "w", encoding="utf-8") as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=2)

# with open(test_out, "w", encoding="utf-8") as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=2)

# print(f"✅ Saved {len(train_data)} train samples to {train_out}")
# print(f"✅ Saved {len(test_data)} test samples to {test_out}")
import json

# Input and output paths
input_path = "/home/work/austin/Dataset/vqa-rad/VQA_RAD Dataset Public.json"
train_out = "VQARAD_train_new.json"
test_out = "VQARAD_test_new.json"

# Load dataset
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def make_item(item, question, phrase_type, qid=None):
    return {
        "qid": qid if qid is not None else item.get("qid"),
        "phrase_type": phrase_type,
        "image_name": item.get("image_name"),
        "image_organ": item.get("image_organ"),
        "question": question,
        "answer": item.get("answer"),
        "answer_type": item.get("answer_type"),
        "conversations": [
            {"role": "user", "value": str(question)},
            {"role": "assistant", "value": str(item.get("answer", ""))}
        ]
    }

train_data, test_data = [], []

# start new IDs after max qid
max_id = max(int(d["qid"]) for d in data if str(d.get("qid")).isdigit())
new_id = max_id + 1

for d in data:
    ptype = d.get("phrase_type", "")
    q = d.get("question", "")

    # train samples
    if ptype in ["freeform", "para"]:
        train_data.append(make_item(d, q, ptype))

    # test samples
    elif ptype in ["test_freeform", "test_para"]:
        test_data.append(make_item(d, q, ptype))

    # add ALL frame questions to train (no matter phrase_type)
    q_frame = d.get("question_frame", "")
    if q_frame and q_frame != "NULL":
        train_data.append(make_item(d, q_frame, "frame", qid=new_id))
        new_id += 1

print(f"Train size: {len(train_data)}")   # should be ~3064
print(f"Test size: {len(test_data)}")     # should be 451
print(f"Total size: {len(train_data) + len(test_data)}")
print(f"Last assigned qid: {new_id-1}")

# Save outputs
with open(train_out, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(test_out, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)





import json
import re
from pathlib import Path
from rapidfuzz.distance import Levenshtein

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# -------------------
# CONFIG
# -------------------

MODEL_PATH = "merged-vizwiz"

DATASET_JSON = "/Users/ailabiitbhu14/Desktop/Projects/glm ocr/GLM-OCR/dataset/vizwiz/val_glmocr_full.json"

IMAGE_ROOT = "/Users/ailabiitbhu14/Desktop/Projects/glm ocr/GLM-OCR/dataset/vizwiz"

LIMIT = 100

char_distance = 0
char_total = 0

word_distance = 0
word_total = 0

# -------------------
# LOAD MODEL
# -------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using:", device)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

model = model.to(device)

print("Model loaded.")

# -------------------
# HELPERS
# -------------------

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# -------------------
# LOAD DATASET
# -------------------

with open(DATASET_JSON, "r") as f:
    data = json.load(f)

data = data[:LIMIT]

correct = 0
total = 0

# -------------------
# EVALUATE
# -------------------

for idx, sample in enumerate(data):

    image_path = Path(IMAGE_ROOT) / sample["images"][0]

    image = Image.open(image_path).convert("RGB")
    MAX_SIZE = 1024

    image.thumbnail((MAX_SIZE, MAX_SIZE))

    question = sample["messages"][0]["content"].replace("<image>\n", "")
    reference = sample["messages"][1]["content"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    try:
     with torch.no_grad():
         outputs = model.generate(
             **inputs,
             max_new_tokens=20
         )
 
     generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

     prediction = processor.batch_decode(
         generated_ids,
         skip_special_tokens=True
     )[0].strip()

    except Exception as e:
     print(f"[{idx+1}/{LIMIT}] FAILED: {e}")
     continue


    pred_norm = normalize(prediction)
    ref_norm = normalize(reference)

    char_distance += Levenshtein.distance(
    pred_norm,
    ref_norm
    )

    char_total += max(len(ref_norm), 1)

    pred_words = pred_norm.split()
    ref_words = ref_norm.split()

    word_distance += Levenshtein.distance(
     pred_words,
     ref_words 
    )

    word_total += max(len(ref_words), 1)

    match = pred_norm == ref_norm

    if match:
        correct += 1

    total += 1

    print(
        f"[{idx+1}/{LIMIT}] "
        f"GT='{reference}' "
        f"PRED='{prediction}' "
        f"MATCH={match}"
    )

accuracy = correct / total
cer = char_distance / char_total
wer = word_distance / word_total


print("\n===================")
print("EXACT MATCH:", accuracy)
print("CORRECT:", correct)
print("TOTAL:", total)
print("CER:", cer)
print("WER:", wer)
print("===================")

import os
import json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torch

# === CONFIG ===
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
OUTPUT_VOCAB = "vocab.json"
OUTPUT_DATASET = "dataset.pt"
IMAGE_SIZE = (128, 128)  # you can change this later

# === Image Preprocessing ===
img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()  # (C, H, W) with values between 0 and 1
])

# === Vocab and tokenizer ===
token_to_id = {}
id_to_token = {}

def tokenize(text):
    # Remove commas and split by whitespace
    return text.replace(",", " ").split()

# === Dataset list ===
dataset = []

# === Process all images/labels ===
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.endswith(".png"):
        continue

    # === Load image ===
    img_path = os.path.join(IMG_DIR, fname)
    image = Image.open(img_path).convert("RGB")
    image_tensor = img_transform(image)  # shape: (3, 128, 128)

    # === Load label ===
    label_name = fname.replace(".png", ".txt")
    label_path = os.path.join(LABEL_DIR, label_name)

    if not os.path.exists(label_path):
        print(f"[WARNING] Missing label for {fname}")
        continue

    with open(label_path, "r") as f:
        tokens = tokenize(f.read().strip())

    # === Add tokens to vocab ===
    token_ids = []
    for token in tokens:
        if token not in token_to_id:
            token_id = len(token_to_id) + 1  # start from 1
            token_to_id[token] = token_id
            id_to_token[token_id] = token
        token_ids.append(token_to_id[token])

    # === Add sample to dataset ===
    dataset.append((image_tensor, torch.tensor(token_ids)))

print(f"✅ Processed {len(dataset)} samples")
print(f"📚 Vocab size: {len(token_to_id)} tokens")

# === Save vocab ===
with open(OUTPUT_VOCAB, "w") as f:
    json.dump(token_to_id, f)
print(f"💾 Saved vocab → {OUTPUT_VOCAB}")

# === Save dataset ===
torch.save(dataset, OUTPUT_DATASET)
print(f"💾 Saved dataset → {OUTPUT_DATASET}")

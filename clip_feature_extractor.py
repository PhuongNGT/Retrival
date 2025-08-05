import os
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import numpy as np
# ==== Cấu hình ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dir = Path("dataset/oxbuild_images")
output_dir = Path("clip_outputs")
output_dir.mkdir(exist_ok=True)

# ==== Load mô hình CLIP ====
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ==== Load danh sách ảnh và caption ====
image_paths = sorted([str(p) for p in dataset_dir.glob("*.jpg")])
captions = ["a photo of a building"] * len(image_paths)

# ==== Trích đặc trưng ảnh và caption ====
image_features = []
text_features = []

print("🚀 Đang trích đặc trưng bằng CLIP...")

for img_path, caption in tqdm(zip(image_paths, captions), total=len(image_paths)):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=caption, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        img_feat = outputs.image_embeds[0].cpu().numpy()
        txt_feat = outputs.text_embeds[0].cpu().numpy()

    image_features.append(img_feat)
    text_features.append(txt_feat)

# ==== Lưu output ====
torch.save(torch.tensor(image_features), output_dir / "clip_image_features.npy")
torch.save(torch.tensor(text_features), output_dir / "clip_text_features.npy")

with open(output_dir / "clip_image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

with open(output_dir / "clip_captions.pkl", "wb") as f:
    pickle.dump(captions, f)

print("✅ Lưu trích đặc trưng thành công vào thư mục:", output_dir)


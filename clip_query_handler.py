import torch
from PIL import Image
import numpy as np
import pickle
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from clip_retriever import search_clip_image, search_clip_text
import matplotlib.pyplot as plt 


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load features
clip_image_features = torch.load("clip_outputs/clip_image_features.npy").numpy()
clip_text_features = torch.load("clip_outputs/clip_text_features.npy").numpy()


with open("clip_outputs/clip_image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

with open("clip_outputs/clip_captions.pkl", "rb") as f:
    captions = pickle.load(f)

def handle_image_query(image_path, top_k=10):
    """Xử lý truy vấn bằng ảnh"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_feature = model.get_image_features(**inputs).cpu().numpy().flatten()

    indices, scores = search_clip_image(image_feature, clip_image_features, top_k)
    return [image_paths[i] for i in indices], scores

def handle_text_query(text, top_k=10):
    """Xử lý truy vấn bằng văn bản"""
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        text_feature = model.get_text_features(**inputs).cpu().numpy().flatten()

    indices, scores = search_clip_text(text_feature, clip_image_features, top_k)
    return [image_paths[i] for i in indices], scores
    
def visualize_results(image_paths, scores, title="Results"):
    """Hiển thị ảnh kết quả"""
    plt.figure(figsize=(15, 5))
    for i, (path, score) in enumerate(zip(image_paths, scores)):
        img = Image.open(path)
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img)
        plt.title(f"{score:.2f}")
        plt.axis("off")
    plt.suptitle(title)
    plt.savefig("clip_query_result.png")
    print("✅ Đã lưu kết quả hiển thị vào: clip_query_result.png")   
    
if __name__ == "__main__":
    print("🔍 Truy vấn bằng văn bản:")
    query_text = "a photo of a cathedral"
    results, scores = handle_text_query(query_text, top_k=5)
    for path, score in zip(results, scores):
        print(f"{path} → Score: {score:.4f}")
    visualize_results(results, scores, title=f"Text Query: {query_text}")
    
    print("\n🖼 Truy vấn bằng ảnh:")
    query_image = image_paths[0]  # Hoặc thay bằng ảnh bất kỳ trong oxbuild_images
    results, scores = handle_image_query(query_image, top_k=5)
    for path, score in zip(results, scores):
        print(f"{path} → Score: {score:.4f}")
    visualize_results(results, scores, title=f"Image Query: {Path(query_image).name}")    

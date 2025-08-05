# viz_clip.py
import torch
from transformers import CLIPModel, CLIPProcessor
from torchviz import make_dot

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Dummy input
from PIL import Image
img = Image.new("RGB", (224, 224))
inputs = processor(images=img, return_tensors="pt")
outputs = model.get_image_features(**inputs)

dot = make_dot(outputs, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("clip_architecture")
print("✅ Đã lưu sơ đồ kiến trúc: clip_architecture.png")


import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from model import ConvAutoencoder_v2

# ==== Config ====
MODEL_PATH = "conv_autoencoderv2_oxford.pt"
IMAGE_DIR = "dataset/oxbuild_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_SHAPE = (256, 8, 8)  # Output shape of encoder
BATCH_SIZE = 1  # encode từng ảnh (vì ảnh không cố định batch)

# ==== Load model ====
model = ConvAutoencoder_v2().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==== Load ảnh ====
image_paths = sorted([
    os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# ==== Trích đặc trưng ====
features = np.zeros((len(image_paths), *FEATURE_SHAPE), dtype=np.float32)

with torch.no_grad():
    for i, path in enumerate(tqdm(image_paths, desc="Extracting features")):
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        encoded = model.encoder(tensor).cpu().numpy()
        features[i] = encoded[0]

# ==== Lưu output ====
np.save("latent_features.npy", features)

with open("features.pkl", "wb") as f:
    pickle.dump(features, f)

with open("image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

print("✅ Saved latent_features.npy, features.pkl, image_paths.pkl")

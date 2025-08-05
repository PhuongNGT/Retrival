# build_index_clip.py
import numpy as np
import pickle
import os
import torch
# ==== File paths ====
IMAGE_FEATURES_PATH = "clip_outputs/clip_image_features.npy"
TEXT_FEATURES_PATH = "clip_outputs/clip_text_features.npy"
IMAGE_PATHS_PKL = "clip_outputs/clip_image_paths.pkl"
CAPTIONS_PKL = "clip_outputs/clip_captions.pkl"

# ==== Kiểm tra tồn tại ====
assert os.path.exists(IMAGE_FEATURES_PATH), f"❌ Missing: {IMAGE_FEATURES_PATH}"
assert os.path.exists(TEXT_FEATURES_PATH), f"❌ Missing: {TEXT_FEATURES_PATH}"
assert os.path.exists(IMAGE_PATHS_PKL), f"❌ Missing: {IMAGE_PATHS_PKL}"
assert os.path.exists(CAPTIONS_PKL), f"❌ Missing: {CAPTIONS_PKL}"

# ==== Load data ====
image_features = torch.load(IMAGE_FEATURES_PATH).numpy()
text_features = torch.load(TEXT_FEATURES_PATH).numpy()

with open(IMAGE_PATHS_PKL, "rb") as f:
    image_paths = pickle.load(f)

with open(CAPTIONS_PKL, "rb") as f:
    captions = pickle.load(f)

# ==== Kiểm tra lại số lượng ====
assert len(image_paths) == image_features.shape[0], "Mismatch in image features"
assert len(captions) == text_features.shape[0], "Mismatch in text features"

print(f"✅ Loaded {len(image_paths)} image features and {len(captions)} text features.")

# ==== (Optional) Lưu chuẩn hóa nếu cần ====
# from sklearn.preprocessing import normalize
# image_features = normalize(image_features)
# text_features = normalize(text_features)

# ==== Lưu lại (nếu cần) ====
# np.save("clip_image_features_norm.npy", image_features)
# np.save("clip_text_features_norm.npy", text_features)


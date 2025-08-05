import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
from PIL import Image

# ==== Load dữ liệu ====
labels = np.load("cluster_labels.npy")
features = np.load("latent_features.npy")
image_paths = pickle.load(open("image_paths.pkl", "rb"))
model = joblib.load("cluster_model.pkl")

# ==== Tạo output folder ====
os.makedirs("static/cluster_reps", exist_ok=True)

# ==== 1. Biểu đồ số lượng ảnh mỗi cụm ====
def plot_cluster_distribution(labels, save_path="static/cluster_bar.png"):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8,5))
    plt.bar(unique, counts, color='skyblue')
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Images")
    plt.title("📊 Number of Images per Cluster")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved bar chart: {save_path}")

# ==== 2. Ảnh đại diện cụm ====
def save_cluster_representatives(features, labels, model, image_paths):
    for cluster_id in range(model.n_clusters):
        indices = np.where(labels == cluster_id)[0]
        cluster_feats = features[indices]
        center = model.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_feats - center, axis=1)
        rep_idx = indices[np.argmin(dists)]

        img = Image.open(image_paths[rep_idx]).convert("RGB")
        img = img.resize((128, 128))
        save_path = f"static/cluster_reps/cluster_{cluster_id}.jpg"
        img.save(save_path)

    print(f"✅ Saved representative images to: static/cluster_reps/")

# ==== Gọi ====
plot_cluster_distribution(labels)
save_cluster_representatives(features, labels, model, image_paths)


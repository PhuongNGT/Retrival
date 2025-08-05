import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib

# ==== Load đặc trưng ====
latent_features = np.load("latent_features.npy")
latent_features1d = latent_features.reshape(latent_features.shape[0], -1)

# ==== Elbow Method ====
def elbow_method(data, k_range=range(4, 11), save_path="elbow.png"):
    distortions = []
    for k in tqdm(k_range, desc="Elbow"):
        km = KMeans(n_clusters=k, random_state=0).fit(data)
        distortion = np.mean(np.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1))
        distortions.append(distortion)
    
    plt.figure()
    plt.plot(list(k_range), distortions, 'bx-')
    plt.xlabel("K")
    plt.ylabel("Distortion")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Saved elbow plot to {save_path}")

# ==== Silhouette ====
def silhouette_analysis(data, k_range=range(3, 10), save_prefix="silhouette_"):
    for k in tqdm(k_range, desc="Silhouette"):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        sil_avg = silhouette_score(data, labels)
        print(f"Silhouette (K={k}): {sil_avg:.4f}")

        sil_vals = silhouette_samples(data, labels)
        fig, ax = plt.subplots()
        y_lower = 10
        for i in range(k):
            ith = sil_vals[labels == i]
            ith.sort()
            y_upper = y_lower + len(ith)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, alpha=0.7)
            ax.text(-0.05, y_lower + len(ith) / 2, str(i))
            y_lower = y_upper + 10
        ax.axvline(sil_avg, color="red", linestyle="--")
        ax.set_title(f"Silhouette plot (K={k})")
        plt.savefig(f"{save_prefix}{k}.png")
        plt.close()

# ==== Huấn luyện mô hình cuối ====
def train_final_kmeans(data, best_k):
    model = KMeans(n_clusters=best_k, random_state=0).fit(data)
    labels = model.labels_

    # Lưu kết quả
    joblib.dump(model, "cluster_model.pkl")
    np.save("cluster_labels.npy", labels)
    np.save("latent_features.npy", data)  # Ghi đè dạng 1D
    print("✅ Saved: cluster_model.pkl, cluster_labels.npy, latent_features.npy")

# ==== Chạy chính ====
if __name__ == "__main__":
    elbow_method(latent_features1d)
    silhouette_analysis(latent_features1d)

    # 🧠 Tạm thời chọn K tốt nhất bằng mắt (hoặc dùng silhouette score lớn nhất)
    best_k = 7  # Bạn có thể thay đổi sau khi xem kết quả .png
    train_final_kmeans(latent_features1d, best_k)


import numpy as np
import pickle
from PIL import Image
import torch
from torchvision import transforms
from model import ConvAutoencoder_v2
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load index ====
features = pickle.load(open("features.pkl", "rb"))  # shape (N, 256, 8, 8)
image_paths = pickle.load(open("image_paths.pkl", "rb"))

# ==== Load model encoder ====
model = ConvAutoencoder_v2().to(DEVICE)
model.load_state_dict(torch.load("conv_autoencoderv2_oxford.pt", map_location=DEVICE))
model.eval()

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==== Trích đặc trưng truy vấn ====
def extract_query_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        encoded = model.encoder(tensor).cpu().numpy()
    return encoded.reshape(-1)  # flatten

# ==== Tính khoảng cách Euclidean và truy xuất ====
def perform_search(query_feat, index_feats, top_k=10):
    index_feats_flat = index_feats.reshape(len(index_feats), -1)
    dists = np.linalg.norm(index_feats_flat - query_feat, axis=1)
    top_idxs = np.argsort(dists)[:top_k]
    return top_idxs

# ==== Hiển thị kết quả ====
def show_results(query_path, result_paths, save_path="retrieval_result.png"):
    fig, ax = plt.subplots(1, len(result_paths) + 1, figsize=(15, 3))
    ax[0].imshow(Image.open(query_path))
    ax[0].set_title("Query")
    ax[0].axis("off")

    for i, path in enumerate(result_paths):
        ax[i+1].imshow(Image.open(path))
        ax[i+1].set_title(f"#{i+1}")
        ax[i+1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📸 Kết quả truy xuất đã lưu vào {save_path}")
    plt.close()

def handle_autoencoder_query(image_path, top_k=10):
    query_feat = extract_query_feature(image_path)
    top_idxs = perform_search(query_feat, features, top_k=top_k)
    top_paths = [image_paths[i] for i in top_idxs]
    return top_paths, top_idxs
# ==== Thử nghiệm ====
if __name__ == "__main__":
    query_path = image_paths[42]  # thay bằng ảnh bất kỳ
    query_feat = extract_query_feature(query_path)
    top_idxs = perform_search(query_feat, features, top_k=10)
    top_paths = [image_paths[i] for i in top_idxs]

    show_results(query_path, top_paths)


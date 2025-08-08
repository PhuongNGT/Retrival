import os
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics import average_precision_score
from retriever import extract_query_feature, perform_search
from clip_query_handler import handle_image_query
import torch

# Load index Autoencoder
ae_features = pickle.load(open("features.pkl", "rb"))
ae_paths = pickle.load(open("image_paths.pkl", "rb"))

# Load index CLIP
clip_image_features = torch.load("clip_outputs/clip_image_features.npy").numpy()
clip_paths = pickle.load(open("clip_outputs/clip_image_paths.pkl", "rb"))

# Đọc ảnh truy vấn và ground truth
def load_queries(gt_dir="dataset/gt_files"):
    queries = []
    for fname in sorted(os.listdir(gt_dir)):
        if fname.endswith("_query.txt"):
            with open(os.path.join(gt_dir, fname)) as f:
                line = f.readline()
                # Loại bỏ prefix 'oxc1_'
                qimg = line.split()[0].split("oxc1_")[-1] + ".jpg"
                base = fname.replace("_query.txt", "")
                gt_file = os.path.join(gt_dir, f"{base}_good.txt")
                if os.path.exists(gt_file):
                    with open(gt_file) as gtf:
                        gt_list = [f"dataset/oxbuild_images/{line.strip().split('oxc1_')[-1]}.jpg" for line in gtf.readlines()]
                    queries.append((f"dataset/oxbuild_images/{qimg}", gt_list))
    return queries

# Đánh giá P@k
def precision_at_k(results, gt_list, k=10):
    hits = sum(1 for path in results[:k] if path in gt_list)
    return hits / k

# Chạy đánh giá
def evaluate():
    queries = load_queries()
    results = []

    for qimg, gt in queries:
        # Autoencoder
        q_feat = extract_query_feature(qimg)
        ae_indices = perform_search(q_feat, ae_features, top_k=10)
        ae_result_paths = [ae_paths[i] for i in ae_indices]

        # CLIP
        clip_result_paths, _ = handle_image_query(qimg, top_k=10)

        # Precision@5
        ae_prec = precision_at_k(ae_result_paths, gt, k=5)
        clip_prec = precision_at_k(clip_result_paths, gt, k=5)

        results.append({
            "query": os.path.basename(qimg),
            "AE@5": ae_prec,
            "CLIP@5": clip_prec
        })

    # Lưu kết quả
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("static/evaluation_results.csv", index=False)
    print("✅ Đã lưu: evaluation_results.csv")

if __name__ == "__main__":
    evaluate()


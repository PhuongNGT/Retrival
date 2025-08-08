import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from retriever import extract_query_feature, perform_search
from clip_query_handler import handle_image_query
import torch

# =========================
# Config
# =========================
GT_DIR = "dataset/gt_files"
IMG_DIR = "dataset/oxbuild_images"
INCLUDE_OK_AS_RELEVANT = True   # Oxford thường tính good+ok là relevant

TOP_K_LIST = [1, 5, 10]         # các K muốn đánh giá

# =========================
# Load indexes
# =========================
# Autoencoder
ae_features = pickle.load(open("features.pkl", "rb"))
ae_paths    = pickle.load(open("image_paths.pkl", "rb"))
ae_paths = [os.path.normpath(p) for p in ae_paths]

# CLIP
clip_image_features = torch.load("clip_outputs/clip_image_features.npy").numpy()
clip_paths = pickle.load(open("clip_outputs/clip_image_paths.pkl", "rb"))
clip_paths = [os.path.normpath(p) for p in clip_paths]

# =========================
# Ground truth loader (Oxford)
# =========================
def load_queries(gt_dir=GT_DIR):
    """Trả về list (query_image_path, relevant_list) với relevant là list path ảnh ground-truth."""
    queries = []
    for fname in sorted(os.listdir(gt_dir)):
        if fname.endswith("_query.txt"):
            base = fname.replace("_query.txt", "")
            # Đọc file query: dòng có "oxc1_<name> x y w h"
            with open(os.path.join(gt_dir, fname), "r") as f:
                line = f.readline().strip()
                qname = line.split()[0].split("oxc1_")[-1]  # bỏ tiền tố
                qimg  = os.path.join(IMG_DIR, f"{qname}.jpg")

            # Relevance: good (+ ok nếu bật)
            rel = []
            good_file = os.path.join(gt_dir, f"{base}_good.txt")
            ok_file   = os.path.join(gt_dir, f"{base}_ok.txt")
            junk_file = os.path.join(gt_dir, f"{base}_junk.txt")

            def read_list(fp):
                if not os.path.exists(fp): return []
                with open(fp, "r") as fin:
                    items = []
                    for line in fin:
                        name = line.strip().split("oxc1_")[-1]  # có file dùng tiền tố
                        items.append(os.path.normpath(os.path.join(IMG_DIR, f"{name}.jpg")))
                    return items

            rel += read_list(good_file)
            if INCLUDE_OK_AS_RELEVANT:
                rel += read_list(ok_file)

            # (tuỳ chọn) có thể loại junk khỏi tập ảnh, nhưng ở đây chỉ dùng để tính relevance nên không cần

            queries.append((os.path.normpath(qimg), sorted(set(rel))))
    return queries

# =========================
# Metrics
# =========================
def make_relevance_vector(rank_paths, relevant_set, k=None):
    """Trả về mảng 0/1 cho danh sách kết quả theo relevant_set. Nếu k!=None → cắt top-k."""
    if k is not None:
        rank_paths = rank_paths[:k]
    return np.array([1 if p in relevant_set else 0 for p in rank_paths], dtype=np.int32)

def precision_at_k(rank_paths, relevant_set, k):
    rel = make_relevance_vector(rank_paths, relevant_set, k)
    return rel.sum() / max(k, 1)

def recall_at_k(rank_paths, relevant_set, k):
    R = max(len(relevant_set), 1)
    rel = make_relevance_vector(rank_paths, relevant_set, k)
    return rel.sum() / R

def f1_at_k(rank_paths, relevant_set, k):
    p = precision_at_k(rank_paths, relevant_set, k)
    r = recall_at_k(rank_paths, relevant_set, k)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

def hit_rate_at_k(rank_paths, relevant_set, k):
    rel = make_relevance_vector(rank_paths, relevant_set, k)
    return float(rel.sum() > 0)

def mrr(rank_paths, relevant_set):
    for i, p in enumerate(rank_paths, start=1):
        if p in relevant_set:
            return 1.0 / i
    return 0.0

def average_precision(rank_paths, relevant_set):
    """AP = mean precision@i ở các vị trí i có hit; chuẩn IR cổ điển."""
    rel = make_relevance_vector(rank_paths, relevant_set, None)
    if rel.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, r in enumerate(rel, start=1):
        if r == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0

def r_precision(rank_paths, relevant_set):
    """Precision@R với R = số ảnh relevant thực tế (good+ok)."""
    R = max(len(relevant_set), 1)
    return precision_at_k(rank_paths, relevant_set, R)

def dcg_at_k(rel, k):
    rel = np.asfarray(rel[:k])
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float((rel * discounts).sum())

def ndcg_at_k(rank_paths, relevant_set, k):
    rel = make_relevance_vector(rank_paths, relevant_set, None)  # full list
    ideal = np.sort(rel)[::-1]
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0.0 else dcg / idcg

# =========================
# Search wrappers
# =========================
def search_autoencoder(query_img_path, top_k=10):
    q_feat = extract_query_feature(query_img_path)             # vector đặc trưng
    idxs = perform_search(q_feat, ae_features, top_k=top_k)    # trả index
    return [ae_paths[i] for i in idxs]

def search_clip(query_img_path, top_k=10):
    # handler trả (paths, scores) -> lấy paths
    paths, _ = handle_image_query(query_img_path, top_k=top_k)
    return [os.path.normpath(p) for p in paths]

# =========================
# Evaluate
# =========================
def evaluate():
    queries = load_queries()
    per_query_rows = []

    # để tính mAP, MRR trung bình theo phương pháp
    ap_ae, ap_clip = [], []
    mrr_ae, mrr_clip = [], []

    for qimg, gt_list in queries:
        rel_set = set(gt_list)

        # AE
        ae_rank = search_autoencoder(qimg, top_k=1000)  # đủ dài cho AP/MRR
        # CLIP
        clip_rank = search_clip(qimg, top_k=1000)

        row = {
            "query": os.path.basename(qimg),
            "R": len(rel_set),
            "AE_AP": average_precision(ae_rank, rel_set),
            "CLIP_AP": average_precision(clip_rank, rel_set),
            "AE_MRR": mrr(ae_rank, rel_set),
            "CLIP_MRR": mrr(clip_rank, rel_set),
            "AE_R-Prec": r_precision(ae_rank, rel_set),
            "CLIP_R-Prec": r_precision(clip_rank, rel_set),
        }

        # mAP/MRR tổng hợp
        ap_ae.append(row["AE_AP"])
        ap_clip.append(row["CLIP_AP"])
        mrr_ae.append(row["AE_MRR"])
        mrr_clip.append(row["CLIP_MRR"])

        # theo nhiều K
        for k in TOP_K_LIST:
            row[f"AE_P@{k}"]   = precision_at_k(ae_rank, rel_set, k)
            row[f"CLIP_P@{k}"] = precision_at_k(clip_rank, rel_set, k)

            row[f"AE_R@{k}"]   = recall_at_k(ae_rank, rel_set, k)
            row[f"CLIP_R@{k}"] = recall_at_k(clip_rank, rel_set, k)

            row[f"AE_F1@{k}"]   = f1_at_k(ae_rank, rel_set, k)
            row[f"CLIP_F1@{k}"] = f1_at_k(clip_rank, rel_set, k)

            row[f"AE_Hit@{k}"]   = hit_rate_at_k(ae_rank, rel_set, k)
            row[f"CLIP_Hit@{k}"] = hit_rate_at_k(clip_rank, rel_set, k)

            row[f"AE_nDCG@{k}"]   = ndcg_at_k(ae_rank, rel_set, k)
            row[f"CLIP_nDCG@{k}"] = ndcg_at_k(clip_rank, rel_set, k)

        per_query_rows.append(row)

    # Lưu per-query
    df = pd.DataFrame(per_query_rows)
    os.makedirs("static", exist_ok=True)
    df.to_csv("static/evaluation_results.csv", index=False)
    print("✅ Đã lưu: static/evaluation_results.csv")

    # Tổng hợp (mean)
    summary = {
        "mAP_AE":   float(np.mean(ap_ae)) if ap_ae else 0.0,
        "mAP_CLIP": float(np.mean(ap_clip)) if ap_clip else 0.0,
        "MRR_AE":   float(np.mean(mrr_ae)) if mrr_ae else 0.0,
        "MRR_CLIP": float(np.mean(mrr_clip)) if mrr_clip else 0.0,
    }
    # trung bình các metric @K
    for k in TOP_K_LIST:
        for metric in ["P", "R", "F1", "Hit", "nDCG"]:
            ae_col = f"AE_{metric}@{k}"
            cl_col = f"CLIP_{metric}@{k}"
            summary[f"{ae_col}_mean"]  = float(df[ae_col].mean())
            summary[f"{cl_col}_mean"]  = float(df[cl_col].mean())

    # In nhanh
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    # Lưu summary CSV
    pd.DataFrame([summary]).to_csv("static/evaluation_summary.csv", index=False)
    print("✅ Đã lưu: static/evaluation_summary.csv")

if __name__ == "__main__":
    evaluate()

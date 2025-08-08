# app.py
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os, shutil, time

# ==== search handlers (giữ nguyên theo dự án của bạn) ====
from clip_query_handler import handle_image_query as handle_clip_query
from retriever import handle_autoencoder_query

# ==== dashboard deps ====
import numpy as np, joblib, pickle
from PIL import Image

# Matplotlib: bắt buộc dùng backend không GUI khi chạy server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Flask app & paths
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # giảm cache static khi dev

BASE = os.path.dirname(os.path.abspath(__file__))
def P(*parts): return os.path.join(BASE, *parts)

UPLOAD_FOLDER = P("static", "uploads")
RESULT_FOLDER = P("static", "results")
STATIC_DIR    = P("static")
REPS_DIR      = P("static", "cluster_reps")
BAR_PATH      = P("static", "cluster_bar.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Dashboard helpers
# -----------------------------------------------------------------------------
def plot_cluster_distribution(labels, save_path=BAR_PATH):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8,5))
    plt.bar(unique, counts)
    plt.xlabel("Cluster ID"); plt.ylabel("Number of Images")
    plt.title("📊 Number of Images per Cluster"); plt.grid(True)

    # tmp phải giữ đúng extension (png, jpg, ...)
    root, ext = os.path.splitext(save_path)            # vd: ("static/cluster_bar", ".png")
    tmp = f"{root}.tmp{ext}"                           # -> "static/cluster_bar.tmp.png"

    plt.savefig(tmp, bbox_inches="tight")
    plt.clf()
    plt.close("all")

    os.replace(tmp, save_path)

def save_cluster_representatives(features, labels, model, image_paths):
    for cluster_id in range(model.n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        cluster_feats = features[indices]
        center = model.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_feats - center, axis=1)
        rep_idx = indices[np.argmin(dists)]

        img = Image.open(image_paths[rep_idx]).convert("RGB")
        img = img.resize((128, 128))

        out = os.path.join(REPS_DIR, f"cluster_{cluster_id}.jpg")
        root, ext = os.path.splitext(out)              # -> (...."/cluster_0", ".jpg")
        tmp = f"{root}.tmp{ext}"                       # -> ..."/cluster_0.tmp.jpg"

        img.save(tmp)                                  # PIL thấy .jpg OK
        os.replace(tmp, out)                           # ghi đè an toàn


def ensure_dashboard_assets(force: bool = False):
    """Đảm bảo có cluster_bar.png và ảnh đại diện cụm. Nếu force=True thì ép tạo lại."""
    labels = np.load(P("cluster_labels.npy"))
    model  = joblib.load(P("cluster_model.pkl"))

    need_bar  = force or (not os.path.exists(BAR_PATH))
    expected  = [os.path.join(REPS_DIR, f"cluster_{i}.jpg") for i in range(model.n_clusters)]
    need_reps = force or any(not os.path.exists(p) for p in expected)

    if need_bar:
        plot_cluster_distribution(labels, BAR_PATH)

    if need_reps:
        features    = np.load(P("latent_features.npy"))
        image_paths = pickle.load(open(P("image_paths.pkl"), "rb"))
        save_cluster_representatives(features, labels, model, image_paths)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # 'clip' hoặc 'autoencoder'
    method = request.form['method']
    image  = request.files['image']
    filename = secure_filename(image.filename)
    if not filename:
        return "Chưa chọn file ảnh.", 400

    # Lưu ảnh truy vấn
    query_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(query_path)

    # Query
    if method == 'clip':
        results, _ = handle_clip_query(query_path, top_k=5)
    else:
        results, _ = handle_autoencoder_query(query_path, top_k=5)

    # Copy kết quả vào static/results để render dễ
    result_files = []
    for i, src in enumerate(results):
        dest = os.path.join(RESULT_FOLDER, f"result_{i}.jpg")
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy(src, dest)
        result_files.append(f"result_{i}.jpg")

    # Ép regenerate dashboard sau mỗi lần search
    ensure_dashboard_assets(force=True)

    # Bust cache cho biểu đồ trên trang results
    bar_version = int(os.path.getmtime(BAR_PATH)) if os.path.exists(BAR_PATH) else 0

    return render_template(
        'results.html',
        query_image=os.path.join("uploads", filename),  # để template dùng url_for('static', filename=query_image)
        result_images=result_files,
        bar_version=bar_version
    )

@app.route('/dashboard')
def dashboard():
    # Đảm bảo asset tồn tại
    ensure_dashboard_assets()

    # Danh sách ảnh đại diện cụm (kèm version theo mtime để phá cache)
    kmeans = joblib.load(P("cluster_model.pkl"))
    rep_images = []
    for i in range(kmeans.n_clusters):
        fpath = os.path.join(REPS_DIR, f"cluster_{i}.jpg")
        v = int(os.path.getmtime(fpath)) if os.path.exists(fpath) else 0
        rep_images.append(url_for("static", filename=f"cluster_reps/cluster_{i}.jpg", v=v))

    # Bar chart kèm version
    bar_mtime = int(os.path.getmtime(BAR_PATH)) if os.path.exists(BAR_PATH) else 0
    bar_img = url_for("static", filename="cluster_bar.png", v=bar_mtime)

    # (tuỳ chọn) focus cụm được truyền qua query string: /dashboard?focus=3
    focus = request.args.get("focus", default=None, type=int)

    return render_template('cluster_dashboard.html', rep_images=rep_images, bar_img=bar_img, focus=focus)

@app.route('/report')
def report():
    return render_template('report.html')

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # debug=True chỉ dùng dev
    app.run(debug=True)

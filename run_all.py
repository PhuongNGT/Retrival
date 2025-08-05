import os
import subprocess
import webbrowser
import time

# Check if files exist
def file_exists(f):
    return os.path.exists(f)

# Skip nếu file output đã tồn tại
SKIP_CHECKS = {
    "train.py": ["conv_autoencoderv2_oxford.pt"],
    "build_index.py": ["features.pkl", "image_paths.pkl"],
    "cluster_analysis.py": ["cluster_labels.npy", "cluster_model.pkl"],
    "generate_dashboard.py": ["static/cluster_bar.png"],
    "clip_feature_extractor.py": ["clip_outputs/clip_image_features.npy", "clip_outputs/clip_text_features.npy"],
    "build_index_clip.py": ["clip_outputs/clip_image_paths.pkl"],
    "evaluate_clip_vs_ae.py": ["evaluation_results.csv"],
    "report_generator.py": ["precision_comparison.png", "report.html"]
}

# Danh sách các file chính sẽ chạy tuần tự
main_files = [
    "train.py",                   # huấn luyện autoencoder
    "build_index.py",             # trích đặc trưng autoencoder
    "clip_feature_extractor.py", # trích đặc trưng CLIP
    "build_index_clip.py",       # index CLIP
    "cluster_analysis.py",       # clustering
    "generate_dashboard.py",     # biểu đồ cluster
    "evaluate_clip_vs_ae.py",    # đánh giá
    "report_generator.py",       # vẽ đồ thị và sinh báo cáo HTML
]

def run_step(file):
    print(f"\n🚀 Running: {file}")
    subprocess.run(["python", file])

def main():
    for f in main_files:
        if f in SKIP_CHECKS:
            if all(file_exists(out) for out in SKIP_CHECKS[f]):
                print(f"✅ Skipped {f} (outputs exist)")
                continue
        run_step(f)


    print("\n🌐 [4] Launching Flask app (http://127.0.0.1:5000)")
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")
    os.system("python app.py")  # Run Flask app (waits for Ctrl+C)

if __name__ == "__main__":
    main()


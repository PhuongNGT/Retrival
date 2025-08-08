# report_generator.py
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

csv_path = "static/evaluation_results.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("Chưa thấy static/evaluation_results.csv — chạy evaluate.py trước.")

df = pd.read_csv(csv_path)

# Tương thích tên cột
col_ae = "AE_P@5" if "AE_P@5" in df.columns else ("AE@5" if "AE@5" in df.columns else None)
col_cl = "CLIP_P@5" if "CLIP_P@5" in df.columns else ("CLIP@5" if "CLIP@5" in df.columns else None)
if col_ae is None or col_cl is None:
    raise ValueError("Không tìm thấy cột Precision@5 trong evaluation_results.csv")

ae_mean   = df[col_ae].mean()
clip_mean = df[col_cl].mean()

# Vẽ và lưu biểu đồ (nếu bạn đã có sẵn cũng OK — đoạn này có thể giữ)
plot_path = "static/precision_comparison.png"
plt.figure(figsize=(8, 5))
plt.bar(["Autoencoder", "CLIP"], [ae_mean, clip_mean])
plt.ylabel("Precision@5")
plt.title("So sánh Precision@5: CLIP vs Autoencoder")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

# cache-busting
plot_mtime = int(os.path.getmtime(plot_path))
csv_mtime  = int(os.path.getmtime(csv_path))

# (tuỳ chọn) summary
summary_path = "static/evaluation_summary.csv"
summary_html = ""
if os.path.exists(summary_path):
    s = pd.read_csv(summary_path).round(4)
    summary_html = s.to_html(index=False, border=1)

# ✅ Không dùng Jinja nữa — nhúng URL tuyệt đối tới /static
plot_url = f"/static/precision_comparison.png?v={plot_mtime}"
csv_url  = f"/static/evaluation_results.csv?v={csv_mtime}"

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Báo cáo So sánh CLIP vs Autoencoder</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #2c3e50; line-height: 1.6; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px 0; min-width: 300px; }}
    table, th, td {{ border: 1px solid #bbb; }}
    th, td {{ padding: 8px 12px; text-align: center; }}
    th {{ background-color: #ecf0f1; }}
    img {{ max-width: 680px; height: auto; border: 1px solid #ccc; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .nav a {{ display: inline-block; margin-right: 12px; padding: 6px 12px; text-decoration: none; color: #fff; font-weight: bold; background-color: #2980b9; border-radius: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
  </style>
</head>
<body>
  <div class="nav">
    <a href="/">Trang chủ</a>
    <a href="/dashboard">Cụm ảnh</a>
  </div>

  <h1>📊 Báo cáo So sánh CLIP vs Autoencoder</h1>

  <h2>🎯 Precision@5 trung bình</h2>
  <table>
    <tr><th>Phương pháp</th><th>Precision@5</th></tr>
    <tr><td>Autoencoder</td><td>{ae_mean:.4f}</td></tr>
    <tr><td>CLIP</td><td>{clip_mean:.4f}</td></tr>
  </table>

  <div class="grid">
    <div>
      <h2>📈 Biểu đồ</h2>
      <img src="{plot_url}" alt="Precision Comparison">
    </div>
    <div>
      <h2>📂 Tệp kết quả</h2>
      <p><a href="{csv_url}" download>evaluation_results.csv</a></p>
      {"<h3>Tóm tắt</h3>" + summary_html if summary_html else ""}
    </div>
  </div>
</body>
</html>"""

with open("templates/report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Đã tạo templates/report.html — mở qua Flask tại: http://localhost:5000/report")

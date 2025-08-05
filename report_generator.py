import pandas as pd
import matplotlib.pyplot as plt
import webbrowser

# Đọc kết quả đánh giá
df = pd.read_csv("evaluation_results.csv")
ae_mean = df["AE@5"].mean()
clip_mean = df["CLIP@5"].mean()

print(f"🎯 Autoencoder P@5: {ae_mean:.4f}")
print(f"🎯 CLIP         P@5: {clip_mean:.4f}")

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.bar(["Autoencoder", "CLIP"], [ae_mean, clip_mean], color=["blue", "green"])
plt.ylabel("Precision@5")
plt.title("So sánh Precision@5 giữa CLIP và Autoencoder")
plt.savefig("precision_comparison.png")
print("📊 Đã lưu: precision_comparison.png")

# Tạo nội dung HTML báo cáo
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Báo cáo So sánh CLIP vs Autoencoder</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 40px;
    }}
    h1 {{
      color: #2c3e50;
    }}
    table {{
      width: 50%;
      border-collapse: collapse;
      margin-bottom: 30px;
    }}
    table, th, td {{
      border: 1px solid #bbb;
    }}
    th, td {{
      padding: 8px 12px;
      text-align: center;
    }}
    th {{
      background-color: #ecf0f1;
    }}
    img {{
      max-width: 600px;
      height: auto;
      border: 1px solid #ccc;
    }}
  </style>
</head>
<body>

  <h1>📊 Báo cáo So sánh CLIP vs Autoencoder</h1>

  <h2>🎯 Precision@5 Trung bình</h2>
  <table>
    <tr>
      <th>Phương pháp</th>
      <th>Precision@5</th>
    </tr>
    <tr>
      <td>Autoencoder</td>
      <td>{ae_mean:.4f}</td>
    </tr>
    <tr>
      <td>CLIP</td>
      <td>{clip_mean:.4f}</td>
    </tr>
  </table>

  <h2>📈 Biểu đồ</h2>
  <img src="../precision_comparison.png" alt="Precision Comparison">

  <p style="margin-top: 40px">✅ File kết quả chi tiết: <strong>evaluation_results.csv</strong></p>

</body>
</html>
"""

# Ghi file HTML
with open("templates/report.html", "w") as f:
    f.write(html_content)

print("📝 Đã tạo báo cáo: report.html")

# Mở bằng trình duyệt
webbrowser.open("templates/report.html")


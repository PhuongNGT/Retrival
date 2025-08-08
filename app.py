from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
from clip_query_handler import handle_image_query as handle_clip_query
from retriever import handle_autoencoder_query
from PIL import Image
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    method = request.form['method']  # 'clip' or 'autoencoder'
    image = request.files['image']
    filename = secure_filename(image.filename)
    query_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(query_path)

    # Query
    if method == 'clip':
        results, _ = handle_clip_query(query_path, top_k=5)
    else:
        results, _ = handle_autoencoder_query(query_path, top_k=5)

    # Copy results to static/results for display
    for i, path in enumerate(results):
        dest = os.path.join(RESULT_FOLDER, f"result_{i}.jpg")
        shutil.copy(path, dest)

    return render_template('results.html', query_image=filename, result_images=[f"result_{i}.jpg" for i in range(len(results))])

@app.route('/dashboard')
def dashboard():
    cluster_dir = 'static/cluster_reps'
    clusters = sorted(os.listdir(cluster_dir))
    return render_template('cluster_dashboard.html', clusters=clusters)

@app.route('/report')
def report():
    return render_template('report.html')
    
if __name__ == '__main__':
    app.run(debug=True)


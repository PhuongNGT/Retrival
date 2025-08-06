# Image Retrieval System: Autoencoder vs CLIP

This project benchmarks two visual encoding approaches for image retrieval:
- A convolutional autoencoder
- The CLIP model (image-only retrieval)

Dataset: [Oxford Buildings Dataset (~5K images)](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)

---

## 🔧 Project Structure

### 1. Autoencoder Branch
- `train.py`: Train convolutional autoencoder on `oxbuild_images/`
- `build_index.py`: Extract latent features and save to `features.pkl`
- `cluster_analysis.py`: Cluster features with KMeans
- `generate_dashboard.py`: Create bar chart and cluster representatives
- `retriever.py`: Perform image retrieval (Euclidean distance)

### 2. CLIP Branch
- `clip_feature_extractor.py`: Extract image features using `openai/clip-vit-base-patch32`
- `build_index_clip.py`: Save features and image paths
- `clip_query_handler.py`: Retrieve similar images using cosine similarity
- `clip_retriever.py`: Search utilities for CLIP

### 3. Web Interface
- `app.py`: Flask web app with 3 routes:
  - `/` → Upload query image
  - `/search` → Display retrieved results
  - `/dashboard` → Cluster summary
- `templates/`: HTML pages for web interface

### 4. Evaluation
- `evaluate_clip_vs_ae.py`: Compare retrieval accuracy on standard queries
- `report_generator.py`: Output comparative metrics and visuals

---

## 🖼️ Input
- Folder: `oxbuild_images/`
- Query set: `gt_files_170407/`
- ⚠️ No captions → text-to-image retrieval is disabled

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt

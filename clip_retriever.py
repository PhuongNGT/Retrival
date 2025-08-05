# clip_retriever.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_clip_image(query_feature, image_features, top_k=10):
    """
    Truy vấn bằng đặc trưng ảnh → trả về top K ảnh tương tự nhất
    """
    sims = cosine_similarity([query_feature], image_features)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return top_indices, sims[top_indices]

def search_clip_text(query_text_feature, image_features, top_k=10):
    """
    Truy vấn bằng đặc trưng văn bản → trả về top K ảnh tương ứng
    """
    sims = cosine_similarity([query_text_feature], image_features)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return top_indices, sims[top_indices]


import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt

# Load FAISS index and data
import faiss
import numpy as np

from search import search_similar_images 
from features.model import extract_features

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load previously saved data
features = np.load("features/image_features.npy")
image_paths = np.load("features/image_paths.npy")
class_names = np.load("features/class_names.npy")
index = faiss.read_index("features/image_retrieval.index")

def gradio_search(query_image):
    # Extract features for query image
    query_feat = extract_features(query_image)
    query_feat = np.expand_dims(query_feat.astype('float32'), axis=0)
    
    # Search
    results = search_similar_images(query_feat, index, image_paths, k=5)
    
    # Prepare images and captions in the correct format
    result_images = []
    for path, score in results:
        img = Image.open(path).resize((128, 128))
        caption = f"{os.path.basename(path)}\nScore: {score:.2f}"
        result_images.append((img, caption))  # Correct format: (image, caption)
    
    return result_images

# Gradio interface
iface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Image(type="filepath", label="Upload Query Image"),
    outputs=gr.Gallery(label="Top 5 Similar Images"),
    title="Image-Based Retrieval with FAISS Created by Mahdi Savoji",
)

iface.launch()

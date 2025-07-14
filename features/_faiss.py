import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import faiss
import os
import numpy as np
from tqdm import tqdm
from model import extract_features

# Root setup
root_path = r"D:\Course\Filoger-Advanced-CV\Projects\proj4"
image_dir = os.path.join(root_path, "images")
save_dir = os.path.join(root_path, "features")  

image_paths = []
class_names = []

# Collect image paths and class names
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)
            class_name = os.path.basename(root)
            class_names.append(class_name)

print(f"Found {len(image_paths)} images across {len(set(class_names))} classes")

# Feature extraction
features = []
valid_image_paths = []
valid_class_names = []

for path, cls in tqdm(zip(image_paths, class_names), total=len(image_paths)):
    try:
        feat = extract_features(path)
        features.append(feat)
        valid_image_paths.append(path)
        valid_class_names.append(cls)
    except Exception as e:
        print(f"Error processing {path}: {e}")

features = np.array(features).astype('float32')

# Save features and metadata
np.save(os.path.join(save_dir, "image_features.npy"), features)
np.save(os.path.join(save_dir, "image_paths.npy"), np.array(valid_image_paths))
np.save(os.path.join(save_dir, "class_names.npy"), np.array(valid_class_names))

# Load back for indexing
features = np.load(os.path.join(save_dir, "image_features.npy"))

print(features.shape)
# Create FAISS index
dimension = features.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add vectors and save index
index.add(features)



# dimension = features.shape[1]

# # Parameters for IVF + PQ
# nlist = min(10, features.shape[0])  # number of Voronoi cells (clusters), <= training samples
# m = 16           # number of subquantizers (must divide dimension)
# nbits = 8        # bits per code

# quantizer = faiss.IndexFlatL2(dimension)  # coarse quantizer

# index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

# assert not index.is_trained
# training_sample = features[np.random.choice(features.shape[0], nlist, replace=False)]
# index.train(training_sample)

# index.add(features)




faiss.write_index(index, os.path.join(save_dir, "image_retrieval.index"))


# ViT-FAISS Image Retrieval

A deep learning-based **Image Retrieval System** built using:

- **Vision Transformer (ViT)** for feature extraction
- **FAISS (Facebook AI Similarity Search)** for efficient similarity indexing and searching.

------

## 🚀 Features

- Extracts powerful deep features from images using pretrained ViT models.
- Indexes feature vectors with **FAISS** for fast similarity search.
- Supports **IndexFlatL2** and **IndexIVFPQ** for scalable approximate search.
- Stores extracted features and metadata for reuse.
- Example code for querying similar images (coming soon).

------

## 🎬 Demo

![Image-Retrieva-FAISS-Demo](output.gif)

------

## 📦 Installation

```bash
git clone https://github.com/Mahdi-Savoji/ImageRetrieval-FAISS.git
cd ImageRetrieval-FAISS
pip install -r requirements.txt
```

------

## ⚙️ Usage

### 1. Feature Extraction

Extract features from a folder of images:

```bash
python features/_faiss.py
```

### 2. Image Search Example *(WIP)*

You can retrieve similar images by querying the FAISS index with the extracted feature of a query image.

#### Steps:

1. Load the query image and extract its feature vector using the same ViT model.
2. Load the pre-built FAISS index and corresponding image paths.
3. Use the provided `search_similar_images` function to retrieve the top-k similar images.
4. The function returns image paths along with their distances.

------

## 📂 Project Structure

```
.
├── features/
│   ├── _faiss.py         # Feature extraction and FAISS indexing
│   └── model.py          # feature extraction model
├── images/               # Input images dataset
├── saved_indices/        # Directory to store FAISS index and feature data
├── requirements.txt
└── README.md
```

------

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- transformers
- faiss
- numpy
- Pillow
- tqdm

Install them with:

```bash
pip install torch torchvision transformers faiss-cpu numpy pillow tqdm
```

------

## 🤖 Models

- Vision Transformer (ViT-B/16) from Hugging Face Transformers library.


import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt

import faiss
import numpy as np

from search import search_similar_images 
from features.model import extract_features

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_path = r"D:\Course\Filoger-Advanced-CV\Projects\proj4"
image_dir = os.path.join(root_path, "images")

# Load previously saved data
features = np.load("features/image_features.npy")
image_paths = np.load("features/image_paths.npy").tolist()  
class_names = np.load("features/class_names.npy").tolist() 
index = faiss.read_index("features/image_retrieval.index")

from collections import Counter

last_query_info = {}

def gradio_search(query_image):
    query_feat = extract_features(query_image)
    query_feat = np.expand_dims(query_feat.astype('float32'), axis=0)

    results, predicted_classes = search_similar_images(query_feat, index, image_paths, class_names, k=5)

    class_count = Counter(predicted_classes)
    predicted_class = class_count.most_common(1)[0][0]

    result_images = []
    for path, score in results:
        img = Image.open(path).resize((128, 128))
        caption = f"{os.path.basename(path)}\distance: {score:.2f}"
        result_images.append((img, caption))

    last_query_info['query_image'] = query_image
    last_query_info['query_feat'] = query_feat
    last_query_info['predicted_class'] = predicted_class
    last_query_info['results'] = results 

    return predicted_class, result_images


def add_image_to_dataset():
    if not last_query_info:
        return "No query image to add."

    predicted_class = last_query_info['predicted_class']
    query_image = last_query_info['query_image']
    results = last_query_info.get('results', None)

    # Check if closest distance is zero (exact duplicate)
    if results and results[0][1] == 0:
        return "This photo already exists in the dataset."

    if isinstance(query_image, str):
        img = Image.open(query_image).convert("RGB")
    elif isinstance(query_image, np.ndarray):
        img = Image.fromarray(query_image.astype('uint8')).convert('RGB')
    else:
        return "Query image is invalid."

    save_dir = os.path.join(image_dir, predicted_class)
    os.makedirs(save_dir, exist_ok=True)
    img_name = f"augmented_{len(image_paths)}.jpg"
    save_path = os.path.join(save_dir, img_name)
    img.save(save_path)

    image_paths.append(save_path)
    class_names.append(predicted_class)

    global features, index
    features = np.vstack([features, last_query_info['query_feat']])
    index.add(last_query_info['query_feat'])

    np.save("features/image_features.npy", features)
    np.save("features/image_paths.npy", np.array(image_paths))
    np.save("features/class_names.npy", np.array(class_names))
    faiss.write_index(index, "features/image_retrieval.index")

    return f"Image added to dataset under class '{predicted_class}'."



with gr.Blocks(title="Image Retrieval") as demo:
    gr.Markdown("## 🔎 Image Retrieval System")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Query Image", height=400)
            search_button = gr.Button("🔍 Search Similar Images")
        with gr.Column(scale=2):
                predicted_label = gr.Markdown("**Predicted Class:** _None yet_")
                gallery_output = gr.Gallery(label="Top 5 Similar Images", columns=5, height=400)

    search_button.click(gradio_search, inputs=image_input, outputs=[predicted_label, gallery_output])

    gr.Markdown("---")

    add_button = gr.Button("✅ Confirm & Add to Dataset")

    def notify_addition():
        message = add_image_to_dataset()
        gr.Info(message)

    add_button.click(notify_addition, inputs=[], outputs=[])

demo.launch()






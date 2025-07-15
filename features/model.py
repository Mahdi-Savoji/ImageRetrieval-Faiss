import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Preprocessing
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# def extract_features(image_path):
#     model = models.resnet50(pretrained=True)
#     model = torch.nn.Sequential(*(list(model.children())[:-1]))
#     model.eval()
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = preprocess(image).unsqueeze(0)
#     with torch.no_grad():
#         features = model(image_tensor)
#     return features.squeeze().numpy()


def extract_features(image_path):
    model_name = 'google/vit-base-patch16-224-in21k'
    
    #Loads the feature extractor, which resizes, crops, normalizes, and converts the image to the expected tensor format.
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
   
    #Loads the pretrained ViT model (without a classification head)
    model = ViTModel.from_pretrained(model_name)
    
    image = Image.open(image_path)
    
    # inputs is a dictionary returned by ViTFeatureExtractor, usually containing:
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # last_hidden_state: a tensor of shape [batch_size, sequence_length, hidden_size] — for ViT Base: [1, 197, 768].
    # 197 = 1 CLS token + 14x14 patches = 196 tokens
    outputs = model(**inputs)
    
    # token index 0 is CLS 
    return outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy() 
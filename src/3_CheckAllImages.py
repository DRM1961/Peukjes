# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 23:36:06 2025

@author: dirkm
"""
import torch
from PIL import Image
from common import FeatureExtractor18, FeatureExtractor34, FeatureExtractor50,MLPClassifier, transform

from sklearn.metrics import roc_curve
import joblib
import os
import numpy as np
import pandas as pd

USED_RESNET = 34
feature_extractor = None
fnscaler = ''
fnallfeatures = ''
fnalllabels = ''
fnmodel = ''

def extract_features(image_path, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = feature_extractor(image)
    return(feature)


def find_best_threshold(test_features, test_labels):
    all_scores = []  # Store output[0, 0] for each image
    true_labels = []  # Store 0 for Good, 1 for Bad

    with torch.no_grad():
        for img_feature, label in zip(test_features, test_labels):
            img_feature = torch.tensor(img_feature, dtype=torch.float32).unsqueeze(0)
            output = model(img_feature)
            all_scores.append(output[0, 0].item())  # Store the first score
            true_labels.append(label)  # Store ground truth

    # Compute the best threshold
    fpr, tpr, thresholds = roc_curve(true_labels, all_scores)
    optimal_idx = (tpr - fpr).argmax()  # Best trade-off between TP and FP
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")


print(f'load model resnet{USED_RESNET}')
if USED_RESNET == 50:
    feature_extractor = FeatureExtractor50()
    input_dim = 2048  # Adjust based on ResNet feature size: 34 = 512, 50 = 2048
    fnscaler = '../Data/scaler_50.pkl'
    fnallfeatures = '../Data/all_features_50.npy'
    fnalllabels = '../Data/all_labels_50.npy'
    fnmodel = '../Data/mlp_model_50.pth'
elif USED_RESNET == 34:
    feature_extractor = FeatureExtractor34()
    input_dim = 512  # Adjust based on ResNet feature size: 34 = 512, 50 = 2048
    fnscaler = '../Data/scaler_34.pkl'
    fnallfeatures = '../Data/all_features_34.npy'
    fnalllabels = '../Data/all_labels_34.npy'
    fnmodel = '../Data/mlp_model_34.pth'
elif USED_RESNET == 18:
    feature_extractor = FeatureExtractor18()
    input_dim = 512
    fnscaler = '../Data/scaler_18.pkl'
    fnallfeatures = '../Data/all_features_18.npy'
    fnalllabels = '../Data/all_labels_18.npy'
    fnmodel = '../Data/mlp_model_18.pth'
else:
    print(f"error: Resnet{USED_RESNET} not defined")
    exit


scaler = joblib.load(fnscaler)
model = MLPClassifier(input_dim)
model.load_state_dict(torch.load(fnmodel))  # Load saved weights
model.eval()

print("Model loaded successfully and ready for inference!")


# select test dataset = all images or a handpicked selection
test_dirs = ["../Images/DataSet"]

for test_dir in test_dirs:
    test_images = []
    test_tensors = []
    test_probabilities = []

    for filename in os.listdir(test_dir):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(test_dir, filename)
            test_images.append(img_path)
    print(f'checking {len(test_images)} images')

    for img_path in test_images:
        image_feature = extract_features(img_path, feature_extractor)
        image_feature = scaler.transform(image_feature.reshape(1, -1))  # Ensure correct shape

        with torch.no_grad():
            sample_feature = torch.tensor(image_feature, dtype=torch.float32)

            if sample_feature.ndim == 1:  # Ensure correct shape (batch size 1)
                sample_feature = sample_feature.unsqueeze(0)  # Convert [feature_dim] → [1, feature_dim]

            output = model(sample_feature)
            predicted_label = torch.argmax(output, dim=1)

            test_tensors.append([img_path, output[0,0].item(),output[0,1].item(), predicted_label])

            '''
            good_score = output[0, 0].item()
            bad_score = output[0, 1].item()
            threshold = 0.5
            predicted_label = 0 if good_score > threshold else 1
            '''

            probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
            predicted_label = torch.argmax(probabilities, dim=1).item()
            test_probabilities.append([img_path, probabilities[0,0].item(),probabilities[0,1].item(), predicted_label])
            '''
            print(f"==> label {'Good' if predicted_label == 0 else 'Bad'} {probabilities}")
            '''

    df = pd.DataFrame(test_tensors)
    df.to_csv(f"../Debug/{USED_RESNET}_tensors.csv", header=False, index=False)

    df = pd.DataFrame(test_probabilities)
    df.to_csv(f"../Debug/{USED_RESNET}_probabilities.csv", header=False, index=False)

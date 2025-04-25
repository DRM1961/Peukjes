# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 21:29:04 2025

@author: dirkm
"""
import torch
import torch.nn as nn
import torch.optim as optim

from common import FeatureExtractor18, FeatureExtractor34, FeatureExtractor50,MLPClassifier, transform

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

USE_RESNET = 34
REFERENCE_IMAGE_FOLDER = "../Images/Dataset"

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


def get_features(resnet, good_images, bad_images):
    if os.path.exists(fnallfeatures) and os.path.exists(fnalllabels):
        all_features = np.load(fnallfeatures)
        all_labels = np.load(fnalllabels)
        print("Loaded precomputed features and labels")
    else:
        print("Compute good features")
        good_features = [extract_features(img, resnet).numpy() for img in good_images]
        print("Compute bad features")
        bad_features = [extract_features(img, resnet).numpy() for img in bad_images]

        # Stack all features into a (num_samples, num_features) matrix
        all_features = np.vstack(good_features + bad_features)
        all_labels = np.array([0] * len(good_features) + [1] * len(bad_features))

        print("save data")
        np.save(fnallfeatures, all_features)
        np.save(fnalllabels, all_labels)

    X_train, X_test, y_train, y_test = \
        train_test_split(all_features, all_labels, test_size=0.2, random_state=65)

    # Train SupportVectorClassifier
    #{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    # try linear first
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Classifier Accuracy: {accuracy:.2f}")

    #if result is not good enough, add scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train_scaled, y_train)
    accuracy = clf.score(X_test_scaled, y_test)
    print(f"Classifier Accuracy after Scaling: {accuracy:.2f}")

    #if result not good enough, try radial basis function kernel
    clf = SVC(kernel='rbf', C=1, gamma='auto', probability=True)
    clf.fit(X_train_scaled, y_train)
    accuracy = clf.score(X_test_scaled, y_test)
    print(f"RBF SVM Accuracy: {accuracy:.2f}")

    return(X_train, X_test, y_train, y_test)


def get_good_bad_images(class_dir):
    good_images = []
    bad_images = []
    for filename in os.listdir(class_dir):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(class_dir, filename)
            if "Good" in filename:
                good_images.append(img_path)
            elif "Bad" in filename:
                bad_images.append(img_path)

    return(good_images, bad_images)

# ----------MAIN ENTRYPOINT--------------

if USE_RESNET == 50:
    feature_extractor = FeatureExtractor50()
    fnscaler = '../Data/scaler_50.pkl'
    fnallfeatures = '../Data/all_features_50.npy'
    fnalllabels = '../Data/all_labels_50.npy'
    fnmodel = '../Data/mlp_model_50.pth'
elif USE_RESNET == 34:
    feature_extractor = FeatureExtractor34()
    fnscaler = '../Data/scaler_34.pkl'
    fnallfeatures = '../Data/all_features_34.npy'
    fnalllabels = '../Data/all_labels_34.npy'
    fnmodel = '../Data/mlp_model_34.pth'
elif USE_RESNET == 18:
    feature_extractor = FeatureExtractor18()
    fnscaler = '../Data/scaler_18.pkl'
    fnallfeatures = '../Data/all_features_18.npy'
    fnalllabels = '../Data/all_labels_18.npy'
    fnmodel = '../Data/mlp_model_18.pth'
else:
    print(f"error: Resnet{USE_RESNET} not defined")
    exit

good_images, bad_images = get_good_bad_images(REFERENCE_IMAGE_FOLDER)
X_train, X_test, y_train, y_test = get_features(feature_extractor, good_images, bad_images)

# Feature scaling  ==> don't forget to apply during runtime
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, fnscaler)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Initialize model
input_dim = X_train.shape[1]  # Number of features from ResNet
print(f'input dim = {input_dim}')
model = MLPClassifier(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the network
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
print("Training complete.")

# Calculate statistics for model using test_set
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == y_test_tensor).float().mean().item()

torch.save(model.state_dict(), fnmodel)
print(f"MLP Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, predicted.numpy())

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix resnet{USE_RESNET}")
plt.show()

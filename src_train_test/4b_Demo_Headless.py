# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:05:59 2025

@author: dirkm
Modified to process images from folder instead of camera
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from common import FeatureExtractor18, FeatureExtractor34, FeatureExtractor50, MLPClassifier, transform
from common_vision import ProfileFeature
from common_vision import InitCamera, ReadInputImage, MakeHorizontalProfile, MakeVerticalProfile
from PIL import Image

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from time import sleep

# Configuration
USED_RESNET = 34
INPUT_FOLDER = '../InputImages/'  # Folder to monitor for new images
PROCESSED_FOLDER = '../ProcessedImages/'  # Where to move processed images
DEBUG_FOLDER = '../Debug/'  # Where to save debug images

# Create folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

def get_image_profiles(img, direction=3, show=False):
    hbase, wbase = img.shape[:2]
    TL = (0, 0)
    BR = (wbase, hbase)

    if direction & 0x01:
        yprf_avg = MakeVerticalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_AVERAGE)
        yprf_min = MakeVerticalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_MIN)
        yprf_max = MakeVerticalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_MAX)
        xval = np.arange(0,len(yprf_avg))

        if show:
            plt.plot(yprf_min, xval, label='min')
            plt.plot(yprf_avg, xval, label='avg')
            plt.plot(yprf_max, xval, label='max')
            plt.title('vertical profile: avg/min/max per row')
            plt.ylabel("row")
            plt.xlabel("GV")
            plt.legend()
            plt.show()
    else:
        yprf_avg = np.zeros(hbase)
        yprf_min = np.zeros(hbase)
        yprf_max = np.zeros(hbase)

    if direction & 0x02:
        xprf_avg = MakeHorizontalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_AVERAGE)
        xprf_min = MakeHorizontalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_MIN)
        xprf_max = MakeHorizontalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_MAX)
        yval = np.arange(0,len(xprf_avg))

        if show:
            plt.plot(yval, xprf_min, label='min')
            plt.plot(yval, xprf_avg, label='avg')
            plt.plot(yval, xprf_max, label='max')
            plt.title('horizontal profile: avg/min/max per column')
            plt.ylabel("GV")
            plt.xlabel("col")
            plt.legend()
            plt.show()
    else:
        xprf_avg = np.zeros(wbase)
        xprf_min = np.zeros(wbase)
        xprf_max = np.zeros(wbase)

    return(yprf_avg, yprf_min, yprf_max, xprf_avg, xprf_min, xprf_max)

def process_image(frame, feature_extractor, scaler, model):
    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        image_feature = feature_extractor(image)
        image_feature = scaler.transform(image_feature.reshape(1, -1))

        sample_feature = torch.tensor(image_feature, dtype=torch.float32)

        if sample_feature.ndim == 1:
            sample_feature = sample_feature.unsqueeze(0)

        output = model(sample_feature)
        good_score = output[0, 0].item()
        bad_score = output[0, 1].item()
        threshold = 0.9
        obj_is_good = True if good_score > threshold else False

    return obj_is_good, good_score, bad_score

def load_model():
    print(f'Loading model resnet{USED_RESNET}')
    if USED_RESNET == 50:
        feature_extractor = FeatureExtractor50()
        input_dim = 2048
        fnscaler = '../Data/scaler_50.pkl'
        fnmodel = '../Data/mlp_model_50.pth'
    elif USED_RESNET == 34:
        feature_extractor = FeatureExtractor34()
        input_dim = 512
        fnscaler = '../Data/scaler_34.pkl'
        fnmodel = '../Data/mlp_model_34.pth'
    elif USED_RESNET == 18:
        feature_extractor = FeatureExtractor18()
        input_dim = 512
        fnscaler = '../Data/scaler_18.pkl'
        fnmodel = '../Data/mlp_model_18.pth'
    else:
        print(f"Error: Resnet{USED_RESNET} not defined")
        exit()

    scaler = joblib.load(fnscaler)
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(fnmodel))
    model.eval()
    print("Model loaded successfully and ready for inference!")
    return feature_extractor, scaler, model

def process_image_file(image_path, feature_extractor, scaler, model):
    try:
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame.copy()
        
        # Process image
        is_good, val_good, val_bad = process_image(frame, feature_extractor, scaler, model)
        
        # Save results
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds").replace(":", "-")
        result = "good" if is_good else "bad"
        debug_filename = f"{DEBUG_FOLDER}{timestamp}_{result}_g{val_good:.2f}_b{val_bad:.2f}.jpg"
        cv2.imwrite(debug_filename, frame)
        
        print(f"Processed {image_path}: is_good={is_good}, scores (good={val_good:.2f}, bad={val_bad:.2f})")
        
        # Move processed image
        processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
        os.rename(image_path, processed_path)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def monitor_folder(feature_extractor, scaler, model):
    print(f"Monitoring folder {INPUT_FOLDER} for new images...")
    print("Press Ctrl+C to stop")
    
    # Load reference empty image
    empty_path = '../Images/empty.png'
    if os.path.exists(empty_path):
        empty, _, _, _, _ = ReadInputImage(empty_path)
        ref_ravg, ref_rmin, ref_rmax, ref_cavg, ref_cmin, ref_cmax = get_image_profiles(empty, direction=0x03)
        mean_ref_cavg = np.mean(ref_cavg)
    else:
        print("Warning: empty.png reference image not found")
        mean_ref_cavg = None
    
    processed_files = set()
    
    try:
        while True:
            # Get list of files in input folder
            files = [f for f in os.listdir(INPUT_FOLDER) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Process new files
            for filename in files:
                if filename not in processed_files:
                    filepath = os.path.join(INPUT_FOLDER, filename)
                    process_image_file(filepath, feature_extractor, scaler, model)
                    processed_files.add(filename)
            
            # Sleep for a bit before checking again
            sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping folder monitoring")

if __name__ == '__main__':
    # Load the model
    feature_extractor, scaler, model = load_model()
    
    # Start monitoring the folder
    monitor_folder(feature_extractor, scaler, model)
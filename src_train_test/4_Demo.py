# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:05:59 2025

@author: dirkm
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from common import FeatureExtractor18, FeatureExtractor34, FeatureExtractor50,MLPClassifier, transform
from common_vision import ProfileFeature
from common_vision import InitCamera, ReadInputImage, MakeHorizontalProfile, MakeVerticalProfile
from cam_illum import set_illum_white, flash_illum, init_camera
from PIL import Image

from picamera2 import Picamera2
import cv2

import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from time import sleep

USED_RESNET = 34
feature_extractor = None
fnscaler = ''
fnmodel = ''

mycounter = 1 # or 2
counter = 0


def get_image_profiles(img, direction=3, show=False):
    hbase, wbase = img.shape[:2]
    #TL and BR are (col,row)
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
        image_feature = scaler.transform(image_feature.reshape(1, -1))  # Ensure correct shape

        sample_feature = torch.tensor(image_feature, dtype=torch.float32)

        if sample_feature.ndim == 1:  # Ensure correct shape (batch size 1)
            sample_feature = sample_feature.unsqueeze(0)  # Convert [feature_dim] → [1, feature_dim]

        output = model(sample_feature)
        #predicted_label = torch.argmax(output, dim=1)
        good_score = output[0, 0].item()
        bad_score = output[0, 1].item()
        threshold = 0.9
        obj_is_good  = True if good_score > threshold else False

        '''
        probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_label = torch.argmax(probabilities, dim=1).item()
        test_probabilities.append([img_path, probabilities[0,0].item(),probabilities[0,1].item(), predicted_label])
        '''

    return(obj_is_good, good_score, bad_score)

def main(cam, cameratype):
    if cameratype != 'picamera2':
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

            cam.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        else:
            print(f'cannot open camera {CAMID}')
            return()

        cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(cam_w, cam_h)


    #init nnmodel
    print(f'load model resnet{USED_RESNET}')
    if USED_RESNET == 50:
        feature_extractor = FeatureExtractor50()
        input_dim = 2048  # Adjust based on ResNet feature size: 34 = 512, 50 = 2048
        fnscaler = '../Data/scaler_50.pkl'
        fnmodel = '../Data/mlp_model_50.pth'
    elif USED_RESNET == 34:
        feature_extractor = FeatureExtractor34()
        input_dim = 512  # Adjust based on ResNet feature size: 34 = 512, 50 = 2048
        fnscaler = '../Data/scaler_34.pkl'
        fnmodel = '../Data/mlp_model_34.pth'
    elif USED_RESNET == 18:
        feature_extractor = FeatureExtractor18()
        input_dim = 512
        fnscaler = '../Data/scaler_18.pkl'
        fnmodel = '../Data/mlp_model_18.pth'
    else:
        print(f"error: Resnet{USED_RESNET} not defined")
        exit

    scaler = joblib.load(fnscaler)
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(fnmodel))  # Load saved weights
    model.eval()
    print("Model loaded successfully and ready for inference!")

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    print('check position of cup below the camera; press ESC to exit and start processing real images')
    
    brightness = 0.3
    set_illum_white(brightness)
    focussed = False
    while True:
        if cameratype != 'picamera2':
            ret, frame = cam.read()
            if not ret:
                break
        else:
            if not focussed:
                try:
                    cam.autofocus_cycle()
                except:
                    print("Autofocus not available - using fixed focus")
                focussed = True
            frame = cam.capture_array()

        cv2.imshow('img', frame)
        # Press ESC to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if cv2.getWindowProperty('img', 1) < 0:
            break

    #init empty image for runtime detection of object present or not
    cv2.imwrite('../Images/empty.png', frame)
    empty,_,_,_,_ = ReadInputImage('../Images/empty.png')
    ref_ravg, ref_rmin, ref_rmax, ref_cavg, ref_cmin, ref_cmax = get_image_profiles(empty, direction=0x03)
    mean_ref_cavg = np.mean(ref_cavg)
    mean_ref_ravg = np.mean(ref_ravg)

    print('starting the main loop: press ESC to exit')
    prev_np_mean_diff = 0
    while True:
        if cameratype != 'picamera2':
            ret, frame = cam.read()
            if not ret:
                break
        else:
            frame = cam.capture_array()

        cv2.imshow('img', frame)
        # Press ESC to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if cv2.getWindowProperty('img', 1) < 0:
            break

        if len(frame.shape) == 3:  # if not gray already
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ravg, rmin, rmax, cavg, cmin, cmax = get_image_profiles(img, direction=0x03)
        mean_cavg = np.mean(cavg)
        mean_ravg = np.mean(ravg)

        np_mean_diff = max(int(mean_ref_cavg - mean_cavg), int(mean_ref_ravg-mean_ravg))
        if abs(prev_np_mean_diff - np_mean_diff) > 2:
            print(f'diff of mean = {int(np_mean_diff)}')
            prev_np_mean_diff = np_mean_diff

        if abs(np_mean_diff) > 40:
            for i in range(10):  #read buffered frames debounce
                if cameratype != 'picamera2':
                    _, frame = cam.read()
                else:
                    frame = cam.capture_array()

                cv2.imshow('img', frame)

            is_good, val_good, val_bad = process_image(frame, feature_extractor, scaler, model)

            print(f'is good: {is_good}, scores ({val_good}, {val_bad}')
            fn = f'../Debug/{datetime.now().isoformat(sep=" ", timespec="seconds")}_{is_good}.png'
            fn = fn.replace(":", "-")
            cv2.imwrite(fn, frame)

            if is_good:
                flash_illum('green', 2)
            else:
                flash_illum('red', 2)
            set_illum_white(brightness)

            #print("press 'c' to continue, 'c' followed by ESC to quit")
            #if cv2.waitKey(0) & 0xFF == 'c': #flush buffered image
            #    continue

            for i in range(30):  #read buffered frames
                if cameratype != 'picamera2':
                    _, frame = cam.read()
                else:
                    frame = cam.capture_array()
                cv2.imshow('img', frame)

    #cam.set(cv2.CAP_PROP_POS_FRAMES, cam.get(cv2.CAP_PROP_POS_FRAMES))  # Pause stream


if __name__ == '__main__':
    cwd = os.getcwd()
    print(cwd)
    if 'src_train_test' not in cwd:
        os.chdir('src_train_test')

    cameratype = 'picamera2'

    if cameratype != 'picamera2':
        # use opencv camera
        # check if camera is available
        # check if camera is available
        # CAMID = 0 #0 = internal cam, 1..n = external
        CAMID = 1 #0 = internal cam, 1..n = external
                # for external cameras, camid nr doesn't work on ubuntu on rpi, must use /dev/video
                # use: 'lsusb' to check if camera is listed
                # use: 'v4l2-ctl' to get camera connection overview
                # sudo apt-get install v4l-utils
                # v4l2-ctl --list-devices

        print('init camera, this takes a while')

        if os.name=='nt':
            cam = cv2.VideoCapture(CAMID)
        else:
            cam = cv2.VideoCapture('/dev/video0')
    else:
        cam = init_camera()

    main(cam, cameratype)

    # Release the capture and writer objects
    cv2.destroyAllWindows()
    set_illum_white(0.0)
    #cam.release()

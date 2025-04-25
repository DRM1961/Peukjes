# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:05:59 2025

@author: dirkm
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import logging
import torch

from common import mylogger
from common import FeatureExtractor18, FeatureExtractor34, FeatureExtractor50,MLPClassifier, transform
from common_vision import ProfileDirection, ProfileFeature
from common_vision import InitCamera, ReadInputImage, MakeHorizontalProfile, MakeVerticalProfile
from cam_illum import set_illum_white, flash_illum, init_camera
from servo_control import InitServo, SwitchAngle
from PIL import Image

from CMqtt import CMqttClient
from picamera2 import Picamera2
import cv2

import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from time import sleep

from CConfig import CConfig

USED_RESNET = 34
MQTT_STATE_IDLE = 0
MQTT_STATE_ANALYZE = 1
MQTT_STATE_POSITIVE = 2
MQTT_STATE_NEGATIVE = 3

'''
mycounter = 1 # or 2
counter = 0

headless = False

brightness_autofocus = 0.1
brightness_detect = 0.1
brightness_measure = 0.1
flashtime = 1.0

servo_iopin = 5
servo_min_duty = 2.2
servo_max_duty = 9.5

min_mean_difference = 10
'''

def UpdateMQTTCounter(is_good, frame):
    global counter
    if is_good:
        counter = counter+1
        with open("../Debug/counter.txt", "w") as f:
            f.write(str(counter))

    if mqttclient.connected:
        mqttclient.publish(f"counter/{mycounter}", str(counter))
        #mqtt.publish(f"image/{mycounter}", frame, isImage=True)
        mylogger.info(f"publishing counter {mycounter}: counter {counter}")
    else:
        mylogger.info(f"NOT publishing counter {mycounter}: counter {counter}")
        #mqttclient.publish_via_cmd(f"counter/{mycounter}", str(counter))

def UpdateMQTTState(state):
    if mqttclient.connected:
        mqttclient.publish(f"state/{mycounter}", str(state))
        mylogger.info(f"publishing state {mycounter}: state {state}")
    else:
        mylogger.info(f"NOT publishing state {mycounter}: state {state}")
        #mqttclient.publish_via_cmd(f"state/{mycounter}", str(state))

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

def init_NNmodel():
    #init nnmodel
    mylogger.info(f'load model resnet{USED_RESNET}')
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
        mylogger.info(f"error: Resnet{USED_RESNET} not defined")
        exit

    scaler = joblib.load(fnscaler)
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(fnmodel))  # Load saved weights
    model.eval()
    mylogger.info("Model loaded successfully and ready for inference!")

    return(feature_extractor, scaler, model)

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

def get_image_profile_features(frame, saveframe=False):
    if len(frame.shape) > 1:  # if not gray already
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #crop image to ROI
    CROP_X = 0
    CROP_Y = 1
    CROP_W = 2
    CROP_H = 3

    h,w = frame.shape
    #should add checks on cropping area vs actual h/w
    x_start = cropping_area['crop_x']
    y_start = cropping_area['crop_y']
    x_end = cropping_area['crop_w']
    y_end = cropping_area['crop_h']
 
    cropped = frame[y_start:y_end, x_start:x_end].copy()
 
    if saveframe:
        cv2.imwrite('../Debug/empty_orig.png', frame)
        cv2.imwrite('../Debug/empty_cropped.png', cropped)
    ravg, rmin, rmax, cavg, cmin, cmax = get_image_profiles(cropped, direction=0x03, show=False)
    return(np.mean(ravg), np.mean(cavg), cropped)


def IsFrameDifferent(imgV, refV):
    #imgV = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #refV = cv2.cvtColor(imgprev, cv2.COLOR_BGR2GRAY)

    imgV = imgV.astype(np.int16)
    refV = refV.astype(np.int16)
    diffV = np.abs(cv2.add(imgV, -refV))
    diffV = diffV.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diffV = cv2.erode(diffV, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    diffV = cv2.dilate(diffV, kernel, iterations = 1)
    th, img_blobs = cv2.threshold(diffV, 20, 255, cv2.THRESH_BINARY)
    Contours, Hierarchy = cv2.findContours(img_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #mylogger.info(f'contour count = {len(Contours)}')

    reject_count = 0
    reject_area = 0
    result = imgV.copy()
    for c in Contours:
        #mylogger.info(f'found contour {cv2.boundingRect(c)}')
        x,y,w,h = cv2.boundingRect(c)
        if x>0 and y>0 and w>0 and h>0:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.line(result, (0,0), (int(x+w/2), int(y+h/2)), (0, 0, 255), 1)
            reject_count+=1
            reject_area = max(reject_area, w*h)

    return(reject_count, reject_area, result)

def app(cam, servo, mqtt, feature_extactor, scaler, model):
    global mycounter, headless
    global brightness_autofocus, brightness_detect, brightness_measure, flashtime
    global servo_iopin, servo_min_duty, servo_max_duty, min_mean_difference

    set_illum_white(brightness_autofocus)
    try:
        cam.autofocus_cycle()
        mylogger.info("Autofocus finished")
    except:
        mylogger.info("Autofocus not available - using fixed focus")
    focussed = True
    frame = cam.capture_array()
    mylogger.info(f'frame shape = {frame.shape}')

    if not headless:
        cv2.imshow('img', frame)

    #init empty image for detection of object present or not
    mean_ref_ravg, mean_ref_cavg, ref_img = get_image_profile_features(frame, saveframe=True)
    prev_np_mean_diff = 0

    if not headless:
        mylogger.info('starting the main loop: press ESC to exit')

    UpdateMQTTState(MQTT_STATE_IDLE)

    while True:
        frame = cam.capture_array()
        if not headless:
            cv2.imshow('img', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
		
        #get features from image to see if something is present
        mean_ravg, mean_cavg, run_img = get_image_profile_features(frame)
        if detection_method == 'profiles':
            np_mean_diff = max(int(abs(mean_ref_cavg - mean_cavg)), int(abs(mean_ref_ravg-mean_ravg)))
            if abs(prev_np_mean_diff - np_mean_diff) > 2:
                mylogger.info(f'diff of mean = {int(np_mean_diff)}')
                prev_np_mean_diff = np_mean_diff

            if abs(np_mean_diff) > detection_threshold['profile_difference']:
                do_analysis = True
            else:
                do_analysis = False
        else:
            reject_count, reject_area, detection_image = IsFrameDifferent(run_img, ref_img)
            if reject_count > 0 and reject_area > detection_threshold['area_difference']:
                mylogger.info(f'diff count = {reject_count}, max area = {reject_area}')
                do_analysis = True
            else:
                do_analysis = False

        if do_analysis:
            UpdateMQTTState(MQTT_STATE_ANALYZE)
            set_illum_white(brightness_measure)
            for i in range(10):  #read buffered frames debounce
                frame = cam.capture_array()
                if not headless:
                    cv2.imshow('img', frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            is_good, val_good, val_bad = process_image(frame, feature_extractor, scaler, model)

            fn = f'../Debug/{datetime.now().isoformat(sep=" ", timespec="seconds")}_{is_good}'
            fn = fn.replace(":", "-")
            cv2.imwrite(f'{fn}_processed.png', frame)
            cv2.imwrite(f'{fn}_detected.jpg', detection_image)
            mylogger.info(f'{fn}: is good: {is_good}, scores ({val_good}, {val_bad}')
            
            is_good = True ## TODO : Fix above and remove override
            if is_good:
                flash_illum('green', flashtime)
                UpdateMQTTState(MQTT_STATE_POSITIVE)
            else:
                flash_illum('red', flashtime)
                UpdateMQTTState(MQTT_STATE_NEGATIVE)

            UpdateMQTTCounter(is_good, frame)

            if servo:
                SwitchAngle(servo, servo_iopin)
                sleep(1)
            else:
                if not headless:
                    print("press 'c' to continue")
                    if cv2.waitKey(0) & 0xFF == 'c': #flush buffered image
                        continue

            set_illum_white(brightness_detect)
            for i in range(30):  #read buffered frames
                frame = cam.capture_array()
                if not headless:
                    cv2.imshow('img', frame)

            #update reference features
            mean_ref_ravg, mean_ref_cavg, ref_img = get_image_profile_features(frame)

            UpdateMQTTState(MQTT_STATE_IDLE)


if __name__ == '__main__':
    global mycounter, headless, counter
    global brightness_autofocus, brightness_detect, brightness_measure, flashtime
    global servo_iopin, servo_min_duty, servo_max_duty, min_mean_difference
    global cropping_area, detection_method

    cwd = os.getcwd()
    print(cwd)
    if 'src_runtime' not in cwd:
        os.chdir('src_runtime')

    if os.path.exists("../Debug/counter.txt"):
        with open("../Debug/counter.txt", "r") as f:
            counter = int(f.read())
    else:
        counter = 0


    myconfig = CConfig()
    mycounter = myconfig.GetMyCounter()
    headless = myconfig.GetHeadlessMode()

    mqtt_settings = myconfig.GetMQTT()
    mqtt_broker = mqtt_settings['broker']
    mqtt_user = mqtt_settings['user']
    mqtt_pw = mqtt_settings['pw']

    illum = myconfig.GetIllum()
    brightness_autofocus = illum['brightness_autofocus']
    brightness_detect = illum['brightness_detect']
    brightness_measure = illum['brightness_measure']
    flashtime = illum['flashtime']

    servo_settings = myconfig.GetServo()
    servo_iopin = servo_settings['iopin']
    servo_min_duty = servo_settings['min_duty']
    servo_max_duty = servo_settings['max_duty']

    detection_method = myconfig.GetDetectionMethod()
    detection_threshold = myconfig.GetDetectionThreshold()
    cropping_area = myconfig.GetDetectionCroppingArea()

    cam = init_camera()

    #init servo
    servo = InitServo(servo_iopin, servo_min_duty, servo_max_duty)
    SwitchAngle(servo, servo_iopin)
    sleep(2)
    SwitchAngle(servo, servo_iopin)

    #init mqtt
    mqttclient = CMqttClient(userdata="rpi", user=mqtt_user, pw=mqtt_pw, broker=mqtt_broker)
    mqttclient.subscribe(f"set_counter/{mycounter}")

    #init nnmodel
    feature_extractor, scaler, model = init_NNmodel()

    #init display
    if not headless:	
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    app(cam, servo, mqttclient, feature_extractor, scaler, model)

    # Release the capture and writer objects
    if not headless:
        cv2.destroyAllWindows()

    set_illum_white(0.0)
    #cam.release()
	
    SwitchAngle(servo, servo_iopin)
    sleep(2)








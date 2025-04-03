# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:11:31 2025

@author: dirkm
"""

from enum import Enum
import cv2
import numpy as np

class ProfileDirection(Enum):
    E_PRF_NONE = 0
    E_PRF_HORIZONTAL = 1
    E_PRF_VERTICAL = 2
    E_PRF_HORIZONTAL_MIN = 3
    E_PRF_VERTICAL_MIN = 4
    E_PRF_HORIZONTAL_MAX = 5
    E_PRF_VERTICAL_MAX = 6

class ProfileFeature(Enum):
    E_PRF_AVERAGE = 0
    E_PRF_MIN = 1
    E_PRF_MAX = 2

def InitCamera(camid):
    vid = cv2.VideoCapture(camid, cv2.CAP_DSHOW)
    if vid.isOpened():
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

        vid.set(cv2.CAP_PROP_BRIGHTNESS, 0)

    return(vid)

def ReadInputImage(fn, split=True, convertogray=True):
    iB = []
    iG = []
    iR = []
    iA = []
    img_main = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    inpimg = img_main

    if (len(img_main.shape) > 2):  # if not gray
        if split:
            if img_main.shape[2] == 4:
                iR, iG, iB, iA = cv2.split(img_main)
            elif img_main.shape[2] == 3:
                iR, iG, iB = cv2.split(img_main)
                iA = iB.copy()
            else:
                print(f'input image shape nok {img_main.shape}')
    else:
        if split:
            iR = img_main.copy()
            iG = img_main.copy()
            iB = img_main.copy()
            iA = img_main.copy()

    if convertogray and (len(img_main.shape) == 3):  # if not gray already
        inpimg = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)

    return inpimg, iR, iG, iB, iA

def MakeHorizontalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_AVERAGE):  #TL and BR are (col,row)
    if (TL[1] == BR[1]):  # if single row
        lprfx = img[TL[1], TL[0]:BR[0]]
    else:  # else calc profile
        img_roi = img[TL[1]:BR[1], TL[0]:BR[0]]
        match(feature):
            case ProfileFeature.E_PRF_AVERAGE:
                lprfx = cv2.reduce(img_roi, 0, cv2.REDUCE_AVG, dtype=cv2.CV_16U)
            case ProfileFeature.E_PRF_MIN:
                lprfx = cv2.reduce(img_roi, 0, cv2.REDUCE_MIN, dtype=cv2.CV_8U)
            case ProfileFeature.E_PRF_MAX:
                lprfx = cv2.reduce(img_roi, 0, cv2.REDUCE_MAX, dtype=cv2.CV_8U)
            case _:
                print(f'unknown prf feature {feature}, using average instead')
                lprfx = cv2.reduce(img_roi, 0, cv2.REDUCE_AVG, dtype=cv2.CV_16U)

        lprfx = lprfx[0]
    return lprfx


def MakeVerticalProfile(img, TL, BR, feature=ProfileFeature.E_PRF_AVERAGE):
    if (TL[0] == BR[0]):  # if single col
        lprfy = img[TL[1]:BR[1], TL[0]]
    else:  # else calc profile
        img_roi = img[TL[1]:BR[1], TL[0]:BR[0]]
        match(feature):
            case ProfileFeature.E_PRF_AVERAGE:
                lprfy = cv2.reduce(img_roi, 1, cv2.REDUCE_AVG, dtype=cv2.CV_16U)
            case ProfileFeature.E_PRF_MIN:
                lprfy = cv2.reduce(img_roi, 1, cv2.REDUCE_MIN, dtype=cv2.CV_8U)
            case ProfileFeature.E_PRF_MAX:
                lprfy = cv2.reduce(img_roi, 1, cv2.REDUCE_MAX, dtype=cv2.CV_8U)
            case _:
                print(f'unknown prf feature {feature}, using average instead')
                lprfy = cv2.reduce(img_roi, 1, cv2.REDUCE_AVG, dtype=cv2.CV_16U)
        lprfy = np.transpose(lprfy)
        lprfy = lprfy[0]
    return lprfy

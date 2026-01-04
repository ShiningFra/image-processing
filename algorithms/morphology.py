import cv2
import numpy as np

kernel = np.ones((3,3), np.uint8)

def erosion(img):
    return cv2.erode(img, kernel)

def dilation(img):
    return cv2.dilate(img, kernel)

def opening(img):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

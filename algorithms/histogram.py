import cv2
import numpy as np

def histogram_equalization(gray):
    return cv2.equalizeHist(gray)

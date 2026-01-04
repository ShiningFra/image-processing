import cv2
import numpy as np

def apply_kernel(img, kernel):
    return cv2.filter2D(img, -1, kernel)

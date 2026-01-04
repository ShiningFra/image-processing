import cv2

def mean_filter(img, k=3):
    return cv2.blur(img, (k, k))

def gaussian_filter(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def median_filter(img, k=5):
    return cv2.medianBlur(img, k)

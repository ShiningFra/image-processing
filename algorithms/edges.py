import cv2

def sobel_edges(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(sx + sy)

def laplacian(gray):
    return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))

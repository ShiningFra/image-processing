# Chargement / conversion images
import cv2
import numpy as np
from PIL import Image

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image invalide")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def to_pil(img):
    return Image.fromarray(img)

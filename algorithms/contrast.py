import numpy as np

def linear_contrast(img):
    imin, imax = img.min(), img.max()
    return ((img - imin) / (imax - imin) * 255).astype(np.uint8)

def gamma_correction(img, gamma=1.5):
    return np.uint8(255 * (img / 255) ** gamma)

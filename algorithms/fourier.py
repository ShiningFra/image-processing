import numpy as np

def low_pass_filter(gray, radius=30):
    f = np.fft.fftshift(np.fft.fft2(gray))
    rows, cols = gray.shape
    mask = np.zeros_like(gray)
    cx, cy = rows//2, cols//2
    mask[cx-radius:cx+radius, cy-radius:cy+radius] = 1
    filtered = f * mask
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return img.astype(np.uint8)

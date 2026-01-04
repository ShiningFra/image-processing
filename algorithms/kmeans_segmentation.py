import numpy as np
from sklearn.cluster import KMeans

def kmeans_segment(img, k=3):
    Z = img.reshape((-1, 3))
    km = KMeans(n_clusters=k).fit(Z)
    return km.cluster_centers_[km.labels_].reshape(img.shape).astype(np.uint8)

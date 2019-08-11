import pandas as pd
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import cv2 as cv

from sklearn.cluster import DBSCAN 
import sys

def elbow_plot(data, maxK=10, seed_centroids=None):
    sse = {}
    for k in range(1, maxK):
        print("k: ", k)
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
            data["clusters"] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            data["clusters"] = kmeans.labels_
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    return

def kmeans(image, numClusters=10):
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    kmeans = KMeans(n_clusters=numClusters, max_iter=300, n_init=100, random_state=0).fit(reshaped)
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),(image.shape[0], image.shape[1]))

    sortedLabels = sorted([n for n in range(numClusters)],
        key=lambda x: -np.sum(clustering == x))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        mask[clustering == label] = int((255) / (numClusters - 1)) * i  

    return mask   


def dbscan(image):
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    db = DBSCAN(eps = 0.0375, min_samples = 50).fit(reshaped) 

    clustering = np.reshape(np.array(db.labels_, dtype=np.uint8),(image.shape[0], image.shape[1]))

    sortedLabels = sorted([n for n in range(numClusters)],
        key=lambda x: -np.sum(clustering == x))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        mask[clustering == label] = int((255) / (numClusters - 1)) * i  

    return mask   


def watershed(mask):
    # Mask to Polygon
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(mask,markers)
    mask[markers == -1] = [0,0,255]
    cv.imshow(image)

image = cv.imread(sys.argv[1])
dbscan(image)    
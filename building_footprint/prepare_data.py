import re
import numpy as np
import cv2 as cv
from multiprocessing import Pool
import psutil

import glob
import os

ROOT_DIR = "./data/"

def read_image(imagefile):
    return cv.imread(imagefile)

def read_polygon(polyfile):
    with open(polyfile) as stream:
        polygon_data = stream.readlines()

    return polygon_data

def polygon_to_image(polygon_data, img):
    label_img = img
    mask_img = np.zeros([1280, 1280, 3])
    y_img = np.zeros([1280, 1280])

    for poly in polygon_data:
        record = poly.strip("/n")
        record = record[1:]
        record = record[:-2]

        building = []
        for x, y in map(lambda x : re.findall(r'\d+', x), re.findall(r'\[.*?\]', record)):
            x, y = int(x), int(y)
            building.append([x,y])

        building = np.array(building)

        label_img = cv.fillPoly(label_img,  [building],  color=(255,255,255))
        mask_img = cv.fillPoly(mask_img,  [building],  color=(255,255,255))
        y_img = cv.fillPoly(y_img,  [building],  1) 

    return label_img, mask_img, y_img

num_cpus = psutil.cpu_count(logical=False)

def f(name):
    image, polygon, mask = f"{ROOT_DIR}/images/{name}.png", f"{ROOT_DIR}/labels_stitched_1280x1280/poly_xy_footprints/{name}.txt", \
                            f"{ROOT_DIR}/mask/{name}.png"
    img = read_image(image)
    polygon_data = read_polygon(polygon)
    label_img, mask_img, y_img = polygon_to_image(polygon_data, img)
    cv.imwrite(mask, mask_img)
    return 0


images = glob.glob(f"{ROOT_DIR}/images/*.png")
names = []
for image in images:
    head, tail = os.path.split(image)
    name = tail.split(".")[0]
    names.append(name)

all_data = [names[i * num_cpus:(i + 1) * num_cpus] for i in range((len(names) + num_cpus - 1) // num_cpus )]  
print(len(all_data))
pool = Pool(num_cpus)

for batch_id, batch in enumerate(all_data):
    print(batch_id)
    pool.map(f, batch)

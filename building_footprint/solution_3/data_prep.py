import numpy as np
np.random.seed(227)
import json
from shapely.geometry import Polygon, shape, Point
import gdal, ogr
import pandas as pd
import scipy.misc
import skimage

import sys, os, re, datetime, multiprocessing
from keras.models import *
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.normalization import *
from keras.optimizers import *
import keras.preprocessing.image
import xgboost as xgb

img_shape = (1280, 1280)
net_shape = (512, 512)
gtf = "labels_stitched_1280x1280/poly_xy_footprints"

ROOT_DIR = "/ws/data/"
def GetFileList(path):
    return [os.path.split(fn)[1].split('.png')[0].replace('.','').strip('_') for fn in os.listdir(path + 'images/') if fn.endswith('.png')]


def draw_polygon(polygon):
    y, x = polygon
    yy, xx = skimage.draw.polygon(y, x, net_shape)
    return yy, xx

def read_polygon(polyfile):
    with open(polyfile) as stream:
        polygon_data = stream.readlines()

    return polygon_data

def get_polygons(fn):
    polyfile = "{0}/{1}/{2}.txt".format(ROOT_DIR, gtf, fn)
    polygon_data = read_polygon(polyfile)
    polygons = np.array(building)
    for poly in polygon_data:
        record = poly.strip("/n")
        record = record[1:]
        record = record[:-2]

        all_x = []
        all_y = []

        for x, y in map(lambda x : re.findall(r'\d+', x), re.findall(r'\[.*?\]', record)):
            x, y = int(x), int(y)
            all_x.append(x)
            all_y.append(y)

        polygons.append([all_y, all_x])
    return polygons

                

def CreateLabel(fn, gtf):
    full_path = "{0}/{1}/{2}.txt".format(ROOT_DIR, gtf, fn)
    polygon_data = read_polygon(full_path)    

    label_fill = np.zeros(img_shape)
    label_edge = np.zeros(img_shape)
    for poly in polygon_data:
        try:
            record = poly.strip("/n")
            record = record[1:]
            record = record[:-2]

            all_x = []
            all_y = []

            for x, y in map(lambda x : re.findall(r'\d+', x), re.findall(r'\[.*?\]', record)):
                x, y = int(x), int(y)
                all_x.append(x)
                all_y.append(y)

            # building = np.array(building)

            yy, xx = skimage.draw.polygon(all_y, all_x)
            label_fill[yy, xx] = 1
            yy, xx = skimage.draw.polygon_perimeter(all_y, all_x)
            label_edge[yy, xx] = 1
        except:
            print("Erro", fn)
    return label_fill, label_edge



def CreateData(path, fn, gtf=None):
    img = skimage.io.imread('{0}/images/{1}.png'.format(path, fn))
    if gtf is not None:
        data = np.zeros(net_shape + (5,), dtype='uint8')
        label_fill, label_edge = CreateLabel(fn, gtf)
        
        data[:, :, 3], data[:, :, 4] = scipy.misc.imresize(data[:, :, 3], net_shape), scipy.misc.imresize(data[:, :, 4], net_shape)
    else:
        data = np.zeros(net_shape + (3,), dtype='uint8')

    
    data_shape = data.shape[:-1]
    
    for ch in range(3):
        data[:data_shape[0], :data_shape[1], ch] = scipy.misc.imresize(img[:data_shape[0], :data_shape[1], ch], net_shape)

    return data


def CreateTrain(path, gtf_path):
    os.makedirs('train', exist_ok=True)
    file_list = GetFileList(path)
    for i, fn in enumerate(file_list):
        
        try:
            print("Success", i, fn)
            CreateData(path, fn, gtf).dump('train/{}.npz'.format(fn))
        except:
            print("Error: ", i, fn)

def get_val_and_generator():
    path = 'train/'
    files = [path + f for f in sorted(os.listdir(path)) if f.endswith('.npz')]
    tr = []
    val = []
    for i, f in enumerate(files):
        if i % 50 == 0: val.append(f)
        else: tr.append(f)
    val_img = np.concatenate([np.load(f)[None, :] for f in val])

    def gen():
        batch_size = 8
        rnd = np.random.RandomState(0)
        while True:
            rnd.shuffle(tr)
            for i in range(0, len(tr), batch_size):
                t = tr[i:min(i+batch_size, len(tr))]
                img = np.concatenate([np.load(f)[None, :] for f in t])
                yield img[:, :, :, :-2], [img[:, :, :, -2:-1], img[:, :, :, -1:]]

    return (val_img[:, :, :, :-2], [val_img[:, :, :, -2:-1], val_img[:, :, :, -1:]]), gen()


if __name__ == "__main__":
    CreateTrain(ROOT_DIR, gtf)
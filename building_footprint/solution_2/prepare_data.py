import re
import numpy as np
import cv2 as cv
from multiprocessing import Pool
import psutil
import skimage.draw
import glob
import os
import json

ROOT_DIR = "/ws/data"

def read_image(imagefile):
    return cv.imread(imagefile)

def read_polygon(polyfile):
    with open(polyfile) as stream:
        polygon_data = stream.readlines()


    regions = {}
    
    for id, poly in enumerate(polygon_data):
        record = poly.strip("/n")
        record = record[1:]
        record = record[:-2]

        shape_attributes = dict(all_points_x = [], all_points_y = [])
        for x, y in map(lambda x : re.findall(r'\d+', x), re.findall(r'\[.*?\]', record)):
            x, y = int(x), int(y)
            shape_attributes['all_points_x'].append(x)
            shape_attributes['all_points_y'].append(y)


        region = {
            "shape_attributes" : shape_attributes,
            "region_attributes": {
                "name": "building"
            }
        }  
        regions[str(id)] = region
    return  regions  



def generate_mask(image_id):
    image = read_image(f"{ROOT_DIR}/images/{image_id}.png")
    with open(f"{ROOT_DIR}/region_data.json") as stream:
        polygon_data = json.load(stream)

    polygons = polygon_data[image_id]['regions'].values()
    
    for i, p in enumerate(polygons):
        rr, cc = skimage.draw.polygon(p['shape_attributes']['all_points_y'], p['shape_attributes']['all_points_x'])
        image[rr, cc] = (255, 255, 255)

    cv.imwrite(f"{ROOT_DIR}/mask/{image_id}.png",image)    

if __name__ == "__main__":
    if not os.path.exists(f'{ROOT_DIR}/train'):
        os.makedirs(f'{ROOT_DIR}/train')

    if not os.path.exists(f'{ROOT_DIR}/valid'):
        os.makedirs(f'{ROOT_DIR}/valid')

    from shutil import copyfile

    get_label_by_id = lambda x: f"{ROOT_DIR}/labels_stitched_1280x1280/poly_xy_footprints/{x}.txt"
    region_data = {}

    import glob
    all_files = [os.path.split(f)[-1] for f in glob.glob(f"{ROOT_DIR}/images/*.png")]
    import random
    random.shuffle(all_files)
    train_id = all_files[:700]
    valid_ids = all_files[700:]
    print(len(all_files),len(train_id), len(valid_ids))
    data = {'train': train_id, 'valid': valid_ids}
    for dataset_type, data_ids in data.items():
        region_data = {}
        for imp in data_ids:
            image_id, ext  = os.path.splitext(imp)
            
            if ext == ".png":
                print(f"processing {image_id}")
                print(get_label_by_id(image_id))
                regions = read_polygon(get_label_by_id(image_id))
                region_data[image_id] = {
                    "fileref": "",
                    "size": os.stat(f"{ROOT_DIR}/images/{imp}").st_size,
                    "filename": imp,
                    "base64_img_data": "",
                    "file_attributes": {},
                    "regions" : regions
                    } 
                copyfile(f"{ROOT_DIR}/images/{imp}", f"{ROOT_DIR}/{dataset_type}/{imp}")
        with open(f'{ROOT_DIR}/{dataset_type}/region_data.json', 'w') as fp:
            json.dump(region_data, fp)





    # for image_id in region_data.keys():
    #     generate_mask(image_id)

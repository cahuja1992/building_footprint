import cv2 as cv
import glob
import os
from tqdm import tqdm

ROOT_DIR = "./data"

images_path = f"{ROOT_DIR}/images-512"
mask_path = f"{ROOT_DIR}/mask-512"

os.makedirs(images_path) if not os.path.exists(images_path) else None
os.makedirs(mask_path) if not os.path.exists(mask_path) else None
    
images = glob.glob(f"{ROOT_DIR}/images/*.png")

names = []
for image in tqdm(images):
    head, tail = os.path.split(image)
    processed_images = f"{ROOT_DIR}/images-512/{tail}"
    
    mask = f"{ROOT_DIR}/mask/{tail}"
    processed_mask = f"{ROOT_DIR}/mask-512/{tail}"

    try:
        img = cv.imread(image)
        img = cv.resize(img, (512, 512))
        

        mask = cv.imread(mask)
        mask = cv.resize(mask, (512, 512))

        cv.imwrite(processed_images, img)
        cv.imwrite(processed_mask, mask)
        
    except:
        print(f'Error {image}')
        
        
from fastai.utils.collect_env import *
from fastai.vision import *
import rasterio
from rasterio import Affine
from rasterio.windows import Window
from rasterio.transform import from_bounds

from pathlib import Path
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from tqdm import tqdm

defaults.device = 'cpu'

model_path = '/ws/model'
path = Path('/ws/data')
path_img = path/'images-512'
path_lbl = path/'mask-512'
OUTPUT = path/'outputs'

get_y_fn = lambda x: path_lbl/f'{x.stem}.png'

codes = np.array(['Empty','Building'])
unet_sz = 768
bs=1

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
    
class SegItemListCustom(ImageList):
    _label_cls = SegLabelListCustom

data = (SegmentationItemList.from_folder(path_img)
        .random_split_by_pct()
        .label_from_func(get_y_fn, classes=codes)
        .transform(get_transforms(), tfm_y=True, size=unet_sz)
        .databunch(bs=bs)
        .normalize(imagenet_stats))        

learn = unet_learner(data, models.resnet34)
learn.load(f'{model_path}/models/resnet34unet-img512-combineloss-unfreeze-best')
learn.model.eval()

if __name__ == "__main__":
    image = path_img/'133761tileX_194565tileY_0302222301010000203quadKey_19zoomLevel_5XStitchStride_5YStitchStride.png'
    image = cv.imread(image) 

    # Step 1: Image Segmentation
    image = Image(pil2tensor(image,np.float32))
    outputs = learn.predict(image)
    mask = (outputs[2][1]).numpy()

    cv.imwrite("mask.png", mask)

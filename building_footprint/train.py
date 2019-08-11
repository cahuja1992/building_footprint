from fastai.vision import *
from fastai.utils.collect_env import *
from fastai.callbacks import SaveModelCallback
import numpy as np

fastai.torch_core.defaults.device = 'cpu'

path = Path('./data')
path.ls()

path_lbl = path/'masks-512'
path_img = path/'images-512'

fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)
fnames[:3], lbl_names[:3] 

print(len(fnames), len(lbl_names))

size = 512
bs = 8

np.random.shuffle(fnames)
get_y_fn = lambda x: path_lbl/f'{x.stem}.png'

valid_idx = np.random.choice(len(fnames), size=100, replace=False)

img_f = fnames[-1]
img = open_image(img_f)
mask = open_mask(get_y_fn(img_f), div=True)

fig,ax = plt.subplots(1,1, figsize=(10,10))
img.show(ax=ax)
mask.show(ax=ax, alpha=0.5)

src_size = np.array(mask.shape[1:])
print(src_size)
codes = np.array(['Empty','Building'])

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
    
class SegItemListCustom(ImageList):
    _label_cls = SegLabelListCustom

src = (SegItemListCustom.from_folder(path_img)
        .split_by_idx(valid_idx)
        .label_from_func(get_y_fn, classes=codes))    

tfms = get_transforms(flip_vert=True, max_warp=0, max_zoom=1.2, max_lighting=0.3)
data = (src.transform(tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))    
            

# print(data)       
# print(data.valid_ds.items) 
# print(data.train_ds.y[1].shape)
# data.show_batch(2,figsize=(6,6), ds_type=DatasetType.Valid, alpha=0.7)
# print(data.classes)

def dice_loss(input, target):
    smooth = 1.
    input = input[:,1,None].sigmoid()
    iflat = input.contiguous().view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()
    return (1 - ((2. * intersection + smooth) / ((iflat + tflat).sum() +smooth)))

def combo_loss(pred, targ):
    bce_loss = CrossEntropyFlat(axis=1)
    return bce_loss(pred,targ) + dice_loss(pred,targ)

def acc_fixed(input, targs):
    n = targs.shape[0]
    targs = targs.squeeze(1)
    targs = targs.view(n,-1)
    input = input.argmax(dim=1).view(n,-1)
    return (input==targs).float().mean()

def acc_thresh(input:Tensor, target:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: input = input.sigmoid()
    n = input.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    target = target.view(n,-1)
    return ((input>thresh)==target.byte()).float().mean()    

metrics = [dice_loss, accuracy_thresh, dice]   
learn = unet_learner(data, models.resnet34, metrics=metrics)
learn.metrics = metrics
learn.loss_func = combo_loss



learn.fit_one_cycle(10, max_lr=1e-3, callbacks=[SaveModelCallback(learn, monitor='dice', mode='max', name='/models/resnet34unet-img512-comboloss')])

learn.load('./models/resnet34unet-img512-comboloss.pth')
learn.model.train()
learn.unfreeze()
learn.lr_find()

learn.fit_one_cycle(40, max_lr=max_lr=slice(1e-6,1e-4), callbacks=[SaveModelCallback(learn, monitor='dice', mode='max', name='/models/resnet34unet-img512-combineloss')])
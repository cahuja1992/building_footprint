# Building Footprint Detection


## Solution
I applied a resnet34 U-Net model, one of deep neural network model for image segmentation + DBScan

## Limitations:
- Low precision for small object.
- My model is unable to recognize multiple buildings that are close in distance as one building footprint. 

## RUN

-  Data Preparation
```python
python -m building_footprint.prepare_data
```

- Resize Data
```python
python -m building_footprint.resize
```

- Train
```python
python -m building_footprint.train
```

- Predict
```python
python -m building_footprint.predict
```

- Instansation
```python
python -m building_footprint.instancing
```
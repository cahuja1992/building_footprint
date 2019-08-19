# Instance Segmentation


This approach consists of 3 parts listed below.

1. Probability of being inside and edge of buildings of each pixel are predicted by Fully Convolutional Network (FCN).
2. Building polygons are created from probability heat map by using greedy algorithm for cluster growing.
3. Confidences of being a building for each polygon are predicted by Gradient Boosting Decision Tree (GBDT) and polygons with less confidence are removed.

## Part1: Segmentation
A FCN model is used for predicting probability heat map of being inside of building. Keras library was used. properties of the model are listed below.

Structure of FCN: Input layer and output layer were connected with several dozens of convolution, pooling, upsampling layers. Batch normalization was applied for each layer and LeakyReLU activation was used for every hidden layer because recently these are widely used.

```
    python3 -m building_footprint.solution_3.data_prep
```

```
    python3 -m building_footprint.solution_3.segmentation_model train
```

## Part 2: Creating building polygons

1. Define B to be the set of points in the heatmap that are predicted to be inside the building (positive distance from the boundary). Let b1 be the point in the heatmap with the largest value.
2. Define the cluster, c1, as the largest connected component of B containing b1 such that there is a path of decreasing values between b1 and every point in c1. The cluster can be computationally identified with a nearest neighbor search.
3. Remove c1 from B and repeat to find b2 and c2.

## Part3: Building classifier
Polygons created by previous part have features such as cluster size, intensity distribution in the cluster, cluster size growing history and so on. For polygons created from the training images, IOU scores can be evaluated. By learning relevance between the features as input features and IOU score as label, model for predicting IOU score can be created. 

```
    python3 -m building_footprint.solution_3.building_model train
```
import building_footprint
import os
# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
config = building_footprint.CustomConfig()
CUSTOM_DIR = "/ws/data"
print(CUSTOM_DIR)

dataset = building_footprint.CustomDataset()
dataset.load_custom(CUSTOM_DIR, "valid")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))



dataset = building_footprint.CustomDataset()
dataset.load_custom(CUSTOM_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))    

import sys
import random
import os, numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from scipy.misc import imread, imresize
from skimage.transform import resize
from scipy.sparse import csr_matrix
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json

CITYSCAPES_CLASSES = ('dynamic', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'person', 'rider',
    'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle')

class CityscapesDataset(data.Dataset):
    def __init__(self, dataset_folder, mode, transform, random_crops=0):
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.transform = transform
        self.random_crops = random_crops
        self._init_classes()
        self._read_json()
        
    def __getitem__(self, index):
        obj_dict = self.json_data[str(index)]
        img = imread('{}/{}'.format(self.dataset_folder, obj_dict['file_name']), mode='RGB')
        img_np = img.copy()
        img = Image.fromarray(img)

        x1, y1, x2, y2 = obj_dict['bbox']
        x = img.crop((x1, y1, x2, y2))
        x_np = img_np[y1:y2, x1:x2, :]

        scale = 1
        w = int(x.size[0] * scale)
        h = int(x.size[1] * scale)
        if min(w, h) < 227:
            scale = 227 / min(w, h)
            w = int(x.size[0] * scale)
            h = int(x.size[1] * scale)

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.class_to_ind[obj_dict['category_id']]
        
        return x, y, x_np

    def __len__(self):
        return len(self.json_data)

    def _init_classes(self):
        self.classes = CITYSCAPES_CLASSES
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

    def _read_json(self):
        with open('{}/{}_object_data.json'.format(self.dataset_folder, self.mode)) as f:
            data = json.load(f)
        self.json_data = data.copy()


class LostAndFoundDataset(data.Dataset):
    def __init__(self, transform, dataset_folder='/home/yimeng/ARGO_datasets/Lost_and_Found'):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self._read_json()

    def __getitem__(self, index):
        obj_dict = self.json_data[index]
        img = imread('{}/{}'.format(self.dataset_folder, obj_dict['file_name']), mode='RGB')
        img_np = img.copy()
        img = Image.fromarray(img)

        x1, y1, x2, y2 = obj_dict['bbox']
        x = img.crop((x1, y1, x2, y2))
        x_np = img_np[y1:y2, x1:x2, :]
        #x = np.array(x)
        #x = cv2.resize(x, (227, 227))
        x = self.transform(x)
        y = 0

        return x, y, x_np

    def __len__(self):
        return len(self.json_data)

    def _read_json(self):
        with open('{}/{}_data_annotation.json'.format(self.dataset_folder, 'Lost_and_Found')) as f:
            json_data = json.load(f)
        # since some images has two outlier regions, I'd love to build a new dictionary
        new_json = {}
        count_obj = 0
        for _, v in enumerate(json_data.values()):
            file_name = v['file_name']
            img_id = v['image_id']
            height = v['height']
            width = v['width']
            regions = v['regions']
            for _, region in regions.items():
                obj = {}
                obj['file_name'] = file_name
                obj['image_id'] = img_id
                obj['height'] = height
                obj['width'] = width

                px = region['all_points_x']
                py = region['all_points_y']
                bbox = [min(px), min(py), max(px), max(py)]
                obj['bbox'] = bbox
                new_json[count_obj] = obj
                count_obj += 1

        self.json_data = new_json

'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
# Transforms applied to the testing data
test_transform = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            normalize,
        ])

ds_val = CityscapesDataset('/home/yimeng/Datasets/Cityscapes','train',test_transform)

obj_dict = ds_val.json_data[str(5)]
print(obj_dict)
img = imread('{}/{}'.format(ds_val.dataset_folder, obj_dict['file_name']), mode='RGB')
img = Image.fromarray(img)

x1, y1, x2, y2 = obj_dict['bbox']
x = img.crop((x1, y1, x2, y2))
plt.imshow(x)
plt.show()
'''

'''
outlier_val = LostAndFoundDataset()
obj_dict = outlier_val.json_data[21]
img = imread('{}/{}'.format(outlier_val.dataset_folder, obj_dict['file_name']), mode='RGB')
img = Image.fromarray(img)
x1, y1, x2, y2 = obj_dict['bbox']
x = img.crop((x1, y1, x2, y2))
x = np.array(x)
x = cv2.resize(x, (227, 227))
plt.imshow(x)
plt.show()
'''
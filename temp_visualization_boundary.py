import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead
from sseg_model import SSegHead
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset
from dataloaders.lostAndFound_proposals import LostAndFoundProposalsDataset
from dataloaders.fishyscapes_proposals import FishyscapesProposalsDataset
from dataloaders.roadAnomaly_proposals import RoadAnomalyProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax
import cv2

style = 'duq'
dataset = 'cityscapes' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'SSeg' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
ignore_background_uncertainty = False

#for dataset in ['cityscapes', 'lostAndFound', 'roadAnomaly', 'fishyscapes']:
#	for rep_style in ['both', 'ObjDet', 'SSeg']:

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'visualization/vis_road_centroids_2'
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
trained_model_dir = 'trained_model/all_props/{}/{}'.format(style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
	os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
	os.mkdir(saved_folder)

if dataset == 'cityscapes':
	dataset_folder = '/home/yimeng/ARGO_datasets/Cityscapes'
	ds_val = CityscapesProposalsDataset(dataset_folder, 'train', rep_style=rep_style)
elif dataset == 'lostAndFound':
	dataset_folder = '/home/yimeng/ARGO_datasets/Lost_and_Found'
	ds_val = LostAndFoundProposalsDataset(dataset_folder, rep_style=rep_style)

num_classes = ds_val.NUM_CLASSES

def find_boundary(grid_img, bg_list=[0, 1, 3, 4], ignore_value=255):
	H, W = grid_img.shape
	mask_boundary = np.zeros((H, W), dtype=np.bool)
	for i in range(H):
		for j in range(W):
			center = (i, j)
			if grid_img[center] == ignore_value or grid_img[center] in bg_list:
				mask_boundary[center] = 1
				continue
			left   = (i, max(j-1, 0))
			right  = (i, min(j+1, W-1))
			top    = (max(i-1, 0), j)
			bottom = (min(i+1, H-1), j)

			if (grid_img[left] == grid_img[center] or grid_img[left]==ignore_value) and \
			(grid_img[right] == grid_img[center] or grid_img[right]==ignore_value) and \
			(grid_img[top] == grid_img[center] or grid_img[top]==ignore_value) and \
			(grid_img[bottom] == grid_img[center]  or grid_img[bottom]==ignore_value):
				mask_boundary[center] = 1

	return mask_boundary

i = 0
j = 0
with torch.no_grad():

	for i in range(10):
		for j in range(100):
			patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

			a = batch_sseg_label.numpy()[0]
			color_a = apply_color_map(a)

			mask_boundary = find_boundary(a)
			color_b = color_a.copy()
			color_b[mask_boundary==0, :] = 255

			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,9))
			ax[0].imshow(color_a)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb proposal")
			ax[1].imshow(color_b)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("sseg_label_proposal")

			fig.tight_layout()
			plt.show()
			plt.close()
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt 
import glob
from PIL import Image 
import os
import torch.utils.data as data
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

device = torch.device('cuda')

class RoadAnomalyClassificationDataset(data.Dataset):
	def __init__(self, dataset_dir, rep_style='both', mask_dir=None):

		self.dataset_dir = dataset_dir
		self.rep_style = rep_style

		self.void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
		self.valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
		self.class_names = ['background', 'pole', 'person', 'car', 'train']

		self.ignore_index = 255
		self.NUM_CLASSES = len(self.class_names)
		self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

		print("Found {} images".format(60))

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals/roadAnomaly'
		self.mask_ft_folder  = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/proposal_mask_features/roadAnomaly'
		self.sseg_ft_folder  = '/projects/kosecka/yimeng/Datasets/RoadAnomaly/deeplab_ft_8_classes/'

		if mask_dir:
			self.mask_folder = mask_dir
		else:
			assert 1==2

	def __len__(self):
		return 60

	def encode_segmap(self, mask):
		#merge ambiguous classes
		mask[mask == 6] = 7 # ground -> road
		mask[mask == 8] = 7 # sidewalk -> road
		mask[mask == 9] = 7 # parking -> road
		mask[mask == 22] = 21 # terrain -> vegetation
		mask[mask == 25] = 24 # rider -> person
		mask[mask == 32] = 24 # motorcycle -> person
		mask[mask == 33] = 24 # bicycle -> person
		mask[mask == 27] = 26 # truck -> car
		mask[mask == 28] = 26 # bus -> car
		mask[mask == 29] = 26 # caravan -> car
		mask[mask == 30] = 26 # trailer -> car
		mask[mask == 12] = 11 # wall -> building
		mask[mask == 13] = 11 # fence -> building
		mask[mask == 19] = 17 # traffic light -> pole
		mask[mask == 20] = 17 # traffic sign -> pole
		mask[mask == 18] = 17 # pole group -> pole

		# Put all void classes to zero
		for _voidc in self.void_classes:
			mask[mask == _voidc] = self.ignore_index
		for _validc in self.valid_classes:
			mask[mask == _validc] = self.class_map[_validc]
		return mask

	def get_proposal(self, i, j=0):
		img_path = '{}/{}.png'.format(self.dataset_dir, i)
		lbl_path = '{}/{}_label.png'.format(self.dataset_dir, i)

		# read segmentation result mask and label
		mask_and_label = np.load('{}/img_{}_class_and_mask.npy'.format(self.mask_folder, i), allow_pickle=True).item()

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		_, H, W = sseg_feature.shape
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))

		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256
		
		index = np.array([j])
		proposals = proposals[index] # B x 4
		mask_feature = torch.tensor(mask_feature[index]).to(device) # B x 256 x 14 x 14

		sseg_mask = torch.tensor(mask_and_label['mask'][j]).unsqueeze(0)
		class_label = int(mask_and_label['class'][j])

		batch_prop_boxes = torch.zeros((1, 4))
		
		x1, y1, x2, y2 = proposals[0]
		prop_x1 = int(round(x1))
		prop_y1 = int(round(y1))
		prop_x2 = int(round(x2))
		prop_y2 = int(round(y2))

		img_proposal = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]

		batch_prop_boxes[0, 0] = prop_x1
		batch_prop_boxes[0, 1] = prop_y1
		batch_prop_boxes[0, 2] = prop_x2
		batch_prop_boxes[0, 3] = prop_y2

		batch_prop_boxes = batch_prop_boxes.to(device)
		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/8.0, aligned=True)

		if self.rep_style == 'both':
			patch_feature = torch.cat((mask_feature, batch_sseg_feature), dim=1) # B x 512 x 14 x 14
		elif self.rep_style == 'ObjDet':
			patch_feature = mask_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature

		return patch_feature, sseg_mask, img_proposal, class_label
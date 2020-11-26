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

class FishyscapesProposalsDataset(data.Dataset):
	def __init__(self, dataset_dir, rep_style='both'):

		self.dataset_dir = dataset_dir
		self.rep_style = rep_style

		self.data_json_file = json.load(open('{}/{}_data_annotation.json'.format(self.dataset_dir, 'Fishyscapes_Static')))

		self.void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
		self.valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
		self.class_names = ['unlabelled', 'road', 'building', \
							'pole', 'vegetation', 'sky', 'person', 'car', 'train', ]

		self.ignore_index = 255
		self.NUM_CLASSES = len(self.valid_classes)
		self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

		print("Found {} images".format(len(self.data_json_file)))

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals/fishyscapes'
		self.mask_ft_folder  = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/proposal_mask_features/fishyscapes'
		self.sseg_ft_folder  = '/projects/kosecka/yimeng/Datasets/Fishyscapes_Static/deeplab_ft_8_classes'

	def __len__(self):
		return len(self.data_json_file)

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

	def get_whole_img(self, i):
		img_path = '{}/{}.png'.format(self.dataset_dir, i)
		lbl_path = '{}/{}_label.png'.format(self.dataset_dir, i)

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) # 1024 x 2048
		H, W = sseg_label.shape
		#print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		regular_proposals = np.load('{}/{}_regular_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		proposals = np.concatenate((proposals, regular_proposals), axis=0)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True) # 100 x 256 x 14 x 14
		regular_mask_feature = np.load('{}/{}_regular_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		mask_feature = np.concatenate((mask_feature, regular_mask_feature), axis=0)
		#print('mask_feature.shape = {}'.format(mask_feature.shape))
		
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))

		#===================================== create whole image sseg feature ============================
		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256
		mask_feature = torch.tensor(mask_feature).to(device) # 100 x 256 x 14 x 14
		obj_feature  = torch.zeros((1, 256, H, W)).to(device) # 1 x 256 x 1024 x 2048
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))
		#assert 1==2

		num_props = proposals.shape[0]
		#print('num_props = {}'.format(num_props))
		for j in [_ for _ in range(num_props)][::-1]: # there are 100 proposals
			#print('j = {}'.format(j))
			x1, y1, x2, y2 = proposals[j]
			prop_x1 = int(max(round(x1), 0))
			prop_y1 = int(max(round(y1), 0))
			prop_x2 = int(min(round(x2), 2048-1))
			prop_y2 = int(min(round(y2), 1024-1))
			prop_w  = prop_x2 - prop_x1
			prop_h  = prop_y2 - prop_y1

			prop_feature = F.interpolate(mask_feature[j].unsqueeze(0), size=(prop_h, prop_w), mode='bilinear', align_corners=False)
			obj_feature[0, :, prop_y1:prop_y2, prop_x1:prop_x2] = prop_feature[0]

		#=================================== reduce the resolution by half ==================================
		H = int(H/2)
		W = int(W/2)

		rgb_img = cv2.resize(rgb_img, (W, H)) # 512 x 1024 x 3
		sseg_label = cv2.resize(sseg_label, (W, H), interpolation=cv2.INTER_NEAREST) # 512 x 1024
		obj_feature = F.interpolate(obj_feature, size=(H, W), mode='bilinear', align_corners=False) # 1 x 256 x 512 x 1024
		sseg_feature = F.interpolate(sseg_feature, size=(H, W), mode='bilinear', align_corners=False) # 1 x 256 x 512 x 1024
		whole_feature = torch.cat((obj_feature, sseg_feature), dim=1) # 512 x 512 x 1024

		return whole_feature, rgb_img, sseg_label
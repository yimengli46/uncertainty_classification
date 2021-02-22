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
from .pooler import ROIPooler

device = torch.device('cuda')

class CityscapesProposalsDataset(data.Dataset):
	def __init__(self, dataset_dir, split='train', batch_size=3, rep_style='both'):

		self.dataset_dir = dataset_dir
		self.split = split
		self.mode = split
		self.batch_size = batch_size
		self.rep_style = rep_style

		self.img_list = np.load('{}/{}_img_list.npy'.format(self.dataset_dir, self.mode), allow_pickle=True).tolist()

		self.void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
		self.valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
		self.class_names = ['unlabelled', 'road', 'building', \
							'pole', 'vegetation', 'sky', 'person', 'car', 'train', ]

		self.ignore_index = 255
		self.NUM_CLASSES = len(self.valid_classes)
		self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

		print("Found {} {} images".format(len(self.img_list), self.split))

		self.pooler = ROIPooler(
			output_size=14, 
			scales=[1.0/4, 1.0/8, 1.0/16, 1.0/32],
			sampling_ratio=0)

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals_whole/cityscapes_{}'.format(self.mode)
		self.mask_ft_folder  = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/whole_features/cityscapes_{}'.format(self.mode)
		self.sseg_ft_folder  = '/home/yimeng/ARGO_datasets/Cityscapes/deeplab_ft_8_classes/{}'.format(self.mode)

	def __len__(self):
		return len(self.img_list)

	def read_proposals(self, proposal_results, img_H, img_W):
		H = proposal_results['height']
		W = proposal_results['width']
		proposals = proposal_results['proposal']

		scale_x, scale_y = (img_W / W, img_H / H)

		feature_proposals = proposals.copy().astype(np.float32)
		img_proposals = proposals.copy().astype(np.float32)
		#print('img_proposals.dtype = {}'.format(img_proposals.dtype))

		img_proposals[:, 0::2] *= scale_x
		img_proposals[:, 1::2] *= scale_y

		np.clip(img_proposals[:, 0], 0, img_W, out=img_proposals[:, 0])
		np.clip(img_proposals[:, 1], 0, img_H, out=img_proposals[:, 1])
		np.clip(img_proposals[:, 2], 0, img_W, out=img_proposals[:, 2])
		np.clip(img_proposals[:, 3], 0, img_H, out=img_proposals[:, 3])

		# check if proposal size is large enough
		w_diff, h_diff = (img_proposals[:, 2] - img_proposals[:, 0], img_proposals[:, 3] - img_proposals[:, 1])
		mask = (w_diff >= 2) & (h_diff >= 2) # x2 - x1 should be at least 2.

		img_proposals = img_proposals[mask]
		feature_proposals = feature_proposals[mask] 

		return img_proposals, feature_proposals

	def __getitem__(self, i):
		img_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['left_img'])
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['semSeg'])

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		H, W, _ = rgb_img.shape
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) # 1024 x 2048
		#print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposal_results = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True).item()
		img_proposals, feature_proposals = self.read_proposals(proposal_results, H, W)

		# read mask features
		whole_feature = np.load('{}/{}_whole_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True).item()
		# 'p6' is not used 
		obj_feature = [torch.tensor(whole_feature[k]).to(device) for k in ['p2', 'p3', 'p4', 'p5']]

		#print('obj_feature.shape = {}'.format(obj_feature.shape))
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))
		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256

		N, _ = img_proposals.shape
		#if self.split == 'train':
		#	N = 100 # training stage only pick from the top 100
		index = np.random.choice(N, self.batch_size, replace=False)
		img_proposals = img_proposals[index] # B x 4
		feature_proposals = feature_proposals[index]

		batch_sseg_label = torch.zeros((self.batch_size, 28, 28))
		batch_prop_boxes = torch.tensor(img_proposals).to(device)
		batch_feature_prop_boxes = torch.tensor(feature_proposals).to(device)

		for j in range(self.batch_size):
			x1, y1, x2, y2 = img_proposals[j]

			prop_x1 = int(round(x1))
			prop_y1 = int(round(y1))
			prop_x2 = int(round(x2))
			prop_y2 = int(round(y2))

			img_patch = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
			sseg_label_patch = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

			'''
			# visualize for test
			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
			ax[0].imshow(img_patch)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb image")
			ax[1].imshow(sseg_label_patch, vmin=0.0, vmax=8.0)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("sseg label")
			plt.show()
			'''

			# rescale sseg label to 28x28
			sseg_label_patch = cv2.resize(sseg_label_patch, (28, 28), interpolation=cv2.INTER_NEAREST) # 28 x 28
			#print('sseg_label_patch.shape = {}'.format(sseg_label_patch.shape))
			batch_sseg_label[j] = torch.tensor(sseg_label_patch)

		#print('batch_prop_boxes = {}'.format(batch_prop_boxes))
		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/8.0, aligned=True)
		batch_obj_feature  = self.pooler(obj_feature, [batch_feature_prop_boxes])
		#print('batch_obj_feature.shape = {}'.format(batch_obj_feature.shape))
		#print('batch_obj_feature = {}'.format(batch_obj_feature))
		#print('batch_sseg_feature.shape = {}'.format(batch_sseg_feature.shape))

		if self.rep_style == 'both':
			patch_feature = torch.cat((batch_obj_feature, batch_sseg_feature), dim=1) # B x 512 x 14 x 14
		elif self.rep_style == 'ObjDet':
			patch_feature = batch_obj_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature 
		#print('patch_feature.shape = {}'.format(patch_feature.shape))

		return patch_feature, batch_sseg_label
		#assert 1==2

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
		img_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['left_img'])
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['semSeg'])

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		H, W, _ = rgb_img.shape
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) # 1024 x 2048
		#print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposal_results = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True).item()
		img_proposals, feature_proposals = self.read_proposals(proposal_results, H, W)

		# read mask features
		whole_feature = np.load('{}/{}_whole_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True).item()
		# 'p6' is not used 
		obj_feature = [torch.tensor(whole_feature[k]).to(device) for k in ['p2', 'p3', 'p4', 'p5']]

		#print('obj_feature.shape = {}'.format(obj_feature.shape))
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))
		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256

		N, _ = img_proposals.shape
		#index = np.random.choice(N, self.batch_size, replace=False)
		img_proposals = np.expand_dims(img_proposals[j], axis=0) # B x 4
		feature_proposals = np.expand_dims(feature_proposals[j], axis=0)
		#print('img_proposals.shape = {}'.format(img_proposals.shape))
		#print('feature_proposals.shape = {}'.format(feature_proposals.shape))

		batch_sseg_label = torch.zeros((self.batch_size, 28, 28))
		batch_prop_boxes = torch.tensor(img_proposals).to(device)
		batch_feature_prop_boxes = torch.tensor(feature_proposals).to(device)

		x1, y1, x2, y2 = img_proposals[0]

		prop_x1 = int(round(x1))
		prop_y1 = int(round(y1))
		prop_x2 = int(round(x2))
		prop_y2 = int(round(y2))

		img_proposal = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
		sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

		'''
		# visualize for test
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
		ax[0].imshow(img_patch)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("rgb image")
		ax[1].imshow(sseg_label_patch, vmin=0.0, vmax=8.0)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("sseg label")
		plt.show()
		'''

		# rescale sseg label to 28x28
		sseg_label_patch = cv2.resize(sseg_label_proposal, (28, 28), interpolation=cv2.INTER_NEAREST) # 28 x 28
		#print('sseg_label_patch.shape = {}'.format(sseg_label_patch.shape))
		batch_sseg_label[0] = torch.tensor(sseg_label_patch)

		#print('batch_prop_boxes = {}'.format(batch_prop_boxes))
		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/8.0, aligned=True)
		batch_obj_feature  = self.pooler(obj_feature, [batch_feature_prop_boxes])
		#print('batch_obj_feature.shape = {}'.format(batch_obj_feature.shape))
		#print('batch_obj_feature = {}'.format(batch_obj_feature))
		#print('batch_sseg_feature.shape = {}'.format(batch_sseg_feature.shape))

		if self.rep_style == 'both':
			patch_feature = torch.cat((batch_obj_feature, batch_sseg_feature), dim=1) # B x 512 x 14 x 14
		elif self.rep_style == 'ObjDet':
			patch_feature = batch_obj_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature 
		#print('patch_feature.shape = {}'.format(patch_feature.shape))

		return patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal
		#assert 1==2

'''
cityscapes_train = CityscapesProposalsDataset('/projects/kosecka/yimeng/Datasets/Cityscapes', 'train', batch_size=3)
#a = cityscapes_train[1]
#c = cityscapes_train[2]
b = cityscapes_train.get_proposal(0, 2)
'''
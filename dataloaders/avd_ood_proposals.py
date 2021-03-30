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

class AvdOODProposalsDataset(data.Dataset):
	def __init__(self, dataset_dir, batch_size=3, rep_style='both'):

		self.dataset_dir = dataset_dir
		self.batch_size = batch_size
		self.rep_style = rep_style

		self.img_list = np.load('{}/avd_files.npy'.format(self.dataset_dir), allow_pickle=True).tolist()

		self.valid_classes = [x for x in range(1, 33)]
		self.in_dist_classes = [35, 36, 38, 52, 46, 47, 39, 44, 45, 53]
		self.void_classes = []
		for i in range(0, 86): #Avd has 86
			if i not in (self.valid_classes + self.in_dist_classes):
				self.void_classes.append(i)
		self.class_names = ['vase', 'lamp']

		self.ignore_index = 255
		self.NUM_CLASSES = 12

		print("Found {} images".format(len(self.img_list)))

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals_ADE20K/AVD'
		self.mask_ft_folder  = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/proposal_mask_features_ADE20K/AVD'
		self.sseg_ft_folder  = '/home/yimeng/ARGO_datasets/AVD/deeplab_ft/avd'

	def __len__(self):
		return len(self.img_list)

	def encode_segmap(self, mask):
		for _validc in self.valid_classes:
			mask[mask == _validc] = 1 # ood label
		# Put all void classes to zero
		for _voidc in self.void_classes:
			mask[mask == _voidc] = 2
		for _validc in self.in_dist_classes:
			mask[mask == _validc] = 0 # training label
		
		return mask

	def select_ood_props(self, i):
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['anno'])
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) 
		sseg_label[sseg_label == 2] = 0
		H, W = sseg_label.shape
		num_props = proposals.shape[0]

		list_props_idx = np.zeros(num_props, dtype=np.int)

		for i in range(num_props):
			x1, y1, x2, y2 = proposals[i]
			prop_x1 = int(max(round(x1), 0))
			prop_y1 = int(max(round(y1), 0))
			prop_x2 = int(min(round(x2), W-1))
			prop_y2 = int(min(round(y2), H-1))
			sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]
			#print('sseg_label_proposal = {}'.format(sseg_label_proposal))

			sum_label = np.sum(sseg_label_proposal)
			area_prop = (x2-x1)*(y2-y1)
			ratio = 1.0 * sum_label / area_prop

			if ratio > 0.25 and area_prop > 900:
				#print('sum_label = {}'.format(sum_label))
				list_props_idx[i] = 1

		return list_props_idx

	def get_proposal(self, i, j=0):
		img_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['img'])
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['anno'])

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		H, W, _ = rgb_img.shape
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		'''
		fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,10))
		ax[0].imshow(rgb_img)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("rgb image")
		ax[1].imshow(sseg_label)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("sseg label")
		plt.show()
		plt.close()
		'''
		sseg_label = self.encode_segmap(sseg_label) 
		print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))
		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256

		index = np.array([j])
		proposals = proposals[index] # B x 4
		mask_feature = torch.tensor(mask_feature[index]).to(device) # B x 256 x 14 x 14

		batch_sseg_label = torch.zeros((1, 28, 28))
		batch_prop_boxes = torch.zeros((1, 4))
		
		x1, y1, x2, y2 = proposals[0]
		prop_x1 = int(max(round(x1), 0))
		prop_y1 = int(max(round(y1), 0))
		prop_x2 = int(min(round(x2), W-1))
		prop_y2 = int(min(round(y2), H-1))

		img_proposal = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
		sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

		batch_prop_boxes[0, 0] = prop_x1
		batch_prop_boxes[0, 1] = prop_y1
		batch_prop_boxes[0, 2] = prop_x2
		batch_prop_boxes[0, 3] = prop_y2

		# rescale sseg label to 28x28
		sseg_label_patch = cv2.resize(sseg_label_proposal, (28, 28), interpolation=cv2.INTER_NEAREST) # 28 x 28
		sseg_label_patch = sseg_label_patch.astype('int')
		#print('sseg_label_patch = {}'.format(sseg_label_patch))
		batch_sseg_label[0] = torch.tensor(sseg_label_patch)

		batch_prop_boxes = batch_prop_boxes.to(device)
		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/4.0, aligned=True)

		if self.rep_style == 'ObjDet':
			patch_feature = mask_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature 

		return patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal

'''
cityscapes_train = AvdOODProposalsDataset('/projects/kosecka/yimeng/Datasets/Cityscapes', 'train', batch_size=32, rep_style='both')
a = cityscapes_train[2097]
#b = cityscapes_train.get_proposal(0, 2)
#'''
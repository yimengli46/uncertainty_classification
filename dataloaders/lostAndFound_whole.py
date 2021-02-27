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

class LostAndFoundProposalsDataset(data.Dataset):
	def __init__(self, dataset_dir, rep_style='both'):

		self.dataset_dir = dataset_dir
		self.rep_style = rep_style

		self.data_json_file = json.load(open('{}/{}_data_annotation.json'.format(self.dataset_dir, 'Lost_and_Found')))

		self.void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
		self.valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
		self.sseg_class_names = ['road', 'building', 'pole', 'vegetation', 'sky', 'person', 'car', 'train']
		self.cls_class_names = ['background', 'pole', 'person', 'car', 'train']

		self.ignore_index = 255
		self.NUM_CLASSES = len(self.valid_classes)
		self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

		print("Found {} images".format(len(self.data_json_file)))

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals_whole/lostAndFound'
		self.mask_ft_folder  = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/whole_features/lostAndFound'
		self.sseg_ft_folder  = '/home/yimeng/ARGO_datasets/Lost_and_Found/deeplab_ft_8_classes'

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

	def get_num_proposal(self, i):
		v = self.data_json_file[str(i)]
		return len(v['regions'])

	def get_all_proposals(self, i, inlier_props=20):
		img_path = '{}/{}.png'.format(self.dataset_dir, i)
		lbl_path = '{}/{}_label.png'.format(self.dataset_dir, i)

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		#print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		num_outlier_props = proposals.shape[0]
		regular_proposals = np.load('{}/{}_regular_proposal.npy'.format(self.proposal_folder, i),
			allow_pickle=True) # 100 x 4
		#print('regular_proposals.shape = {}'.format(regular_proposals.shape))
		
		proposals = np.concatenate((proposals, regular_proposals), axis=0)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		regular_mask_feature = np.load('{}/{}_regular_proposal_mask_features.npy'.format(
			self.mask_ft_folder, i), allow_pickle=True) # 100 x 256 x 14 x 14
		mask_feature = np.concatenate((mask_feature, regular_mask_feature), axis=0)
		#print('regular_mask_feature.shape = {}'.format(regular_mask_feature.shape))
		#assert 1==2
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))

		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256

		num_total_props = num_outlier_props+inlier_props
		index = np.array(list(range(num_total_props)))
		proposals = proposals[index] # B x 4
		mask_feature = torch.tensor(mask_feature[index]).to(device) # B x 256 x 14 x 14

		batch_sseg_label = torch.zeros((num_total_props, 28, 28))
		batch_prop_boxes = torch.tensor(proposals).to(device)
		img_proposals = []
		sseg_label_proposals = []
		proposals_coords = []

		for j in range(num_total_props):
			x1, y1, x2, y2 = proposals[j]

			prop_x1 = int(round(x1))
			prop_y1 = int(round(y1))
			prop_x2 = int(round(x2))
			prop_y2 = int(round(y2))

			img_proposal = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
			sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]
			img_proposals.append(img_proposal)
			sseg_label_proposals.append(sseg_label_proposal)
			proposals_coords.append([prop_x1, prop_y1, prop_x2, prop_y2])

			# rescale sseg label to 28x28
			sseg_label_patch = cv2.resize(sseg_label_proposal, (28, 28), interpolation=cv2.INTER_NEAREST) # 28 x 28
			sseg_label_patch = sseg_label_patch.astype('int')
			#print('sseg_label_patch = {}'.format(sseg_label_patch))
			batch_sseg_label[j] = torch.tensor(sseg_label_patch)

		batch_prop_boxes = batch_prop_boxes.to(device).float()
		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/8.0, aligned=True)

		if self.rep_style == 'both':
			patch_feature = torch.cat((mask_feature, batch_sseg_feature), dim=1) # B x 512 x 14 x 14
		elif self.rep_style == 'ObjDet':
			patch_feature = mask_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature

		#print('patch_feature.shape = {}'.format(patch_feature.shape))

		return patch_feature, batch_sseg_label, img_proposals, sseg_label_proposals, rgb_img, proposals_coords

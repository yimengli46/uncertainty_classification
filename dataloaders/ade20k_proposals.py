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

class ADE20KProposalsDataset(data.Dataset):
	def __init__(self, dataset_dir, split='train', batch_size=3, rep_style='both'):

		self.dataset_dir = dataset_dir
		self.split = split
		self.mode = split
		self.batch_size = batch_size
		self.rep_style = rep_style

		self.img_list = np.load('{}/{}_img_list.npy'.format(self.dataset_dir, self.mode), allow_pickle=True).tolist()

		self.valid_classes = [1, 4, 6, 8, 9, 11, 15, 16, 19, 20, 23, 24, 25, 28, 29, 38, 40, 48, 51, 66]
		self.void_classes = []
		for i in range(1, 151): #ADE has 150 semantic categories
			if i not in self.valid_classes:
				self.void_classes.append(i)
		self.class_names = ['wall', 'floor', 'ceiling', 'bed', 'window', 'cabinet', 'door', 'table', 'curtain', 'chair', 'painting', 'sofa', 'shelf', 'mirror', 'carpet', 'bathtub', 'cushion', 'sink', 'fridge', 'toilet']
		assert len(self.valid_classes) == len(self.class_names)

		self.ignore_index = 255
		self.NUM_CLASSES = len(self.valid_classes)
		self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

		print("Found {} {} images".format(len(self.img_list), self.split))

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals_ADE20K/ADE20K_{}'.format(self.mode)
		self.mask_ft_folder  = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/proposal_mask_features_ADE20K/ADE20K_{}'.format(self.mode)
		self.sseg_ft_folder  = '/projects/kosecka/yimeng/Datasets/ADE20K/Semantic_Segmentation/deeplab_ft//{}'.format(self.mode)

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, i):
		img_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['img'])
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['anno'])
		#print('img_path = {}'.format(img_path))

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		H, W, _ = rgb_img.shape
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) # 1024 x 2048
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		#print('mask_feature.shape = {}'.format(mask_feature.shape))
		#assert 1==2
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))
		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256

		index = np.random.choice(30, self.batch_size, replace=False)
		#index = np.array([0,1,2])
		proposals = proposals[index] # B x 4
		mask_feature = torch.tensor(mask_feature[index]).to(device) # B x 256 x 14 x 14

		#print('proposals.shape = {}'.format(proposals.shape))
		#print('mask_feature.shape = {}'.format(mask_feature.shape))

		batch_sseg_feature = torch.zeros((self.batch_size, 256, 14, 14))
		batch_sseg_label = torch.zeros((self.batch_size, 28, 28))
		batch_prop_boxes = torch.zeros((self.batch_size, 4)).to(device)

		for j in range(self.batch_size):
			x1, y1, x2, y2 = proposals[j]

			# in case the proposal is very small
			x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
			if x2 - x1 < 2:
				x2 = x1 + 2
			if y2 - y1 < 2:
				y2 = y1 + 2

			prop_x1 = int(max(round(x1), 0))
			prop_y1 = int(max(round(y1), 0))
			prop_x2 = int(min(round(x2), W))
			prop_y2 = int(min(round(y2), H))

			# in case the proposal is outside
			if prop_x2 == W:
				prop_x1 = min(prop_x1, prop_x2 - 2)
			if prop_y2 == H:
				prop_y1 = min(prop_y1, prop_y2 - 2)

			#print('prop_x1 = {}, prop_y1 = {}, prop_x2 = {}, prop_y2 = {}'.format(prop_x1, prop_y1, prop_x2, prop_y2))

			img_patch = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
			sseg_label_patch = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

			batch_prop_boxes[j, 0] = prop_x1
			batch_prop_boxes[j, 1] = prop_y1
			batch_prop_boxes[j, 2] = prop_x2
			batch_prop_boxes[j, 3] = prop_y2
			
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

		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/4.0, aligned=True)
		batch_obj_feature = mask_feature

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
		mask[mask==31] = 20 # armchair -> chair
		mask[mask==32] = 20 # seat -> chair
		mask[mask==34] = 16 # desk -> table
		mask[mask==36] = 11 # wardrobe -> cabinet
		mask[mask==41] = 1  # base -> wall
		mask[mask==42] = 1  # pillar -> wall
		mask[mask==45] = 11 # chest -> cabinet
		mask[mask==54] = 4  # stairs -> floor
		mask[mask==58] = 40 # pillow -> cushion
		mask[mask==65] = 16 # coffee table -> table
		mask[mask==67] = 18 # flower -> plant

		# Put all void classes to zero
		for _voidc in self.void_classes:
			mask[mask == _voidc] = self.ignore_index
		for _validc in self.valid_classes:
			mask[mask == _validc] = self.class_map[_validc]
		return mask

	def get_proposal(self, i, j=0):
		img_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['img'])
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['anno'])

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		H, W, _ = rgb_img.shape
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) 
		print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)

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

		if self.rep_style == 'ObjDet':
			patch_feature = mask_feature

		return patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal

'''
cityscapes_train = CityscapesProposalsDataset('/projects/kosecka/yimeng/Datasets/Cityscapes', 'train', batch_size=32, rep_style='both')
a = cityscapes_train[2097]
#b = cityscapes_train.get_proposal(0, 2)
'''
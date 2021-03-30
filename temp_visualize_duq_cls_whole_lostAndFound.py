import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image 
from classifier_model import DuqHead as cls_duq
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset
from dataloaders.lostAndFound_proposals import LostAndFoundProposalsDataset
from dataloaders.fishyscapes_proposals import FishyscapesProposalsDataset
from dataloaders.roadAnomaly_proposals import RoadAnomalyProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map, select_props, round_prop_coords
from scipy.stats import entropy
from scipy.special import softmax
import cv2
import random

style = 'duq'
dataset = 'fishyscapes' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
ignore_background_uncertainty = False
ignore_boundary_uncertainty = False

thresh_mask_obj = 0.3
thresh_obj_uncertainty = 0.3



print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'visualization/whole_all_props'
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
cls_model_dir = 'trained_model/prop_cls_more_class_old/{}/{}'.format(style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
	os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
	os.mkdir(saved_folder)

if dataset == 'cityscapes':
	dataset_folder = '/home/yimeng/ARGO_datasets/Cityscapes'
	ds_val = CityscapesProposalsDataset(dataset_folder, 'val', rep_style=rep_style)
elif dataset == 'lostAndFound':
	dataset_folder = '/home/yimeng/ARGO_datasets/Lost_and_Found'
	ds_val = LostAndFoundProposalsDataset(dataset_folder, rep_style=rep_style)
elif dataset == 'fishyscapes':
	dataset_folder = '/home/yimeng/ARGO_datasets/Fishyscapes_Static'
	ds_val = FishyscapesProposalsDataset(dataset_folder, rep_style=rep_style)
elif dataset == 'roadAnomaly':
	dataset_folder = '/home/yimeng/ARGO_datasets/RoadAnomaly'
	ds_val = RoadAnomalyProposalsDataset(dataset_folder, rep_style=rep_style)
num_classes = 10

if rep_style == 'both':
	input_dim = 512
else:
	input_dim = 256

device = torch.device('cuda')

#========================================= load models ===================================================

cls_head = cls_duq(num_classes, input_dim).to(device)
cls_head.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(cls_model_dir, style)))
cls_head.eval()

#========================================= start evaluation ===============================================
big_outlier_list_lostAndFound = [0, 2, 3, 4, 7, 8, 9, 10, 11, 15, 16, 20, 22, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 50, 51, 52, 54, 56, 57, 58, 60, 61, 63, 65, 67, 68, 71, 72, 73, 74, 76, 80, 83, 84, 85, 86, 89, 91, 92, 93, 95, 96, 98, ]
#big_outlier_list_lostAndFound = [16]
if dataset == 'lostAndFound':
	img_id_list = big_outlier_list_lostAndFound
else: 
	img_id_list = list(range(len(ds_val)))

with torch.no_grad():
	for i in img_id_list:

		print('i = {}'.format(i))

		img_path = '{}/{}.png'.format(ds_val.dataset_dir, i)
		lbl_path = '{}/{}_label.png'.format(ds_val.dataset_dir, i)

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)

		h, w, _ = rgb_img.shape
		uncertainty_img = np.zeros((h, w))

		#========================================== select regular props through IoU ================================
		proposals = np.load('{}/{}_regular_proposal.npy'.format(ds_val.proposal_folder, i), allow_pickle=True)
		prop_idx_array = select_props(proposals)
		#assert 1==2
		num_regular_props = 500 #proposals.shape[0]
		#==========================================process regular prop =============================
		all_uncertainty = np.load('gen_object_mask/obj_sseg_duq/SSeg/{}_regular_props/img_{}_regular_prop_mask.npy'.format(dataset, i), allow_pickle=True).item()['uncertainty_bg']


		for j in reversed(list(range(num_regular_props))):
			if prop_idx_array[j] > 0:
				patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_regular_proposals(i, j)

				H, W = sseg_label_proposal.shape
				if H > 0 and W > 0:
					patch_feature = patch_feature.to(device)
					
					uncertainty = all_uncertainty[j]
					mask_obj = (all_uncertainty[j] > thresh_mask_obj)
					#================================ proposal classification ================================
					tensor_mask_obj = torch.tensor(mask_obj).unsqueeze(0).to(device)

					cls_logits = cls_head(patch_feature, tensor_mask_obj)
					# ignore uncertainty on the background class
					cls_logits = cls_logits.cpu().numpy()[0, [2,3,4,5,6,7,8,9]]
					cls_pred = np.argmax(cls_logits)
					cls_uncertainty = 1 - cls_logits[cls_pred]
					#assert 1==2

					if cls_uncertainty > thresh_obj_uncertainty:
						#cls_uncertainty = random.random() * 0.5
						uncertainty_cp = uncertainty * cls_uncertainty
					else:
						uncertainty_cp = uncertainty * 0.0
					uncertainty_cp = cv2.resize(uncertainty_cp, (W, H))

					#============================= draw proposal uncertainty and pred on whole img ==============
					x1, y1, x2, y2 = round_prop_coords(proposals[j])
					
					uncertainty_img[y1:y2, x1:x2] = uncertainty_cp
		
		#==========================================process ood prop =============================
		num_ood_props = ds_val.get_num_proposal(i)
		#print('num_ood_props = {}'.format(num_ood_props))
		for j in range(num_ood_props):
			
			patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal, prop_coords = ds_val.get_ood_proposals(i, j)
			H, W = sseg_label_proposal.shape
			patch_feature = patch_feature.to(device)

			mask_npy = np.load('gen_object_mask/obj_sseg_duq/SSeg/{}/img_{}_proposal_{}_mask.npy'.format(dataset, i, j), allow_pickle=True).item()

			uncertainty = mask_npy['uncertainty_bg']
			mask_obj = (uncertainty > thresh_mask_obj)

			#================================ proposal classification ================================
			tensor_mask_obj = torch.tensor(mask_obj).unsqueeze(0).to(device)

			cls_logits = cls_head(patch_feature, tensor_mask_obj)
			# ignore uncertainty on the background class
			cls_logits = cls_logits.cpu().numpy()[0, [2,3,4,5,6,7,8,9]]
			cls_pred = np.argmax(cls_logits)
			cls_uncertainty = 1 - cls_logits[cls_pred]
			#assert 1==2

			#uncertainty_cp = uncertainty
			#'''
			if cls_uncertainty > thresh_obj_uncertainty:
				uncertainty_cp = uncertainty * cls_uncertainty
			else:
				uncertainty_cp = uncertainty * 0.0
			#'''
			uncertainty_cp = cv2.resize(uncertainty_cp, (W, H))

			#============================= draw proposal uncertainty and pred on whole img ==============
			x1, y1, x2, y2 = prop_coords
			
			uncertainty_img[y1:y2, x1:x2] = uncertainty_cp

		#assert 1==2
		#===================================== visualize the whole img ==================================
		#color_sseg_label = apply_color_map(sseg_img)

		fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(30, 40))
		ax[0].imshow(rgb_img)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("rgb_img")
		ax[1].imshow(sseg_label)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("sseg label")
		ax[2].imshow(uncertainty_img, vmin=0.0, vmax=1.0)
		ax[2].get_xaxis().set_visible(False)
		ax[2].get_yaxis().set_visible(False)
		ax[2].set_title("uncertainty")

		fig.tight_layout()
		#plt.show()
		fig.savefig('{}/img_{}.jpg'.format(saved_folder, i))
		plt.close()

		result = {}
		result['sseg'] = sseg_label
		result['uncertainty'] = uncertainty_img
		np.save('{}/img_{}.npy'.format(saved_folder, i), result)

		#assert 1==2


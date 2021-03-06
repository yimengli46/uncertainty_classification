import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead
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
dataset = 'roadAnomaly' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
ignore_background_uncertainty = False
ignore_boundary_uncertainty = False

thresh_mask_obj = 0.3

#for dataset in ['cityscapes', 'lostAndFound', 'roadAnomaly', 'fishyscapes']:
#	for rep_style in ['both', 'ObjDet', 'SSeg']:

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'gen_object_mask'
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
trained_model_dir = 'trained_model/no_conv_maskrcnn_coco/{}/{}'.format(style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
	os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
	os.mkdir(saved_folder)

if dataset == 'cityscapes':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Cityscapes'
	ds_val = CityscapesProposalsDataset(dataset_folder, 'val', rep_style=rep_style)
elif dataset == 'lostAndFound':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Lost_and_Found'
	ds_val = LostAndFoundProposalsDataset(dataset_folder, rep_style=rep_style)
elif dataset == 'fishyscapes':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Fishyscapes_Static'
	ds_val = FishyscapesProposalsDataset(dataset_folder, rep_style=rep_style)
elif dataset == 'roadAnomaly':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/RoadAnomaly'
	ds_val = RoadAnomalyProposalsDataset(dataset_folder, rep_style=rep_style)
num_classes = ds_val.NUM_CLASSES

if rep_style == 'both':
	input_dim = 512
else:
	input_dim = 256

device = torch.device('cuda')

#classifier = DuqHead(num_classes, input_dim).to(device)
classifier = DuqHead(num_classes, input_dim).to(device)
#classifier = SSegHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(trained_model_dir, style)))
classifier.eval()

big_outlier_list_lostAndFound = [0, 2, 3, 4, 7, 8, 9, 10, 11, 15, 16, 20, 22, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 50, 51, 52, 54, 56, 57, 58, 60, 61, 63, 65, 67, 68, 71, 72, 73, 74, 76, 80, 83, 84, 85, 86, 89, 91, 92, 93, 95, 96, 98, ]
#big_outlier_list_lostAndFound = [72]
if dataset == 'lostAndFound':
	img_id_list = big_outlier_list_lostAndFound
else: 
	img_id_list = list(range(len(ds_val)))

with torch.no_grad():
	for i in img_id_list:
		if dataset == 'cityscapes':
			num_proposals = 100
		elif dataset == 'lostAndFound':
			num_proposals = ds_val.get_num_proposal(i)
		elif dataset == 'fishyscapes':
			num_proposals = ds_val.get_num_proposal(i)# + 10
		elif dataset == 'roadAnomaly':
			num_proposals = 20
		
		for j in range(num_proposals):
			if dataset == 'roadAnomaly':
				patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)
			else:
				patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_all_proposals(i, j)

			patch_feature = patch_feature.to(device)
			logits = classifier(patch_feature)
			#logits = logits[0]

			H, W = sseg_label_proposal.shape
			#logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

# ================================== estimate mask_obj based on obj classes ==========================
			logit = logits.cpu().numpy()[0] # 4 x H x W
			sseg_pred = np.argmax(logit, axis=0)

			logit = logit[[5, 6, 7]]
			uncertainty_obj = 1 - np.amax(logit, axis=0)
			mask_obj_from_obj = (uncertainty_obj < 0.1)
			
# ================================== estimate mask_obj based on background classes ==========================
			logits = logits.cpu().numpy()[0, [0, 1, 3, 4]] # 4 x H x W
			uncertainty = 1.0 - np.amax(logits, axis=0)

			if ignore_boundary_uncertainty:
				uncertainty[:2, :] = 0 #np.repeat(uncertainty[[5], :], 5, axis=0)
				uncertainty[-2:, :] = 0 #np.repeat(uncertainty[[-5], :], 5, axis=0)
				uncertainty[:, :2] = 0 #np.repeat(uncertainty[:, [5]], 5, axis=1)
				uncertainty[:, -2:] = 0 #np.repeat(uncertainty[:, [-5]], 5, axis=1)
			
			mask_obj = (uncertainty > thresh_mask_obj)
			uncertainty[mask_obj < 1] = 0

			if dataset == 'cityscapes':
				color_sseg_label_proposal = apply_color_map(sseg_label_proposal)
			else:
				color_sseg_label_proposal = sseg_label_proposal
			color_sseg_pred = apply_color_map(sseg_pred)
			#assert 1==2

			if save_option == 'both' or save_option == 'image':
				fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(18,15))
				ax[0][0].imshow(img_proposal)
				ax[0][0].get_xaxis().set_visible(False)
				ax[0][0].get_yaxis().set_visible(False)
				ax[0][0].set_title("rgb proposal")
				ax[0][1].imshow(color_sseg_pred)
				ax[0][1].get_xaxis().set_visible(False)
				ax[0][1].get_yaxis().set_visible(False)
				ax[0][1].set_title("color_sseg_pred")
				ax[1][0].imshow(uncertainty, vmin=0.0, vmax=1.0)
				ax[1][0].get_xaxis().set_visible(False)
				ax[1][0].get_yaxis().set_visible(False)
				ax[1][0].set_title("uncertainty from bg")
				ax[1][1].imshow(mask_obj, vmin=0.0, vmax=1.0)
				ax[1][1].get_xaxis().set_visible(False)
				ax[1][1].get_yaxis().set_visible(False)
				ax[1][1].set_title("mask_obj from bg")
				ax[2][0].imshow(uncertainty_obj, vmin=0.0, vmax=1.0)
				ax[2][0].get_xaxis().set_visible(False)
				ax[2][0].get_yaxis().set_visible(False)
				ax[2][0].set_title("uncertainty from obj")
				ax[2][1].imshow(mask_obj_from_obj, vmin=0.0, vmax=1.0)
				ax[2][1].get_xaxis().set_visible(False)
				ax[2][1].get_yaxis().set_visible(False)
				ax[2][1].set_title("mask_obj_from_obj")

				fig.tight_layout()
				fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
				plt.close()

			if save_option == 'both' or save_option == 'npy':

				result = {}
				result['sseg'] = sseg_pred
				result['uncertainty_bg'] = uncertainty
				result['mask_obj_from_bg'] = mask_obj
				result['uncertainty_obj'] = uncertainty_obj
				result['mask_obj_from_obj'] = mask_obj_from_obj
				np.save('{}/img_{}_proposal_{}_mask.npy'.format(saved_folder, i, j), result)

				#assert 1==2


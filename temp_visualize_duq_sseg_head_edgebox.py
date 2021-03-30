import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead
from sseg_model import SSegHead
from dataloaders.lostAndFound_proposals import LostAndFoundProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax
import cv2

style = 'duq'
dataset = 'lostAndFound' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'SSeg' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
ignore_background_uncertainty = False
ignore_boundary_uncertainty = False

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'visualization/all_props_edgebox'
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
num_classes = ds_val.NUM_CLASSES

if rep_style == 'both':
	input_dim = 512
else:
	input_dim = 256

device = torch.device('cuda')

classifier = DuqHead(num_classes, input_dim).to(device)
#classifier = SSegHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(trained_model_dir, style)))
classifier.eval()

big_outlier_list_lostAndFound = [0, 2, 3, 4, 7, 8, 9, 10, 11, 15, 16, 20, 22, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 50, 51, 52, 54, 56, 57, 58, 60, 61, 63, 65, 67, 68, 71, 72, 73, 74, 76, 80, 83, 84, 85, 86, 89, 91, 92, 93, 95, 96, 98, ]
#big_outlier_list_lostAndFound = [16]
if dataset == 'lostAndFound':
	img_id_list = big_outlier_list_lostAndFound
else: 
	img_id_list = list(range(len(ds_val)))



with torch.no_grad():
	for i in range(len(ds_val)):
		print('i = {}'.format(i))

		if dataset == 'lostAndFound':
			num_proposals = ds_val.get_num_edgebox_ood_proposal(i)
		elif dataset == 'fishyscapes':
			num_proposals = ds_val.get_num_proposal(i)
		elif dataset == 'roadAnomaly':
			num_proposals = 5

		i = 83
		
		for j in range(num_proposals):
			j = 3

			patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_edgebox_ood_proposal(i, j)

			patch_feature = patch_feature.cpu().numpy()
			result = {}
			result['patch_feature'] = patch_feature
			result['img_proposal'] = img_proposal
			result['sseg_label_proposal'] = sseg_label_proposal

			np.save('ex_1.npy', result)

			assert 1==2
			patch_feature = patch_feature.to(device)
			logits = classifier(patch_feature)
			#logits = logits[0]

			H, W = sseg_label_proposal.shape
			#'''
			# compute uncertainty on background classes
			logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

			#logits = logits.cpu().numpy()[0]
			logits = logits.cpu().numpy()[0, [0, 1, 3, 4]] # 4 x H x W
			uncertainty = 1.0 - np.amax(logits, axis=0)
			
			sseg_pred = np.argmax(logits, axis=0)
			sseg_pred[sseg_pred == 0] = 0 # road
			sseg_pred[sseg_pred == 1] = 1 # building
			sseg_pred[sseg_pred == 2] = 3 # vegetation
			sseg_pred[sseg_pred == 3] = 4 # sky
			sseg_pred[uncertainty > 0.3] = 6

			'''
			logits = logits[[0, 1, 2, 3, 4, 5, 6, 7]] # 4 x H x W
			uncertainty = 1.0 - np.amax(logits, axis=0)

			uncertainty[sseg_pred == 0] = 0 # road
			uncertainty[sseg_pred == 1] = 0 # building
			uncertainty[sseg_pred == 3] = 0 # vegetation
			uncertainty[sseg_pred == 4] = 0 # sky
			'''


			if dataset == 'cityscapes':
				color_sseg_label_proposal = apply_color_map(sseg_label_proposal)
			else:
				color_sseg_label_proposal = sseg_label_proposal
			color_sseg_pred = apply_color_map(sseg_pred)
			#assert 1==2

			if save_option == 'both' or save_option == 'image':
				fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18,10))
				ax[0][0].imshow(img_proposal)
				ax[0][0].get_xaxis().set_visible(False)
				ax[0][0].get_yaxis().set_visible(False)
				ax[0][0].set_title("rgb proposal")
				ax[0][1].imshow(color_sseg_label_proposal)
				ax[0][1].get_xaxis().set_visible(False)
				ax[0][1].get_yaxis().set_visible(False)
				ax[0][1].set_title("sseg_label_proposal")
				ax[1][0].imshow(color_sseg_pred)
				ax[1][0].get_xaxis().set_visible(False)
				ax[1][0].get_yaxis().set_visible(False)
				ax[1][0].set_title("sseg pred")
				ax[1][1].imshow(uncertainty, vmin=0.0, vmax=1.0)
				ax[1][1].get_xaxis().set_visible(False)
				ax[1][1].get_yaxis().set_visible(False)
				ax[1][1].set_title("uncertainty")

				fig.tight_layout()
				fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
				plt.close()

			if save_option == 'both' or save_option == 'npy':

				# remove uncertainty on the image boundary
				if ignore_boundary_uncertainty:
					uncertainty[:5, :] = 0 #np.repeat(uncertainty[[5], :], 5, axis=0)
					uncertainty[-5:, :] = 0 #np.repeat(uncertainty[[-5], :], 5, axis=0)
					uncertainty[:, :5] = 0 #np.repeat(uncertainty[:, [5]], 5, axis=1)
					uncertainty[:, -5:] = 0 #np.repeat(uncertainty[:, [-5]], 5, axis=1)

				result = {}
				result['sseg'] = sseg_pred
				result['uncertainty'] = uncertainty
				np.save('{}/img_{}_proposal_{}.npy'.format(saved_folder, i, j), result)

				#assert 1==2


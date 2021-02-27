import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead
from sseg_model import SSegHead
from dataloaders.ade20k_proposals import ADE20KProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax
import cv2

style = 'duq'
dataset = 'ade20k'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
ignore_background_uncertainty = False
ignore_boundary_uncertainty = True

#for dataset in ['cityscapes', 'lostAndFound', 'roadAnomaly', 'fishyscapes']:
#	for rep_style in ['both', 'ObjDet', 'SSeg']:

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'visualization/ade20k'
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
trained_model_dir = 'trained_model/ade20k/{}/{}'.format(style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
	os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
	os.mkdir(saved_folder)

if dataset == 'ade20k':
	dataset_folder = '/home/yimeng/ARGO_datasets/ADE20K/Semantic_Segmentation'
	ds_val = ADE20KProposalsDataset(dataset_folder, 'val', rep_style=rep_style)
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


with torch.no_grad():
	for i in range(len(ds_val)):
		if dataset == 'ade20k':
			num_proposals = 5
		
		for j in range(num_proposals):
			patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

			patch_feature = patch_feature.to(device)
			logits = classifier(patch_feature)
			logits = logits[0]

			H, W = sseg_label_proposal.shape

			#'''
			# compute uncertainty on all classes
			logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)

			#logits = logits.cpu().numpy()[0]
			logits = logits.cpu().numpy()[0] # 4 x H x W
			uncertainty = 1.0 - np.amax(logits, axis=0)
			
			sseg_pred = np.argmax(logits, axis=0)
			#'''

			color_sseg_label_proposal = apply_color_map(sseg_label_proposal, num_classes='ade20k')
			color_sseg_pred = apply_color_map(sseg_pred, num_classes='ade20k')
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
				plt.show()
				#fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
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


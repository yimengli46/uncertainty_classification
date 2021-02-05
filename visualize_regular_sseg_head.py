import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import SSegHead
from dataloaders.ade20k_proposals import ADE20KProposalsDataset
#from dataloaders.lostAndFound_proposals import LostAndFoundProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax

style = 'regular'
dataset = 'ade20k' #'lostAndFound', 'cityscapes', 'fishyscapes'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
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
	dataset_folder = '/projects/kosecka/yimeng/Datasets/ADE20K/Semantic_Segmentation'
	ds_val = ADE20KProposalsDataset(dataset_folder, 'val', rep_style=rep_style)

num_classes = ds_val.NUM_CLASSES

if rep_style == 'both':
    input_dim = 512
else:
    input_dim = 256

device = torch.device('cuda')

classifier = SSegHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/regular_classifier.pth'.format(trained_model_dir)))

#print('aaaaaaaaaaaaaa')
#ssert 1==2

with torch.no_grad():
	for i in range(len(ds_val)):
		if dataset == 'ade20k':
			num_proposals = 15
		elif dataset == 'lostAndFound':
			num_proposals = ds_val.get_num_proposal(i)
		
		for j in range(5, 15):
			print('i = {}, j = {}'.format(i, j))
			patch_feature, _, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

			patch_feature = patch_feature.to(device)
			logits = classifier(patch_feature)

			H, W = sseg_label_proposal.shape

			logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
			sseg_pred = torch.argmax(logits, dim=1)

			logits = logits.cpu().numpy()[0]
			sseg_pred = sseg_pred.cpu().numpy()[0]

			uncertainty = entropy(softmax(logits, axis=0), axis=0, base=2)

			'''
			if dataset == 'cityscapes':
				color_sseg_label_proposal = apply_color_map(sseg_label_proposal)
			else:
				color_sseg_label_proposal = sseg_label_proposal
			color_sseg_pred = apply_color_map(sseg_pred)
			#assert 1==2
			'''

			uncertainty[sseg_pred == 0] = 0 # road
			uncertainty[sseg_pred == 1] = 0 # building
			uncertainty[sseg_pred == 2] = 0 # vegetation
			uncertainty[sseg_pred == 4] = 0 # sky
			uncertainty[sseg_pred == 5] = 0
			uncertainty[sseg_pred == 6] = 0

			if save_option == 'both' or save_option == 'image':
				fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18,10))
				ax[0][0].imshow(img_proposal)
				ax[0][0].get_xaxis().set_visible(False)
				ax[0][0].get_yaxis().set_visible(False)
				ax[0][0].set_title("rgb proposal")
				ax[0][1].imshow(sseg_label_proposal, vmin=0, vmax=num_classes)
				ax[0][1].get_xaxis().set_visible(False)
				ax[0][1].get_yaxis().set_visible(False)
				ax[0][1].set_title("sseg_label_proposal")
				ax[1][0].imshow(sseg_pred, vmin=0, vmax=num_classes)
				ax[1][0].get_xaxis().set_visible(False)
				ax[1][0].get_yaxis().set_visible(False)
				ax[1][0].set_title("sseg pred")
				ax[1][1].imshow(uncertainty, vmin=0.0, vmax=3.0)
				ax[1][1].get_xaxis().set_visible(False)
				ax[1][1].get_yaxis().set_visible(False)
				ax[1][1].set_title("uncertainty")

				fig.tight_layout()
				fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
				plt.close()

			if save_option == 'both' or save_option == 'npy':
				result = {}
				result['sseg'] = sseg_pred
				result['uncertainty'] = uncertainty
				np.save('{}/img_{}_proposal_{}.npy'.format(saved_folder, i, j), result)
		

		#assert 1==2


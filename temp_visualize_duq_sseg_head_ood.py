import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead
from dataloaders.ade20k_ood_proposals import ADE20KOODProposalsDataset
from dataloaders.avd_ood_proposals import AvdOODProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax
from PIL import Image 

style = 'duq'
dataset = 'avd' #'ade20k' 
rep_style = 'SSeg' #'both', 'ObjDet', 'SSeg' 
save_option = 'npy' #'image', 'npy'

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'visualization/ade20k_ood'
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
trained_model_dir = 'trained_model/ade20k/{}/{}'.format(style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
    os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
    os.mkdir(saved_folder)

if rep_style == 'both':
    input_dim = 512
else:
    input_dim = 256

if dataset == 'ade20k':
	dataset_folder = '/home/yimeng/ARGO_datasets/ADE20K/Semantic_Segmentation'
	ds_val = ADE20KOODProposalsDataset(dataset_folder, split='val', rep_style=rep_style)
elif dataset == 'avd':
	dataset_folder = '/home/yimeng/ARGO_datasets/AVD'
	ds_val = AvdOODProposalsDataset(dataset_folder, rep_style=rep_style)
num_classes = ds_val.NUM_CLASSES

device = torch.device('cuda')
classifier = DuqHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(trained_model_dir, style)))
classifier.eval()

with torch.no_grad():
	for i in range(len(ds_val)):
		print('i = {}'.format(i))
		num_proposals = 100
		ood_prop_array = ds_val.select_ood_props(i)
		
		for j in range(num_proposals):
			if ood_prop_array[j] > 0:
				print('i = {}, j = {}'.format(i, j))
				patch_feature, _, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

				patch_feature = patch_feature.to(device)
				logits = classifier(patch_feature)

				H, W = sseg_label_proposal.shape

				logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
				
				#logits = logits.cpu().numpy()[0, [0,1,2,3,4]]
				logits = logits.cpu().numpy()[0]
				sseg_pred = np.argmax(logits, axis=0)
				#assert 1==2

				uncertainty = 1.0 - np.amax(logits, axis=0)

				
				color_sseg_pred = apply_color_map(sseg_pred, num_classes=20)
				#assert 1==2

				if save_option == 'both' or save_option == 'image':
					fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18,10))
					ax[0][0].imshow(img_proposal)
					ax[0][0].get_xaxis().set_visible(False)
					ax[0][0].get_yaxis().set_visible(False)
					ax[0][0].set_title("rgb proposal")
					ax[0][1].imshow(sseg_label_proposal, vmin=0.0, vmax=2.0)
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
					#plt.show()
					fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
					plt.close()

				if save_option == 'both' or save_option == 'npy':
					result = {}
					result['sseg'] = sseg_pred
					result['uncertainty'] = uncertainty
					np.save('{}/img_{}_proposal_{}.npy'.format(saved_folder, i, j), result)

			#assert 1==2


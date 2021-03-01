import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead
from dataloaders.lostAndFound_proposals import LostAndFoundProposalsDataset
from dataloaders.fishyscapes_proposals import FishyscapesProposalsDataset
from dataloaders.roadAnomaly_proposals import RoadAnomalyProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map, gen_mask
from scipy.stats import entropy
from scipy.special import softmax

method = 'all_props'
style = 'duq'
dataset = 'roadAnomaly' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
save_option = 'npy' #'image', 'npy'
ignore_background_uncertainty = True

print('method = {}, style = {}, rep_style = {},  dataset = {}'.format(method, style, rep_style, dataset))

base_folder = 'classification_label_mask/{}'.format(method)
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
trained_model_dir = 'trained_model/{}/{}/{}'.format(method, style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
	os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
	os.mkdir(saved_folder)


if dataset == 'lostAndFound':
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

classifier = DuqHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(trained_model_dir, style)))
#classifier.eval()

with torch.no_grad():
	for i in range(len(ds_val)):
		if dataset == 'lostAndFound':
			num_proposals = ds_val.get_num_proposal(i)
		elif dataset == 'fishyscapes':
			num_proposals = ds_val.get_num_proposal(i)
		elif dataset == 'roadAnomaly':
			num_proposals = 20

		if save_option == 'image':
			for j in range(num_proposals):
				print('i = {}, j = {}'.format(i, j))
				patch_feature, patch_label, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

				patch_feature = patch_feature.to(device)
				logits = classifier(patch_feature) # 1 x 8 x 28 x 28

				sseg_pred = torch.argmax(logits, dim=1)
				sseg_pred = sseg_pred.cpu().numpy()[0] # 28 x 28
				patch_label = patch_label.cpu().numpy()[0] # 28 x 28
				#print('patch_label.shape = {}'.format(patch_label.shape))

				object_mask = gen_mask(sseg_pred)

				if dataset == 'cityscapes':
					color_sseg_label_proposal = apply_color_map(sseg_label_proposal)
				else:
					color_sseg_label_proposal = sseg_label_proposal
				color_sseg_pred = apply_color_map(sseg_pred)

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
				ax[1][1].imshow(object_mask)
				ax[1][1].get_xaxis().set_visible(False)
				ax[1][1].get_yaxis().set_visible(False)
				ax[1][1].set_title("gt object mask")

				fig.tight_layout()
				fig.suptitle('class = {}, dist = {}'.format(gt_class_label, class_dist))
				fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
				plt.close()
				#assert 1==2

		if save_option == 'npy':

			all_object_mask = np.zeros((num_proposals, 28, 28), dtype=np.bool)
			all_class_label = np.zeros(num_proposals)

			for j in range(num_proposals):
				print('i = {}, j = {}'.format(i, j))
				patch_feature, patch_label, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

				patch_feature = patch_feature.to(device)
				logits = classifier(patch_feature) # 1 x 8 x 28 x 28

				sseg_pred = torch.argmax(logits, dim=1)
				sseg_pred = sseg_pred.cpu().numpy()[0] # 28 x 28
				patch_label = patch_label.cpu().numpy()[0] # 28 x 28
				#print('patch_label.shape = {}'.format(patch_label.shape))

				object_mask = gen_mask(sseg_pred)
				all_object_mask[j] = object_mask
				all_class_label[j] = 5

			result = {}
			result['mask'] = all_object_mask
			result['class'] = all_class_label
			np.save('{}/img_{}_class_and_mask.npy'.format(saved_folder, i), result)

			#assert 1==2


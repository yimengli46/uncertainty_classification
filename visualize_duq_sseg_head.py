import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from classifier_model import DuqHead
from dataloaders.cityscapes_classification import CityscapesClassificationDataset
from dataloaders.lostAndFound_classification import LostAndFoundClassificationDataset
from dataloaders.fishyscapes_classification import FishyscapesClassificationDataset
from dataloaders.roadAnomaly_classification import RoadAnomalyClassificationDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax

style = 'duq'
dataset = 'roadAnomaly' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
method = 'prop_classification'

#for dataset in ['roadAnomaly']:#'lostAndFound', 'fishyscapes']:
#	for rep_style in ['both', 'ObjDet', 'SSeg']:

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'visualization/{}'.format(method)
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
trained_model_dir = 'trained_model/{}/{}/{}'.format(method, style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
	os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
	os.mkdir(saved_folder)

if dataset == 'cityscapes':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Cityscapes'
	mask_folder_val = 'classification_label_mask/all_props/obj_sseg_duq/ObjDet/cityscapes_{}'.format('val')
	ds_val = CityscapesClassificationDataset(dataset_folder, 'val', rep_style=rep_style, mask_dir=mask_folder_val)
elif dataset == 'lostAndFound':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Lost_and_Found'
	mask_folder_val = 'classification_label_mask/all_props/obj_sseg_duq/ObjDet/lostAndFound'
	ds_val = LostAndFoundClassificationDataset(dataset_folder, rep_style=rep_style, mask_dir=mask_folder_val)
elif dataset == 'fishyscapes':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Fishyscapes_Static'
	mask_folder_val = 'classification_label_mask/all_props/obj_sseg_duq/ObjDet/fishyscapes'
	ds_val = FishyscapesClassificationDataset(dataset_folder, rep_style=rep_style, mask_dir=mask_folder_val)
elif dataset == 'roadAnomaly':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/RoadAnomaly'
	mask_folder_val = 'classification_label_mask/all_props/obj_sseg_duq/ObjDet/roadAnomaly'
	ds_val = RoadAnomalyClassificationDataset(dataset_folder, rep_style=rep_style, mask_dir=mask_folder_val)
num_classes = ds_val.NUM_CLASSES

if rep_style == 'both':
	input_dim = 512
else:
	input_dim = 256

device = torch.device('cuda')

classifier = DuqHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(trained_model_dir, style)))
#classifier.eval()
#assert 1==2

with torch.no_grad():
	for i in range(len(ds_val)):
		if dataset == 'cityscapes':
			num_proposals = 100
		elif dataset == 'lostAndFound':
			num_proposals = ds_val.get_num_proposal(i)
		elif dataset == 'fishyscapes':
			num_proposals = ds_val.get_num_proposal(i)
		elif dataset == 'roadAnomaly':
			num_proposals = 20
		
		for j in range(num_proposals):
			print('i = {}, j = {}'.format(i, j))
			if dataset == 'cityscapes':
				patch_feature, sseg_mask, img_proposal, class_label, N  = ds_val.get_proposal(i, j)
			else:
				patch_feature, sseg_mask, img_proposal, class_label  = ds_val.get_proposal(i, j)

			patch_feature = patch_feature.to(device)
			sseg_mask = sseg_mask.to(device)
			logits = classifier(patch_feature, sseg_mask)

			sseg_pred = torch.argmax(logits, dim=1)
			logits = logits.cpu().numpy()[0]
			sseg_pred = sseg_pred.cpu().numpy()[0]
			uncertainty = 1.0 - np.amax(logits, axis=0)
			flag_class = (class_label == sseg_pred)

			sseg_mask = sseg_mask.cpu().numpy()[0]

			if save_option == 'both' or save_option == 'image':
				fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
				ax[0].imshow(img_proposal)
				ax[0].get_xaxis().set_visible(False)
				ax[0].get_yaxis().set_visible(False)
				ax[0].set_title("rgb proposal")
				ax[1].imshow(sseg_mask)
				ax[1].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].set_title("sseg_mask")
				fig.suptitle('flag = {}, pred = {}, class = {}, uncertainty = {}'.format(flag_class, sseg_pred, class_label, uncertainty))
				fig.tight_layout()
				fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
				plt.close()

			if save_option == 'both' or save_option == 'npy':
				result = {}
				result['sseg'] = sseg_pred
				result['uncertainty'] = uncertainty
				np.save('{}/img_{}_proposal_{}.npy'.format(saved_folder, i, j), result)

			#assert 1==2


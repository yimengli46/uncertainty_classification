import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from classifier_model import DuqHead as cls_duq
from dataloaders.cityscapes_allClasses_classification import CityscapesAllClassesClassificationDataset
import torch.nn.functional as F
from utils import apply_color_map, convert_pred_and_label, convert_pred_and_label_sseg
from scipy.stats import entropy
from scipy.special import softmax
import cv2

trained_classes = 10

style = 'duq'
dataset = 'cityscapes' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
save_option = 'npy' #'image', 'npy'
ignore_background_uncertainty = False
ignore_boundary_uncertainty = False

thresh_mask_obj = 0.3

#for dataset in ['cityscapes', 'lostAndFound', 'roadAnomaly', 'fishyscapes']:
#	for rep_style in ['both', 'ObjDet', 'SSeg']:

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'cls_results/prop_cls_more_class_old'
#base_folder = 'cls_results/prop_classification'
saved_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset)
cls_model_dir = 'trained_model/prop_cls_more_class_old/{}/{}'.format(style, rep_style)
#cls_model_dir = 'trained_model/prop_classification/{}/{}'.format(style, rep_style)

# check if folder exists
if not os.path.exists('{}/obj_sseg_{}'.format(base_folder, style)):
	os.mkdir('{}/obj_sseg_{}'.format(base_folder, style))
if not os.path.exists('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style)): 
	os.mkdir('{}/obj_sseg_{}/{}'.format(base_folder, style, rep_style))
if not os.path.exists(saved_folder): 
	os.mkdir(saved_folder)

if dataset == 'cityscapes':
	dataset_folder = '/home/yimeng/ARGO_datasets/Cityscapes'
	ds_val = CityscapesAllClassesClassificationDataset(dataset_folder, 'val', batch_size=64, rep_style=rep_style)

num_classes = trained_classes
cls_class_names = ds_val.cls_class_names

if rep_style == 'both':
	input_dim = 512
else:
	input_dim = 256

device = torch.device('cuda')

#========================================= load models ===================================================

cls_head = cls_duq(num_classes, input_dim).to(device)
cls_head.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(cls_model_dir, style)))
cls_head.eval()

img_id_list = list(range(len(ds_val)))

with torch.no_grad():
	for i in img_id_list:
		print('i = {}'.format(i))
		patch_feature, sseg_mask, class_label = ds_val.get_top_proposals(i)

		# only take the object classes
		patch_feature = patch_feature[(class_label > 1)]
		sseg_mask = sseg_mask[(class_label > 1)]
		class_label = class_label[(class_label > 1)]

		patch_feature = patch_feature.to(device)

		#================================ proposal classification ================================
		tensor_mask_obj = torch.tensor(sseg_mask).to(device)


		cls_logits = cls_head(patch_feature, tensor_mask_obj)
		# ignore uncertainty on the background class
		if trained_classes > 5:
			cls_logits = cls_logits.cpu().numpy()[:, [2,3,4,5,6,7,8,9]]
		elif trained_classes == 5:
			cls_logits = cls_logits.cpu().numpy()[:, [2,3,4]]
		cls_pred = np.argmax(cls_logits, axis=1)
		cls_uncertainty = 1 - np.amax(cls_logits, axis=1)
		cls_pred = cls_pred + 2
		class_label = class_label.numpy().astype('int')

		#assert 1==2
		if  trained_classes > 5:
			cls_pred = convert_pred_and_label(cls_pred)
			class_label = convert_pred_and_label(class_label)
		elif trained_classes == 5:
			class_label = convert_pred_and_label_sseg(class_label)
		#assert 1==2

		'''
		mask_obj = cv2.resize(mask_obj.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
		if dataset == 'cityscapes':
			color_sseg_label_proposal = apply_color_map(sseg_label_proposal)
		else:
			color_sseg_label_proposal = sseg_label_proposal
		color_sseg_pred = apply_color_map(sseg_pred)
		#assert 1==2
		'''

		'''
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
			ax[1][0].imshow(uncertainty, vmin=0.0, vmax=1.0)
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title("uncertainty")
			ax[1][1].imshow(mask_obj)
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title("class= {}, uncertainty = {:.4f}".format(cls_class_names[cls_pred+2], \
				cls_uncertainty))

			fig.tight_layout()
			#plt.show()
			fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
			plt.close()
		'''

		if save_option == 'both' or save_option == 'npy':

			result = {}
			result['pred'] = cls_pred
			result['uncertainty'] = cls_uncertainty
			result['label'] = class_label
			np.save('{}/img_{}.npy'.format(saved_folder, i), result)

			#assert 1==2


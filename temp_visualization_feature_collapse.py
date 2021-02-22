import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DuqHead
from sseg_model import SSegHead
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset
from dataloaders.lostAndFound_proposals import LostAndFoundProposalsDataset
from dataloaders.fishyscapes_proposals import FishyscapesProposalsDataset
from dataloaders.roadAnomaly_proposals import RoadAnomalyProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax
import cv2

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

style = 'regular'
dataset = 'lostAndFound' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'SSeg' #'both', 'ObjDet', 'SSeg' 
save_option = 'image' #'image', 'npy'
ignore_background_uncertainty = False

#for dataset in ['cityscapes', 'lostAndFound', 'roadAnomaly', 'fishyscapes']:
#	for rep_style in ['both', 'ObjDet', 'SSeg']:

print('style = {}, rep_style = {},  dataset = {}'.format(style, rep_style, dataset))

base_folder = 'visualization/temp_all_props'
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

num_classes = ds_val.NUM_CLASSES

if rep_style == 'both':
	input_dim = 512
else:
	input_dim = 256

device = torch.device('cuda')

if style == 'duq':
	classifier = DuqHead(num_classes, input_dim).to(device)
	classifier.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(trained_model_dir, style)))
elif style == 'regular':
	classifier = SSegHead(num_classes, input_dim).to(device)
	classifier.load_state_dict(torch.load('{}/{}_classifier.pth'.format(trained_model_dir, style)))
classifier.eval()


i = 3
j = 1
with torch.no_grad():

	patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

	# visualize input feature
	input_patch_feature = patch_feature.cpu().numpy()[0] # 256 x 14 x 14
	batch_sseg_label = batch_sseg_label.cpu().numpy()[0]
	
	#'''
	input_batch_sseg_label = cv2.resize(batch_sseg_label, (14, 14), interpolation=cv2.INTER_NEAREST)
	background_feature = input_patch_feature[:, input_batch_sseg_label==0].transpose() # N x 256
	#assert 1==2
	input_patch_feature = input_patch_feature.transpose((1, 2, 0)) # 14 x 14 x 256
	input_patch_feature = normalized(input_patch_feature, axis=2)
	background_feature = normalized(background_feature, axis=1)
	H, W, _ = input_patch_feature.shape
	vis_input = np.zeros((H, W))
	for i in range(H):
		for j in range(W):
			ft = input_patch_feature[i, j]
			#dist = np.sum(np.square(ft - background_feature), axis=1).mean() # N
			dist = np.linalg.norm(ft - background_feature[0])
			vis_input[i, j] = dist

	H, W = sseg_label_proposal.shape
	vis_input = cv2.resize(vis_input, (W, H))
	#'''
	#assert 1==2

	# visualize output feature
	if style == 'duq':
		logits, z, _ = classifier(patch_feature)
	elif style == 'regular':
		logits, z = classifier(patch_feature)
	print('z.shape = {}'.format(z.shape))
	z = z.cpu().numpy().reshape(28, 28, -1)# 28 x 28 x 256

	background_feature = z[batch_sseg_label==0, :] # N x 256
	print('background_feature.shape = {}'.format(background_feature.shape))
	output_patch_feature = z
	output_patch_feature = normalized(output_patch_feature, axis=2)
	background_feature = normalized(background_feature, axis=1)
	H, W, _ = output_patch_feature.shape
	vis_output = np.zeros((H, W))
	for i in range(H):
		for j in range(W):
			ft = output_patch_feature[i, j]
			#dist = np.sum(np.square(ft - ft), axis=1).mean() # N
			dist = np.linalg.norm(ft - background_feature[0])
			vis_output[i, j] = dist

	H, W = sseg_label_proposal.shape
	vis_output = cv2.resize(vis_output, (W, H))


	'''
	H, W = sseg_label_proposal.shape

	logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
	logits = logits.cpu().numpy()[0, [0, 1, 3, 4]] # 4 x H x W
	uncertainty = 1.0 - np.amax(logits, axis=0)
	sseg_pred = np.argmax(logits, axis=0)
	sseg_pred[sseg_pred == 0] = 0 # road
	sseg_pred[sseg_pred == 1] = 1 # building
	sseg_pred[sseg_pred == 2] = 3 # vegetation
	sseg_pred[sseg_pred == 3] = 4 # sky
	sseg_pred[uncertainty > 0.3] = 6
	'''

	if save_option == 'both' or save_option == 'image':
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
		ax[0].imshow(vis_input)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("input {} ft".format(rep_style))
		ax[1].imshow(vis_output)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("output {} ft".format(rep_style))
		
		fig.tight_layout()
		plt.show()

		#fig.tight_layout()
		#fig.savefig('{}/img_{}_proposal_{}_eval_{}.jpg'.format(saved_folder, i, j, flag_eval))
		#plt.close()




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

style = 'duq'
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

classifier = DuqHead(num_classes, input_dim).to(device)
#classifier = SSegHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/{}_classifier_0.0.pth'.format(trained_model_dir, style)))
flag_eval = True
if flag_eval:
	classifier.eval()
#W = classifier.W.detach().cpu().numpy()
#np.save('./duq_{}_W.npy'.format(rep_style), W)
#assert 1==2

i = 7
j = 1
with torch.no_grad():

	patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)

	'''
	patch_feature = patch_feature.cpu().numpy()[0] # 256 x 14 x 14
	batch_sseg_label = batch_sseg_label.cpu().numpy()[0]
	batch_sseg_label = cv2.resize(batch_sseg_label, (14, 14), interpolation=cv2.INTER_NEAREST)

	background_feature = patch_feature[:, batch_sseg_label==1].transpose() # N x 256
	patch_feature = patch_feature.transpose((1, 2, 0)) # 14 x 14 x 256
	H, W, _ = patch_feature.shape
	vis = np.zeros((H, W))
	for i in range(H):
		for j in range(W):
			ft = patch_feature[i, j]
			dist = np.sum(np.square(ft - background_feature), axis=1).mean() # N
			vis[i, j] = dist

	H, W = sseg_label_proposal.shape
	vis = cv2.resize(vis, (W, H))
	assert 1==2
	'''

	'''
	patch_feature = patch_feature.to(device)
	logits, z = classifier(patch_feature)

	z = z.cpu().numpy().reshape(28, 28, -1)# 28 x 28 x 256
	batch_sseg_label = batch_sseg_label.cpu().numpy()[0] # 28 x 28
	background_feature = z[batch_sseg_label==0, :] # N x 256
	patch_feature = z
	H, W, _ = patch_feature.shape
	vis = np.zeros((H, W))
	for i in range(H):
		for j in range(W):
			ft = patch_feature[i, j]
			dist = np.sum(np.square(ft - background_feature), axis=1).mean() # N
			vis[i, j] = dist

	H, W = sseg_label_proposal.shape
	vis = cv2.resize(vis, (W, H))
	plt.imshow(vis)
	plt.show()
	assert 1==2
	'''

	'''
	z = z.permute(0, 2, 3, 1).cpu().numpy()[0]
	batch_sseg_label = batch_sseg_label.cpu().numpy()[0]
	logit = logits.permute(0, 2, 3, 1).cpu().numpy()[0]

	result = {}
	result['ft'] = z
	result['logit'] = logit
	result['label'] = batch_sseg_label
	result['predictor_weight'] = classifier.state_dict()['predictor.weight'].cpu().numpy()
	result['predictor_bias'] = classifier.state_dict()['predictor.bias'].cpu().numpy()
	np.save('{}/regular_img_{}_proposal_{}_eval_{}.npy'.format(saved_folder, i, j, flag_eval), result)
	assert 1==2
	'''

	patch_feature = patch_feature.to(device)
	logits, z, dis_to_centroids = classifier(patch_feature)

	dis_to_centroids = dis_to_centroids.cpu().numpy().reshape((28, 28, 512, 8))
	picked_centroid_idxs_for_bg = np.array([7, 30, 167, 265, 288, 323, 338, 348, 359, 379, 427, 432, 459, 466])
	dis_to_road_centroids = dis_to_centroids[:, :, picked_centroid_idxs_for_bg, 0]
	logits = np.exp(np.mean(np.abs(dis_to_road_centroids), axis=2)/(2*0.1**2)*(-1))
	H, W = sseg_label_proposal.shape
	
	logits = cv2.resize(logits, (W, H)) 
	
	uncertainty = 1.0 - logits
	sseg_pred = np.zeros(logits.shape)
	sseg_pred[sseg_pred == 0] = 0 # road
	sseg_pred[sseg_pred == 1] = 1 # building
	sseg_pred[sseg_pred == 2] = 3 # vegetation
	sseg_pred[sseg_pred == 3] = 4 # sky
	sseg_pred[uncertainty > 0.3] = 6


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
	

	'''
	uncertainty[sseg_pred == 0] = 0 # road
	uncertainty[sseg_pred == 1] = 0 # building
	uncertainty[sseg_pred == 3] = 0 # vegetation
	uncertainty[sseg_pred == 4] = 0 # sky
	'''

	'''
	sseg_pred = torch.argmax(logits, dim=1)

	logits = logits.cpu().numpy()[0]
	sseg_pred = sseg_pred.cpu().numpy()[0]

	#uncertainty = 1.0 - np.amax(logits, axis=0)
	uncertainty = entropy(softmax(logits, axis=0), axis=0, base=2)
	
	if ignore_background_uncertainty:
		# ignore uncertainty on the background pixels
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
		plt.show()

		#fig.tight_layout()
		#fig.savefig('{}/img_{}_proposal_{}_eval_{}.jpg'.format(saved_folder, i, j, flag_eval))
		#plt.close()

	'''
	if save_option == 'both' or save_option == 'npy':
		result = {}
		result['sseg'] = sseg_pred
		result['uncertainty'] = uncertainty
		np.save('{}/img_{}_proposal_{}.npy'.format(saved_folder, i, j), result)
	'''
		#assert 1==2


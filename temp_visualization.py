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

base_folder = 'visualization/vis_road_centroids_2'
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
	#assert 1==2

	patch_feature = patch_feature.to(device)
	logits, z, dis_to_centroids = classifier(patch_feature)

	batch_sseg_label = batch_sseg_label.cpu().numpy()[0] # 28 x 28
	dis_to_centroids = dis_to_centroids.cpu().numpy()
	logits = logits.cpu().numpy()[0, [0, 1, 3, 4]] # 4 x H x W
	sseg_pred = np.argmax(logits, axis=0)[batch_sseg_label==1]
	
	dis_to_background_centroids = dis_to_centroids[batch_sseg_label.flatten()==1, :][:, :, [0, 1, 3, 4]] # 141 x 512 x 4
	# since most (138/141) of sseg_pred value is 0, the ground 
	temp = dis_to_background_centroids[:, :, 0]
	temp = np.abs(temp)
	top_centroids_idxs = np.argsort(temp, axis=1)[:, :10]
	unique, counts = np.unique(top_centroids_idxs, return_counts=True)
	picked_centroid_idxs_for_obj = unique[counts > 50] # array([ 39,  70, 115, 328, 365, 477])

	dis_to_background_centroids = dis_to_centroids[batch_sseg_label.flatten()==0, :][:, :, [0, 1, 3, 4]] # 141 x 512 x 4
	# since most (138/141) of sseg_pred value is 0, the ground 
	temp = dis_to_background_centroids[:, :, 0]
	temp = np.abs(temp)
	top_centroids_idxs = np.argsort(temp, axis=1)[:, :10]
	unique, counts = np.unique(top_centroids_idxs, return_counts=True)
	#picked_centroid_idxs_for_bg = unique[counts > 50]
	picked_centroid_idxs_for_bg = np.array([7, 30, 167, 265, 288, 323, 338, 348, 359, 379, 427, 432, 459, 466])

# search on cityscapes dataset
dataset_folder = '/home/yimeng/ARGO_datasets/Cityscapes'
ds_cityscapes = CityscapesProposalsDataset(dataset_folder, 'train', rep_style=rep_style)

with torch.no_grad():
	for i in range(len(ds_cityscapes)):
		for j in range(0, 500, 10):
			print('i = {}, j = {}'.format(i, j))
			patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal = ds_cityscapes.get_proposal(i, j)
			patch_feature = patch_feature.to(device)
			logits, z, dis_to_centroids = classifier(patch_feature)

			batch_sseg_label = batch_sseg_label.cpu().numpy()[0] # 28 x 28
			dis_to_centroids = dis_to_centroids.cpu().numpy().reshape((28, 28, 512, 8))

			dis_to_road_centroids = dis_to_centroids[:, :, :, 0] # 28 x 28 x 512
			logits = logits.cpu().numpy()[0] # 4 x H x W
			sseg_pred = np.argmax(logits, axis=0) # 28 x 28
		
			#.div(2 * self.sigma **2).mul(-1).exp()
			dis_to_picked_centroids_for_obj = np.exp(np.mean(np.abs(dis_to_road_centroids[:, :, picked_centroid_idxs_for_obj]), axis=2)/(2*0.1**2)*(-1))
			dis_to_picked_centroids_for_bg = np.exp(np.mean(np.abs(dis_to_road_centroids[:, :, picked_centroid_idxs_for_bg]), axis=2)/(2*0.1**2)*(-1))
			mask_road = (batch_sseg_label == 0)

			if np.sum(mask_road) > 0:
				dis_to_picked_centroids_for_obj[mask_road<1] = 0
				dis_to_picked_centroids_for_bg[mask_road<1] = 0

				color_batch_sseg_label = apply_color_map(batch_sseg_label)
				color_sseg_pred = apply_color_map(sseg_pred)

				fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
				ax[0][0].imshow(img_proposal)
				ax[0][0].get_xaxis().set_visible(False)
				ax[0][0].get_yaxis().set_visible(False)
				ax[0][0].set_title("rgb proposal")
				ax[0][1].imshow(color_batch_sseg_label)
				ax[0][1].get_xaxis().set_visible(False)
				ax[0][1].get_yaxis().set_visible(False)
				ax[0][1].set_title("sseg_label_proposal")
				ax[0][2].imshow(color_sseg_pred)
				ax[0][2].get_xaxis().set_visible(False)
				ax[0][2].get_yaxis().set_visible(False)
				ax[0][2].set_title("sseg pred")
				ax[1][0].imshow(dis_to_picked_centroids_for_obj, vmin=0.0, vmax=1.0)
				ax[1][0].get_xaxis().set_visible(False)
				ax[1][0].get_yaxis().set_visible(False)
				ax[1][0].set_title("picked_centroids_for_obj")
				ax[1][1].imshow(dis_to_picked_centroids_for_bg, vmin=0.0, vmax=1.0)
				ax[1][1].get_xaxis().set_visible(False)
				ax[1][1].get_yaxis().set_visible(False)
				ax[1][1].set_title("picked_centroids_for_bg")
				#plt.show()
				fig.tight_layout()
				fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
				plt.close()
				#assert 1==2
	'''
	# visualize the difference between obj features and background features
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
		plt.show()

		#fig.tight_layout()
		#fig.savefig('{}/img_{}_proposal_{}_eval_{}.jpg'.format(saved_folder, i, j, flag_eval))
		#plt.close()
	'''

	


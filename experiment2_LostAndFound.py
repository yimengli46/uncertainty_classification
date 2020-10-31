import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import matplotlib as mpl
import cv2
from utils import find_IoU, find_IoU_and_interArea
import json
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
import statistics
from sklearn import manifold, datasets
import random
from statistics import mean

# cityscapes imports
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import labels, name2label

thing_list = ['pole', 'polegroup', 'traffic light', 'traffic sign', 'person', 'rider',
	'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']

thing_map = {}
for idx, name in enumerate(thing_list):
	thing_map[name] = idx

interesting_labels = [17, 18, 19, 20]

selected_features = None
selected_classes = None
count_outlier = 0
count_background = 0


#'''
# process LostAndFound Dataset and find outlier points
base_folder = '/home/yimeng/work/detectron2/my_projects/Bayesian_MaskRCNN/new_generated_bbox_features_LostAndFound'
saved_folder = 'experiment_2_LostAndFound'

Fishyscapes_folder = '/home/yimeng/Datasets/Lost_and_Found'
json_file = '{}/{}_data_annotation.json'.format(Fishyscapes_folder, 'Lost_and_Found')
with open(json_file) as f:
	imgs_anns = json.load(f)

# go through each image
for _, v in enumerate(imgs_anns.values()):
	img_id = int(v['file_name'][:-4])
	print('img_id = {}'.format(img_id))

	img = cv2.imread('{}/{}.png'.format(Fishyscapes_folder, img_id), 1)[:, :, ::-1]
		
	# find the gt bbox
	annos = v['regions']
	num_bbox = len(list(annos.keys())) 
	#print('num_bbox = {}'.format(num_bbox))
	gt_bbox = np.zeros((num_bbox, 4))
	# loop over all objects
	for j, anno in annos.items():
		px = anno['all_points_x']
		py = anno['all_points_y']

		gt_bbox[int(j)] = np.min(px), np.min(py), np.max(px), np.max(py)

	#============================================================================================================

	num_pass = 1
	IoU_thresh = 0.1
	thresh_list = [x/10.0 for x in range(1, 10)]

	pred_all_features = None 
	pred_all_classes = None
	pred_all_boxes = None
	num_all_outliers = 0
	
	# go through each pass
	for j in range(num_pass):
		#print('img_id = {}, pass = {}'.format(img_id, j))
		
		npy_file = '{}/{}_forpass_{}.npy'.format(base_folder, img_id, j)
		current_npy = np.load(npy_file, allow_pickle=True).item()
		
		pred_boxes = current_npy['boxes']
		pred_features = current_npy['features']
		pred_classes = current_npy['classes'].reshape(1000, 14)

		if j == 0:
			pred_all_features = pred_features
			pred_all_boxes = pred_boxes
			pred_all_classes = pred_classes
		else:
			pred_all_boxes = np.concatenate([pred_all_boxes, pred_boxes], axis=0)
			pred_all_classes = np.concatenate([pred_all_classes, pred_classes], axis=0)
			pred_all_features = np.concatenate([pred_all_features, pred_features], axis=0)

	for idx in range(pred_all_classes.shape[0]):
		current_bboxes = pred_all_boxes[idx*14:(idx+1)*14]
		current_bbox = mean(current_bboxes[:, 0]), mean(current_bboxes[:, 1]), mean(current_bboxes[:, 2]), mean(current_bboxes[:, 3])

		if idx == 0:
			temp_all_boxes = current_bbox
		else:
			temp_all_boxes = np.concatenate([temp_all_boxes, current_bbox], axis=0)

	pred_all_boxes = temp_all_boxes.reshape(-1, 4)
	
	#=============================================================================================================
	# check how many outlier bbox having entropy above the threshold
	outlier_entropy_list = []
	outlier_diff_thresh_list = [0 for _ in range(9)]

	for idx in range(pred_all_classes.shape[0]):

		max_IoU = 0.0
		#for box_idx in range(14):
		current_bbox = pred_all_boxes[idx]

		# go through the gt boxes
		for i_gt_bbox in range(num_bbox):
			IoU = find_IoU(current_bbox, gt_bbox[i_gt_bbox])
			if IoU > max_IoU:
				max_IoU = IoU
			
		if max_IoU >= IoU_thresh:
			outlier_entropy_list.append(idx)

		for i_thresh, current_thresh in enumerate(thresh_list):
			if max_IoU >= current_thresh:
				outlier_diff_thresh_list[i_thresh] += 1

	count_outlier += len(outlier_entropy_list)

	#=============================================================================================================
	#'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
	ax.imshow(img)

	for idx in range(pred_all_classes.shape[0]):
		current_bbox = pred_all_boxes[idx]
		x1, y1, x2, y2 = current_bbox
		rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
		ax.add_patch(rect)

	if len(outlier_entropy_list) > 0:
		num_detections = len(outlier_entropy_list)
		outlier_boxes = pred_all_boxes[outlier_entropy_list]
		for idx in range(num_detections):
			x1, y1, x2, y2 = outlier_boxes[idx]
			rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='b', facecolor='none')
			ax.add_patch(rect)

	for idx in range(gt_bbox.shape[0]):
		x1, y1, x2, y2 = gt_bbox[idx]
		rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='k', facecolor='none')
		ax.add_patch(rect)

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	proposal_patch = patches.Patch(color='r', label='Proposals')
	outlier_patch = patches.Patch(color='b', label='Outlier')
	gt_patch = patches.Patch(color='k', label='gt-Outlier')
	ax.legend(handles=[proposal_patch, outlier_patch, gt_patch], loc=1)

	title_str = ''
	for t in range(len(outlier_diff_thresh_list)):
		title_str += '{},'.format(str(outlier_diff_thresh_list[t]))

	plt.title(title_str)

	fig.tight_layout()
	fig.savefig('{}/{}_outlier_boxes.jpg'.format(saved_folder, img_id))
	plt.close()
	#'''


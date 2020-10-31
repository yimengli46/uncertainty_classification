import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import matplotlib as mpl
import cv2
import json
from scipy.stats import entropy
from scipy.special import softmax
from utils import compute_iou, find_IoU, find_IoU_batch2batch
from math import ceil

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

detectron2_base_folder = '/home/yimeng/work/detectron2/my_projects/Bayesian_RetinaNet'
dataset_base_folder = '/home/yimeng/Datasets/{}'.format('Lost_and_Found')#('Fishyscapes_Static')

#saved_folder = 'RetinaNet_Fishyscapes_Static_Results'
saved_folder = 'RetinaNet_LostAndFound_Static_Results_new'

thing_list = ['person', 'rider',
	'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']

thing_map = {}
for idx, name in enumerate(thing_list):
	thing_map[name] = idx

#npy_folder = '{}/{}'.format(detectron2_base_folder,'retinaNet_Fishyscapes_results')
npy_folder = '{}/{}'.format(detectron2_base_folder,'retinaNet_LostAndFound_results')

num_forward = 10
top_uncertain_boxes = 200
thresh_IoU_objects = 0.1
thresh_IoU_background = 0.5

uncertainty_type = 'variance' #'variance' # 'entropy'

big_outlier_list = [2, 3, 4, 7, 10, 11, 15, 16, 25, 27, 31, 33, 34, 35, 38, 40, 45, 46, 48, 50, 51, 54, 57, 60, 61, 63, 65, 
	68, 71, 72, 74, 76, 83, 84, 85, 86, 91, 93, 95]

for i in big_outlier_list:
	im = cv2.imread('{}/{}.png'.format(dataset_base_folder, i, 1))[:,:,::-1]
	print('im = {}'.format(i))

	# read first npy file to get to know box numbers
	npy_file = '{}/{}_{}.npy'.format(npy_folder, i, 0)
	current_npy = np.load(npy_file, allow_pickle=True).item()

	pred_boxes = current_npy['boxes']
	pred_scores = current_npy['scores']
	pred_classes = current_npy['classes']

	r, num_classes = pred_scores.shape
	assert num_classes == len(thing_list)

	pred_all_scores = np.zeros((num_forward, r, num_classes))
	pred_all_classes = np.zeros((num_forward, r, num_classes))
	pred_all_boxes = np.zeros((num_forward, r, 4))

	pred_all_scores[0] = pred_scores
	pred_all_classes[0] = pred_classes
	pred_all_boxes[0] = pred_boxes

	#print('pred_scores.shape = {}'.format(pred_scores.shape))

	for j in range(1, num_forward):
		npy_file = '{}/{}_{}.npy'.format(npy_folder, i, j)
		current_npy = np.load(npy_file, allow_pickle=True).item()

		pred_boxes = current_npy['boxes']
		pred_scores = current_npy['scores']
		pred_classes = current_npy['classes']

		pred_all_scores[j] = pred_scores
		pred_all_classes[j] = pred_classes
		pred_all_boxes[j] = pred_boxes

		#print('pred_scores.shape = {}'.format(pred_scores.shape))

	mean_boxes = np.mean(pred_all_boxes, axis=0)

	# remove unwanted boxes
	widths = mean_boxes[:, 2] - mean_boxes[:, 0]
	heights = mean_boxes[:, 3] - mean_boxes[:, 1]
	keep = (widths > 0) & (heights > 0)

	mean_boxes = mean_boxes[keep]
	pred_all_scores = pred_all_scores[:, keep, :]
	mean_scores = np.mean(pred_all_scores, axis=0)
	#---------------------------------------------------------------------------------------------------------------------------
	# remove normal objects
	mean_scores_among_objects = np.mean(pred_all_scores, axis=0)
	max_mean_scores_among_objects = np.max(mean_scores_among_objects, axis=1)
	normal_objects_idx = max_mean_scores_among_objects >= 0.5
	normal_objects_boxes = mean_boxes[normal_objects_idx]

	ious = find_IoU_batch2batch(mean_boxes, normal_objects_boxes)
	max_ious = np.max(ious, axis=1)
	normal_objects_list = (max_mean_scores_among_objects >= 0.5) | (max_ious > thresh_IoU_objects)
	non_normal_objects_list = (~normal_objects_list)

	'''
	non_normal_objects_list = []
	for j in range(mean_boxes.shape[0]):
		print('j = {}'.format(j))
		if max_mean_scores_among_objects[j] < 0.5:
			left_box = mean_boxes[j]
			ious = batch_find_IoU(left_box, normal_objects_boxes)
			max_iou = np.max(ious)
			if max_iou > thresh_IoU_objects:
				non_normal_objects_list.append(False)
			else:
				non_normal_objects_list.append(True)
		else:
			non_normal_objects_list.append(False)

	assert len(non_normal_objects_list) == mean_boxes.shape[0]
	non_normal_objects_list = np.array(non_normal_objects_list)
	'''

	mean_scores = mean_scores[non_normal_objects_list]
	mean_boxes = mean_boxes[non_normal_objects_list]
	pred_all_scores = pred_all_scores[:, non_normal_objects_list, :]
	print('mean_boxes.shape = {}'.format(mean_boxes.shape))
	#assert 1==2

	#---------------------------------------------------------------------------------------------------------------------------
	# remove background anchor boxes
	max_mean_scores = np.max(mean_scores, axis=1)
	background_idx = max_mean_scores < 0.05
	background_boxes = mean_boxes[background_idx]
	print('background_boxes.shape={}'.format(background_boxes.shape))

	non_background_list = (~background_idx)
	mean_scores = mean_scores[non_background_list]
	pred_all_scores = pred_all_scores[:, non_background_list, :]
	mean_boxes = mean_boxes[non_background_list]
	print('mean_boxes.shape = {}'.format(mean_boxes.shape))

	#--------------------------------------------------------------------------------------------------------------------------
	# Shouldn't remove anchor boxes overlaps with certain background anchor boxes
	# the result will be very bad
	'''
	num_batches = ceil(mean_boxes.shape[0]/1000)
	print('num_batches = {}'.format(num_batches))
	for j in range(num_batches-1):
		print('j = {}'.format(j))
		current_ious = find_IoU_batch2batch(mean_boxes[j*1000:(j+1)*1000], background_boxes)
		if j == 0:
			ious = current_ious
		else:
			ious = np.concatenate((ious, current_ious), axis=0)
	current_ious = find_IoU_batch2batch(mean_boxes[(num_batches-1)*1000:], background_boxes)
	ious = np.concatenate((ious, current_ious), axis=0)
	#ious = find_IoU_batch2batch(mean_boxes, background_boxes)
	assert ious.shape[0] == mean_boxes.shape[0]
	
	max_ious = np.max(ious, axis=1)
	background_list = max_ious > thresh_IoU_background
	non_background_list = (~background_list)
	'''

	'''
	non_background_list = []
	for j in range(mean_boxes.shape[0]):
		print('j = {}'.format(j))
		if max_mean_scores[j] < 0.05:
			left_box = mean_boxes[j]
			ious = batch_find_IoU(left_box, background_boxes)
			max_iou = np.max(ious)
			if max_iou > thresh_IoU_background:
				non_background_list.append(False)
			else:
				non_background_list.append(True)
		else:
			non_background_list.append(False)

	assert len(non_background_list) == mean_boxes.shape[0]
	non_background_list = np.array(non_background_list)
	'''

	'''
	mean_scores = mean_scores[non_background_list]
	pred_all_scores = pred_all_scores[:, non_background_list, :]
	mean_boxes = mean_boxes[non_background_list]
	'''

	#--------------------------------------------------------------------------------------------------------------------------
	# entropy or variance?
	if uncertainty_type == 'entropy':
		uncertainty_scores = entropy(softmax(mean_scores, axis=1), axis=1,base=2)
	else:
		uncertainty_scores = np.max(np.std(pred_all_scores, axis=0), axis=1)

	idx_large_uncertainty = np.argsort(uncertainty_scores)[::-1]

	if top_uncertain_boxes > uncertainty_scores.shape[0]:
		temp = uncertainty_scores.shape[0]
	else:
		temp = top_uncertain_boxes
	picked_boxes = mean_boxes[idx_large_uncertainty[:top_uncertain_boxes]]

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
	ax.imshow(im)

	for idx in range(picked_boxes.shape[0]):
		current_bbox = picked_boxes[idx]
		x1, y1, x2, y2 = current_bbox
		rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
		ax.add_patch(rect)

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	fig.tight_layout()
	fig.savefig('{}/{}_uncertain_boxes_{}.jpg'.format(saved_folder, i, uncertainty_type))
	plt.close()

	assert 1==2
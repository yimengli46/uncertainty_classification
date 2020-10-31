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

# process Cityscapes Dataset and find ordinary object and random background datapoints
base_folder = '/home/yimeng/work/detectron2/my_projects/Bayesian_MaskRCNN/new_generated_bbox_features_Cityscapes'
saved_folder = 'object_embedding_tsne_Cityscapes_and_LostAndFound_one_point'

Fishyscapes_folder = '/home/yimeng/Datasets/Cityscapes'
np_file = np.load('{}/{}_img_list.npy'.format(Fishyscapes_folder, 'val'), allow_pickle=True)

#'''
# go through each image
#for img_id, v in enumerate(np_file):
for img_id in range(100):
	v = np_file[img_id]

	print('img_id = {}'.format(img_id))

	img = cv2.imread('{}/{}'.format(Fishyscapes_folder, v['rgb_path']), 1)[:, :, ::-1]

	# load the instance segmentation
	inJson = '{}/{}'.format(Fishyscapes_folder, v['polygon_path'])
	annotation = Annotation()
	annotation.fromJsonFile(inJson)

	objs = []
	# loop over all objects
	for obj in annotation.objects:
		label = obj.label
		#print('label = {}'.format(label))
		polygon = obj.polygon

		if obj.deleted:
			continue

		# if the label is not known, but ends with a 'group' (e.g. cargroup)
		# try to remove the s and see if that works
		# also we know that this polygon describes a group
		isGroup = False
		if ( not label in name2label ) and label.endswith('group'):
			label = label[:-len('group')]
			isGroup = True

		labelTuple = name2label[label]
		id = labelTuple.id

		# check if we should include this label or not
		if labelTuple.hasInstances or id in interesting_labels:
			px = [polygon[i].x for i in range(len(polygon))]
			py = [polygon[i].y for i in range(len(polygon))]

			obj_dict = {
				'bbox': [np.min(px), np.min(py), np.max(px), np.max(py)],
				'category_id': thing_map[label],
			}
			#print('obj_dict: {}'.format(obj_dict))

			objs.append(obj_dict)

	# find the gt bbox
	num_bbox = len(objs) 
	#print('num_bbox = {}'.format(num_bbox))
	gt_bbox = np.zeros((num_bbox, 4))
	gt_labels = []
	# loop over all objects
	for j in range(len(objs)):
		bbox = objs[j]['bbox']
		gt_bbox[j] = bbox[0], bbox[1], bbox[2], bbox[3]
		gt_labels.append(objs[j]['category_id'])

	num_pass = 1
	IoU_thresh = 0.8 # select only the very good ones

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

	# check how many outlier bbox having entropy above the threshold
	
	count_above_thresh_outlier = 0
	object_list = []
	class_list = []
	background_list = []
	for idx in range(pred_all_classes.shape[0]):
		current_bbox = pred_all_boxes[idx]

		max_IoU = 0.0
		max_class = -1
		min_interArea = 10000
		#for box_idx in range(14):
		#current_bbox = current_bboxes[box_idx]

		# go through the gt boxes
		for i_gt_bbox in range(num_bbox):
			IoU, interArea = find_IoU_and_interArea(current_bbox, gt_bbox[i_gt_bbox])
			if IoU > max_IoU:
				max_IoU = IoU
				max_class = gt_labels[i_gt_bbox]
				min_interArea = interArea

		if max_IoU >= IoU_thresh:
			#if current_entropy > thresh_entropy:
			object_list.append(idx)
			class_list.append(max_class)
		#elif min_interArea < 10 and max_IoU < 0.0001:
		elif max_IoU < 0.01:
			background_list.append(idx)

	print('len(object_list) = {}, len(background_list) = {}'.format(len(object_list), len(background_list)))

	##=========================================================================================================

	object_list = np.array(object_list)
	class_list = np.array(class_list)
	if len(object_list) > 100:
		chosen_idx = random.sample([x for x in range(len(object_list))], 100)
	else:
		chosen_idx = [x for x in range(len(object_list))]
	object_list = object_list[chosen_idx]
	object_classes = class_list[chosen_idx]
	if len(object_list) > 0:
		# filter out the object boxes if they have large IoU
		select_one_list = []
		temp_object_classes = []
		for m in range(len(object_list)):
			index = object_list[m]
			left_box = pred_all_boxes[index]
			# compare with boxes in select_one_list
			flag_one = True
			for right_index in select_one_list:
				right_box = pred_all_boxes[right_index]
				iou = find_IoU(left_box, right_box)
				if iou >= 0.5:
					flag_one = False
					break

			if flag_one:
				select_one_list.append(index)
				temp_object_classes.append(object_classes[m])

		# done with filtering
		object_features = pred_all_features[select_one_list]
		object_classes = temp_object_classes
		object_boxes = pred_all_boxes[select_one_list]

	##==========================================================================================================

	background_list = np.array(background_list)
	if len(background_list) > 50:
		chosen_idx = random.sample([x for x in range(len(background_list))], 50)
	else:
		chosen_idx = [x for x in range(len(background_list))]
	chosen_idx = [x for x in range(len(background_list))]
	
	flag_background = False
	if len(background_list) > 0:
		background_features = pred_all_features[background_list[chosen_idx]]
		background_classes = np.ones((len(chosen_idx)), np.int16) * (len(thing_list) + 1)
		background_boxes = pred_all_boxes[background_list[chosen_idx]]
		flag_background = True

	##==========================================================================================================
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
	ax.imshow(img)

	for idx in range(pred_all_classes.shape[0]):
		current_bbox = pred_all_boxes[idx]
		x1, y1, x2, y2 = current_bbox
		rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
		ax.add_patch(rect)

	if flag_background:
		num_detections = background_boxes.shape[0]
		for idx in range(num_detections):
			x1, y1, x2, y2 = background_boxes[idx]
			rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='b', facecolor='none')
			ax.add_patch(rect)

	for idx in range(object_boxes.shape[0]):
		x1, y1, x2, y2 = object_boxes[idx]
		rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='k', facecolor='none')
		ax.add_patch(rect)

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	proposal_patch = patches.Patch(color='r', label='Proposals')
	background_patch = patches.Patch(color='b', label='Background')
	object_patch = patches.Patch(color='k', label='Objects')
	ax.legend(handles=[proposal_patch, background_patch, object_patch], loc=1)

	fig.tight_layout()
	fig.savefig('{}/{}_proposal_boxes_background_boxes_object_boxes.jpg'.format(saved_folder, img_id))
	plt.close()

	if img_id == 0:
		selected_features = object_features
		selected_classes = object_classes
	else:
		selected_features = np.concatenate([selected_features, object_features], axis=0)
		selected_classes = np.concatenate([selected_classes, object_classes], axis=0)

	if flag_background:
		selected_features = np.concatenate([selected_features, background_features], axis=0)
		selected_classes = np.concatenate([selected_classes, background_classes], axis=0)

	print('selected_features.shape = {}, selected_classes.shape = {}'.format(selected_features.shape, selected_classes.shape))
#'''



#'''
# process LostAndFound Dataset and find outlier points
base_folder = '/home/yimeng/work/detectron2/my_projects/Bayesian_MaskRCNN/new_generated_bbox_features_LostAndFound'

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

	num_pass = 10
	IoU_thresh = 0.3

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

	if len(outlier_entropy_list) > 0:
		# filter out the object boxes if they have large IoU
		select_one_list = []
		
		for m in range(len(outlier_entropy_list)):
			index = outlier_entropy_list[m]
			left_box = pred_all_boxes[index]
			# compare with boxes in select_one_list
			flag_one = True
			for right_index in select_one_list:
				right_box = pred_all_boxes[right_index]
				iou = find_IoU(left_box, right_box)
				if iou >= 0.1:
					flag_one = False
					break
			if flag_one:
				select_one_list.append(index)
	
		outlier_entropy_list = select_one_list


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

	fig.tight_layout()
	fig.savefig('{}/{}_outlier_boxes.jpg'.format(saved_folder, img_id))
	plt.close()
	#'''

	#=============================================================================================================
	if len(outlier_entropy_list) > 0:
		outlier_features = pred_all_features[outlier_entropy_list]
		outlier_classes = np.ones((len(outlier_entropy_list)), np.int16) * len(thing_list)
		#outlier_features = np.expand_dims(outlier_features[0], axis=0)
		#outlier_classes = np.expand_dims(outlier_classes[0], axis=0)

	if len(outlier_entropy_list) > 0:
		selected_features = np.concatenate([selected_features, outlier_features], axis=0)
		selected_classes = np.concatenate([selected_classes, outlier_classes], axis=0)


	print('selected_features.shape = {}, selected_classes.shape = {}'.format(selected_features.shape, selected_classes.shape))


	#if img_id > 0:
	#	break
#'''

# visualize the bbox features through t-sne
X = selected_features
y = selected_classes.astype(np.int16)
N = len(thing_list) + 2

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, verbose=1)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig = plt.figure(figsize=(10, 10))
scat = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap=cmap, norm=norm)
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
plt.xticks([])
plt.yticks([])
plt.title('number of points = {}, number of outlier = {}'.format(X.shape[0], count_outlier))
#plt.show()
fig.tight_layout()
fig.savefig('{}/all_images_{}_passes.jpg'.format(saved_folder, num_pass))
plt.close()



#assert 1==2


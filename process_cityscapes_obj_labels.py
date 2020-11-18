from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels	 import labels, name2label
import numpy as np
import json

dataset_dir = '/home/reza/ARGO_datasets/Cityscapes'
mode = 'train'

data_np_file = np.load('{}/{}_img_list.npy'.format(dataset_dir, mode), allow_pickle=True).tolist()
img_obj_lst = []

for idx in range(len(data_np_file)):
	v = data_np_file[idx]
	# load the instance segmentation
	inJson = '{}/{}'.format(dataset_dir, v['polygon_path'])
	annotation = Annotation()
	annotation.fromJsonFile(inJson)

	targets = []
	count_obj = 0
	# loop over all objects
	for obj in annotation.objects:
		label = obj.label
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
		if labelTuple.hasInstances:
			px = [polygon[i].x for i in range(len(polygon))]
			py = [polygon[i].y for i in range(len(polygon))]

			x1 = max(np.min(px), 0)
			y1 = max(np.min(py), 0)
			x2 = min(np.max(px)+1, 2048-1)
			y2 = min(np.max(py)+1, 1024-1)

			#print('label = {}'.format(label))

			if label == 'trailer' or label == 'caravan':
				label = 'car'

			
			print('x1 = {}, y1 = {}, x2 = {}, y2 = {}, class_label = {}'.format(x1, y1, x2, y2, label))
			target = {}
			target['id'] = count_obj
			target['x1'] = int(x1)
			target['y1'] = int(y1)
			target['x2'] = int(x2)
			target['y2'] = int(y2)
			target['class'] = label

			targets.append(target)
			count_obj += 1

	img_obj = {}
	img_obj['id'] = idx
	img_obj['objects'] = targets
	img_obj_lst.append(img_obj)

with open('cityscapes_{}_img_obj.json'.format(mode), 'w') as json_file:
	json.dump(img_obj_lst, json_file)

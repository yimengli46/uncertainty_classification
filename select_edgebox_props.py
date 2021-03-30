import numpy as np
import scipy.io
import json
from utils import find_IoU_one2batch

dataset = 'Lost_and_Found'
folder_dir = '/home/yimeng/ARGO_datasets/{}'.format(dataset)
thresh_iou = 0.5

mat_prop_file_dir = '{}/edgebox_props.mat'.format(folder_dir)
mat_prop_file = scipy.io.loadmat(mat_prop_file_dir)['all_boxes'][0]

if dataset == 'Lost_and_Found':
	num_imgs = 100
elif dataset == 'RoadAnomaly':
	num_imgs = 60

data_json_file = json.load(open('{}/{}_data_annotation.json'.format(folder_dir, 'Lost_and_Found')))

results = {}
for i in range(num_imgs):
	print('i = {}'.format(i))

	#================================== load edgebox props ==========================================
	all_boxes = list(mat_prop_file[i])
	subtractor = np.array((1, 1, 0, 0, 0))[np.newaxis, :]
	all_boxes = [boxes - subtractor for boxes in all_boxes]

	num_props = len(all_boxes)
	edgebox_props = np.zeros((num_props, 4), dtype=int)
	for j in range(num_props):
		y1, x1, y2, x2, score = all_boxes[j][0]
		edgebox_props[j] = np.array((x1, y1, x2, y2))

	#================================= load ood props ===============================================
	v = data_json_file[str(i)]
	# load the instance segmentation
	regions = v['regions']
	num_proposals = len(regions)

	self_sampled_proposals = []
	if len(regions) > 0:
		#print('regions = {}'.format(regions))
		for idx in range(len(regions)):
			region = regions[str(idx)]
			x1 = int(min(region['all_points_x']))
			y1 = int(min(region['all_points_y']))
			x2 = int(max(region['all_points_x']))
			y2 = int(max(region['all_points_y']))
			#print('x1 = {}, y1 = {}, x2 = {}, y2 = {}'.format(x1, y1, x2, y2))
			self_sampled_proposals.append([x1, y1, x2, y2])
	self_sampled_proposals = np.array(self_sampled_proposals)
	#================================ select_props ==================================================
	result_props = []

	for j in range(len(self_sampled_proposals)):
		current_prop = self_sampled_proposals[j]
		batch_iou = find_IoU_one2batch(current_prop, edgebox_props)
		left_props = edgebox_props[batch_iou >= thresh_iou]
		for u in range(left_props.shape[0]):
			result_props.append(left_props[u].tolist())
		#assert 1==2
	results[i] = np.array(result_props)
	#assert 1==2
np.save('edgebox_ood_props.npy', results)
#assert 1==2
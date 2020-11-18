import numpy as np
import cv2
import json
import matplotlib.pyplot as plt 
import glob
from PIL import Image 
import os
from utils import find_IoU, encode_segmap, stuff_class_names

dataset = 'cityscapes'
mode = 'train'

# where original proposals are saved
proposal_folder = '/home/reza/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals/{}_{}'.format(dataset, mode)
saved_folder = 'processed_proposals/{}_{}'.format(dataset, mode)
dataset_dir = '/home/reza/ARGO_datasets/Cityscapes'
data_np_file = np.load('{}/{}_img_list.npy'.format(dataset_dir, mode), allow_pickle=True).tolist()
num_stuff = len(stuff_class_names)

count_mixed = 0

# read obj label
with open('{}/{}_img_obj.json'.format(dataset_dir, mode)) as f:
	img_obj_json = json.load(f)

for i in range(2): #len(data_np_file)):
	v = data_np_file[i]
	#assert 1==2
	im = np.array(Image.open('{}/{}'.format(dataset_dir, v['rgb_path'])).convert('RGB'))
	# read semantic segmentation label image
	sseg_label = np.array(Image.open('{}/{}'.format(dataset_dir, v['semSeg_path'])), dtype=np.uint8)
	sseg_label = encode_segmap(sseg_label)
	#plt.imshow(sseg_label)
	#plt.show()

	proposals = np.load('{}/{}_proposal.npy'.format(proposal_folder, i), allow_pickle=True)

	prop_lst = []

#================================= check if the proposal is an object or not ==========================
	obj_label = img_obj_json[i]['objects']

	N, _ = proposals.shape
	for j in range(N):
		print('process img {}, proposal {} ......'.format(i, j))
		prop_result = {}

		x1, y1, x2, y2 = proposals[j]
		prop_x1 = int(max(round(x1), 0))
		prop_y1 = int(max(round(y1), 0))
		prop_x2 = int(min(round(x2), 2048-1))
		prop_y2 = int(min(round(y2), 1024-1))

		img_patch = im[prop_y1:prop_y2, prop_x1:prop_x2]
		plt.imshow(img_patch)

		flag_obj = False

		for m in range(len(obj_label)):
			obj_x1 = obj_label[m]['x1']
			obj_y1 = obj_label[m]['y1']
			obj_x2 = obj_label[m]['x2']
			obj_y2 = obj_label[m]['y2']
			category = obj_label[m]['class']

			iou = find_IoU([prop_x1, prop_y1, prop_x2, prop_y2], [obj_x1, obj_y1, obj_x2, obj_y2])
			if iou >= 0.5:
				print('==> obj {}, iou = {:.3f}, class = {}, type = {}'.format(m, iou, category, 'obj'))
				prop_result['id'] = j
				prop_result['type'] = 'obj'
				prop_result['x1'] = prop_x1
				prop_result['y1'] = prop_y1
				prop_result['x2'] = prop_x2
				prop_result['y2'] = prop_y2
				prop_result['class'] = category
				prop_result['obj_x1'] = obj_x1
				prop_result['obj_y1'] = obj_y1
				prop_result['obj_x2'] = obj_x2
				prop_result['obj_y2'] = obj_y2
				prop_result['iou'] = iou
				flag_obj = True
				break
			elif iou >= 0.4:
				print('==> obj {}, iou = {:.3f}, class = {}, type = {}'.format(m, iou, category, 'ignored'))
				prop_result['id'] = j
				prop_result['type'] = 'ignored'
				prop_result['x1'] = prop_x1
				prop_result['y1'] = prop_y1
				prop_result['x2'] = prop_x2
				prop_result['y2'] = prop_y2
				prop_result['class'] = category
				prop_result['obj_x1'] = obj_x1
				prop_result['obj_y1'] = obj_y1
				prop_result['obj_x2'] = obj_x2
				prop_result['obj_y2'] = obj_y2
				prop_result['iou'] = iou
				flag_obj = True
				break

		if flag_obj:
			print('proposal {} is an object!'.format(j))
			prop_lst.append(prop_result)
			plt.show()
			continue

#================================= check if the proposal is stuff or not ===============================
		# compute iou of all the 8 classes
		stuff_iou = np.zeros(num_stuff)

		proposal_sseg = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]
		H, W = proposal_sseg.shape
		area = H*W - np.sum(proposal_sseg==255)
		for s in range(num_stuff):
			iou = np.sum(proposal_sseg==s)/area
			stuff_iou[s] = iou
		max_iou_id = np.argmax(stuff_iou)
		max_iou = stuff_iou[max_iou_id]
		print('stuff_iou = {}, max_iou = {:.3f}'.format(stuff_iou, max_iou))

		prop_result['type'] = 'sseg'
		prop_result['id'] = j
		prop_result['x1'] = prop_x1
		prop_result['y1'] = prop_y1
		prop_result['x2'] = prop_x2
		prop_result['y2'] = prop_y2
		prop_result['iou'] = float(iou)
		if max_iou >= 0.5:
			prop_result['class'] = stuff_class_names[max_iou_id]
		else:
			prop_result['class'] = 'mixed'
			count_mixed +=1 
			#plt.show()
		print('==> sseg class = {}'.format(prop_result['class']))
		prop_lst.append(prop_result)

		plt.show()

#=============================== save the image proposal result============================
	with open('{}/{}_processed_proposals.json'.format(saved_folder, i), 'w') as json_file:
		json.dump(prop_lst, json_file)

print('count_mixed = {}'.format(count_mixed))
#assert 1==2
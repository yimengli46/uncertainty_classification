import numpy as np 

sseg_color_list_19 = [(128, 64, 128), (244, 36, 232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), 
	(250,170, 30), (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), 
	(  0,  0,142), (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)]
sseg_color_list_8 = [(128, 64, 128), ( 70, 70, 70), (153, 153, 153), (107, 142, 35), (70, 130, 180), (220, 20, 60),
	(0, 0, 142), (0, 80, 100)]
sseg_color_list_ade20k = [(120, 120, 120), (80, 50, 50), (120, 120, 80), (204, 5, 255), (230, 230, 230), (224, 5, 255), 
	(8, 255, 51), (255, 6, 82), (255, 51, 7), (204, 70, 3), (255, 6, 51), (11, 102, 255), (255, 7, 71), (220, 220, 220),
	(255, 9, 92), (102, 8, 255), (255, 194, 7), (0, 163, 255), (20, 255, 0), (0, 255, 133)]
	
def apply_color_map(image_array, num_classes=8):
	if num_classes == 19:
		sseg_color_list = sseg_color_list_19
	elif num_classes == 8:
		sseg_color_list = sseg_color_list_8
	elif num_classes == 'ade20k':
		sseg_color_list = sseg_color_list_ade20k

	color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
	for label_id, color in enumerate(sseg_color_list):
		color_array[image_array == label_id] = color
	return color_array

def find_IoU_one2batch(box, boxes):
	box = np.expand_dims(box, axis=0)
	iou = find_IoU_batch2batch(box, boxes)
	#print('iou.shape = {}'.format(iou.shape))
	return iou[0]

def find_IoU_batch2batch(bboxes1, bboxes2):
	x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
	x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = np.maximum(x11, np.transpose(x21))
	yA = np.maximum(y11, np.transpose(y21))
	xB = np.minimum(x12, np.transpose(x22))
	yB = np.minimum(y12, np.transpose(y22))

	# compute the area of intersection rectangle
	interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

	iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

	return iou

# return a list of prop index that are selected
def select_props(proposals, thresh_iou = 0.3):
	num_props = proposals.shape[0]

	list_props_idx = np.ones(num_props, dtype=np.int)

	for i in range(num_props):
		if list_props_idx[i] > 0:
			current_prop = proposals[i]
			batch_iou = find_IoU_one2batch(current_prop, proposals)
			#print('batch_iou.shape = {}'.format(batch_iou.shape))
			for j in range(i+1, num_props):
				if batch_iou[j] > thresh_iou:
					list_props_idx[j] = 0
	return list_props_idx

def round_prop_coords(prop):
	x1, y1, x2, y2 = prop
	prop_x1 = int(round(x1))
	prop_y1 = int(round(y1))
	prop_x2 = int(round(x2))
	prop_y2 = int(round(y2))
	return prop_x1, prop_y1, prop_x2, prop_y2
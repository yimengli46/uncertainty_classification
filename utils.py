import numpy as np 

sseg_color_list_19 = [(128, 64, 128), (244, 36, 232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), 
	(250,170, 30), (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), 
	(  0,  0,142), (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)]
sseg_color_list_8 = [(128, 64, 128), ( 70, 70, 70), (153, 153, 153), (107, 142, 35), (70, 130, 180), (220, 20, 60),
	(0, 0, 142), (0, 80, 100)]
	
def apply_color_map(image_array, num_classes=8):
	if num_classes == 19:
		sseg_color_list = sseg_color_list_19
	elif num_classes == 8:
		sseg_color_list = sseg_color_list_8

	color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
	for label_id, color in enumerate(sseg_color_list):
		color_array[image_array == label_id] = color
	return color_array

# sseg_pred is of shape 28 x 28
def gen_mask_and_class_label(sseg_pred, patch_label, num_classes=8):
	object_mask = np.zeros(sseg_pred.shape, dtype=np.bool)
	object_mask = np.where(sseg_pred == 2, True, object_mask) # pole
	object_mask = np.where(sseg_pred == 5, True, object_mask) # person
	object_mask = np.where(sseg_pred == 6, True, object_mask) # car
	object_mask = np.where(sseg_pred == 7, True, object_mask) # train

	gt_label = patch_label[object_mask]
	#print('gt_label.shape = {}'.format(gt_label.shape))

	class_dist = np.zeros(num_classes)
	for i in range(num_classes):
		class_dist[i] = np.sum(gt_label==i)
	max_class = np.argmax(class_dist)
	
	# compute max_class label
	gt_class_label = 0 # background
	if max_class == 0 or max_class == 1 or max_class == 3 or max_class == 4:
		gt_class_label = 0 # background
	elif max_class == 2:
		gt_class_label = 1 # pole
	elif max_class == 5:
		gt_class_label = 2 # person
	elif max_class == 6:
		gt_class_label = 3 # car
	elif max_class == 7:
		gt_class_label = 4 # train

	#print('class_dist = {}, max_class = {}'.format(class_dist, max_class))
	return object_mask, gt_class_label, class_dist

def gen_mask(sseg_pred, num_classes=8):
	object_mask = np.zeros(sseg_pred.shape, dtype=np.bool)
	object_mask = np.where(sseg_pred == 2, True, object_mask) # pole
	object_mask = np.where(sseg_pred == 5, True, object_mask) # person
	object_mask = np.where(sseg_pred == 6, True, object_mask) # car
	object_mask = np.where(sseg_pred == 7, True, object_mask) # train

	return object_mask


def convert_pred_and_label(mask):
	mask[mask == 3] = 2 # rider -> person
	mask[mask == 8] = 2 # motorcycle -> person
	mask[mask == 9] = 2 # bicycle -> person
	mask[mask == 5] = 4 # truck -> car
	mask[mask == 6] = 4 # bus -> car
	return mask

def convert_pred_and_label_sseg(mask):
	mask[mask == 3] = 2 # rider -> person
	mask[mask == 8] = 2 # motorcycle -> person
	mask[mask == 9] = 2 # bicycle -> person
	mask[mask == 5] = 3 # truck -> car
	mask[mask == 6] = 3 # bus -> car
	mask[mask == 4] = 3 # car -> car
	mask[mask == 7] = 4 # train -> train 
	return mask

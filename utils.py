import numpy as np
from sklearn.metrics import confusion_matrix

def find_IoU(box1, box2):
	x11, y11, x12, y12 = box1
	x21, y21, x22, y22 = box2

	xA = np.maximum(x11, x21)
	yA = np.maximum(y11, y21)
	xB = np.minimum(x12, x22)
	yB = np.minimum(y12, y22)

	interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

	iou = interArea / (boxAArea + boxBArea - interArea)

	return iou

def find_IoU_one2batch(box, boxes):
	ww = np.maximum(np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) -
		np.maximum(box[0], boxes[:, 0]),
		0)
	hh = np.maximum(np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) -
		np.maximum(box[1], boxes[:, 1]),
		0)
	uu = box[2] * box[3] + boxes[:, 2] * boxes[:, 3]
	return ww * hh / (uu - ww * hh)

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


def find_IoU_and_interArea(box1, box2):
	x11, y11, x12, y12 = box1
	x21, y21, x22, y22 = box2

	xA = np.maximum(x11, x21)
	yA = np.maximum(y11, y21)
	xB = np.minimum(x12, x22)
	yB = np.minimum(y12, y22)

	interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

	iou = interArea / (boxAArea + boxBArea - interArea)

	return iou, interArea

# compute mIoU, all IoU, index where union is not zero
def compute_iou(y_pred, y_true, num_classes):
	y_pred = y_pred.flatten()
	y_true = y_true.flatten()
	current = confusion_matrix(y_true, y_pred, labels=[x for x in range(num_classes)])
	# compute mean iou
	intersection = np.diag(current)
	ground_truth_set = current.sum(axis=1)
	predicted_set = current.sum(axis=0)
	union = ground_truth_set + predicted_set - intersection
	# pick out labels with union larger than 0
	union = union.astype(np.float32)
	intersection = intersection.astype(np.float32)
	idx_not_zero = union > 0
	#print('union: {}'.format(union))
	#print('intersection: {}'.format(intersection))
	IoU_no_zero = intersection[idx_not_zero] / union[idx_not_zero]
	IoU = intersection / (union+0.00001)
	#print(IoU)
	return np.mean(IoU_no_zero), IoU, idx_not_zero


void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
stuff_class_names = ['road', 'building', 'pole', 'vegetation', 'sky', 'person', 'car', 'train']
class_map = dict(zip(valid_classes, range(len(valid_classes))))

def encode_segmap(mask, ignore_index=255):
    #merge ambiguous classes
    mask[mask == 6] = 7 # ground -> road
    mask[mask == 8] = 7 # sidewalk -> road
    mask[mask == 9] = 7 # parking -> road
    mask[mask == 22] = 21 # terrain -> vegetation
    mask[mask == 25] = 24 # rider -> person
    mask[mask == 32] = 24 # motorcycle -> person
    mask[mask == 33] = 24 # bicycle -> person
    mask[mask == 27] = 26 # truck -> car
    mask[mask == 28] = 26 # bus -> car
    mask[mask == 29] = 26 # caravan -> car
    mask[mask == 30] = 26 # trailer -> car
    mask[mask == 12] = 11 # wall -> building
    mask[mask == 13] = 11 # fence -> building
    mask[mask == 19] = 17 # traffic light -> pole
    mask[mask == 20] = 17 # traffic sign -> pole
    mask[mask == 18] = 17 # pole group -> pole

    # Put all void classes to zero
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask
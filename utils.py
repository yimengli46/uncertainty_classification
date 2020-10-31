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
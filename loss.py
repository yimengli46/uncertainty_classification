import torch
import torch.nn as nn
import torch.nn.functional as F

def BinaryCrossEntropyLoss(logit, target, num_classes=8):
	B, C, H, W = logit.shape
	logit = logit.permute(0, 2, 3, 1).reshape(-1, C)

	y_targets = target.reshape(-1, 1).long().squeeze(1)
	idx_ignored = (y_targets == 255)
	y_targets[idx_ignored] = 0 # convert boundary points to class 0
	#print('y_targets.shape = {}'.format(y_targets.shape))
	y_targets = F.one_hot(y_targets, num_classes).float()
	#print('y_targets.shape = {}'.format(y_targets.shape))
	y_targets[idx_ignored, 0] = 0
	#print('y_targets.shape = {}'.format(y_targets.shape))

	#print('logit.shape = {}, y_targets.shape = {}'.format(logit.shape, y_targets.shape))

	loss = F.binary_cross_entropy(logit, y_targets, reduction='sum').div(num_classes * logit.shape[0])
	return loss
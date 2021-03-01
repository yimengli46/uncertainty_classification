import torch
import torch.nn as nn
import torch.nn.functional as F

def BinaryCrossEntropyLoss(logit, target, num_classes=8):
	#B, C, H, W = logit.shape
	#logit = logit.permute(0, 2, 3, 1).reshape(-1, C)

	y_targets = target.long()
	y_targets = F.one_hot(y_targets, num_classes).float()

	loss = F.binary_cross_entropy(logit, y_targets, reduction='sum').div(num_classes * logit.shape[0])
	return loss
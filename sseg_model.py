import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSegHead(nn.Module):
	def __init__(self, num_classes=8):
		super(SSegHead, self).__init__()
		self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(256)
		self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0)
		self.bn5 = nn.BatchNorm2d(256)
		self.predictor = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)


	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.deconv(x)))
		x = self.predictor(x)
		return x




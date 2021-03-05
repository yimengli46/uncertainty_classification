import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSegHead(nn.Module):
	def __init__(self, num_classes=8, input_dim=512):
		super(SSegHead, self).__init__()
		self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
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

class DropoutHead(nn.Module):
	def __init__(self, num_classes=8, input_dim=512):
		super(DropoutHead, self).__init__()
		self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(256)
		self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0)
		self.bn5 = nn.BatchNorm2d(256)
		self.predictor = nn.Linear(256, num_classes)


	def forward(self, x, mask):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn5(self.deconv(x)))
		x = F.dropout2d(x, p=0.2, training=True)

		# put through mask
		B, C, H, W = x.shape
		mask = mask.unsqueeze(1).repeat(1, C, 1, 1) # B*H*W -> B*1*H*W -> B*C*H*W
		block_value = x.min()  # min values of whole tensor [B*C*H*W]
		x = torch.where(mask < 1, block_value, x)
		x = F.max_pool2d(x, [H, W]) # 64 x 256 x 1 x 1

		#print('x.shape = {}'.format(x.shape))
		x = x.view(x.size(0),-1) # 64 x 256
		#print('x.shape = {}'.format(x.shape))

		x = self.predictor(x)
		return x

class ResidualBlock(nn.Module):
	def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
		super(ResidualBlock,self).__init__()
		self.left=nn.Sequential(
			nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
			nn.BatchNorm2d(outchannel)
		)
		self.right=shortcut
		
	def forward(self,x):
		out = self.left(x)
		residual = x if self.right is None else self.right(x)
		out += residual
		return F.relu(out)

class DuqHead(nn.Module):
	def __init__(self, num_classes=8, input_dim=512):
		super(DuqHead, self).__init__()
		self.num_classes = num_classes

		self.pre=nn.Sequential(
			nn.Conv2d(3,64,7,2,3,bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3,2,1)
		)

		self.layer1=self._make_layer(64,128,3)
		self.layer2=self._make_layer(128,256,4,stride=2)

		self.fc = nn.Linear(1024, 256)
		
		#==========================================================================================================
		self.duq_centroid_size = 512
		self.duq_model_output_size = 256
		self.gamma = 0.999
		self.duq_length_scale = 0.1

		self.W = nn.Parameter(torch.zeros(self.duq_centroid_size, self.num_classes, self.duq_model_output_size))
		nn.init.kaiming_normal_(self.W, nonlinearity='relu')
		self.register_buffer('N', torch.ones(self.num_classes)*20)
		self.register_buffer('m', torch.normal(torch.zeros(self.duq_centroid_size, self.num_classes), 0.05))
		self.m = self.m *self.N
		self.sigma = self.duq_length_scale

	def _make_layer(self,inchannel,outchannel,block_num,stride=1):
		shortcut=nn.Sequential(
			nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
			nn.BatchNorm2d(outchannel))
 
		layers=[ ]
		layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
		
		for i in range(1,block_num):
			layers.append(ResidualBlock(outchannel,outchannel))
		return nn.Sequential(*layers)

	def rbf(self, z):
		z = torch.einsum('ij,mnj->imn', z, self.W)
		embeddings = self.m / self.N.unsqueeze(0)
		diff = z - embeddings.unsqueeze(0)
		diff = (diff ** 2).mean(1).div(2 * self.sigma **2).mul(-1).exp()
		return diff

	def forward(self, x, mask):
		x = self.pre(x)
		#print('x.shape = {}'.format(x.shape))
		x = self.layer1(x)
		#print('x.shape = {}'.format(x.shape))
		x = self.layer2(x) # B x 128 x 28 x 28
		#print('x.shape = {}'.format(x.shape))
		#assert 1==2

		# put through mask
		B, C, H, W = x.shape
		mask = mask.unsqueeze(1).repeat(1, C, 1, 1) # B*H*W -> B*1*H*W -> B*C*H*W
		block_value = x.min()  # min values of whole tensor [B*C*H*W]
		x = torch.where(mask < 1, block_value, x)
		x = F.max_pool2d(x, [14, 14]) # 64 x 256 x 1 x 1

		#print('x.shape = {}'.format(x.shape))
		x = x.view(x.size(0),-1) # 64 x 256
		#print('x.shape = {}'.format(x.shape))
		x = self.fc(x)

		z = x

		y_pred = self.rbf(z) # 64 x 5
		#print('y_pred.shape = {}'.format(y_pred.shape))
		
		return y_pred

	def update_embeddings(self, x, mask, y_targets):
		
		y_targets = F.one_hot(y_targets, self.num_classes).float()

		self.N = self.gamma * self.N + (1-self.gamma) * y_targets.sum(0)

		x = self.pre(x)
		#print('x.shape = {}'.format(x.shape))
		x = self.layer1(x)
		#print('x.shape = {}'.format(x.shape))
		x = self.layer2(x) # B x 128 x 28 x 28
		#print('x.shape = {}'.format(x.shape))
		#assert 1==2

		# put through mask
		B, C, H, W = x.shape
		mask = mask.unsqueeze(1).repeat(1, C, 1, 1) # B*H*W -> B*1*H*W -> B*C*H*W
		block_value = x.min()  # min values of whole tensor [B*C*H*W]
		x = torch.where(mask < 1, block_value, x)
		x = F.max_pool2d(x, [14, 14])

		#print('x.shape = {}'.format(x.shape))
		x = x.view(x.size(0),-1)
		#print('x.shape = {}'.format(x.shape))
		x = self.fc(x)

		z = x

		z = torch.einsum('ij,mnj->imn', z, self.W)
		embedding_sum = torch.einsum('ijk,ik->jk', z, y_targets)

		self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum


def calc_gradient_penalty(x, y_pred):
	B, H, W, C = y_pred.shape
	y_pred = y_pred.reshape(B, -1)

	gradients = torch.autograd.grad(
		outputs=y_pred,
		inputs = x,
		grad_outputs=torch.ones_like(y_pred)/(1.0*H*W),
		create_graph=True,
	)[0]

	gradients = gradients.flatten(start_dim=1)

	grad_norm = gradients.norm(2, dim=1)

	gradient_penalty = ((grad_norm-1)**2).mean()

	return gradient_penalty
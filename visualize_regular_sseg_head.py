import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import SSegHead
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset

saved_folder = 'visualization/obj_sseg_regular'
trained_model_dir = 'trained_model/regular'
num_proposals = 100

dataset_folder = '/home/reza/ARGO_datasets/Cityscapes'
ds_val = CityscapesProposalsDataset(dataset_folder, 'val')
num_classes = ds_val.NUM_CLASSES

device = torch.device('cuda')

classifier = SSegHead(num_classes).to(device)
classifier.load_state_dict(torch.load('{}/regular_classifier.pth'.format(trained_model_dir)))

with torch.no_grad():
	for i in range(len(ds_val)):
		patch_feature, _, img_patch, sseg_label_patch = ds_val.get_proposal(i, 0)

		patch_feature = patch_feature.to(device)
		logits = classifier(patch_feature)

		assert 1==2


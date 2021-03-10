import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from PIL import Image
import matplotlib.pyplot as plt

trained_classes = 10
rep_style = 'ObjDet'

# read all cityscapes files
dataset = 'cityscapes'
if trained_classes > 5:
	result_folder = 'cls_results/prop_cls_more_class_old/obj_sseg_duq/{}/{}'.format(rep_style, dataset)
elif trained_classes == 5:
	result_folder = 'cls_results/prop_classification/obj_sseg_duq/{}/{}'.format(rep_style, dataset)
num_files = 50

acc_list = np.zeros(num_files)
for i in range(num_files):
	result = np.load('{}/img_{}.npy'.format(result_folder, i), allow_pickle=True).item()

	#pred = result['pred']
	uncertainty = result['uncertainty']
	label = np.zeros(uncertainty.shape[0])

	if i == 0:
		all_uncertainty = uncertainty
		all_label = label
	else:
		all_uncertainty = np.concatenate((all_uncertainty, uncertainty))
		all_label = np.concatenate((all_label, label))

#'''
# read lost and found files
dataset = 'lostAndFound'
if trained_classes > 5:
	result_folder = 'cls_results/prop_cls_more_class_old/obj_sseg_duq/{}/{}'.format(rep_style, dataset)
elif trained_classes == 5:
	result_folder = 'cls_results/prop_classification/obj_sseg_duq/{}/{}'.format(rep_style, dataset)
big_outlier_list_lostAndFound = [0, 2, 3, 4, 7, 8, 9, 10, 11, 15, 16, 20, 22, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 50, 51, 52, 54, 56, 57, 58, 60, 61, 63, 65, 67, 68, 71, 72, 73, 74, 76, 80, 83, 84, 85, 86, 89, 91, 92, 93, 95, 96, 98, ]

img_id_list = big_outlier_list_lostAndFound

for i in img_id_list:
	result = np.load('{}/img_{}.npy'.format(result_folder, i), allow_pickle=True).item()

	uncertainty = result['uncertainty']
	label = np.ones(uncertainty.shape[0])

	all_uncertainty = np.concatenate((all_uncertainty, uncertainty))
	all_label = np.concatenate((all_label, label))

auroc_score = roc_auc_score(all_label, all_uncertainty)

#compute AP
ap = average_precision_score(all_label, all_uncertainty)
#'''

'''
# read roadAnomaly files
dataset = 'roadAnomaly'
if trained_classes > 5:
	result_folder = 'cls_results/prop_cls_more_class_old/obj_sseg_duq/{}/{}'.format(rep_style, dataset)
elif trained_classes == 5:
	result_folder = 'cls_results/prop_classification/obj_sseg_duq/{}/{}'.format(rep_style, dataset)
proposal_folder = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals_whole/roadAnomaly'
dataset_folder = '/home/yimeng/ARGO_datasets/RoadAnomaly'
num_props = 20

img_id_list = list(range(60))

for i in img_id_list:
	result = np.load('{}/img_{}.npy'.format(result_folder, i), allow_pickle=True).item()

	uncertainty = result['uncertainty']
	label = np.zeros(uncertainty.shape[0])

	proposals = np.load('{}/{}_proposal.npy'.format(proposal_folder, i), allow_pickle=True)
	lbl_path = '{}/{}_label.png'.format(dataset_folder, i)
	sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)

	for proposal_id in range(num_props):
		x1, y1, x2, y2 = proposals[proposal_id]
		prop_x1 = int(round(x1))
		prop_y1 = int(round(y1))
		prop_x2 = int(round(x2))
		prop_y2 = int(round(y2))

		sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]
		ood_area = np.sum(sseg_label_proposal==1)
		prop_area = (prop_y2 - prop_y1) * (prop_x2-prop_x1)
		ratio = 1.0 * ood_area / prop_area
		if ratio > 0.5:
			label[proposal_id] = 1
	
	uncertainty = uncertainty[label==1]
	label = label[label==1]
	print('uncertainty.shape = {}'.format(uncertainty.shape))

	all_uncertainty = np.concatenate((all_uncertainty, uncertainty))
	all_label = np.concatenate((all_label, label))

auroc_score = roc_auc_score(all_label, all_uncertainty)

#compute AP
ap = average_precision_score(all_label, all_uncertainty)
'''
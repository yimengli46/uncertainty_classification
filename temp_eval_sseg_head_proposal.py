import numpy as np
from PIL import Image 
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from utils import compute_fpr

style = 'duq' # 'duq', 'dropout'
dataset = 'roadAnomaly' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg' 
add_regular_props = False # add mask-rcnn proposals to lost&found and fishyscapes or not

print('style = {}, dataset = {}, rep_style = {}, add_regular = {}'.format(style, dataset, rep_style, add_regular_props))

#base_folder = '/home/yimeng/work/uncertainty_classification_all_props/visualization/temp_all_props_bg_other'
#base_folder = '/home/yimeng/ARGO_scratch/uncertainty_classification/visualization/bg_only'
base_folder = '/home/yimeng/work/uncertainty_classification_all_props/visualization/all_props_gradientPenalty'
#base_folder = '/home/yimeng/ARGO_scratch/uncertainty_classification/visualization/no_conv'
#base_folder = '/home/yimeng/work/uncertainty_classification_all_props/visualization/ignore_boundary'
#base_folder = '/home/yimeng/work/uncertainty_classification_all_props/visualization/more_bg_props'
#base_folder = '/home/yimeng/work/uncertainty_classification_all_props/visualization/all_props_all_classes'
#base_folder = '/home/yimeng/work/uncertainty_classification_all_props/visualization/adversarial_boundary'


result_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset) # where sseg npy file is saved
proposal_folder = '/home/yimeng/ARGO_scratch/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals_whole/{}'.format(dataset)
if dataset == 'cityscapes':
	dataset_folder = '/home/yimeng/ARGO_datasets/Cityscapes'
elif dataset == 'lostAndFound':
	dataset_folder = '/home/yimeng/ARGO_datasets/Lost_and_Found'
	num_images = 100
elif dataset == 'fishyscapes':
	dataset_folder = '/home/yimeng/ARGO_datasets/Fishyscapes_Static'
	num_images = 30
elif dataset == 'roadAnomaly':
	dataset_folder = '/home/yimeng/ARGO_datasets/RoadAnomaly'
	num_images = 60

big_outlier_list_lostAndFound = [0, 2, 3, 4, 7, 8, 9, 10, 11, 15, 16, 20, 22, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 50, 51, 52, 54, 56, 57, 58, 60, 61, 63, 65, 67, 68, 71, 72, 73, 74, 76, 80, 83, 84, 85, 86, 89, 91, 92, 93, 95, 96, 98, ]
#big_outlier_list_lostAndFound = [7]

auroc_score_list = []
ap_list = []
fpr_list = []

if dataset == 'lostAndFound':
	img_id_list = big_outlier_list_lostAndFound
else: 
	img_id_list = list(range(num_images))

for img_id in img_id_list:
	
	#print('img_id = {}'.format(img_id))
	# read in proposal file
	proposals = np.load('{}/{}_proposal.npy'.format(proposal_folder, img_id), allow_pickle=True)

	# read in gt label
	lbl_path = '{}/{}_label.png'.format(dataset_folder, img_id)
	sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
	#assert 1==2

	if dataset == 'lostAndFound' or dataset == 'fishyscapes':
		num_props = proposals.shape[0]
		if add_regular_props:
			regular_proposals = np.load('{}/{}_regular_proposal.npy'.format(proposal_folder, img_id), allow_pickle=True)
			proposals = np.concatenate((proposals, regular_proposals), axis=0)
			num_props += 20
	elif dataset == 'roadAnomaly':
		num_props = 20

	#print('num_props = {}'.format(num_props))

	for proposal_id in range(num_props):
		x1, y1, x2, y2 = proposals[proposal_id]
		prop_x1 = int(round(x1))
		prop_y1 = int(round(y1))
		prop_x2 = int(round(x2))
		prop_y2 = int(round(y2))
		if dataset == 'roadAnomaly':
			prop_x2 = int(round(x2))
			prop_y2 = int(round(y2))

		# road anomaly use detected proposals. These proposals have weired shape. So ignore them.
		if dataset == 'roadAnomaly' or add_regular_props:
			prop_W = prop_x2 - prop_x1
			prop_H = prop_y2 - prop_y1
			if prop_W / prop_H < 0.25 or prop_W / prop_H < 0.25:
				continue

		sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

		#print('proposal_id = {}'.format(proposal_id))
		# read in detection results
		result = np.load('{}/img_{}_proposal_{}.npy'.format(result_folder, img_id, proposal_id), allow_pickle=True).item()
		result_sseg = result['sseg']
		result_uncertainty = result['uncertainty']

		result_uncertainty[sseg_label_proposal==2] = 0 # change the ignored pixels into inlier class
		sseg_label_proposal[sseg_label_proposal==2] = 0 # change the ignored pixels into inlier class

		if proposal_id == 0:
			img_sseg_label = sseg_label_proposal.ravel()
		else:
			img_sseg_label = np.concatenate((img_sseg_label, sseg_label_proposal.ravel()))

		if proposal_id == 0:
			img_result_uncertainty = result_uncertainty.ravel()
		else:
			img_result_uncertainty = np.concatenate((img_result_uncertainty, result_uncertainty.ravel()))

	# road anomaly has proposals that doesn't have positive labels
	if (img_sseg_label == 1).sum() == 0 or (img_sseg_label == 0).sum() == 0:
		continue
	else:
		# compute the roc-auc score
		auroc_score = roc_auc_score(img_sseg_label, img_result_uncertainty)

		#compute AP
		ap = average_precision_score(img_sseg_label, img_result_uncertainty)

		# compute fpr at 50% tpr
		fpr_score = compute_fpr(img_sseg_label, img_result_uncertainty, 0.95)

		auroc_score_list.append(auroc_score)
		ap_list.append(ap)
		fpr_list.append(fpr_score)


print('===>mean auroc_score is {:.3f}'.format(np.array(auroc_score_list).mean()))
print('===>mean ap is {:.3f}'.format(np.array(ap_list).mean()))
print('===>mean fpr is {:.3f}'.format(np.array(fpr_list).mean()))
print('--------------------------------------------------------------------------')
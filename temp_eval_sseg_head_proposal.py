import numpy as np
from PIL import Image 
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from utils import compute_fpr
from dataloaders.ade20k_ood_proposals import ADE20KOODProposalsDataset
from dataloaders.avd_ood_proposals import AvdOODProposalsDataset

style = 'duq' # 'duq', 'dropout'
dataset = 'avd' #'ade20k' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'
rep_style = 'SSeg' #'both', 'ObjDet', 'SSeg' 

print('style = {}, dataset = {}, rep_style = {}'.format(style, dataset, rep_style))

base_folder = '/home/yimeng/work/uncertainty_classification-ade20k/visualization/ade20k_ood'

result_folder = '{}/obj_sseg_{}/{}/{}'.format(base_folder, style, rep_style, dataset) # where sseg npy file is saved

if dataset == 'ade20k':
	dataset_folder = '/home/yimeng/ARGO_datasets/ADE20K/Semantic_Segmentation'
	ds_val = ADE20KOODProposalsDataset(dataset_folder, 'val', rep_style=rep_style)
elif dataset == 'avd':
	dataset_folder = '/home/yimeng/ARGO_datasets/AVD'
	ds_val = AvdOODProposalsDataset(dataset_folder, rep_style=rep_style)

auroc_score_list = []
ap_list = []
fpr_list = []

count = 0

for i in range(len(ds_val)):
	num_proposals = 100
	ood_prop_array = ds_val.select_ood_props(i)
	
	for j in range(num_proposals):
		if ood_prop_array[j] > 0:
			print('i = {}, j = {}'.format(i, j))
			_, _, _, sseg_label_proposal = ds_val.get_proposal(i, j)

			#print('proposal_id = {}'.format(proposal_id))
			# read in detection results
			result = np.load('{}/img_{}_proposal_{}.npy'.format(result_folder, i, j), allow_pickle=True).item()
			result_sseg = result['sseg']
			result_uncertainty = result['uncertainty']

			result_uncertainty[sseg_label_proposal==2] = 0 # change the ignored pixels into inlier class
			sseg_label_proposal[sseg_label_proposal==2] = 0 # change the ignored pixels into inlier class


			img_sseg_label = sseg_label_proposal.ravel()
			img_result_uncertainty = result_uncertainty.ravel()


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
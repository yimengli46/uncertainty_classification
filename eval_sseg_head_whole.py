import numpy as np
from PIL import Image 
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from utils import compute_fpr
import cv2

style = 'duq' # 'duq', 'dropout'
dataset = 'fishyscapes' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'

for style in ['duq', 'dropout']:
	for dataset in ['lostAndFound', 'roadAnomaly', 'fishyscapes']:
		for tpr_thresh in [0.5, 0.7]:

			print('style = {}, dataset = {}, tpr_thresh = {}'.format(style, dataset, tpr_thresh))

			base_folder = '/home/reza/ARGO_scratch/uncertainty_classification/visualization/whole'
			result_folder = '{}/obj_sseg_{}/{}'.format(base_folder, style, dataset) # where sseg npy file is saved

			if dataset == 'cityscapes':
				dataset_folder = '/home/reza/ARGO_datasets/Cityscapes'
			elif dataset == 'lostAndFound':
				dataset_folder = '/home/reza/ARGO_datasets/Lost_and_Found'
				num_images = 100
			elif dataset == 'fishyscapes':
				dataset_folder = '/home/reza/ARGO_datasets/Fishyscapes_Static'
				num_images = 30
			elif dataset == 'roadAnomaly':
				dataset_folder = '/home/reza/ARGO_datasets/RoadAnomaly'
				num_images = 60

			auroc_score_list = []
			ap_list = []
			fpr_list = []
			for img_id in range(num_images):
				#print('img_id = {}'.format(img_id))

				# read in gt label
				lbl_path = '{}/{}_label.png'.format(dataset_folder, img_id)
				sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
				H, W = sseg_label.shape
				#assert 1==2

				#print('proposal_id = {}'.format(proposal_id))
				# read in detection results
				result = np.load('{}/img_{}.npy'.format(result_folder, img_id), allow_pickle=True).item()
				result_sseg = result['sseg']
				result_uncertainty = result['uncertainty']
				result_uncertainty = cv2.resize(result_uncertainty, (W, H))

				result_uncertainty = result_uncertainty.ravel()
				sseg_label = sseg_label.ravel()

				result_uncertainty = result_uncertainty[sseg_label<2]
				sseg_label = sseg_label[sseg_label<2]

				# road anomaly has proposals that doesn't have positive labels
				if (sseg_label == 1).sum() == 0 or (sseg_label == 0).sum() == 0:
					continue
				else:
					# compute the roc-auc score
					auroc_score = roc_auc_score(sseg_label, result_uncertainty)

					# compute fpr at 95% tpr
					fpr_score = compute_fpr(sseg_label, result_uncertainty)

					#compute AP
					ap = average_precision_score(sseg_label, result_uncertainty)

					auroc_score_list.append(auroc_score)
					ap_list.append(ap)
					fpr_list.append(fpr_score)


			print('===>mean auroc_score is {:.3f}'.format(np.array(auroc_score_list).mean()))
			print('===>mean fpr_list is {:.3f}'.format(np.array(fpr_list).mean()))
			print('===>mean ap is {:.3f}'.format(np.array(ap_list).mean()))
			print('--------------------------------------------------------------------------')
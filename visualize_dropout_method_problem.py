import numpy as np
from PIL import Image 
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from utils import compute_fpr
import cv2

style = 'gan' #'gan', 'deeplab'
dataset = 'lostAndFound' #'lostAndFound', 'cityscapes', 'fishyscapes', 'roadAnomaly'

gan_result_folder = '{}/visualization_{}'.format('/home/yimeng/work/detecting-the-unexpected_GG', dataset) 
deeplab_result_folder = '{}/resNet_{}'.format('/home/yimeng/work/Anomaly_Detection_SSeg-duq/results_duq', dataset)

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


img_id = 9
proposal_id = 0

if dataset == 'lostAndFound':
	img_id_list = big_outlier_list_lostAndFound
else: 
	img_id_list = list(range(num_images))

for img_id in img_id_list:



		# read in gt label
		img_path = '{}/{}.png'.format(dataset_folder, img_id)
		lbl_path = '{}/{}_label.png'.format(dataset_folder, img_id)
		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		H, W = sseg_label.shape
		#assert 1==2



		# read in deeplab results
		deeplab_result = np.load('{}/{}_result.npy'.format(deeplab_result_folder, img_id), allow_pickle=True).item()
		deeplab_result_sseg = deeplab_result['sseg']
		deeplab_result_uncertainty = deeplab_result['uncertainty']
		deeplab_result_uncertainty = cv2.resize(deeplab_result_uncertainty, (W, H))

		

		fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
		ax[0][0].imshow(rgb_proposal)
		ax[0][0].get_xaxis().set_visible(False)
		ax[0][0].get_yaxis().set_visible(False)
		ax[0][0].set_title("object proposal")
		ax[0][1].imshow(my_prop_uncertainty, vmin=0.0, vmax=1.0)
		ax[0][1].get_xaxis().set_visible(False)
		ax[0][1].get_yaxis().set_visible(False)
		ax[0][1].set_title("ours")
		ax[1][0].imshow(gan_result_prop_uncertainty, vmin=0.0, vmax=1.0)
		ax[1][0].get_xaxis().set_visible(False)
		ax[1][0].get_yaxis().set_visible(False)
		ax[1][0].set_title("gan")
		ax[1][1].imshow(deeplab_result_prop_uncertainty, vmin=0.0, vmax=1.0)
		ax[1][1].get_xaxis().set_visible(False)
		ax[1][1].get_yaxis().set_visible(False)
		ax[1][1].set_title("Dropout")
		plt.show()
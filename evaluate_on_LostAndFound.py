import numpy as np
from utils import compute_iou
import cv2
import matplotlib.pyplot as plt 

#result_base_folder = '/home/yimeng/ARGO_scratch/detectron2/my_projects/BayesianSSeg/visualization_lostAndFound'
result_base_folder = '/home/yimeng/work/detecting-the-unexpected/visualization_lostAndFound'
dataset_base_folder = '/home/yimeng/Datasets/{}'.format('Lost_and_Found')

#uncertainty_threshold_list = [x/10.0 for x in range(5, 50)]
uncertainty_threshold_list = [x/100.0 for x in range(0, 100, 5)]

big_outlier_list = [2, 3, 4, 7, 10, 11, 15, 16, 25, 27, 31, 33, 34, 35, 38, 40, 45, 46, 48, 50, 51, 54, 57, 60, 61, 63, 65, 
	68, 71, 72, 74, 76, 83, 84, 85, 86, 91, 93, 95]

num_test_imgs = len(big_outlier_list)
all_mIoU = np.zeros(num_test_imgs)

for uncertainty_threshold in uncertainty_threshold_list:
	for i, img_id in enumerate(big_outlier_list):
		#print('img_id = {}'.format(img_id))
		label_img = cv2.imread('{}/{}_label.png'.format(dataset_base_folder, img_id), 0)
		# outlier have label 1, others have label 0
		label_img = np.where(label_img != 1, 0, label_img)

		result = np.load('{}/{}_result.npy'.format(result_base_folder, img_id), allow_pickle=True).item()
		uncertainty_result = result['uncertainty']
		uncertainty_result = np.where(uncertainty_result < uncertainty_threshold, 0, 1)

		# in case the result is got from downsampled images
		h, w = uncertainty_result.shape
		label_img = cv2.resize(label_img, (w,h))

		mIoU, all_IoU, idx_not_zero = compute_iou(label_img, uncertainty_result, 2)
		# only take the IoU on the outlier, rather than the background
		#print('all_IoU = {}'.format(all_IoU))
		all_mIoU[i] = all_IoU[1]

	print('uncertainty_threshold = {}'.format(uncertainty_threshold))
	print('mean IoU over {} imgs from LostAndFound is {:.4f}'.format(num_test_imgs, np.mean(all_mIoU)))
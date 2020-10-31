import numpy as np
from utils import compute_iou
import cv2
import matplotlib.pyplot as plt 

#result_base_folder = '{}/visualization_RoadAnomaly'.format('/home/yimeng/work/detectron2/my_projects/BayesianSSeg')
result_base_folder = '{}/visualization_RoadAnomaly'.format('/home/yimeng/work/detecting-the-unexpected')

dataset_base_folder = '/home/yimeng/Datasets/{}'.format('RoadAnomaly')

num_test_imgs = 60
all_mIoU = np.zeros(num_test_imgs)

#uncertainty_threshold_list = [x/10.0 for x in range(5, 10)]
uncertainty_threshold_list = [x/100.0 for x in range(0, 50, 5)]

for uncertainty_threshold in uncertainty_threshold_list:
	for i in range(num_test_imgs):
		#print('i = {}'.format(i))
		label_img = cv2.imread('{}/{}_label.png'.format(dataset_base_folder, i), 0)
		# outlier have label 1, others have label 0
		label_img = np.where(label_img != 1, 0, label_img)

		result = np.load('{}/{}_result.npy'.format(result_base_folder, i), allow_pickle=True).item()
		uncertainty_result = result['uncertainty']
		uncertainty_result = np.where(uncertainty_result < uncertainty_threshold, 0, 1)

		# in case the result is got from downsampled images
		h, w = uncertainty_result.shape
		label_img = cv2.resize(label_img, (w,h))

		mIoU, all_IoU, idx_not_zero = compute_iou(label_img, uncertainty_result, 2)
		# only take the IoU on the outlier, rather than the background
		#print('all_IoU = {}'.format(all_IoU))
		all_mIoU[i] = all_IoU[1]
		#assert 1==2
	print('uncertainty_threshold = {}'.format(uncertainty_threshold))
	print('mean IoU over {} imgs from RoadAnomaly is {:.4f}'.format(num_test_imgs, np.mean(all_mIoU)))
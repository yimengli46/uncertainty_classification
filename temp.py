import numpy as np
import cv2
import json
import matplotlib.pyplot as plt 
import glob
from PIL import Image 
import os

color_list = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), 
	(244, 36, 232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153), (180,165,180),
	(150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35),
	(152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
	(  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32)]

def apply_color_map(image_array):
	color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
	for label_id, color in enumerate(color_list):
		color_array[image_array == label_id] = color
	return color_array

detectron2_base_folder = '/home/yimeng/work/detectron2'
dataset_base_folder = '/home/yimeng/Datasets/{}'.format('Lost_and_Found')

big_outlier_list = [2, 3, 4, 7, 10, 11, 15, 16, 25, 27, 31, 33, 34, 35, 38, 40, 45, 46, 48, 50, 51, 54, 57, 60, 61, 63, 65, 
	68, 71, 72, 74, 76, 83, 84, 85, 86, 91, 93, 95]


#for i in range(100):
for i in big_outlier_list:
	print('i = {}'.format(i))
	rgb_img = cv2.imread('{}/{}.png'.format(dataset_base_folder, i))[:,:,::-1]

	np_result = np.load('{}/{}_result.npy'.format('/home/yimeng/work/detectron2/my_projects/BayesianSSeg/visualization_lostAndFound', i), allow_pickle=True).item()
	uncertainty_entropy=np_result['uncertainty']
	class_prediction = np_result['sseg']

	colored_prediction_array = apply_color_map(class_prediction)

	det_img = cv2.imread('{}/{}_uncertain_boxes_variance.jpg'.format('RetinaNet_LostAndFound_Static_Results/500', i), 1)[:,:,::-1]

	# visualization
	fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
	ax[0][0].imshow(rgb_img)
	ax[0][0].get_xaxis().set_visible(False)
	ax[0][0].get_yaxis().set_visible(False)
	ax[0][0].set_title('RGB Image')
	ax[0][1].imshow(colored_prediction_array)
	ax[0][1].get_xaxis().set_visible(False)
	ax[0][1].get_yaxis().set_visible(False)
	ax[0][1].set_title('Semantic Segmentation')
	ax[1][0].imshow(det_img)
	ax[1][0].get_xaxis().set_visible(False)
	ax[1][0].get_yaxis().set_visible(False)
	ax[1][0].set_title('Outlier Detection')
	ax[1][1].imshow(uncertainty_entropy)
	ax[1][1].get_xaxis().set_visible(False)
	ax[1][1].get_yaxis().set_visible(False)
	ax[1][1].set_title('Uncertainty Entropy')

	fig.tight_layout()
	fig.savefig('{}/{}_result.jpg'.format('merge_det_sseg', i))
	plt.close()




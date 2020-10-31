import numpy as np
from utils import compute_iou
import cv2
import matplotlib.pyplot as plt 

#result_base_folder = '/home/yimeng/ARGO_scratch/detectron2/my_projects/BayesianSSeg/visualization_cityscapes'
result_base_folder = '{}/visualization_cityscapes'.format('/home/yimeng/work/detectron2/my_projects/FPN_SSeg')
dataset_base_folder = '/home/yimeng/Datasets/{}'.format('Cityscapes')

# write a function that loads the dataset into detectron2's standard format
def get_cityscapes_sseg_dicts(dataset_dir, mode='train'):
	np_file = np.load('{}/{}_img_list.npy'.format(dataset_dir, mode), allow_pickle=True)

	dataset_dics = []
	for idx, v in enumerate(np_file):
		record = {}

		filename = '{}/{}'.format(dataset_dir, v['rgb_path'])
		sseg_filename = '{}/{}'.format(dataset_dir, v['semSeg_path'])

		record['file_name'] = filename
		record['image_id'] = idx
		record['height'] = 1024
		record['width'] = 2048
		record['sem_seg_file_name'] = sseg_filename

		dataset_dics.append(record)
	return dataset_dics

dataset_dicts = get_cityscapes_sseg_dicts(dataset_base_folder, 'val')

num_test_imgs = 100
all_mIoU = np.zeros(num_test_imgs)

for i in range(num_test_imgs):
	print('i = {}'.format(i))
	d = dataset_dicts[i]
	label_img = cv2.imread(d['sem_seg_file_name'], 0)

	result = np.load('{}/{}_result.npy'.format(result_base_folder, i), allow_pickle=True).item()
	sseg_result = result['sseg']

	mIoU, all_IoU, idx_not_zero = compute_iou(label_img, sseg_result, 34)
	all_mIoU[i] = mIoU


print('mean IoU over {} imgs from Cityscapes is {:.4f}'.format(num_test_imgs, np.mean(all_mIoU)))
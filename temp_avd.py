import glob
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

dataset_folder = '/home/yimeng/ARGO_datasets/AVD'
scene_names = ['Home_004_1', 'Home_006_1', 'Home_007_1', 'Home_010_1', 'Home_011_1', 'Home_016_1']
results = []

for scene in scene_names:
	mat_files = [os.path.basename(x) for x in glob.glob('{}/{}/final_label/*.mat'.format(dataset_folder, scene))]
	
	for mat_file in mat_files:
		img_id = mat_file[:-4]
		img = cv2.imread('{}/{}/selected_images/{}.jpg'.format(dataset_folder, scene, img_id), 1)[:,:,::-1] 

		anno = sio.loadmat('{}/{}/final_label/{}.mat'.format(dataset_folder, scene, img_id))['mapLabel']
		cv2.imwrite('{}/{}/final_label/{}.png'.format(dataset_folder, scene, img_id), anno)
		#assert 1==2
		
		result = {}
		result['img'] = '{}/selected_images/{}.jpg'.format(scene, img_id)
		result['anno'] = '{}/final_label/{}.png'.format(scene, img_id)
		results.append(result)

np.save('avd_files.npy', results)
import numpy as np 
import matplotlib.pyplot as plt

rep_style = 'ObjDet'
npy_folder = '/home/reza/ARGO_scratch/uncertainty_classification/visualization/all_props/obj_sseg_regular/{}/cityscapes'.format(rep_style)

img_id = 2#list(range(100))
prop_id = 0
flag_eval = True

for img_id in range(100):
	result = np.load('{}/regular_img_{}_proposal_{}_eval_{}.npy'.format(npy_folder, img_id, prop_id, flag_eval), allow_pickle=True).item()

	z = result['ft'].reshape(-1, 256)
	logit = result['logit']
	label = result['label']
	sseg_pred = np.argmax(logit, axis=2)

	sseg_pred[sseg_pred == 0] = 0 # road
	sseg_pred[sseg_pred == 1] = 0 # building
	sseg_pred[sseg_pred == 3] = 0 # vegetation
	sseg_pred[sseg_pred == 4] = 0 # sky
	sseg_pred[sseg_pred > 0] = 1

	# compute mean and sigma of each channel of z
	mu_z = np.mean(z, axis=0).reshape((16, 16))
	std_z = np.std(z, axis=0).reshape((16, 16))

	#plt.imshow(mu_z, vmin=0.0, vmax=1.0)
	#plt.show()

	fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18,10))
	ax[0][0].imshow(sseg_pred)
	ax[0][0].get_xaxis().set_visible(False)
	ax[0][0].get_yaxis().set_visible(False)
	ax[0][0].set_title("object pixel prediction")
	ax[0][1].imshow(label)
	ax[0][1].get_xaxis().set_visible(False)
	ax[0][1].get_yaxis().set_visible(False)
	ax[0][1].set_title('gt label')
	ax[1][0].imshow(mu_z, vmin=0.0, vmax=0.5)
	ax[1][0].get_xaxis().set_visible(False)
	ax[1][0].get_yaxis().set_visible(False)
	ax[1][0].set_title("mu 16x16")
	ax[1][1].imshow(std_z, vmin=0.0, vmax=0.5)
	ax[1][1].get_xaxis().set_visible(False)
	ax[1][1].get_yaxis().set_visible(False)
	ax[1][1].set_title("sigma 16x16")

	fig.tight_layout()
	fig.savefig('{}/img_{}_proposal_{}_eval_{}_analysis.jpg'.format(npy_folder, img_id, prop_id, flag_eval))
	plt.close()
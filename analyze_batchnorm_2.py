import numpy as np 
import matplotlib.pyplot as plt

def change_sseg_pred (sseg_pred):
	sseg_pred[sseg_pred == 0] = 0 # road
	sseg_pred[sseg_pred == 1] = 0 # building
	sseg_pred[sseg_pred == 3] = 0 # vegetation
	sseg_pred[sseg_pred == 4] = 0 # sky
	sseg_pred[sseg_pred > 0] = 1
	return sseg_pred

rep_style = 'SSeg'
npy_folder = '/home/reza/ARGO_scratch/uncertainty_classification/visualization/all_props/obj_sseg_regular/{}/lostAndFound'.format(rep_style)

img_id = 73
prop_id = 0

npy_eval_True = np.load('{}/regular_img_{}_proposal_{}_eval_{}.npy'.format(npy_folder, img_id, prop_id, True), allow_pickle=True).item()
npy_eval_False = np.load('{}/regular_img_{}_proposal_{}_eval_{}.npy'.format(npy_folder, img_id, prop_id, False), allow_pickle=True).item()

predictor_weight = npy_eval_True['predictor_weight'].squeeze((2, 3))
predictor_bias   = np.expand_dims(npy_eval_True['predictor_bias'], axis=0)

# label a for eval_True
z_a = npy_eval_True['ft']
logit_a = npy_eval_True['logit']
label_a = npy_eval_True['label']
sseg_pred_a = change_sseg_pred(np.argmax(logit_a, axis=2))

# label b for eval_False
z_b = npy_eval_False['ft']
logit_b = npy_eval_False['logit']
label_b = npy_eval_False['label']
sseg_pred_b = change_sseg_pred(np.argmax(logit_b, axis=2))

# pick out 3 mask
mask_a = (sseg_pred_a == 0) # eval True, background
mask_b = (sseg_pred_b == 0) # eval False, background
mask_c = (sseg_pred_b == 1) # eval False, object
mask_ab = np.logical_and(mask_a, mask_b) # both background
mask_ac = np.logical_and(mask_a, mask_c) # first background, then object

z_eval_True_bg = z_a[mask_ab == 1]
z_eval_True_obj = z_a[mask_ac == 1]
z_eval_False_bg = z_b[mask_ab == 1]
z_eval_False_obj = z_b[mask_ac == 1]

dist_True_bg = np.mean(z_eval_True_bg.dot(predictor_weight.T) + predictor_bias, axis=0)
dist_True_obj = np.mean(z_eval_True_obj.dot(predictor_weight.T) + predictor_bias, axis=0)
dist_False_bg = np.mean(z_eval_False_bg.dot(predictor_weight.T) + predictor_bias, axis=0)
dist_False_obj = np.mean(z_eval_False_obj.dot(predictor_weight.T) + predictor_bias, axis=0)

dist_classes = predictor_weight.dot(predictor_weight.T) + predictor_bias.T

assert 1==2

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
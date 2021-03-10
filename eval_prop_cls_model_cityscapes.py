import numpy as np

rep_style = 'ObjDet'
dataset = 'cityscapes'

result_folder = 'cls_results/prop_cls_more_class_old/obj_sseg_duq/{}/{}'.format(rep_style, dataset)
num_files = 50

acc_list = np.zeros(num_files)
for i in range(num_files):
	result = np.load('{}/img_{}.npy'.format(result_folder, i), allow_pickle=True).item()

	pred = result['pred']
	label = result['label']

	acc = (1.0*np.sum(pred==label)/pred.shape[0])
	acc_list[i] = acc

mean_acc = np.mean(acc_list)
print('rep_style = {}, dataset = {}, mean_acc = {}'.format(rep_style, dataset, mean_acc))
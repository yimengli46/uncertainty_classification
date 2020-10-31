import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from classifier import ResNet, ResNetDropout, enable_dropout
from cityscapes_dataloader import CityscapesDataset, CITYSCAPES_CLASSES, LostAndFoundDataset
from scipy.stats import entropy
from util import softmax
from tqdm import tqdm

'''
num_classes = len(CITYSCAPES_CLASSES)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])

# Transforms applied to the testing data
test_transform = transforms.Compose([
	transforms.Resize(227),
	transforms.CenterCrop(227),
	transforms.ToTensor(),
	normalize,
	])

ds_cityscapes = CityscapesDataset('/home/yimeng/Datasets/Cityscapes', 'val', test_transform, random_crops=0)
ds_outlier = LostAndFoundDataset(test_transform)

device = torch.device("cuda:1")
BATCH_SIZE = 1
num_forward = 10

cityscapes_loader = torch.utils.data.DataLoader(dataset=ds_cityscapes, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
outlier_loader = torch.utils.data.DataLoader(dataset=ds_outlier, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# load the classifiers
resNet_classifier = ResNet(num_classes).to(device)
resNet_classifier.load_state_dict(torch.load('trained_model/{}.pth'.format('resNet_42_epochs')))
resNet_classifier.eval()
enable_dropout(resNet_classifier)

resNetDropout_classifier = ResNetDropout(num_classes).to(device)
resNetDropout_classifier.load_state_dict(torch.load('trained_model/{}.pth'.format('resNet_all_Dropout_51_epochs')))
resNetDropout_classifier.eval()
enable_dropout(resNetDropout_classifier)
'''

'''
# evaluate on cityscapes
cityscapes_result_matrix = np.zeros((len(cityscapes_loader), 6))
with torch.no_grad():
	for i, (images, labels, images_np) in tqdm(enumerate(cityscapes_loader)):
		#if i % 100 == 0:
		#	print('i = {}'.format(i))
		images, labels, images_np = images.to(device), labels.numpy()[0], images_np.numpy()[0]

		resNet_logits = resNet_classifier(images).cpu().numpy()
		resNet_predicted_class = np.argmax(resNet_logits, axis=1)[0]
		resNet_entropy = entropy(softmax(resNet_logits)[0])

		#print()
		resNetDropout_logits = np.zeros((num_forward, num_classes))
		for j in range(num_forward):
			resNetDropout_logits[j, :] = resNetDropout_classifier(images).cpu().numpy()
		# take the average
		resNetDropout_predicted_class = np.argmax(np.mean(resNetDropout_logits, axis=0))
		# compute variance as uncertainty
		uncertainty_variance = np.mean(np.var(resNetDropout_logits, axis=0))
		# compute entropy as uncertainty
		uncertainty_entropy = entropy(softmax(np.mean(resNetDropout_logits, axis=0)))

		cityscapes_result_matrix[i, 0] = labels
		cityscapes_result_matrix[i, 1] = resNet_predicted_class
		cityscapes_result_matrix[i, 2] = resNet_entropy
		cityscapes_result_matrix[i, 3] = resNetDropout_predicted_class
		cityscapes_result_matrix[i, 4] = uncertainty_variance
		cityscapes_result_matrix[i, 5] = uncertainty_entropy

		## visualize some results
		if i % 200 == 0:
			plt.imshow(images_np)
			plt.title('gt_class = {}\nresNet_class = {}, resNet entropy = {:.4f}\nresNetDropout_class = {}, variance = {:.4f}, entropy = {:.4f}'.format(
				CITYSCAPES_CLASSES[labels], CITYSCAPES_CLASSES[resNet_predicted_class], resNet_entropy,
				CITYSCAPES_CLASSES[resNetDropout_predicted_class], uncertainty_variance, uncertainty_entropy))
			plt.axis('off')
			plt.savefig('{}/{}_cityscapes_result.jpg'.format('generated_images', i))
			plt.close()
		#assert 1==2
np.save('results/{}.npy'.format('cityscapes_classification_result'), cityscapes_result_matrix)
'''

'''
# evaluate on Lost&Found
outlier_result_matrix = np.zeros((len(outlier_loader), 6))
with torch.no_grad():
	for i, (images, labels, images_np) in tqdm(enumerate(outlier_loader)):
		#if i % 100 == 0:
		#	print('i = {}'.format(i))
		images, labels, images_np = images.to(device), labels.numpy()[0], images_np.numpy()[0]

		resNet_logits = resNet_classifier(images).cpu().numpy()
		resNet_predicted_class = np.argmax(resNet_logits, axis=1)[0]
		resNet_entropy = entropy(softmax(resNet_logits)[0])

		#print()
		resNetDropout_logits = np.zeros((num_forward, num_classes))
		for j in range(num_forward):
			resNetDropout_logits[j, :] = resNetDropout_classifier(images).cpu().numpy()
		# take the average
		resNetDropout_predicted_class = np.argmax(np.mean(resNetDropout_logits, axis=0))
		# compute variance as uncertainty
		uncertainty_variance = np.mean(np.var(resNetDropout_logits, axis=0))
		# compute entropy as uncertainty
		uncertainty_entropy = entropy(softmax(np.mean(resNetDropout_logits, axis=0)))

		outlier_result_matrix[i, 0] = labels
		outlier_result_matrix[i, 1] = resNet_predicted_class
		outlier_result_matrix[i, 2] = resNet_entropy
		outlier_result_matrix[i, 3] = resNetDropout_predicted_class
		outlier_result_matrix[i, 4] = uncertainty_variance
		outlier_result_matrix[i, 5] = uncertainty_entropy

		## visualize some results
		if i % 1 == 0:
			plt.imshow(images_np)
			plt.title('gt_class = {}\nresNet_class = {}, resNet entropy = {:.4f}\nresNetDropout_class = {}, variance = {:.4f}, entropy = {:.4f}'.format(
				'outlier', CITYSCAPES_CLASSES[resNet_predicted_class], resNet_entropy,
				CITYSCAPES_CLASSES[resNetDropout_predicted_class], uncertainty_variance, uncertainty_entropy))
			plt.axis('off')
			plt.savefig('{}/{}_lostAndFound_result.jpg'.format('generated_images', i))
			plt.close()

np.save('results/{}.npy'.format('lostAndFound_classification_result'), outlier_result_matrix)
'''

#'''
# analyze cityscapes
classification_result_matrix = np.load('results/{}_classification_result.npy'.format('cityscapes'))

resNet_correctly_classified_idxs = (classification_result_matrix[:, 0] == classification_result_matrix[:, 1])
resNet_correctly_classified_rows = classification_result_matrix[resNet_correctly_classified_idxs]
resNet_wrongly_classified_rows = classification_result_matrix[~resNet_correctly_classified_idxs]

resNetDropout_correctly_classified_idxs = (classification_result_matrix[:, 0] == classification_result_matrix[:, 3])
resNetDropout_correctly_classified_rows = classification_result_matrix[resNetDropout_correctly_classified_idxs]
resNetDropout_wrongly_classified_rows = classification_result_matrix[~resNetDropout_correctly_classified_idxs]

# compute classification accuracy
acc_resNet = resNet_correctly_classified_rows.shape[0] * 1.0 / classification_result_matrix.shape[0]
acc_resNetDropout =  resNetDropout_correctly_classified_rows.shape[0] * 1.0 / classification_result_matrix.shape[0]
print('acc_resNet = {}'.format(acc_resNet))
print('acc_resNetDropout = {}'.format(acc_resNetDropout))


mean_resNet_correct_entropy = np.mean(resNet_correctly_classified_rows, axis=0)[2]
std_resNet_correct_entropy = np.std(resNet_correctly_classified_rows, axis=0)[2]
mean_resNet_wrong_entropy = np.mean(resNet_wrongly_classified_rows, axis=0)[2]
std_resNet_wrong_entropy = np.std(resNet_wrongly_classified_rows, axis=0)[2]
print('resNet:')
print('mean_correct = {}, std_correct = {}, mean_wrong = {}, std_wrong = {}'.format(mean_resNet_correct_entropy, 
	std_resNet_correct_entropy, mean_resNet_wrong_entropy, std_resNet_wrong_entropy))

mean_resNetDropout_correct_uncertainty_variance = np.mean(resNetDropout_correctly_classified_rows, axis=0)[4]
std_resNetDropout_correct_uncertainty_variance = np.std(resNetDropout_correctly_classified_rows, axis=0)[4]
mean_resNetDropout_wrong_uncertainty_variance = np.mean(resNetDropout_wrongly_classified_rows, axis=0)[4]
std_resNetDropout_wrong_uncertainty_variance = np.std(resNetDropout_wrongly_classified_rows, axis=0)[4]
print('resNetDropout uncertainty_variance:')
print('mean_correct = {}, std_correct = {}, mean_wrong = {}, std_wrong = {}'.format(mean_resNetDropout_correct_uncertainty_variance, 
	std_resNetDropout_correct_uncertainty_variance, mean_resNetDropout_wrong_uncertainty_variance, 
	std_resNetDropout_wrong_uncertainty_variance))

mean_resNetDropout_correct_uncertainty_entropy = np.mean(resNetDropout_correctly_classified_rows, axis=0)[5]
std_resNetDropout_correct_uncertainty_entropy = np.std(resNetDropout_correctly_classified_rows, axis=0)[5]
mean_resNetDropout_wrong_uncertainty_entropy = np.mean(resNetDropout_wrongly_classified_rows, axis=0)[5]
std_resNetDropout_wrong_uncertainty_entropy = np.std(resNetDropout_wrongly_classified_rows, axis=0)[5]
print('resNetDropout uncertainty_entropy:')
print('mean_correct = {}, std_correct = {}, mean_wrong = {}, std_wrong = {}'.format(mean_resNetDropout_correct_uncertainty_entropy, 
	std_resNetDropout_correct_uncertainty_entropy, mean_resNetDropout_wrong_uncertainty_entropy, 
	std_resNetDropout_wrong_uncertainty_entropy))

# analyze lostAndFound
classification_result_matrix = np.load('results/{}_classification_result.npy'.format('lostAndFound'))

mean_resNet_entropy = np.mean(classification_result_matrix, axis=0)[2]
std_resNet_entropy = np.std(classification_result_matrix, axis=0)[2]

mean_resNetDropout_uncertainty_variance = np.mean(classification_result_matrix, axis=0)[4]
std_resNetDropout_uncertainty_variance = np.std(classification_result_matrix, axis=0)[4]
mean_resNetDropout_uncertainty_entropy = np.mean(classification_result_matrix, axis=0)[5]
std_resNetDropout_uncertainty_entropy = np.std(classification_result_matrix, axis=0)[5]
print('-------------------------------------------------------------------------------')
print('LostAndFound:')
print('resNet :')
print('mean entropy = {}, std = {}'.format(mean_resNet_entropy, std_resNet_entropy))
print('resNetDropout : ')
print('mean uncertainty variance = {}, std = {}'.format(mean_resNetDropout_uncertainty_variance, std_resNetDropout_uncertainty_variance))
print('mean uncertainty entropy = {}, std = {}'.format(mean_resNetDropout_uncertainty_entropy, std_resNetDropout_uncertainty_entropy))
#'''

fig, axes = plt.subplots(nrows=1, ncols=3)
ax0, ax1, ax2 = axes.flatten()

colors = ['red', 'tan', 'lime']
labels = ['correct', 'wrong', 'outlier']
nbins = 10

# for resNet
correctly_classified = resNet_correctly_classified_rows[:, 2]
wrongly_classified = resNet_wrongly_classified_rows[:, 2]
outlier_classified = classification_result_matrix[:, 2]
x = [correctly_classified, wrongly_classified, outlier_classified]
ax0.hist(x, 10, density=True, histtype='bar', color=colors, label=labels)
ax0.legend(prop={'size': 10})
ax0.set_title('resNet entropy uncertainty')

# for resNetDropout variance
correctly_classified = resNetDropout_correctly_classified_rows[:, 4]
wrongly_classified = resNetDropout_wrongly_classified_rows[:, 4]
outlier_classified = classification_result_matrix[:, 4]
x = [correctly_classified, wrongly_classified, outlier_classified]
ax1.hist(x, 10, density=True, histtype='bar', color=colors, label=labels)
ax1.legend(prop={'size': 10})
ax1.set_title('resNetDropout variance uncertainty')

# for resNetDropout entropy
correctly_classified = resNetDropout_correctly_classified_rows[:, 5]
wrongly_classified = resNetDropout_wrongly_classified_rows[:, 5]
outlier_classified = classification_result_matrix[:, 5]
x = [correctly_classified, wrongly_classified, outlier_classified]
ax2.hist(x, 10, density=True, histtype='bar', color=colors, label=labels)
ax2.legend(prop={'size': 10})
ax2.set_title('resNetDropout entropy uncertainty')

fig.tight_layout()
plt.show()

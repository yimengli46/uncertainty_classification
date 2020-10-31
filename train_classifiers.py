import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from classifier import Classifier, enable_dropout
from cityscapes_dataloader import CityscapesDataset, CITYSCAPES_CLASSES

num_classes = len(CITYSCAPES_CLASSES)

# Transforms applied to the training data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std= [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
            #transforms.Resize(227),
            #transforms.CenterCrop(227),
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

dataset_folder = '/home/yimeng/Datasets/Cityscapes'
ds_train = CityscapesDataset(dataset_folder, 'train', train_transform, random_crops=0)

# ### Loading Validation Data
# Transforms applied to the testing data
test_transform = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            normalize,
        ])

ds_val = CityscapesDataset(dataset_folder, 'val', test_transform)

# # Classification
device = torch.device("cuda:1")
BATCH_SIZE = 128

train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               num_workers=8)

val_loader = torch.utils.data.DataLoader(dataset=ds_val,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               num_workers=8)

def train_classifier(train_loader, classifier, criterion, optimizer):
    classifier.train()
    loss_ = 0.0
    losses = []
    for i, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = classifier(images)
        #print('logits = {}'.format(logits))
        #print('labels = {}'.format(labels))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return torch.stack(losses).mean().item()

def test_classifier(test_loader, classifier, criterion, print_ind_classes=True):
    classifier.eval()
    enable_dropout(classifier) # convert dropout layer to train condition

    losses = []
    with torch.no_grad():
        y_true = np.zeros((0, num_classes))
        y_score = np.zeros((0, num_classes))
        for i, (images, labels, _) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            labels = labels.cpu().numpy()
            lbl = np.zeros((len(labels), num_classes))
            
            for idx, label in enumerate(labels):
                lbl[idx, label] = 1
            y_true = np.concatenate((y_true, lbl), axis=0)
            #print('y_true = {}'.format(y_true))

            logits = classifier(images)
            y_score = np.concatenate((y_score, logits.cpu().numpy()), axis=0)
            #print('y_score.shape = {}'.format(y_score.shape))

        aps = []
        for i in range(0, y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            if print_ind_classes:
                print('-------  Class: {:<12}     AP: {:>8.4f}  -------'.format(CITYSCAPES_CLASSES[i], ap))
            aps.append(ap)
        
        mAP = np.mean(aps)

        print('mAP: {0:.4f}'.format(mAP))
    return mAP, aps

classifier = Classifier(num_classes).to(device)
# You can can use this function to reload a network you have already saved previously
classifier.load_state_dict(torch.load('resNet.pth'))

criterion = nn.CrossEntropyLoss()

import torch.optim as optim
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Training the Classifier
NUM_EPOCHS = 200
TEST_FREQUENCY = 1

for epoch in range(1, NUM_EPOCHS+1):
    print("Starting epoch number " + str(epoch))
    train_loss = train_classifier(train_loader, classifier, criterion, optimizer)
    print("Loss for Training on Epoch " +str(epoch) + " is "+ str(train_loss))
    if(epoch%TEST_FREQUENCY==0):
        mAP_val, _ = test_classifier(val_loader, classifier, criterion, print_ind_classes=True)
        print('Evaluating classifier')
        print("Mean Precision Score for Testing on Epoch " +str(epoch) + " is "+ str(mAP_val))
        # Save the clssifier network
        # Suggestion: you can save checkpoints of your network during training and reload them later
        torch.save(classifier.state_dict(), './temp_classifier.pth')


## Evaluate on test set
ds_test = CityscapesDataset(dataset_folder,'val', test_transform)

test_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                               batch_size=50, 
                                               shuffle=False,
                                               num_workers=1)

mAP_test, test_aps = test_classifier(test_loader, classifier, criterion)


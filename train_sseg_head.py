import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import SSegHead
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset

dataset_folder = '/home/reza/ARGO_datasets/Cityscapes'
ds_train = CityscapesProposalsDataset(dataset_folder, 'train')
num_classes = ds_train.NUM_CLASSES
ds_val = CityscapesProposalsDataset(dataset_folder, 'val')

# # Classification
device = torch.device('cuda')

def train_classifier(train_loader, classifier, criterion, optimizer):
    classifier.train()
    loss_ = 0.0
    losses = []
    for i in range(len(train_loader)):
        images, labels = train_loader[i]
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = classifier(images)
        
        N, C, _, _ = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.reshape(-1)

        print('logits.shape = {}'.format(logits.shape))
        print('labels.shape = {}'.format(labels.shape))

        logits = logits[labels<255]
        labels = labels[labels<255]

        print('logits.shape = {}'.format(logits.shape))
        print('labels.shape = {}'.format(labels.shape))
        print('max_labels = {}'.format(torch.max(labels)))

        loss = criterion(logits, labels.long())
        #print('loss = {}'.format(loss))
        loss.backward()
        optimizer.step()

        loss = loss.item()
        losses.append(loss)
        print('i = {}, loss = {}'.format(i, loss))
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

classifier = SSegHead(num_classes).to(device)
# You can can use this function to reload a network you have already saved previously
#classifier.load_state_dict(torch.load('resNet.pth'))

criterion = nn.CrossEntropyLoss().to(device)

import torch.optim as optim
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Training the Classifier
NUM_EPOCHS = 200
TEST_FREQUENCY = 1

for epoch in range(1, NUM_EPOCHS+1):
    print("Starting epoch number " + str(epoch))
    train_loss = train_classifier(ds_train, classifier, criterion, optimizer)
    print("Loss for Training on Epoch " +str(epoch) + " is "+ str(train_loss))
    if(epoch%TEST_FREQUENCY==0):
        mAP_val, _ = test_classifier(ds_val, classifier, criterion)
        #print('Evaluating classifier')
        #print("Mean Precision Score for Testing on Epoch " +str(epoch) + " is "+ str(mAP_val))
        # Save the clssifier network
        # Suggestion: you can save checkpoints of your network during training and reload them later
        torch.save(classifier.state_dict(), './temp_classifier.pth')


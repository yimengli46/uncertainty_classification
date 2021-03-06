import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from classifier_model import DropoutHead
from dataloaders.cityscapes_classification import CityscapesClassificationDataset

BATCH_SIZE = 64
rep_style = 'ObjDet' #'both', 'ObjDet', 'SSeg'
saved_folder = 'trained_model/prop_classification_old/dropout/{}'.format(rep_style)

print('saved_folder = {}'.format(saved_folder))

if rep_style == 'both':
    input_dim = 512
else:
    input_dim = 256

dataset_folder = '/projects/kosecka/yimeng/Datasets/Cityscapes'
ds_train = CityscapesClassificationDataset(dataset_folder, 'train', batch_size=BATCH_SIZE, rep_style=rep_style)
num_classes = ds_train.NUM_CLASSES
ds_val = CityscapesClassificationDataset(dataset_folder, 'val', batch_size=BATCH_SIZE, rep_style=rep_style)

# # Classification
device = torch.device('cuda')

def train_classifier(train_loader, classifier, criterion, optimizer):
    classifier.train()
    loss_ = 0.0
    epoch_loss = []
    for i in range(len(train_loader)):
        #if i > 30:
        #    break

        images, masks, labels = train_loader[i]
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        #print('mask.shape = {}'.format(mask.shape))
        #print('mask.dtype = {}'.format(mask.dtype))

        optimizer.zero_grad()
        logits = classifier(images, masks) # logits.shape = batch_size x 5
        #print('logits.shape = {}'.format(logits.shape))
        #print('labels.shape = {}'.format(labels.shape))
        #print('labels = {}'.format(labels))
        loss = criterion(logits, labels.long())
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        print('i = {}, loss = {:.4f}'.format(i, loss.item()))
    return np.mean(epoch_loss)

def test_classifier(test_loader, classifier, criterion):
    with torch.no_grad():
        classifier.eval()
        epoch_loss = []
        for i in range(len(test_loader)):
            #if i > 30:
            #    break

            images, masks, labels = test_loader[i]
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)

            logits = classifier(images, masks)

            loss = criterion(logits, labels.long())
            epoch_loss.append(loss.item())
            print('i = {}, loss = {:.3f}'.format(i, loss.item()))

    return np.mean(epoch_loss)

classifier = DropoutHead(num_classes, input_dim).to(device)
# You can can use this function to reload a network you have already saved previously
#classifier.load_state_dict(torch.load('resNet.pth'))

criterion = nn.CrossEntropyLoss().to(device)

import torch.optim as optim
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# Training the Classifier
NUM_EPOCHS = 200

best_loss = 1000.0
for epoch in range(1, NUM_EPOCHS+1):
    print("Starting epoch number " + str(epoch))
    train_loss = train_classifier(ds_train, classifier, criterion, optimizer)
    print("Loss for Training on Epoch " +str(epoch) + " is "+ str(train_loss))

    test_loss = test_classifier(ds_val, classifier, criterion)
    print('Validation: Epoch: {}, Test Loss: {:.3f}'.format(epoch, test_loss))

    new_loss = test_loss
    if new_loss < best_loss:
        best_loss = new_loss
        torch.save(classifier.state_dict(), '{}/dropout_classifier.pth'.format(saved_folder))

    scheduler.step(train_loss)


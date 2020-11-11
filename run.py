#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OPTIONS = [True, False]
MODELS = ['alexnet', 'resnet','densenet']
NUM_EPOCHS = 20

# three different transformations: 'basic', 'crop', 'crop & distorted'
data_transforms = {
    'base': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'train_distorted': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
}


# first read all imgs using three transformations
imgs = datasets.ImageFolder('data', transform=data_transforms['base'])
cropped = datasets.ImageFolder('data', transform=data_transforms['train'])
distorted = datasets.ImageFolder('data', transform=data_transforms['train_distorted'])


# divide data in each class into 5 folds
# using 0, 1, 2, 3, 4 to index each example in every class, and repeat 3 times
splits = np.concatenate([np.random.permutation(
    np.arange(5).repeat(2)) for i in range(3)])


def get_five_folds_indices():
    # then for each fold
    # Overall train: test: valid = 6: 2: 2
    for i in range(5):
        # the indices of testing examples are exactly equal to the folder index
        test_indices = np.where(np.isin(splits, [i, ]))[0]
        # +1 for valid
        valid_indices = np.where(np.isin(splits, [(i+1) % 5, ]))[0]
        # the remaiming for training
        train_indices = np.where(~np.isin(splits, [i, (i+1) % 5, ]))[0]
        yield train_indices, test_indices, valid_indices


# use generate to give splitted data
def get_five_folds_data(data_augmentation):
    for i in range(5):
        # get indices
        train_indices, test_indices, valid_indices = next(
            get_five_folds_indices())

        # split subdataset by respective indices
        train, test, valid = torch.utils.data.Subset(imgs, train_indices),\
            torch.utils.data.Subset(imgs, test_indices),\
            torch.utils.data.Subset(imgs, valid_indices)

        # get augmented train and valid data
        cropped_train = torch.utils.data.Subset(cropped, train_indices)
        distorted_train = torch.utils.data.Subset(distorted, train_indices)
        cropped_valid = torch.utils.data.Subset(cropped, valid_indices)
        distorted_valid = torch.utils.data.Subset(distorted, valid_indices)

        # add additional train and valid data if data_augmentation
        if data_augmentation:
            train = torch.utils.data.ConcatDataset(
                [train, cropped_train, distorted_train])
            valid = torch.utils.data.ConcatDataset(
                [valid, cropped_valid, distorted_valid])

        # converted to data loader
        train, test, valid = [torch.utils.data.DataLoader(
            d, batch_size=2, shuffle=True) for d in [train, test, valid]]

        # return
        yield (train, test, valid)


def train_model(model, criterion, optimizer, scheduler, data, num_epochs=NUM_EPOCHS):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # unpack the data
    train, test, valid = data[0], data[1], data[2]

    # save the loss of each epoch
    epoch_results = []

    for epoch in range(num_epochs):

        # train phase, enable gradient calculation
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train.dataset)
        epoch_acc = running_corrects.float() / len(train.dataset)

        # valid phase, stop gradient calculation
        model.eval()

        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in valid:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        valid_loss = running_loss / len(valid.dataset)
        valid_acc = running_corrects.float() / len(valid.dataset)

        # save the model if get best result on valid set
        if valid_acc >= best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print('Epoch {} --- {} Loss: {:.3f} Acc: {:.3f}; {} Loss: {:.3f} Acc: {:.3f}'.format(
            epoch, 'Train', epoch_loss, epoch_acc, 'Valid', valid_loss, valid_acc), end='\r')
        epoch_results.append([epoch_loss, valid_loss])

    # restore the best parematers
    model.load_state_dict(best_model_wts)

    # Test phase
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in test:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test.dataset)
    test_acc = running_corrects.float() / len(test.dataset)

    print('\n\n{} Loss: {:.4f} Acc: {:.4f}\n'.format(
        'Test', test_loss, test_acc))

    # return the results
    return test_loss, test_acc, epoch_results


def train_one_fold(data, model_name):

    # different pretrained models have different settings for classifier layer
    # reference: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    if model_name == 'resnet':
        model = models.resnet18(pretrained=True)
        fc_in = model.fc.in_features
        model.fc = nn.Linear(fc_in, 3)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        fc_in = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(fc_in, 3)
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        fc_in = model.classifier.in_features
        model.classifier = nn.Linear(fc_in, 3)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # SGD with momentim was the best performer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9)

    # decay the learning rate every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    # return the test acc of this fold
    return train_model(model, criterion, optimizer, exp_lr_scheduler, data,
                       num_epochs=NUM_EPOCHS)

def plot_results():
    for option in OPTIONS:
        one_fold_data = next(get_five_folds_data(data_augmentation=option))
        l, a, r = train_one_fold(one_fold_data, model_name='densenet')
        l, a, r2 = train_one_fold(one_fold_data, model_name='alexnet')
        l, a, r3 = train_one_fold(one_fold_data, model_name='resnet')
        dense = pd.DataFrame(r, columns=['train loss', 'valid loss'])
        alex = pd.DataFrame(r2, columns=['train loss', 'valid loss'])
        res = pd.DataFrame(r3, columns=['train loss', 'valid loss'])
        df=pd.DataFrame({'x': range(1, NUM_EPOCHS+1), 'dense_train': dense['train loss'], 'alex_train': alex['train loss'], 'res_train': res['train loss'], 'dense_valid': dense['valid loss'], 'alex_valid': alex['valid loss'], 'res_valid': res['valid loss']})
        if option:
            fig_title = 'Train/Validation Losses, With Augmentation'
            fig_name = 'aug.png'
        else:
            fig_title = 'Train/Validation Losses, No Augmentation'
            fig_name = 'no_aug.png'
        plt.figure(figsize=(15,10))
        plt.plot('x', 'dense_train', data=df, marker='', markerfacecolor='blue', color='blue', linewidth=2, label="Dense Train")
        plt.plot('x', 'alex_train', data=df, marker='', color='red', linewidth=2, label="Alex Train")
        plt.plot('x', 'res_train', data=df, marker='', color='orange', linewidth=2, label="ResNet Train")
        plt.plot('x', 'dense_valid', data=df, marker='', markerfacecolor='blue', color='blue', linewidth=2, linestyle='dashed', label="Dense valid")
        plt.plot('x', 'alex_valid', data=df, marker='', color='red', linewidth=2, linestyle='dashed', label="Alex valid")
        plt.plot('x', 'res_valid', data=df, marker='', color='orange', linewidth=2, linestyle='dashed', label="ResNet valid")
        plt.title(fig_title, fontsize=18)
        plt.xlabel('Epoch', fontsize=15)
        plt.xticks(np.arange(0, 21, step=4))
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=12)
        plt.savefig(fig_name)

if __name__ == "__main__":
    for option in OPTIONS:
        accs = []
        losses = []
        results = []
        print("Include Data Augmentation? ", option)
        for model in MODELS:
            for one_fold_data in get_five_folds_data(data_augmentation=option):
                l, a, r = train_one_fold(one_fold_data, model_name=model)
                losses.append(l)
                accs.append(a)
                results.append(r)
            print('Model Name: ', model)
            print('The 5-folds accuracy: %.3f' % torch.Tensor(accs).mean().numpy())
            print('The 5-folds loss: %.3f' % torch.Tensor(losses).mean().numpy())
    plot_results()

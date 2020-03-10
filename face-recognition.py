#!/usr/bin/env python
# coding: utf-8

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as I


DATA_DIR = './data/train-test-data/'
TRAIN_DIR = DATA_DIR + 'training/'
TEST_DIR = DATA_DIR + 'test/'


class FacesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.key_pts = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.key_pts)
    
    def __getitem__(self, i):
        image_name = os.path.join(self.root_dir, self.key_pts.iloc[i, 0])
        image = mpimg.imread(image_name)
        
        if (image.shape[2] == 4):
            image = image[:, :, 0:3]
        
        key_pts = self.key_pts.iloc[i, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


class Normalize(object):
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        image_copy = image.copy()
        key_pts_copy = key_pts.copy()
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_copy = image_copy / 255.0
        key_pts_copy = (key_pts_copy - 100) / 50.0
        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        key_pts = key_pts * [new_w / w, new_h / h]
        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3) 
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)   
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       

        self.fc1 = nn.Linear(in_features=51200, out_features=4096)        
        self.fc2 = nn.Linear(in_features=4096, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=136) 
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)


        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.bn6 = nn.BatchNorm2d(num_features=256)
        self.bn7 = nn.BatchNorm2d(num_features=512)
        self.bn8 = nn.BatchNorm2d(num_features=512)
        self.bn9 = nn.BatchNorm1d(num_features=4096)
        self.bn10 = nn.BatchNorm1d(num_features=1000)       

    def forward(self, x):    
        x = F.relu(self.conv1(x))
        x= self.bn1(x)
        x = F.relu(self.conv2(x))
        x= self.bn2(x)
        x= self.pool(x)        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.pool(self.bn6(F.relu(self.conv6(x))))        
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.pool(self.bn8(F.relu(self.conv8(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bn9(x)
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn10(x)
        x = self.dropout5(x)
        
        x = self.fc3(x)
        
        return x


face_dataset = FacesDataset(csv_file=DATA_DIR + 'training_frames_keypoints.csv',
                            root_dir=TRAIN_DIR)

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

transformed_dataset = FacesDataset(csv_file=DATA_DIR + 'training_frames_keypoints.csv',
                                   root_dir=TRAIN_DIR,
                                   transform=data_transform)

batch_size = 16

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)


test_dataset = FacesDataset(csv_file=DATA_DIR + 'test_frames_keypoints.csv',
                            root_dir=TEST_DIR,
                            transform=data_transform)

test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)


def run_model(model, criterion, optimizer, running_mode='train',
              train_loader=None, valid_loader=None, test_loader=None,
              n_epochs=1, stop_thr=1e-4, device=torch.device('cpu')):
    if running_mode == 'train':
        loss = {'train': [], 'valid': []}
        
        prev_loss = np.Inf
        for epoch in range(n_epochs):
            model, train_loss = _train(model, criterion, optimizer, train_loader, device=device)
            loss['train'].append(train_loss)

            if not valid_loader:
                continue
            else:
                valid_loss = _test(model, criterion, valid_loader, device=device)
                loss['valid'].append(valid_loss)

                print(f'====== Epoch {epoch + 1}, training loss {train_loss}, valid loss {valid_loss} ======')
                
                if np.abs(valid_loss - prev_loss) < stop_thr:
                    break
                prev_loss = valid_loss
        return model, loss
    elif running_mode == 'test':
        return _test(model, criterion, test_loader)


def _train(model, criterion, optimizer, data_loader, device=torch.device('cpu')):
    model.to(device)
    model.train()
    
    train_loss = []
    for i, data in enumerate(data_loader):
        images = data['image']
        key_pts = data['keypoints']
        images = images.type(torch.FloatTensor).to(device)
        key_pts = key_pts.view(key_pts.size(0), -1)
        key_pts = key_pts.type(torch.FloatTensor).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, key_pts)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item() * images.data.size(0))

        if i % 50 == 0:
            print(f'Data index {i} training loss {loss.item() * images.data.size(0)}')
    
    return model, np.array(train_loss).mean()


def _test(model, criterion, data_loader, device=torch.device('cpu')):
    model.to(device)
    model.eval()
    
    test_loss = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images = data['image']
            key_pts = data['keypoints']
            images = images.type(torch.FloatTensor).to(device)
            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, key_pts)
            test_loss.append(loss.item() * images.data.size(0))

            if i % 50 == 0:
                print(f'Data index {i} valid loss {loss.item() * images.data.size(0)}')
    
    return np.array(test_loss).mean()


lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


model, loss = run_model(model, criterion, optimizer,
                        train_loader=train_loader, valid_loader=test_loader,
                        n_epochs=1, device=device)

epoch = len(loss['train'])
valid_loss = loss['valid'][-1]
torch.save(model, f'./model/epoch{epoch}_loss{valid_loss}.pth')

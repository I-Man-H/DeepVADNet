import torch
import torch.nn as nn
import time
import os
from PIL import Image
from torch.utils import data
import numpy as np
import pandas as pd
from torchvision import transforms as T
import io
import zipfile
from torchnet import meter
from torch.utils.data import DataLoader
import scipy.io as scio



class MAHNOB(data.Dataset):
    def __init__(self, modal='facebio', kind='train'):
        self.modal = modal
        self.kind = kind
        root = './drive/MyDrive/DeepVADNet/data/MAHNOB/'
        if self.kind == 'train':
            self.data = scio.loadmat(root+'mahnob_train.mat')
        elif self.kind == 'test':
            self.data = scio.loadmat(root+'mahnob_test.mat')
        else:
            self.data = scio.loadmat(root+'mahnob_val.mat')
    
    def __getitem__(self, i):
        face_data = self.data['img'][i]
        bio_data = self.data['bio'][i]
        labels = self.data['labels'][i]
        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'peri':
            data = bio_data[32:]
        elif self.modal == 'bio':
            data = bio_data
        elif self.modal == 'face_eeg':
            data = (face_data, bio_data[:32])
        elif self.modal == 'face_peri':
            data = (face_data, bio_data[32:])
        elif self.modal == 'face_bio':
            data = (face_data, bio_data)

        valence = labels[3]
        arousal = labels[4]
        control = labels[5]
        emotion = labels[6]
        emotion_label_map = {0: 3, 1: 5,
                             2: 2, 3: 6, 4: 1, 5: 0, 6: 7, 11: 4, 12: 8}
        emotion = emotion_label_map[emotion]
        return data, torch.Tensor([valence, arousal, control]), emotion

    def __len__(self):
        return len(self.data['labels'])


class DEAP(data.Dataset):
    def __init__(self, modal='facebio', kind='train'):
        self.modal = modal
        self.kind = kind
        root = './drive/MyDrive/DeepVADNet/data/DEAP/'
        if self.kind == 'train':
            # self.data = scio.loadmat(root+'mahnob_train.mat')
            self.imgs = np.load(root+'./deap_train_imgs.npy')
            self.bios = np.load(root+'./deap_train_bios.npy')
            self.labels = np.load(root+'./deap_train_labels.npy')
        elif self.kind == 'test':
            self.imgs = np.load(root+'./deap_test_imgs.npy')
            self.bios = np.load(root+'./deap_test_bios.npy')
            self.labels = np.load(root+'./deap_test_labels.npy')
            # self.data = scio.loadmat(root+'mahnob_test.mat')
        else:
            self.imgs = np.load(root+'./deap_val_imgs.npy')
            self.bios = np.load(root+'./deap_val_bios.npy')
            self.labels = np.load(root+'./deap_val_labels.npy')
            # self.data = scio.loadmat(root+'mahnob_val.mat')
    
    def __getitem__(self, i):
        face_data = self.imgs[i]
        bio_data = self.bios[i]
        labels = self.labels[i]
        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'peri':
            data = bio_data[32:]
        elif self.modal == 'bio':
            data = bio_data
        elif self.modal == 'face_eeg':
            data = (face_data, bio_data[:32])
        elif self.modal == 'face_peri':
            data = (face_data, bio_data[32:])
        elif self.modal == 'face_bio':
            data = (face_data, bio_data)

        valence = labels[4]
        arousal = labels[5]
        control = labels[6]
        emotion = labels[3]
        # emotion_label_map = {0: 3, 1: 5,
                            #  2: 2, 3: 6, 4: 1, 5: 0, 6: 7, 11: 4, 12: 8}
        # emotion = emotion_label_map[emotion]
        return data, torch.Tensor([valence, arousal, control]), emotion

    def __len__(self):
        return len(self.labels)

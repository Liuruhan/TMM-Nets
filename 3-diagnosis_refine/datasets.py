import glob
import random
import os
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch
from PIL import Image
#import torchvision.transforms as transforms

def get_filename(path, filetype):
    name = []
    for root, dirs, files in os.walk(path):
        for i in files:
            if filetype + ' ' in i + ' ':
                name.append(i)
    return name

class ImageDataset(Dataset):
    def __init__(self, root, size, unaligned=False, mode='train'):
        #self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.FAF_root = os.path.join(root, '%s/FAF' % mode)+'/'
        self.FP_root = os.path.join(root, '%s/FP' % mode)+'/'
        self.FAF_fold_list= []
        self.FP_fold_list = []
        self.size = size

        FAF_fold_list = os.listdir(self.FAF_root)
        for i in range(len(FAF_fold_list)):
            if os.path.isfile(self.FAF_root+FAF_fold_list[i]):
                #print(self.FAF_root+FAF_fold_list[i], FAF_fold_list[i][-3:])
                if FAF_fold_list[i][-3:] == 'png':
                    self.FAF_fold_list.append(FAF_fold_list[i])

        FP_fold_list = os.listdir(self.FP_root)
        for i in range(len(FP_fold_list)):
            if os.path.isfile(self.FP_root + FP_fold_list[i]):
                #print(self.FP_root + FP_fold_list[i], FP_fold_list[i][-3:])
                if FP_fold_list[i][-3:] == 'png':
                    self.FP_fold_list.append(FP_fold_list[i])
        #print('FP len:', len(self.FP_fold_list), 'FA len:', len(self.FAF_fold_list))
        self.resample()
        #print('FP len:', len(self.FP_fold_list), 'FA len:', len(self.FAF_fold_list))

    def resample(self):
        num_list = np.zeros(4)
        list = [[],[],[],[]]
        for i in range(len(self.FAF_fold_list)):
            if self.FAF_fold_list[i][0] == '0':
                num_list[0] += 1
                list[0].append(self.FAF_fold_list[i])
            else:
                num_list[1] += 1
                list[1].append(self.FAF_fold_list[i])
        for i in range(len(self.FP_fold_list)):
            if self.FP_fold_list[i][0] == '0':
                num_list[2] += 1
                list[2].append(self.FP_fold_list[i])
            else:
                num_list[3] += 1
                list[3].append(self.FP_fold_list[i])

        num_list = num_list.astype(int)

        #print("FAF-0:", num_list[0], "FAF-1:", num_list[1], "FP-0:", num_list[2], "FP-1:", num_list[3])
        max_value = np.max(np.array(num_list))
        resample_FP_list = []
        resample_FAF_list = []
        for i in range(2):
            if max_value % num_list[i] == 0:
                for j in range(len(list[i])):
                    resample_FAF_list.append(list[i][j])
            else:
                for j in range(int(max_value/num_list[i])):
                    for k in range(num_list[i]):
                        resample_FAF_list.append(list[i][k])
                for k in range(max_value % num_list[i]):
                    resample_FAF_list.append(list[i][k])

        for i in range(2, 4):
            if max_value % num_list[i] == 0:
                for j in range(len(list[i])):
                    resample_FP_list.append(list[i][j])
            else:
                for j in range(int(max_value/num_list[i])):
                    for k in range(num_list[i]):
                        resample_FP_list.append(list[i][k])
                for k in range(max_value % num_list[i]):
                    resample_FP_list.append(list[i][k])
        #print(len(resample_FAF_list), len(resample_FP_list))

        self.FP_fold_list = resample_FP_list
        self.FAF_fold_list = resample_FAF_list
        return

    def generate_tensor(self, root, name):
        img = cv2.imread(root + name)
        img = np.array(cv2.resize(img, (self.size, self.size)))
        img = img / 255.0
        img = (img - 0.5)/0.5
        img = np.expand_dims(img, axis=0)
        img = np.swapaxes(img, 1, 3)
        img = np.swapaxes(img, 3, 2)
        img_tensor = torch.from_numpy(img)

        if name[0] == '0':
            label = np.array([0, 1])
        elif name[0] == '1':
            label = np.array([1, 0])
        return img_tensor, label

    def __getitem__(self, index):
        FAF_fold_name = self.FAF_fold_list[index]
        FP_fold_name = self.FP_fold_list[index]

        FAF_tensor, FAF_label = self.generate_tensor(self.FAF_root, FAF_fold_name)
        FP_tensor, FP_label = self.generate_tensor(self.FP_root, FP_fold_name)
        #print(FAF_tensor.size(), FP_tensor.size())

        FAF_num = torch.FloatTensor(FAF_label.astype(np.float32))
        FP_num = torch.FloatTensor(FP_label.astype(np.float32))
        #print(FAF_num, FP_num)

        return {'FAF': FAF_tensor, 'FP': FP_tensor, 'FAF_label': FAF_num, 'FP_label': FP_num}

    def __len__(self):
        return max(len(self.FAF_fold_list), len(self.FP_fold_list))
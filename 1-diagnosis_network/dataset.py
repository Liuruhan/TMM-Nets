import re
from os.path import splitext
from os import listdir
import numpy as np
import random as rd
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
#import cv2

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, n_classes, scale=1, resample=True):
        self.imgs_dir = imgs_dir
        #self.img_files = listdir(imgs_dir)
        self.scale = scale
        self.n_classes = n_classes

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        if resample == True:
            self.ids = self.resampler(ids)
        else:
            self.ids = ids

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        #print(pil_img.size)
        img_nd = np.array(pil_img)
        #print(img_nd.shape)

        if len(img_nd.shape) == 2 :
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        #if img_trans.max() == 255:
        img_trans = img_trans / 255

        if img_trans.shape[0] == 4:
            img_return = img_trans[0:3, :, :]
        else:
            img_return = img_trans
        return img_return
    @classmethod
    def resampler(cls, ids):
        np.random.shuffle(ids)
        new_ids = []
        ids_0 = []
        ids_1 = []
        #print('id:', ids)
        for i in range(len(ids)):
            if int(ids[i][:1]) == 0:
                ids_0.append(ids[i])
            elif int(ids[i][:1]) == 1:
                ids_1.append(ids[i])
        high_length = max(len(ids_0), len(ids_1))
        for i in range(high_length):
            if high_length > len(ids_0):
                new_ids.append(ids_0[np.random.randint(low=0, high=len(ids_0) - 1)])
                new_ids.append(ids_1[i])
            elif high_length > len(ids_1):
                new_ids.append(ids_0[i])
                new_ids.append(ids_1[np.random.randint(low=0, high=len(ids_1) - 1)])
            else:
                new_ids.append(ids_1[i])
                new_ids.append(ids_0[i])
        #print('self id:', new_ids)
        return new_ids

    def __getitem__(self, i):
        idx = self.ids[i]
        #print('index:', idx)
        img_file = glob(self.imgs_dir + idx + '.png')
        #print('mask_file:', mask_file)
        #print('img_file:', img_file)
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])
        #img = cv2.imread(img_file[0])

        img = self.preprocess(img, self.scale)
        label = int(idx[:re.search('_', idx).span()[0]])
        #label = torch.tensor(int(idx[:re.search('_', idx).span()[0]]))
        label_one_hot = np.zeros(self.n_classes)
        label_one_hot[label] = 1
        #print(img, label_one_hot)
        #print(np.max(img), np.min(img))
        #print(img.shape, label_one_hot.shape)
        #print(img_file[0], np.array(img).shape)
        #print('img:', img.shape)
        #print('mask:', mask[0].shape)
        #print('mask_M:', np.max(mask[0]))
        #print('img:', torch.from_numpy(img).type(torch.ByteTensor), 'mask:', torch.from_numpy(mask[0]).type(torch.ByteTensor))
        return {'image': torch.from_numpy(img).type(torch.FloatTensor), 'target': torch.from_numpy(label_one_hot).type(torch.FloatTensor)}# label.type(torch.LongTensor)} #}




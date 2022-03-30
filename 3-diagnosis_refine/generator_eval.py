import os
import cv2
import numpy as np

import torch
import torch.nn as nn

from models import Generator

cuda = False
input_nc = 3
output_nc = 3
weight_path = './output/9_netG_A2B.pth'
input_image_path = './patches/'
input_img_list = os.listdir(input_image_path)
output_image_path = './output_result_generator/'

netG_A2B = Generator(input_nc, output_nc)
if cuda:
    netG_A2B.cuda()
    netG_A2B = nn.DataParallel(netG_A2B)

netG_A2B.load_state_dict(torch.load(weight_path))
#print(netG_A2B)

def img2tensor(img):
    trans_img = np.swapaxes(np.array(img), 0, 2)
    trans_img = np.swapaxes(trans_img, 1, 2)
    trans_img = np.expand_dims(trans_img, axis=0)
    trans_img = trans_img / 255.0
    print(trans_img.shape)

    tensor = torch.from_numpy(trans_img)
    tensor = tensor.float()
    print(tensor.size())
    return tensor

def tensor2img(tensor):
    numpy_array = tensor.detach().numpy()
    print(numpy_array.shape)

    output_img = numpy_array[0]
    print(output_img.shape)
    output_img = np.swapaxes(output_img, 1, 2)
    output_img = np.swapaxes(output_img, 0, 2)
    print(output_img.shape)
    output_img = output_img * 255
    return output_img.astype(int)

for i in range(len(input_img_list)):
    input_img = cv2.imread(input_image_path+input_img_list[i])
    print(input_img.shape)

    input_tensor = img2tensor(input_img)
    output_tensor = netG_A2B(input_tensor)
    print('out:', output_tensor.size())

    output_img = tensor2img(output_tensor)
    print('out_img:', output_img.shape)
    cv2.imwrite(output_image_path+input_img_list[i], output_img)
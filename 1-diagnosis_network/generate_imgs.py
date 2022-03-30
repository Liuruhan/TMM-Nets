import os
import cv2
import numpy as np
import random as rd

root = './val_2class_square/'
dir_list = ['0/', '1/', '2/', '3/']

all_imgs_path = []
all_imgs_name = []
all_imgs_label = []
for i in range(len(dir_list)):
    filelist = os.listdir(root+dir_list[i])
    for j in range(len(filelist)):
        if filelist[j][-3:] == 'png':
            all_imgs_path.append(root+dir_list[i]+filelist[j])
            all_imgs_name.append(filelist[j])
            all_imgs_label.append(i)

state = np.random.get_state()
np.random.shuffle(all_imgs_path)
np.random.set_state(state)
np.random.shuffle(all_imgs_name)
np.random.set_state(state)
np.random.shuffle(all_imgs_label)

#print(all_imgs_label[:10])
#print(all_imgs_name[:10])
#print(all_imgs_path[:10])

for i in range(len(all_imgs_name)):
    img = cv2.imread(all_imgs_path[i])
    print(img.shape)
    print(str(all_imgs_label[i])+'_'+all_imgs_name[i])
    cv2.imwrite(root+str(all_imgs_label[i])+'_'+all_imgs_name[i], img)
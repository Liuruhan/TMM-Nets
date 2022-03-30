from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, wide_resnet50_2
from models.pytorch_resnet import wide_Resnet50_2, Resnet18, Resnet34
import cv2
import torch
import numpy as np
import os
cuda = True

def show_cam(cam_mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def wide_resnet50_cam(model_name='resnet18', ouput_name='exampler.png', out_cam_name='cam.png', model_path = './checkpoints/wide_resnet50_2_epoch29.pth', image_path = './FP_dataset/FP_0/test/1_fp_ab_1.png'):
    if model_name == 'wide_resnet50_2':
        model = wide_Resnet50_2(num_classes=2)
    elif model_name == 'resnet18':
        model = Resnet18(num_classes=2)
    elif model_name == 'resnet34':
        model = Resnet34(num_classes=2)

    if cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()

    para_in_save = torch.load(model_path)
    model.load_state_dict(para_in_save)

    target_layer = model.module.layer4[-1]

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    input_np = np.swapaxes(rgb_img, 1, 2)
    input_np = np.swapaxes(input_np, 0, 1)
    input_np = input_np / 255
    print(rgb_img.shape, input_np.shape)
    input_tensor = torch.from_numpy(np.expand_dims(input_np, axis=0)).type(torch.FloatTensor)# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    print('grayscale_cam shape:', grayscale_cam.shape)

    # In this example grayscale_cam has only one image in the batch:
    input_img = np.array(rgb_img) / 255
    input_img.astype(np.float32)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_img, grayscale_cam)
    cam_img = show_cam(grayscale_cam)
    cv2.imwrite(out_cam_name, cam_img)
    cv2.imwrite(ouput_name, visualization)
    return

cam_pth = './cam_wide/'
input_pth = './fp_dataset/test/'
files = os.listdir(input_pth)
ab_files = []
for i in range(len(files)):
    if int(files[i][0]) == 1:
        ab_files.append(files[i])
print(ab_files)
model_name_str = 'resnet34'
output_name = 'UWF'
#E:\SLE\child_classification\results\UWF_FP_training\checkpoints
#E:\SLE\child_classification\results\regular_fundus--UWF_FP_training\checkpoints
#E:\SLE\child_classification\results\diagnosis_with_aug\checkpoints
#E:\SLE\child_classification\weights
if not os.path.exists(cam_pth):
    os.mkdir(cam_pth)

for name in range(49):
    if not os.path.exists(cam_pth+model_name_str+'/'+output_name+'/'+str(name)+'/'):
        os.mkdir(cam_pth+model_name_str+'/'+output_name+'/'+str(name)+'/')
    for i in range(len(ab_files)):
        print(cam_pth+model_name_str+'/'+output_name+'/'+str(name)+'/'+ab_files[i])
        wide_resnet50_cam(model_name=model_name_str, ouput_name=cam_pth+model_name_str+'/'+output_name+'/'+str(name)+'/'+ab_files[i][:-4] + 'heatmap.png',
                          out_cam_name=cam_pth+model_name_str+'/'+output_name+'/'+str(name)+'/'+ab_files[i][:-4] + 'cam.png',
                      #model_path='./weights/'+str(name)+'_finetune_netDiagnosis.pth',
                      model_path='./results/UWF_FP_training/checkpoints/'+model_name_str+'_epoch' + str(name+2) + '.pth',
                      #model_path='./results/regular_fundus--UWF_FP_training/checkpoints/'+model_name_str+'_epoch' + str(
                      #        name + 2) + '.pth',
                      #model_path='./results/diagnosis_with_aug/checkpoints/'+ model_name_str + '_epoch' + str(
                      #      name + 2) + '.pth',
                      image_path=input_pth+ab_files[i])

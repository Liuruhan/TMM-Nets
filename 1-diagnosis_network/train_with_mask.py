import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchnet import meter
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from models.pytorch_resnet import Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, Resnext50_32x4d, Resnext101_32x8d, wide_Resnet50_2, wide_Resnet101_2
from models.pytorch_mask_resnet import mask_Resnet18, mask_Resnet34, mask_Resnet50, mask_Resnet101, mask_Resnet152, mask_Resnext50_32x4d, mask_Resnext101_32x8d, mask_wide_Resnet50_2, mask_wide_Resnet101_2
from models.pytorch_densenet import Densenet121, Densenet161, Densenet169, Densenet201
from models.pytorch_inceptionV3 import Inception_v3
from models.pytorch_vgg import Vgg11, Vgg11_bn, Vgg13, Vgg13_bn, Vgg16, Vgg16_bn, Vgg19, Vgg19_bn
from models.pytorch_squeezenet import Squeezenet1_1, Squeezenet1_0
from models.pytorch_mobilenetV2 import Mobilenet_v2
from models.scse_unet import SCSERes
#from dataset import BasicDataset
#from dataset_patches import BasicDataset
from dataset_mask import BasicDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from segLoss import SegmentationLosses
ultra_DR_learning_rate = 0.000001
#other_learning_rate = 0.000001
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', type=int, default=4, metavar='B',
                        help='input batch size for training (default: 64)', dest='batch_size')
    parser.add_argument('-tb', '--val-batch-size', type=int, default=2, metavar='TB',
                        help='input batch size for valing (default: 1000)')
    parser.add_argument('-e', '--epochs', type=int, default=50, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-l', '--lr', type=float, default=ultra_DR_learning_rate, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('-c', '--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-log', '--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-p', '--train-prob', type=float, default=0.8, metavar='train_prob',
                        help='the probability of training used')
    parser.add_argument('-cp', '--save-cp', type=bool, default=True, metavar='save_cp',
                        help='is save the model?')
    parser.add_argument('-dir', '--cp-dir', type=str, default='./checkpoints/', metavar='cp_dir',
                        help='the path of saving models')
    parser.add_argument('-nclasses', '--num-classes', type=int, default=2, metavar='Num',
                        help='how many class to identify')
    return parser.parse_args()

def plot_AUC(pred, target, n_classes, epoch, path, din_color, k_fold):
    pred_np = np.argmax(pred, axis=1)

    #np.savetxt('./checkpoints/'+str(epoch)+'pred.csv', pred_np)
    n_target = np.eye(n_classes)[target.astype(int)]
    #print(n_target.shape, pred.shape)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(n_target[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(n_target.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    lw = 2
    # 此部分micro-average ROC curve，即平均的AUC值
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='chocolate', linestyle=':', linewidth=3)
    plt.legend(loc="lower right")

    # 对三个类别进行循环分别得出各自的AUC
    if din_color == plt.cm.Blues:
        colors = cycle(['lightskyblue', 'dodgerblue', 'royalblue', 'midnightblue'])
    elif din_color  == plt.cm.Reds:
        colors = cycle(['mistyrose', 'lightcoral', 'firebrick', 'maroon'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    # 绘图
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # 绘制对角线虚线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    if not os.path.exists('FP_'+str(k_fold)+'/'):
        os.makedirs('FP_'+str(k_fold)+'/')
    if not os.path.exists('FP_'+str(k_fold)+'/'+path+'/'):
        os.makedirs('FP_'+str(k_fold)+'/'+path+'/')
    if din_color == plt.cm.Blues:
        plt.savefig('FP_'+str(k_fold)+'/'+path+'/'+str(epoch)+'_val_AUC.jpg')
    elif din_color == plt.cm.Reds:
        plt.savefig('FP_'+str(k_fold)+'/'+path + '/' + str(epoch) + '_extraVal_AUC.jpg')
    plt.close()
    return

#optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
def plot_Matrix(cm, classes, epoch, path, k_fold, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    #for row in str_cm:
    #    print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if not os.path.exists('FP_'+str(k_fold)+'/'):
        os.makedirs('FP_'+str(k_fold)+'/')
    if not os.path.exists('FP_'+str(k_fold)+'/'+path+'/'):
        os.makedirs('FP_'+str(k_fold)+'/'+path+'/')
    if cmap == plt.cm.Blues:
        plt.savefig('FP_'+str(k_fold)+'/'+path + '/' + str(epoch) + 'val_cm.jpg', dpi=300)
    elif cmap == plt.cm.Reds:
        plt.savefig('FP_'+str(k_fold)+'/'+path + '/' + str(epoch) + 'extraVal_cm.jpg', dpi=300)
    plt.close()
    #plt.show()
    return

def valdata_cal(model, args, loader, png_flag, epoch, path, n_classes, color, modelname, k_fold):
    model.eval()
    val_loss = 0
    correct = 0
    confusion_matrix = meter.ConfusionMeter(n_classes)
    out_np = np.zeros((len(loader)*args.batch_size, n_classes))
    tgt_np = np.zeros(len(loader)*args.batch_size)
    data_idx = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, batch in enumerate(loader):
        data = batch['image']
        mask = batch['mask']
        target = batch['target']
        #data = torch.transpose(data, 3, 4)
        # print(data.size(0), data.size(1), data.size(2)*data.size(3), data.size(4)*data.size(5))
        #data = data.reshape(data.size(0), data.size(1), data.size(2) * data.size(3), data.size(4) * data.size(5))
        #print(data.shape, target.shape)
        if args.cuda:
            data, mask, target = data.cuda(), mask.cuda(), target.cuda()
        data, mask, target = Variable(data), Variable(mask), Variable(target)
        output, mask_out = model(data)

        val_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target[:, 0].data.view_as(pred)).cpu().sum()

        confusion_matrix.add(output.data, target[:, 1].data)
        size = output.data.cpu().numpy().shape[0]
        out_np[data_idx:data_idx+size] = output.data.cpu().numpy()
        tgt_np[data_idx:data_idx+size] = target[:, 1].data.cpu().numpy()
        data_idx += size
    out = np.argmax(out_np, axis=1)
        #print(tgt_np.shape, out.shape)
    output_csv = np.concatenate((np.expand_dims(out, axis=1), np.expand_dims(tgt_np, axis=1)), axis=1)
    if not os.path.exists('FP_'+str(k_fold)+'/'):
        os.makedirs('FP_'+str(k_fold)+'/')
    if not os.path.exists('FP_'+str(k_fold)+'/'+modelname+'/'):
        os.makedirs('FP_'+str(k_fold)+'/'+modelname+'/')
    if color == plt.cm.Blues:
        np.savetxt('FP_'+str(k_fold)+'/'+modelname+'/'+str(epoch)+'val_pred.csv', output_csv, delimiter=',')
    elif color == plt.cm.Reds:
        np.savetxt('FP_'+str(k_fold)+'/'+modelname+'/'+str(epoch)+'val_pred.csv', output_csv, delimiter=',')

    cm_value = confusion_matrix.value()
    if png_flag == True:
        plot_Matrix(cm_value, [1,2], epoch, path, k_fold, title=None, cmap=color)
        plot_AUC(out_np, tgt_np, n_classes, epoch, path, color, k_fold)

    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    specificity = 100. * cm_value[0][0] / (cm_value[0][0] + cm_value[0][1])
    sensitivity = 100. * cm_value[1][1] / (cm_value[1][1] + cm_value[1][0])
    precision = 100. * cm_value[1][1] / (cm_value[1][1] + cm_value[0][1])
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    print('Epoch:', epoch, 'val_loss:', val_loss / len(loader), 'Accuracy:', accuracy, 'Specificity:', specificity, 'Sensitivity/Recall:', sensitivity, 'Precision:', precision, 'f1_score:', f1_score)
    val_loss /= len(loader.dataset)

    return val_loss, accuracy

def plot_evaluation_curve(train, val, path, name, k_fold):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train)), train, color='darkred', label='train')
    plt.plot(range(len(val)), val, color='darkblue', label='val')
    plt.legend()
    if not os.path.exists('FP_'+str(k_fold)+'/'):
        os.makedirs('FP_'+str(k_fold)+'/')
    if not os.path.exists('FP_'+str(k_fold)+'/'+path+'/'):
        os.makedirs('FP_'+str(k_fold)+'/'+path+'/')
    plt.savefig('FP_'+str(k_fold)+'/'+path+'/'+name+'.png')
    return

def model_selected(model_name, n_classes):
    if model_name == 'scse_res':
        model = SCSERes(n_channels=3, n_classes=n_classes, pretrained_model=True, cuda=True)
    else:
        if model_name == 'resnet18':
            resnet = Resnet18(num_classes=n_classes)
            model = mask_Resnet18(num_classes=n_classes)
        elif model_name == 'resnet34':
            resnet = Resnet34(num_classes=n_classes)
            model = mask_Resnet34(num_classes=n_classes)
        elif model_name == 'resnet50':
            resnet = Resnet50(num_classes=n_classes)
            model = mask_Resnet50(num_classes=n_classes)
        elif model_name == 'resnet101':
            resnet = Resnet101(num_classes=n_classes)
            model = mask_Resnet101(num_classes=n_classes)
        elif model_name == 'resnet152':
            resnet = Resnet152(num_classes=n_classes)
            model = mask_Resnet152(num_classes=n_classes)
        elif model_name == 'resnext50_32x4d':
            resnet = Resnext50_32x4d(num_classes=n_classes)
            model = mask_Resnext50_32x4d(num_classes=n_classes)
        elif model_name == 'resnext101_32x8d':
            resnet = Resnext101_32x8d(num_classes=n_classes)
            model = mask_Resnext101_32x8d(num_classes=n_classes)
        elif model_name == 'wide_resnet50_2':
            resnet = wide_Resnet50_2(num_classes=n_classes)
            model = mask_wide_Resnet50_2(num_classes=n_classes)
        elif model_name == 'wide_resnet101_2':
            resnet = wide_Resnet101_2(num_classes=n_classes)
            model = mask_wide_Resnet101_2(num_classes=n_classes)
        elif model_name == 'densenet121':
            resnet = Densenet121(num_classes=n_classes)
            model = Densenet121(num_classes=n_classes)
        elif model_name == 'densenet161':
            resnet = Densenet161(num_classes=n_classes)
            model = Densenet161(num_classes=n_classes)
        elif model_name == 'densenet169':
            resnet = Densenet169(num_classes=n_classes)
            model = Densenet169(num_classes=n_classes)
        elif model_name == 'densenet201':
            resnet = models.densenet201(pretrained=True)
            model = Densenet201(num_classes=n_classes)
        elif model_name == 'mobilenet_v2':
            resnet = models.mobilenet_v2(pretrained=True)
            model = Mobilenet_v2(num_classes=n_classes)
        elif model_name == 'squeezenet1_0':
            resnet = models.squeezenet1_0(pretrained=True)
            model = Squeezenet1_0(num_classes=n_classes)
        elif model_name == 'squeezenet1_1':
            resnet = models.squeezenet1_1(pretrained=True)
            model = Squeezenet1_1(num_classes=n_classes)
        elif model_name == 'vgg11':
            resnet = models.vgg11(pretrained=True)
            model = Vgg11(num_classes=n_classes)
        elif model_name == 'vgg11_bn':
            resnet = models.vgg11_bn(pretrained=True)
            model = Vgg11_bn(num_classes=n_classes)
        elif model_name == 'vgg13':
            resnet = models.vgg13(pretrained=True)
            model = Vgg13(num_classes=n_classes)
        elif model_name == 'vgg13_bn':
            resnet = models.vgg13_bn(pretrained=True)
            model = Vgg13_bn(num_classes=n_classes)
        elif model_name == 'vgg16':
            resnet = models.vgg16(pretrained=True)
            model = Vgg16(num_classes=n_classes)
        elif model_name == 'vgg16_bn':
            resnet = models.vgg16_bn(pretrained=True)
            model = Vgg16_bn(num_classes=n_classes)
        elif model_name == 'vgg19':
            resnet = models.vgg19(pretrained=True)
            model = Vgg19(num_classes=n_classes)
        elif model_name == 'vgg19_bn':
            resnet = models.vgg19_bn(pretrained=True)
            model = Vgg19_bn(num_classes=n_classes)
        elif model_name == 'inception_v3':
            resnet = models.inception_v3(pretrained=True)
            model = Inception_v3(num_classes=n_classes)
    return resnet, model

def train_net(model_name, n_classes, train_loader, val_loader, k_fold, load_path):
    resnet, model = model_selected(model_name, n_classes)
    if args.cuda:
        resnet = nn.DataParallel(resnet)
        model = nn.DataParallel(model)
        resnet.cuda()
        model.cuda()
    resnet.load_state_dict(torch.load(load_path))

    pretrained_dict = resnet.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    criterion = nn.BCEWithLogitsLoss()
    mask_criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    #lambda1 = lambda epoch: np.sin(epoch) / epoch
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-12, momentum=0.95)
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-12)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)

    Train_loss = []
    Train_acc = []
    val_loss = []
    val_acc = []
    confusion_matrix = meter.ConfusionMeter(n_classes)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            data = batch['image']
            mask = batch['mask']
            target = batch['target']

            if args.cuda:
                data, mask, target = data.cuda(), mask.cuda(), target.cuda()
            data, mask, target = Variable(data), Variable(mask), Variable(target)

            optimizer.zero_grad()
            output, mask_out = model(data)
            class_loss = criterion(output, target)
            mask_loss = mask_criterion(mask_out, mask)
            loss = 2.0 * class_loss + mask_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            confusion_matrix.add(output.data, target[:, 1].data)
            #print(loss.item(), train_loss, batch_idx)

            #if batch_idx % 5 == 0:
            #    scheduler.step()
            #if len(train_loader)//(batch_idx+1)  == 0:
            #    print("####################################################")
            #    print('Percentage:', (batch_idx+1) / len(train_loader), 'Train_loss:', train_loss / batch_idx)
            #    val_acc = valdata_cal(model, args, val_loader, True, epoch*10000+batch_idx, model_name, n_classes, plt.cm.Blues)

        print("####################################################")
        cm_value = confusion_matrix.value()
        train_acc = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
        print('Epoch:', epoch, 'Train_loss:', train_loss / len(train_loader), 'Train_acc:', train_acc)
        Train_loss.append(train_loss / len(train_loader))
        Train_acc.append(train_acc)
        valloss, valacc = valdata_cal(model, args, val_loader, True, epoch, model_name, n_classes, plt.cm.Reds, model_name, k_fold)
        val_loss.append(valloss)
        val_acc.append(valacc)
        #val_acc = valdata_cal(model, args, extraVal_loader, True, epoch, model_name, n_classes, plt.cm.Reds)
        print("####################################################")
        #final_acc = (val_acc + val_acc)/2
        #scheduler.step(final_acc)
        #if epoch % 2 == 0:
        #    for p in optimizer.param_groups:
        #        p['lr'] *= 0.9
        if args.save_cp:
            try:
                os.mkdir(args.cp_dir)
            except OSError:
                pass
            torch.save(model.state_dict(),
                       args.cp_dir + model_name + f'_epoch{epoch + 1}.pth')
            print(f'Checkpoint {epoch + 1} saved !')
    plot_evaluation_curve(Train_acc, val_acc, model_name, 'acc', k_fold)
    plot_evaluation_curve(Train_loss, val_loss, model_name, 'loss', k_fold)
    return

def train_k_fold(args, model_list, k):
    tr_dir_img = "./FP_dataset/FP_" + str(k) + "/train/"
    val_dir_img = "./FP_dataset/FP_" + str(k) + "/test/"

    train = BasicDataset(tr_dir_img, args.num_classes, 1, True)
    val = BasicDataset(val_dir_img, args.num_classes, 1, False)
    n_val = len(val)
    n_train = len(train)
    print('samples:', n_train, n_val)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for i in range(len(model_list)):
        print(model_list[i])
        train_net(model_list[i], args.num_classes, train_loader, val_loader, k)
    return

if __name__ == "__main__":
    model_list = ['resnet18', 'resnet34', 'resnet50','wide_resnet50_2']#, 'mobilenet_v2','squeezenet1_0', 'squeezenet1_1']#'mobilenet_v2', 'inception_v3', 'densenet121', 'densenet161']
    #'resnet18',
           #'resnet18', 'resnet34',
           #'resnet34', 'resnet50', 'resnet101','resnext50_32x4d',
           #'resnet152', 'resnext50_32x4d',
           #'resnext101_32x8d',
           #'wide_resnet50_2', 'wide_resnet101_2', 'densenet121',
           #'densenet161', 'densenet169', 'densenet201',
           #'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
           #'vgg13', 'vgg13_bn','vgg16','vgg16_bn',
           #'vgg19', 'vgg19_bn', 'inception_v3', 'scse_res', 'wide_resnet50_2',

    args = get_args()
    print(args.no_cuda, torch.cuda.is_available())
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('cuda available:', args.cuda)

    tr_dir_img = "./fp_mask_dataset/train/"
    val_dir_img = "./fp_mask_dataset/test/"
    tr_mask_img = './fp_mask_dataset/train_mask/'
    val_mask_img = './fp_mask_dataset/test_mask/'

    train = BasicDataset(tr_dir_img, tr_mask_img, args.num_classes, 1, True)
    val = BasicDataset(val_dir_img, val_mask_img, args.num_classes, 1, False)
    n_val = len(val)
    n_train = len(train)
    print('samples:', n_train, n_val)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    regular_model_path = './results/regular_fundus_training/checkpoints/'
    model_num_list = ['4', '4', '4', '4', '4', '3', '26']
    for i in range(len(model_list)):
        load_path = regular_model_path+model_list[i]+'/'+model_list[i]+'_epoch'+model_num_list[i]+'.pth'
        print(model_list[i])
        train_net(model_list[i], args.num_classes, train_loader, val_loader, 22, load_path)



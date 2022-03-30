import torch
import torch.nn as nn
import numpy as np
from model.resnet import resnet18


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class SE_block(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=16):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(in_channel, in_channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channel // reduction, in_channel),
                nn.Sigmoid()
        )
        self.ConvBNReLU = ConvBnRelu(in_channel, out_channel, 1, 1, 0,
                       has_bn=False, norm_layer=nn.BatchNorm2d,
                       has_relu=True, has_bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x + x * y
        y = self.ConvBNReLU(y)
        return y

class UPA_Net(nn.Module):
    def __init__(self, n_channels, n_classes, patches_list, pretrained_model=True):
        super(UPA_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.patches_list = patches_list

        self.network_array = []
        for i in range(len(patches_list)):
            self.network_array.append(resnet18(pretrained=pretrained_model))
        self.adpAvePool = nn.AdaptiveAvgPool2d(1)

        self.SE_attentions = [SE_block(in_channel=64, out_channel=512), SE_block(in_channel=128, out_channel=512),
                              SE_block(in_channel=256, out_channel=512), SE_block(in_channel=512, out_channel=512)]

        self.fuse_conv = ConvBnRelu(512, 512, 1, 1, 0,
                       has_bn=False, norm_layer=nn.BatchNorm2d,
                       has_relu=True, has_bias=False)
        self.fc = nn.Linear(512, n_classes)

    def generate_area(self):
        area_x_range = []
        area_y_range = []
        max_x_length = self.patches_list[0][0]
        max_y_length = self.patches_list[0][1]
        for i in range(1, len(self.patches_list)):
            current_x_length = self.patches_list[i][0]
            current_y_length = self.patches_list[i][1]
            interval_x = int((max_x_length-current_x_length)/2)
            interval_y = int((max_y_length-current_y_length)/2)
            area_x_range.append([interval_x, max_x_length-interval_x])
            area_y_range.append([interval_y, max_y_length-interval_y])

        x_ranges = []
        y_ranges = []
        for i in range(len(area_x_range)):
            if i == 0:
                x_ranges.append([[0, area_x_range[i][0]],[area_x_range[i][1], max_x_length]])
            else:
                x_ranges.append([[area_x_range[i-1][0], area_x_range[i][0]],[area_x_range[i][1], area_x_range[i-1][1]]])
        x_ranges.append([[area_x_range[len(area_x_range)-1][0], area_x_range[len(area_x_range)-1][1]]])
        for i in range(len(area_y_range)):
            if i == 0:
                y_ranges.append([[0, area_y_range[i][0]],[area_y_range[i][1], max_y_length]])
            else:
                y_ranges.append([[area_y_range[i-1][0], area_y_range[i][0]],[area_y_range[i][1], area_y_range[i-1][1]]])
        y_ranges.append([[area_y_range[len(area_y_range)-1][0], area_y_range[len(area_y_range)-1][1]]])
        return x_ranges, y_ranges

    def generate_mark_matrix(self, patch_w, patch_h, x_ranges, y_ranges):
        area_mark = np.zeros((patch_w, patch_h))
        for i in range(patch_w):
            for j in range(patch_h):
                for index in range(len(x_ranges)-1):
                    k = len(x_ranges)-2-index
                    if i >= x_ranges[k][0][0] and i < x_ranges[k][0][1]:
                        area_mark[i, j] = k
                    elif i >= x_ranges[k][1][0] and i < x_ranges[k][1][1]:
                        area_mark[i, j] = k

                    if j >= y_ranges[k][0][0] and j < y_ranges[k][0][1]:
                        area_mark[i, j] = k
                    elif j >= y_ranges[k][1][0] and j < y_ranges[k][1][1]:
                        area_mark[i, j] = k
                k = len(x_ranges)-1
                if i >= x_ranges[k][0][0] and i < x_ranges[k][0][1]:
                    if j >= y_ranges[k][0][0] and j < y_ranges[k][0][1]:
                        area_mark[i, j] = k
        return area_mark.astype(int)

    def forward(self, x):
        # x = (b, patch_w, patch_h, image_h, image_w, image_channel)
        _, patch_w, patch_h, _, _, _= x.size()

        x_ranges, y_ranges = self.generate_area()
        patches_level = len(x_ranges)
        area_marks = self.generate_mark_matrix(patch_w, patch_h, x_ranges, y_ranges)

        for k in range(patches_level):
            if area_marks[0, 0] == k:
                row_out = self.network_array[k](x[:, 0, 0, :, :, :])

        for level in range(len(row_out)):
            tmp_out = row_out[level]
            tmp_out = self.adpAvePool(tmp_out)
            for j in range(1, patch_h):
                for k in range(patches_level):
                    if area_marks[0, j] == k:
                        #print(k, '(', 0, ',', j, ')', area_marks[0, j])
                        tmp = self.adpAvePool(self.network_array[k](x[:, 0, j, :, :, :])[level])
                        tmp_out = torch.cat((tmp_out, tmp), 3)

            for i in range(1, patch_w):
                for k in range(patches_level):
                    if area_marks[i, 0] == k:
                        #print(k, '(', i, ',', 0, ')', area_marks[i, 0])
                        each_row = self.network_array[k](x[:, i, 0, :, :, :])[level]
                        each_row = self.adpAvePool(each_row)
                        for j in range(1, patch_h):
                            for k in range(patches_level):
                                if area_marks[i, j] == k:
                                    #print(k, '(', i, ',', j, ')', area_marks[i, j])
                                    tmp = self.adpAvePool(self.network_array[k](x[:, i, j, :, :, :])[level])
                                    each_row = torch.cat((each_row, tmp), 3)
                tmp_out = torch.cat((tmp_out, each_row), 2)
            #print(tmp_out.size())
            row_out[level] = tmp_out
            row_out[level] = self.SE_attentions[level](row_out[level])

        out = row_out[0]
        for k in range(1, len(row_out)):
            out.add_(row_out[k])

        out = self.fuse_conv(out)
        out = self.adpAvePool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)

if __name__ == "__main__":
    model = UPA_Net(n_channels=3, n_classes=2, patches_list=[[11, 15], [7, 9], [3, 3]])
    model.eval()
    image = torch.randn(1, 11, 15, 3, 256, 256)
    target = torch.tensor(np.array([1.0, 0.0]), requires_grad=True)
    fmap_block = list()
    grad_block = list()

    model.fuse_conv.register_backward_hook(farward_hook)
    model.fuse_conv.register_backward_hook(backward_hook)

    output = model(image)
    idx = np.argmax(output.cpu().data.numpy())

    # backward
    model.zero_grad()
    class_loss = nn.CrossEntropyLoss(output, target)
    class_loss.backward()

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    print(image.shape)
    print("input:", image.shape)
    print("output:", model(image).shape)
    print("gradient:", grads_val.shape)
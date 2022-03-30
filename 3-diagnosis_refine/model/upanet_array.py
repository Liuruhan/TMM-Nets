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
    def __init__(self, n_channels, n_classes, patches_w, patches_h, pretrained_model=True):
        super(UPA_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.network_array = []
        for i in range(patches_w):
            for j in range(patches_h):
                self.network_array.append(resnet18(pretrained=pretrained_model))
        self.adpAvePool = nn.AdaptiveAvgPool2d(1)

        self.SE_attentions = [SE_block(in_channel=64, out_channel=512), SE_block(in_channel=128, out_channel=512),
                              SE_block(in_channel=256, out_channel=512), SE_block(in_channel=512, out_channel=512)]

        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # x = (b, patch_w, patch_h, image_h, image_w, image_channel)
        _, patch_w, patch_h, _, _, _= x.size()

        levels = self.network_array[0](x[:, 0, 0, :, :, :])
        for k in range(len(levels)):
            levels[k] = self.adpAvePool(levels[k])
            for j in range(1, patch_h):
                tmp = self.network_array[j](x[:, 0, j, :, :, :])
                levels[k] = torch.cat((levels[k], self.adpAvePool(tmp[k])), 3)
            for i in range(1, patch_w):
                each_row = self.network_array[i*patch_w](x[:, i, 0, :, :, :])[k]
                each_row = self.adpAvePool(each_row)
                for j in range(1, patch_h):
                    tmp = self.network_array[i*patch_w+j](x[:, i, j, :, :, :])
                    each_row = torch.cat((each_row, self.adpAvePool(tmp[k])), 3)
                levels[k] = torch.cat((levels[k], each_row), 2)
            levels[k] = self.SE_attentions[k](levels[k])

        out = levels[0]
        for k in range(1, len(levels)):
            out.add_(levels[k])

        out = self.adpAvePool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = UPA_Net(n_channels=3, n_classes=2, patches_w=2, patches_h=3)
    model.eval()
    image = torch.randn(1, 2, 3, 3, 256, 256)

    print(image.shape)
    print("input:", image.shape)
    print("output:", model(image).shape)
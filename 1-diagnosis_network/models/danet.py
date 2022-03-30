from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CAM_Module, PAM_Module

from models.dilated_resnet import resnet50

class DANet(nn.Module):
    def __init__(self, n_classes, n_channels, pretrained_model, dilated=True, norm_layer=nn.BatchNorm2d,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='./pretrain_models',
                 multi_grid=False, multi_dilation=None):
        super(DANet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.pretrained = resnet50(pretrained=pretrained_model, dilated=dilated,
                                              norm_layer=norm_layer, root=root,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        self.head = DANetHead(2048, n_classes, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        x = self.head(c4)
        x = list(x)
        x[0] = F.interpolate(x[0], size=imsize)
        x[1] = F.interpolate(x[1], size=imsize)
        x[2] = F.interpolate(x[2], size=imsize)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return  tuple(outputs)


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


if __name__ == "__main__":
    model = DANet(n_classes=8, n_channels=3, pretrained_model=True)
    image = torch.randn(1, 3, 256, 256)
    label = torch.randn(1, 8, 256, 256)
    main_pred = model(image)
    #print(model)
    print(main_pred[0].size(), main_pred[1].size())
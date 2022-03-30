# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import ConvBnRelu
from change_feature.attention import CAM_Module, FeatureFusion
from model.attention_module import PAM_SE_Module
from model.dilated_resnet import resnet50

num_classes = 8
bn_eps = 1e-5
bn_momentum = 0.1


def get():
    return DDHNet(num_classes, None, None)

class DDHNet(nn.Module):
    def __init__(self, n_classes, n_channels, pretrained_model, dilated=True, norm_layer=nn.BatchNorm2d,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='./pretrain_models',
                 multi_grid=False, multi_dilation=None):
        super(DDHNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.context_path = resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)

        self.refine3x3 = ConvBnRelu(2048, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        self.refine1x1 = ConvBnRelu(512, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        # self.fpa = FPA(channels=512)
        self.CA = CAM_Module(in_dim=512)
        self.PA = PAM_SE_Module(in_dim=512, height=32, width=32)
        # self.final_FFM = FeatureFusion(in_planes=1024, out_planes=512)
        self.FFM = FeatureFusion(in_planes=1024, out_planes=512)
        self.low_FFM = FeatureFusion(in_planes=768, out_planes=512)
        self.output_head = BiSeNetHead(514, n_classes, 4, True, norm_layer)

    def forward(self, data):
        # spatial_out = self.spatial_path(data)
        x = self.context_path.conv1(data)
        x = self.context_path.bn1(x)
        x = self.context_path.relu(x)
        x = self.context_path.maxpool(x)
        c1 = self.context_path.layer1(x)
        c2 = self.context_path.layer2(c1)
        c3 = self.context_path.layer3(c2)
        c4 = self.context_path.layer4(c3)

        context_blocks = [c1, c2, c3, c4]
        context_blocks.reverse()

        refine = self.refine3x3(context_blocks[0])
        refine = self.refine1x1(refine)
        # FPA = self.fpa(refine)
        # final_fm = self.final_FFM(refine, FPA)
        ca = self.CA(refine)
        pa = self.PA(refine)
        ffm = self.FFM(ca, pa)
        fm = F.interpolate(ffm, size=context_blocks[3].shape[2:], mode="bilinear", align_corners=False)
        # h = torch.cat((fm, context_blocks[2]), dim=1)
        h = self.low_FFM(fm, context_blocks[3])
        x_range = torch.linspace(-1, 1, h.shape[-1])
        y_range = torch.linspace(-1, 1, h.shape[-2])
        Y, X = torch.meshgrid(y_range, x_range)
        Y = Y.expand([h.shape[0], 1, -1, -1])
        X = X.expand([h.shape[0], 1, -1, -1])
        coord_feat = torch.cat([X, Y], 1)#.cuda()
        h = torch.cat([h, coord_feat], 1)#.cuda()
        h = self.output_head(h)
        return h


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, n_classes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, n_classes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output


if __name__ == "__main__":
    model = DDHNet(n_classes=8, n_channels=3, pretrained_model=True)
    image = torch.randn(1, 3, 256, 256)
    label = torch.randn(1, 8, 256, 256)
    if torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()
        label = label.cuda()
    pred = model(image)
    print(pred.size())

import torch
from torch import nn
import torch.nn.functional as F
from models.resnet import wide_resnet101_2


#import torchvision.models as models

# SCSE模块
class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)  # 16
        self.channel_gate = ChannelGate2d(in_ch)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2  # x = g1*x + g2*x
        return x

# 空间门控
class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


# 通道门控
class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


# 编码连续卷积层
def contracting_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block


# 解码上采样卷积层
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()
        #self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(3, 3), stride=2, padding=1,
        #                             output_padding=1, dilation=1)
        self.fine = nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1)

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels+2, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, d, e=None):
        #d = self.up(d)
        d = F.interpolate(d,  size=e.shape[2:], mode='bilinear', align_corners=True)
        d = self.fine(d)
        # concat
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        out = self.spa_cha_gate(out)
        return out


# 输出层
def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels),
        # nn.BatchNorm2d(out_channels),
        # nn.ReLU()
    )
    return block


# SCSE U-Net
class SCSEUnet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(SCSEUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Encode
        self.conv_encode1 = nn.Sequential(contracting_block(in_channels=n_channels, out_channels=32), SCSE(32))
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = nn.Sequential(contracting_block(in_channels=32, out_channels=64), SCSE(64))
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = nn.Sequential(contracting_block(in_channels=64, out_channels=128), SCSE(128))
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = nn.Sequential(contracting_block(in_channels=128, out_channels=256), SCSE(256))
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SCSE(512)
        )

        # Decode
        self.conv_decode4 = expansive_block(512, 256, 256)
        self.conv_decode3 = expansive_block(256, 128, 128)
        self.conv_decode2 = expansive_block(128, 64, 64)
        self.conv_decode1 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, n_classes)

    def forward(self, x):
        # set_trace()
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)

        # Bottleneck
        bottleneck = self.bottleneck(encode_pool4)

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)

        final_layer = self.final_layer(decode_block1)
        out = torch.sigmoid(final_layer)  # 可注释，根据情况

        return out

class SCSERes(nn.Module):

    def __init__(self, n_channels, n_classes, pretrained_model, cuda):
        super(SCSERes, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.context_path = wide_resnet101_2(pretrained=pretrained_model)
        self.is_cuda = cuda

        # Decode
        self.conv_decode4 = expansive_block(2050, 1026, 1024)
        #self.conv_decode1 = expansive_block(256, 128, 128)
        #self.final_layer = final_block(256, n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, n_classes)

    def generate_coord(self, h, is_cuda):
        x_range = torch.linspace(-1, 1, h.shape[-1])
        y_range = torch.linspace(-1, 1, h.shape[-2])
        Y, X = torch.meshgrid(y_range, x_range)
        Y = Y.expand([h.shape[0], 1, -1, -1])
        X = X.expand([h.shape[0], 1, -1, -1])
        if is_cuda == True:
            coord_feat = torch.cat([X, Y], 1).cuda()
        else:
            coord_feat = torch.cat([X, Y], 1)
        return coord_feat

    def forward(self, x):
        # set_trace()
        # Encode
        context_blocks = self.context_path(x)
        context_blocks.reverse()

        # Decode
        fourth_coord_feat = self.generate_coord(context_blocks[0], self.is_cuda)
        if self.is_cuda == True:
            fourth_layer = torch.cat([context_blocks[0], fourth_coord_feat], 1).cuda()
        else:
            fourth_layer = torch.cat([context_blocks[0], fourth_coord_feat], 1)
        third_coord_feat = self.generate_coord(context_blocks[1], self.is_cuda)
        if self.is_cuda == True:
            third_layer = torch.cat([context_blocks[1], third_coord_feat], 1).cuda()
        else:
            third_layer = torch.cat([context_blocks[1], third_coord_feat], 1)
        decode_block4 = self.conv_decode4(fourth_layer, third_layer)

        #final_layer = self.final_layer(decode_block2)
        out = self.avgpool(decode_block4)
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out

if __name__ == "__main__":
    model = SCSERes(n_channels=3, n_classes=4, pretrained_model=True, cuda=False)
    model.eval()
    image = torch.randn(1, 3, 224, 172)

    print(image.shape)
    print("input:", image.shape)
    print("output:", model(image).shape)
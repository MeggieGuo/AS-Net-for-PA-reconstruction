import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2)
    # if mode == 'transpose':
    #     return nn.ConvTranspose2d(
    #         in_channels,
    #         out_channels,
    #         kernel_size=2,
    #         stride=2)
    # else:
    #     # out_channels is always going to be the same
    #     # as in_channels
    #     return nn.Sequential(
    #         nn.Upsample(mode='bilinear', scale_factor=2),
    #         conv1x1(in_channels, out_channels))


class DownConvNN(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConvNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)

        if self.pooling:
            x = self.pool(x)
        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out #, attention


class BottomNN(nn.Module):
    def __init__(self, in_channels):
        super(BottomNN, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels *2

        self.conv1 = conv3x3(self.in_channels, self.inter_channels)
        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        # self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
        #                        kernel_size=(20, 3), stride=(20, 1), padding=(3, 1))
        # self.bn2 = nn.BatchNorm2d(self.out_channels)
        # self.conv3 = nn.Conv2d(self.out_channels, self.in_channels, 3, padding=1)
        self.conv3 = conv3x3(self.inter_channels, self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        # self.conv3 = conv3x3(self.out_channels, self.in_channels)
        # self.bn2 = nn.BatchNorm2d(self.in_channels)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        # x = self.conv2(x)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.01)

        return x


nonlinearity = partial(F.relu, inplace=True)
class Atrousblock(nn.Module):
    def __init__(self, channel):
        super(Atrousblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.in_channels,
                                mode=self.up_mode)
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.upconv(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x

class Sidelayer(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Sidelayer, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.conv1(x))
        return out


class SidePool(nn.Module):
    def __init__(self):
        super(SidePool, self).__init__()

    def forward(self, feature, x):
        m = x.size(-1)
        f = nn.AdaptiveAvgPool2d((m,m))(feature)
        out = torch.cat((f, x), 1)
        return out


class ConvBlock_keb(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1):
        super().__init__()
        self.conv_keb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1))

    def forward(self, input):
        return self.conv_keb(input)


class Featurelayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Featurelayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv1x1(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv1x1(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x


class PSPblock(nn.Module):
    def __init__(self):
        super(PSPblock, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((8, 8))
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        h, w = 8,8

        self.layer1 = F.interpolate(self.pool1(x), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.interpolate(self.pool2(x), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.interpolate(self.pool3(x), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.interpolate(self.pool4(x), size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat((self.layer1, self.layer2, self.layer3, self.layer4), 1)

        return out


class FinConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FinConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        return x

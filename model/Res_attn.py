# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       KEB_RES_SE
   Project Name:    
   Author :         Hengrong LAN
   Date:            20200527
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2020/5/27:
-------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_block import ContextBlock



class Bottleneck(nn.Module):

    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, ratio): #, stride=1,cardinality=1,dilation=1,bottleneck_width=64):
        super(Bottleneck, self).__init__()
        # group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        # print(group_width)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.gcb = ContextBlock(planes*4,ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.gcb(out)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck_KEB(nn.Module):

    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, ratio=0.5): #, stride=1,cardinality=1,dilation=1,bottleneck_width=64):
        super(Bottleneck_KEB, self).__init__()
        # group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        # print(group_width)
        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.gcb = ContextBlock(planes,ratio)

    def forward(self, x):

        residual = self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)

        out = self.gcb(out)
        out += residual
        out = self.relu(out)

        return out

class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, sub_sample=False, bn_layer=True):
        super(NONLocalBlock2D, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        if self.inter_channels == 0:
            self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.gbf = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.gbf = nn.Sequential(self.gbf, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, bfimg, x):
        """
        :param x: (b, c, h, w)
        :return:
        """

        batch_size = x.size(0)

        # theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = self.theta(bfimg).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # phi_bf = self.phi(bfimg).view(batch_size, self.inter_channels, -1)
        phi_bf = self.phi(bfimg).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_bf)
        f_div_C = F.softmax(f, dim=-1)


        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        g_bf = self.gbf(bfimg).view(batch_size, self.inter_channels, -1)
        g_bf = g_bf.permute(0, 2, 1)

        y_bf = torch.matmul(f_div_C, g_bf)
        y_bf = y_bf.permute(0, 2, 1).contiguous()
        y_bf = y_bf.view(batch_size, self.inter_channels, *bfimg.size()[2:])
        W_y_bf = self.W(y_bf)
        z_bf = W_y_bf + bfimg

        out=torch.cat((z, z_bf), 1)

        return out


"""
if __name__ == '__main__':
    import torch


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.zeros(1, 32, 128, 128).to(device)
    bfimg = torch.zeros(1, 32, 128, 128).to(device)
    net = NONLocalBlock2D(32).to(device)
    out = net(img,bfimg)
    print(out.size())
"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       KEB_RES_SE
   Project Name:    
   Author :         Hengrong LAN
   Date:            20200527
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2020/5/27:
-------------------------------------------------
"""
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from .context_block import ContextBlock
#
#
# class Bottleneck(nn.Module):
#     # pylint: disable=unused-argument
#     expansion = 4
#
#     def __init__(self, inplanes, planes, ratio, stride=1, cardinality=1, dilation=1, bottleneck_width=64):
#         super(Bottleneck, self).__init__()
#         group_width = int(planes * (bottleneck_width / 64.)) * cardinality
#         print(group_width)
#         self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(group_width)
#         self.conv2 = nn.Conv2d(
#             group_width, group_width, kernel_size=3, stride=stride,
#             padding=dilation, dilation=dilation,
#             groups=cardinality, bias=False)
#         self.bn2 = nn.BatchNorm2d(group_width)
#         self.conv3 = nn.Conv2d(
#             group_width, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.gcb = ContextBlock(planes * 4, ratio)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out = self.gcb(out)
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class NONLocalBlock2D(nn.Module):
#     def __init__(self, in_channels, sub_sample=False, bn_layer=True):
#         super(NONLocalBlock2D, self).__init__()
#         self.sub_sample = sub_sample
#         self.in_channels = in_channels
#         self.inter_channels = in_channels // 4
#         if self.inter_channels == 0:
#             self.inter_channels = 1
#
#         conv_nd = nn.Conv2d
#         max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
#         bn = nn.BatchNorm2d
#
#         self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                          kernel_size=1, stride=1, padding=0)
#
#         self.gbf = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                            kernel_size=1, stride=1, padding=0)
#
#         if bn_layer:
#             self.W = nn.Sequential(
#                 conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
#                         kernel_size=1, stride=1, padding=0),
#                 bn(self.in_channels)
#             )
#             nn.init.constant_(self.W[1].weight, 0)
#             nn.init.constant_(self.W[1].bias, 0)
#         else:
#             self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
#                              kernel_size=1, stride=1, padding=0)
#             nn.init.constant_(self.W.weight, 0)
#             nn.init.constant_(self.W.bias, 0)
#
#         self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                              kernel_size=1, stride=1, padding=0)
#         self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                            kernel_size=1, stride=1, padding=0)
#
#         if sub_sample:
#             self.g = nn.Sequential(self.g, max_pool_layer)
#             self.gbf = nn.Sequential(self.gbf, max_pool_layer)
#             self.phi = nn.Sequential(self.phi, max_pool_layer)
#
#     def forward(self, bfimg, x):
#         """
#         :param x: (b, c, h, w)
#         :return:
#         """
#
#         batch_size = x.size(0)
#
#         # theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
#         theta_x = self.theta(bfimg).view(batch_size, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         # phi_bf = self.phi(bfimg).view(batch_size, self.inter_channels, -1)
#         phi_bf = self.phi(bfimg).view(batch_size, self.inter_channels, -1)
#         f = torch.matmul(theta_x, phi_bf)
#         f_div_C = F.softmax(f, dim=-1)
#
#         g_x = self.g(x).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)
#
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *x.size()[2:])
#         W_y = self.W(y)
#         z = W_y + x
#
#         g_bf = self.gbf(bfimg).view(batch_size, self.inter_channels, -1)
#         g_bf = g_bf.permute(0, 2, 1)
#
#         y_bf = torch.matmul(f_div_C, g_bf)
#         y_bf = y_bf.permute(0, 2, 1).contiguous()
#         y_bf = y_bf.view(batch_size, self.inter_channels, *bfimg.size()[2:])
#         W_y_bf = self.W(y_bf)
#         z_bf = W_y_bf + bfimg
#
#         out = torch.cat((z, z_bf), 1)
#
#         return out
#
#
# """
# if __name__ == '__main__':
#     import torch
#
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     img = torch.zeros(1, 32, 128, 128).to(device)
#     bfimg = torch.zeros(1, 32, 128, 128).to(device)
#     net = NONLocalBlock2D(32).to(device)
#     out = net(img,bfimg)
#     print(out.size())
# """
'''
Function:
  Down-Up-CNN model

Main functions:
  * Luma_Q_Net():  luma CTU -> qt depth map
  * Luma_MSBD_Net(): luma CTU + qt depth map ->  mtt depth map + direction map
  * Chroma net is similar with the Luma

Note:
  * Luma model and chroma model are similar. Chorma model has smaller size.

Author: Aolin Feng
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from Metrics import block_qtnode_norm

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
# unused
# class UniverseQuant(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         # b = np.random.uniform(-1, 1)
#         b = 0
#         uniform_distribution = Uniform(-0.5 * torch.ones(x.size())
#                                        * (2 ** b), 0.5 * torch.ones(x.size()) * (2 ** b)).sample().cuda()
#         return torch.round(x + uniform_distribution) - uniform_distribution
#
#     @staticmethod
#     def backward(ctx, g):
#         return g

class Luma_Q_Net(nn.Module):  # luma QT depth prediction
    def __init__(self):
        super(Luma_Q_Net, self).__init__()

        self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
        self.padding_rb = nn.ZeroPad2d((0, 4, 0, 4))
        self.padding_r = nn.ZeroPad2d((0, 4, 0, 0))
        self.padding_b = nn.ZeroPad2d((0, 0, 0, 4))

        self.conv_q1 = nn.Conv2d(1, 32, kernel_size=9, padding=0, stride=1)
        self.resblock_q1 = ResidualBlock(32, 64, kernel_size=5, padding=2)
        self.resblock_q2 = ResidualBlock(64, 64, kernel_size=5, padding=2)
        self.resblock_q3 = ResidualBlock(64, 32, kernel_size=3, padding=1)
        # multi pooling
        self.resblock_q4 = ResidualBlock(128, 32, kernel_size=3, padding=1)
        self.resblock_q5 = ResidualBlock(32, 32, kernel_size=3, padding=1)
        self.resblock_q6 = ResidualBlock(32, 8, kernel_size=3, padding=1)
        self.conv_q2 = nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):  # input 1*68*68
        x1 = self.padding_rb(x)  # 1*72*72
        x2 = F.relu(self.conv_q1(x1))  # 32*64*64
        x3 = F.max_pool2d(self.resblock_q1(x2), 2)  # 64*32*32
        x4 = F.max_pool2d(self.resblock_q2(x3), 2)  # 64*16*16
        x5 = self.resblock_q3(x4)  # 32*16*16
        x5_1 = F.interpolate(F.max_pool2d(x5, 2), scale_factor=2)
        x5_2 = F.interpolate(F.max_pool2d(x5, 4), scale_factor=4)
        x5_3 = F.interpolate(F.max_pool2d(x5, 8), scale_factor=8)
        x6 = torch.cat([x5, x5_1, x5_2, x5_3], 1)  # 128*16*16
        x7 = self.resblock_q4(x6)  # 32*16*16
        x8 = F.max_pool2d(self.resblock_q5(x7), 2)  # 32*8*8
        x9 = self.resblock_q6(x8)  # 8*8*8
        x10 = self.conv_q2(x9)  # 1*8*8 qt depth map
        # if add_noise == 1:
        #     xq10 = UniverseQuant.apply(x10)
        # else:
        #     xq10 = x10
        # start bt depth

        return x10  # qt depth map

class Luma_MSBD_Net(nn.Module):  # luma bt depth and direction prediction
    def __init__(self):
        super(Luma_MSBD_Net, self).__init__()
        self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
        self.padding_rb = nn.ZeroPad2d((0, 4, 0, 4))
        self.padding_r = nn.ZeroPad2d((0, 4, 0, 0))
        self.padding_b = nn.ZeroPad2d((0, 0, 0, 4))

        self.conv_b1_1 = nn.Conv2d(2, 16, kernel_size=(9, 9), padding=0, stride=1)
        self.conv_b1_2 = nn.Conv2d(2, 8, kernel_size=(5, 9), padding=0, stride=1)
        self.conv_b1_3 = nn.Conv2d(2, 8, kernel_size=(9, 5), padding=0, stride=1)
        # M-Main, B-Branch, A-Attention
        self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1))
        self.trunk_M2 = nn.Sequential(ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 64, 3, 1))
        self.trunk_B1 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.trunk_B2 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.trunk_B3 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.conv_B1 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        self.conv_B2 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        self.conv_B3 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        # self.resblock_A1 = ResidualBlock(2, 64, kernel_size=3, padding=1)
        # self.resblock_A2 = ResidualBlock(2, 64, kernel_size=3, padding=1)
        self.trunk_Att1 = nn.Sequential(ResidualBlock(3, 32, 3, 1), ResidualBlock(32, 64, 3, 1))
        self.trunk_Att2 = nn.Sequential(ResidualBlock(3, 32, 3, 1), ResidualBlock(32, 64, 3, 1))

    def forward(self, x, x1):  # input image block + qt depth map
        # x.shape = 2*68*68 if the img-block with variance map
        # x10_2 = self.padding_lu(block_qtnode_norm(qt_map=x10, block=x, isLuma=True))  # 1*68*68
        x1_1 = self.padding_lu(F.interpolate(x1, scale_factor=8))  # 1*68*68
        x2 = torch.cat([x, x1_1], 1)  # 2 or 3*68*68
        x3_1 = F.relu(self.conv_b1_1(self.padding_rb(x2)))  # 16*64*64
        x3_2 = F.relu(self.conv_b1_2(self.padding_r(x2)))  # 8*64*64
        x3_3 = F.relu(self.conv_b1_3(self.padding_b(x2)))  # 8*64*64
        x3 = torch.cat([x3_1, x3_2, x3_3], 1)  # 32*64*64
        x4 = F.max_pool2d(self.trunk_M1(x3), 2)  # 64*32*32 M1 out
        x5 = F.max_pool2d(self.trunk_M2(x4), 2)  # 64*16*16 M2 out
        x6 = self.trunk_B1(x5)  # 8*16*16
        out0 = self.conv_B1(x6)  # 2*16*16
        out0q = torch.cat([F.interpolate(x1, scale_factor=2), out0], 1)
        # B2 branch
        x_att0 = self.trunk_Att1(out0q)  # 64*16*16
        xb1 = x5 * x_att0  # 64*16*16
        xb2 = self.trunk_B2(xb1)  # 8*16*16
        out1 = self.conv_B2(xb2)  # 2*16*16
        out1[:, 0:1, :, :] = out1[:, 0:1, :, :] + out0[:, 0:1, :, :]  # 2*16*16
        out1q = torch.cat([F.interpolate(x1, scale_factor=4), F.interpolate(out1, scale_factor=2)], 1)
        # B3 branch
        x_att1 = self.trunk_Att2(out1q)  # 64*32*32
        xb3 = x4 * x_att1  # 64*32*32
        xb4 = F.max_pool2d(self.trunk_B3(xb3), 2)  # 8*16*16
        out2 = self.conv_B3(xb4)  # 2*16*16
        out2[:, 0:1, :, :] = out2[:, 0:1, :, :] + out1[:, 0:1, :, :]  # 2*16*16

        return out0, out1, out2

class Chroma_Q_Net(nn.Module):  # chroma QT depth prediction
    def __init__(self):
        super(Chroma_Q_Net, self).__init__()

        self.padding_lu = nn.ZeroPad2d((2, 0, 2, 0))
        self.padding_rb = nn.ZeroPad2d((0, 2, 0, 2))
        self.padding_r = nn.ZeroPad2d((0, 2, 0, 0))
        self.padding_b = nn.ZeroPad2d((0, 0, 0, 2))

        self.conv_q1 = nn.Conv2d(3, 32, kernel_size=5, padding=0, stride=1)
        self.resblock_q1 = ResidualBlock(32, 64, kernel_size=3, padding=1)
        self.resblock_q2 = ResidualBlock(64, 64, kernel_size=3, padding=1)
        self.resblock_q3 = ResidualBlock(64, 32, kernel_size=3, padding=1)
        # multi pooling
        self.resblock_q4 = ResidualBlock(128, 32, kernel_size=3, padding=1)
        self.resblock_q5 = ResidualBlock(32, 32, kernel_size=3, padding=1)
        self.resblock_q6 = ResidualBlock(32, 8, kernel_size=3, padding=1)
        self.conv_q2 = nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):  # input 3*34*34
        x1 = self.padding_rb(x)  # 3*36*36
        x2 = F.relu(self.conv_q1(x1))  # 32*32*32
        x3 = self.resblock_q1(x2) # 64*32*32
        x4 = F.max_pool2d(self.resblock_q2(x3), 2)  # 64*16*16
        x5 = self.resblock_q3(x4)  # 32*16*16
        x5_1 = F.interpolate(F.max_pool2d(x5, 2), scale_factor=2)
        x5_2 = F.interpolate(F.max_pool2d(x5, 4), scale_factor=4)
        x5_3 = F.interpolate(F.max_pool2d(x5, 8), scale_factor=8)
        x6 = torch.cat([x5, x5_1, x5_2, x5_3], 1)  # 128*16*16
        x7 = self.resblock_q4(x6)  # 32*16*16
        x8 = F.max_pool2d(self.resblock_q5(x7), 2)  # 32*8*8
        x9 = self.resblock_q6(x8)  # 8*8*8
        x10 = self.conv_q2(x9)  # 1*8*8 qt depth map
        # if add_noise == 1:
        #     xq10 = UniverseQuant.apply(x10)
        # else:
        #     xq10 = x10
        # start bt depth

        return x10  # qt depth map

class Chroma_MSBD_Net(nn.Module):  # chroma bt depth and direction prediction
    def  __init__(self):
        super(Chroma_MSBD_Net, self).__init__()
        self.padding_lu = nn.ZeroPad2d((2, 0, 2, 0))
        self.padding_rb = nn.ZeroPad2d((0, 2, 0, 2))
        self.padding_r = nn.ZeroPad2d((0, 2, 0, 0))
        self.padding_b = nn.ZeroPad2d((0, 0, 0, 2))

        self.conv_b1_1 = nn.Conv2d(4, 16, kernel_size=(5, 5), padding=0, stride=1)
        self.conv_b1_2 = nn.Conv2d(4, 8, kernel_size=(3, 5), padding=0, stride=1)
        self.conv_b1_3 = nn.Conv2d(4, 8, kernel_size=(5, 3), padding=0, stride=1)
        # M-Main, B-Branch, A-Attention
        self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1))
        self.trunk_M2 = nn.Sequential(ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 64, 3, 1))
        self.trunk_B1 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.trunk_B2 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.trunk_B3 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.conv_B1 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        self.conv_B2 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        self.conv_B3 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        # self.resblock_A1 = ResidualBlock(2, 64, kernel_size=3, padding=1)
        # self.resblock_A2 = ResidualBlock(2, 64, kernel_size=3, padding=1)
        self.trunk_Att1 = nn.Sequential(ResidualBlock(3, 32, 3, 1), ResidualBlock(32, 64, 3, 1))
        self.trunk_Att2 = nn.Sequential(ResidualBlock(3, 32, 3, 1), ResidualBlock(32, 64, 3, 1))

    def forward(self, x, x1):  # input image block + qt depth map
        # x.shape = 2*68*68 if the img-block with variance map
        # x10_2 = self.padding_lu(block_qtnode_norm(qt_map=x10, block=x, isLuma=True))  # 1*68*68
        x1_1 = self.padding_lu(F.interpolate(x1, scale_factor=4))  # 1*34*34
        x2 = torch.cat([x, x1_1], 1)  # 3*34*34
        x3_1 = F.relu(self.conv_b1_1(self.padding_rb(x2)))  # 16*32*32
        x3_2 = F.relu(self.conv_b1_2(self.padding_r(x2)))  # 8*32*32
        x3_3 = F.relu(self.conv_b1_3(self.padding_b(x2)))  # 8*32*32
        x3 = torch.cat([x3_1, x3_2, x3_3], 1)  # 32*32*32
        x4 = self.trunk_M1(x3)  # 64*32*32 M1 out
        x5 = F.max_pool2d(self.trunk_M2(x4), 2)  # 64*16*16 M2 out
        x6 = self.trunk_B1(x5)  # 8*16*16
        out0 = self.conv_B1(x6)  # 2*16*16
        out0q = torch.cat([F.interpolate(x1, scale_factor=2), out0], 1)
        # B2 branch
        x_att0 = self.trunk_Att1(out0q)  # 64*16*16
        xb1 = x5 * x_att0  # 64*16*16
        xb2 = self.trunk_B2(xb1)  # 8*16*16
        out1 = self.conv_B2(xb2)  # 2*16*16
        out1[:, 0:1, :, :] = out1[:, 0:1, :, :] + out0[:, 0:1, :, :]  # 2*16*16
        out1q = torch.cat([F.interpolate(x1, scale_factor=4), F.interpolate(out1, scale_factor=2)], 1)
        # B3 branch
        x_att1 = self.trunk_Att2(out1q)  # 64*32*32
        xb3 = x4 * x_att1  # 64*32*32
        xb4 = F.max_pool2d(self.trunk_B3(xb3), 2)  # 8*16*16
        out2 = self.conv_B3(xb4)  # 2*16*16
        out2[:, 0:1, :, :] = out2[:, 0:1, :, :] + out1[:, 0:1, :, :]  # 2*16*16

        return out0, out1, out2

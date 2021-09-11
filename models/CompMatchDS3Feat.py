# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:08:31 2020

@author: B016
"""

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        #        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(in_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        x = self.conv(x)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 is_conv2=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_conv2 = is_conv2
        self.deconv = deconv
        self.is_3d = is_3d

        if deconv and is_3d:
            ks = (3, 3, 3)
            ss = (2, 2, 2)
            pd = (1, 1, 1)
            op = (1, 1, 1)
            #            kernel = 3
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=ks,
                                   stride=ss, padding=pd, output_padding=op)
        elif deconv:
            #            kernel = 4
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=3,
                                   stride=2, padding=1, output_padding=1)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=3,
                                   stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        # print(x.size(), rem.size())
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        if self.is_conv2:
            x = self.conv2(x)
        return x

class HGFeature(nn.Module):
    def __init__(self):
        super(HGFeature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class GetCostVolume(nn.Module):
    def __init__(self, maxdisp):
        super(GetCostVolume, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x, y):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels * 2, self.maxdisp, height, width).zero_()
            for i in range(self.maxdisp):
                if i > 0 :
                    cost[:, :x.size()[1], i, :,i:]   = x[:,:,:,i:]
                    cost[:, x.size()[1]:, i, :,i:]   = y[:,:,:,:-i]
                else:
                    cost[:, :x.size()[1], i, :,:]   = x
                    cost[:, x.size()[1]:, i, :,:]   = y

            cost = cost.contiguous()
        return cost


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)), [1, self.maxdisp, 1, 1])).cuda(),
                            requires_grad=False)
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out


class Disp(nn.Module):

    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = torch.squeeze(x, 1)
        x = self.softmax(x)

        return self.disparity(x)


class CostRefine(nn.Module):
    def __init__(self, maxdisp=192, c=32):
        super(CostRefine, self).__init__()

        ks = (3, 3, 3)
        ss = (2, 2, 2)
        pd = (1, 1, 1)

        c2 = c  # *2
        c4 = c  # *4
        # c8 = c*8

        self.maxdisp = maxdisp

        self.conv_19 = BasicConv(c, c, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_20 = BasicConv(c, c, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_21 = BasicConv(c, c2, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_22 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_23 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_24 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_25 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_26 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_27 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_28 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_29 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_30 = BasicConv(c2, c4, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_31 = BasicConv(c4, c4, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_32 = BasicConv(c4, c4, is_3d=True, kernel_size=ks, padding=pd)

        self.deconv_33 = Conv2x(c4, c2, deconv=True, is_3d=True, concat=False, is_conv2=False)
        self.deconv_34 = Conv2x(c2, c2, deconv=True, is_3d=True, concat=False, is_conv2=False)
        self.deconv_35 = Conv2x(c2, c2, deconv=True, is_3d=True, concat=False, is_conv2=False)
        self.deconv_36 = Conv2x(c2, c, deconv=True, is_3d=True, concat=False, is_conv2=False)

    def forward(self, x):
        bc = x
        bc = self.conv_19(bc)
        bc = self.conv_20(bc)
        rem_20 = bc

        x = self.conv_21(x)
        bc = self.conv_22(x)
        bc = self.conv_23(bc)
        rem_23 = bc

        x = self.conv_24(x)
        bc = self.conv_25(x)
        bc = self.conv_26(bc)
        rem_26 = bc

        x = self.conv_27(x)
        bc = self.conv_28(x)
        bc = self.conv_29(bc)
        rem_29 = bc

        x = self.conv_30(x)
        x = self.conv_31(x)
        x = self.conv_32(x)

        x = self.deconv_33(x, rem_29)
        x = self.deconv_34(x, rem_26)
        x = self.deconv_35(x, rem_23)
        x = self.deconv_36(x, rem_20)

        return x


class Model(nn.Module):
    def __init__(self, maxdisp=192, c=32, training=True):
        super(Model, self).__init__()
        self.training = training
        self.maxdisp = maxdisp
        self.feature = HGFeature()
        self.cv = GetCostVolume(int(self.maxdisp/3))
        self.cost_refine_1 = CostRefine(self.maxdisp, c)
        self.cost_refine_2 = CostRefine(self.maxdisp, c)
        self.cost_refine_3 = CostRefine(self.maxdisp, c)
        self.disp = Disp(self.maxdisp)

        # self.conv_start = BasicConv(3, c, is_3d=True, kernel_size=5, stride=3, padding=2)
        self.conv_start = BasicConv(c * 2, c, is_3d=True, kernel_size=1, padding=0)
        self.deconv_1 = BasicConv(c, 1, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=5, stride=3, padding=2,
                                output_padding=(2, 2, 2))
        self.deconv_2 = BasicConv(c, 1, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=5, stride=3, padding=2,
                                output_padding=(2, 2, 2))
        self.deconv_3 = BasicConv(c, 1, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=5, stride=3, padding=2,
                                output_padding=(2, 2, 2))
        # self.final_conv = BasicConv(c, 1, is_3d=True, bn=True, relu=True, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = self.feature(x)
        y = self.feature(y)
        cv = self.cv(x, y)
        cv = self.conv_start(cv)
        cv1 = self.cost_refine_1(cv)
        cv2 = self.cost_refine_2(cv1)
        cv3 = self.cost_refine_3(cv2)
        # cv = self.final_conv(cv)
        # cv = F.interpolate(cv, [self.maxdisp, cv.size()[3] * 3, cv.size()[4] * 3], mode='trilinear', align_corners=False)

        cv3 = self.deconv_3(cv3)
        exp_disp_3 = self.disp(cv3)

        if self.training:
            cv1 = self.deconv_1(cv1)
            cv2 = self.deconv_2(cv2)
            exp_disp_1 = self.disp(cv1)
            exp_disp_2 = self.disp(cv2)
            return exp_disp_1, exp_disp_2, exp_disp_3
        return exp_disp_3
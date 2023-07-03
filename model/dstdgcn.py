#!/usr/bin/env python
# coding: utf-8
import math
import pdb

import numpy as np
import torch
import torch.nn as nn

from .layers.graph import Graph
from .layers.time import Time


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)


class BatchNorm(nn.Module):

    def __init__(self, feature_channels, joint_dim, time_dim):
        super(BatchNorm, self).__init__()
        self.c = feature_channels
        self.v = joint_dim
        self.t = time_dim
        self.bn = nn.BatchNorm1d(feature_channels * joint_dim)

    def forward(self, x):
        n, c, t, v = x.shape
        assert (c, t, v) == (self.c, self.t, self.v)
        x = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        x = self.bn(x)
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()
        return x


class DSTDGC(nn.Module):

    def __init__(self, in_channels, out_channels, ref_channels, kpt_channels, red_channels=2, mode="spatial"):
        super(DSTDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ref_channels = ref_channels
        self.kpt_channels = kpt_channels
        self.red_channels = red_channels
        self.mode = mode
        assert mode in {"spatial", "temporal"}

        # spatial / temporal modeling
        self.conv_m1 = nn.Conv2d(self.in_channels, self.red_channels, 1)
        self.conv_m2 = nn.Conv2d(self.in_channels, self.red_channels, 1)
        self.conv_rm = nn.Conv2d(self.red_channels * self.ref_channels, self.ref_channels, 1)
        self.tanh = nn.Tanh()

        self.conv_f = nn.Conv2d(self.in_channels, self.out_channels, 1)

        self.init_parameter()

    def init_parameter(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)

    def forward(self, x, A=None, alpha_m=1):
        xf = self.conv_f(x)
        xm1, xm2 = self.conv_m1(x), self.conv_m2(x)
        if self.mode == "spatial":
            n, c, t, v = xm1.shape
            xm = self.tanh(xm1.view(n, c * t, v).unsqueeze(-1) - xm2.view(n, c * t, v).unsqueeze(-2))
            xm = self.conv_rm(xm) * alpha_m + A.unsqueeze(0)
            xfm = torch.einsum("nctv,ntvw->nctw", xf, xm)
        else:
            xm1, xm2 = xm1.permute(0, 1, 3, 2).contiguous(), xm2.permute(0, 1, 3, 2).contiguous()
            n, c, v, t = xm1.shape
            xm = self.tanh(xm1.view(n, c * v, t).unsqueeze(-1) - xm2.view(n, c * v, t).unsqueeze(-2))
            xm = self.conv_rm(xm) * alpha_m + A.unsqueeze(0)
            xfm = torch.einsum("nctv,nvtu->ncuv", xf, xm)
        return xfm


class DSTDGCB(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim, joint_dim, layout="h36m"):
        super(DSTDGCB, self).__init__()

        graph_gen = Graph(layout)
        time_gen = Time(time_dim)
        A_s = graph_gen.get_all_adjacency()
        A_t = time_gen.get_all_adjacency()

        self.A_s = nn.Parameter(torch.FloatTensor(A_s), False)
        self.W_s = nn.Parameter(torch.zeros_like(self.A_s))
        self.R_s = nn.Parameter(torch.FloatTensor(self.A_s))
        # temporal modeling
        self.A_t = nn.Parameter(torch.FloatTensor(A_t), False)
        self.R_t = nn.Parameter(torch.zeros_like(self.A_t))

        self.conv_s = nn.ModuleList()
        self.conv_t = nn.ModuleList()

        if in_channels != out_channels:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                          BatchNorm(out_channels, joint_dim, time_dim))
        else:
            self.residual = lambda x: x

        for _ in range(A_s.shape[0]):
            self.conv_s.append(DSTDGC(in_channels, out_channels, time_dim, joint_dim, mode="spatial"))
        self.alpha_sm = nn.Parameter(torch.zeros(1))
        self.bn = BatchNorm(out_channels, joint_dim, time_dim)

        for _ in range(A_t.shape[0]):
            self.conv_t.append(DSTDGC(out_channels, out_channels, joint_dim, time_dim, mode="temporal"))
        self.alpha_tm = nn.Parameter(torch.zeros(1))

        self.prelu = nn.PReLU()
        self.do = nn.Dropout(0.1)

    def init_parameter(self):
        stdt = 1. / math.sqrt(self.R_t.size(1))
        self.R_t.data.uniform_(-stdt, stdt)
        stdt = 1. / math.sqrt(self.R_s.size(1))
        self.R_s.data.uniform_(-stdt, stdt)

    def forward(self, x):

        r = self.residual(x)
        y = None
        for i, conv in enumerate(self.conv_s):
            a_s = self.A_s[i:i + 1]
            w_s = self.W_s[i:i + 1]
            r_s = self.R_s[i:i + 1]
            z = conv(x, a_s * w_s + r_s, self.alpha_sm)
            y = y + z if y is not None else z
        x = y
        x = self.bn(x)
        x += r
        x = self.prelu(x)

        y = None
        for i, conv in enumerate(self.conv_t):
            a_t = self.A_t[i:i + 1]
            r_t = self.R_t[i:i + 1]
            z = conv(x, a_t + r_t, self.alpha_tm)
            y = y + z if y is not None else z
        x = y
        return x.contiguous()


class ConvTemporalGraphical(nn.Module):

    def __init__(self, time_dim, joints_dim, layout="h36m"):
        super(ConvTemporalGraphical, self).__init__()

        self.A = nn.Parameter(
            torch.FloatTensor(time_dim, joints_dim,
                              joints_dim))
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

        graph_gen = Graph(layout)
        adj = graph_gen.get_adjacency()[np.newaxis, :]
        self.A_fixed = nn.Parameter(torch.FloatTensor(adj), requires_grad=False)

    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        x = torch.einsum('nctv,tvw->nctw', (x, self.A + self.A_fixed))
        return x.contiguous()


class ST_GCNN_layer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 bias=True,
                 refine=False,
                 residual=True,
                 layout="h36m"):

        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        self.refine = refine
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        if refine:
            self.stgcn = nn.ModuleList()
            self.stgcn.append(
                nn.Sequential(*nn.ModuleList([
                    DSTDGCB(in_channels, out_channels, time_dim, joints_dim, layout),
                ])))
        else:
            self.stgcn = nn.Sequential(*nn.ModuleList([
                ConvTemporalGraphical(time_dim, joints_dim, layout),
                nn.Conv2d(in_channels, out_channels, (self.kernel_size[0], self.kernel_size[1]), (stride,
                                                                                                  stride), padding),
            ]))

        if not residual:
            self.residual = None
        elif stride != 1 or in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual = nn.Identity()

        self.apply(weights_init)

    def forward(self, x):
        if self.residual is not None:
            res = self.residual(x)

        if self.refine:
            y = None
            for stb in self.stgcn:
                z = stb(x)
                y = z if y is None else y + z
            x = y
        else:
            x = self.stgcn(x)

        if self.residual is not None:
            x = x + res
        return x


class DSTDGCN(nn.Module):

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 num_feature=64,
                 num_layers=7,
                 layout="h36m"):

        super(DSTDGCN, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.joints_to_consider = joints_to_consider

        self.encoders = nn.ModuleList()

        all_time_frame = input_time_frame + output_time_frame
        self.conv_st_in = ST_GCNN_layer(input_channels, num_feature, [1, 1], 1, all_time_frame, joints_to_consider,
                                        True, True, False, layout)

        self.bn_in = BatchNorm(num_feature, joints_to_consider, all_time_frame)
        self.do_in = nn.Dropout(st_gcnn_dropout)

        for _ in range(num_layers):
            self.encoders.append(
                nn.Sequential(*nn.ModuleList([
                    ST_GCNN_layer(num_feature, num_feature, [1, 1], 1, input_time_frame +
                                  output_time_frame, joints_to_consider, False, True, True, layout),
                    BatchNorm(num_feature, joints_to_consider, all_time_frame),
                    nn.PReLU(),
                ])))

        self.conv_st_out = ST_GCNN_layer(num_feature, input_channels // 2, [1, 1], 1,
                                         input_time_frame + output_time_frame, joints_to_consider, True, True, False,
                                         layout)

        self.prelu = nn.PReLU()

    def forward(self, x):
        # input
        n, t, v, c = x.shape
        assert t == self.input_time_frame + self.output_time_frame

        x_orig = x.clone()
        residual = x[:, -1:]
        # use motion input
        x = x - residual
        x = torch.cat((x_orig, x), dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv_st_in(x)
        x = self.bn_in(x)
        x = self.prelu(x)
        x = self.do_in(x)

        for gcn in (self.encoders):
            x = gcn(x)

        x = self.conv_st_out(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + residual

        return x

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
        if hasattr(m, 'bias') and m.bias is not None and isinstance(
                m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    # elif classname.find('BatchNorm') != -1:
    #     if hasattr(m, 'weight') and m.weight is not None:
    #         m.weight.data.normal_(1.0, 0.02)
    #     if hasattr(m, 'bias') and m.bias is not None:
    #         m.bias.data.fill_(0)


class BatchNorm(nn.Module):

    def __init__(self, feature_channels, joint_dim, time_dim):
        super(BatchNorm, self).__init__()
        self.c = feature_channels
        self.v = joint_dim
        self.t = time_dim
        self.bn = nn.BatchNorm1d(feature_channels * joint_dim)

    def forward(self, x):
        n, t, v, c = x.shape
        assert (c, t, v) == (self.c, self.t, self.v)
        x = x.permute(0, 2, 3, 1).contiguous().view(n, c * v, t)
        x = self.bn(x)
        x = x.view(n, v, c, t).permute(0, 3, 1, 2).contiguous()
        return x


class DSTDGC(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ref_channels,
                 kpt_channels,
                 red_channels=2,
                 mode="spatial"):
        super(DSTDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ref_channels = ref_channels
        self.kpt_channels = kpt_channels
        self.red_channels = red_channels
        self.mode = mode
        assert mode in {"spatial", "temporal"}
        # channel modeling
        # self.conv_c1 = nn.Conv2d(self.out_channels, 16, 1)
        # self.conv_c2 = nn.Conv2d(self.out_channels, 16, 1)
        # self.conv_rc = nn.Conv2d(16, self.out_channels, 1)

        # spatial / temporal modeling
        # self.conv_m1 = nn.Conv2d(self.in_channels, self.red_channels, 1)
        # self.conv_m2 = nn.Conv2d(self.in_channels, self.red_channels, 1)
        # self.conv_rm = nn.Conv2d(self.red_channels * self.ref_channels,
        #                          self.ref_channels, 1)
        # self.tanh = nn.Tanh()
        self.conv_m1 = nn.Conv2d(self.in_channels, self.red_channels, 1)
        self.conv_m2 = nn.Conv2d(self.in_channels, self.red_channels, 1)
        self.conv_rm = nn.Conv2d(self.red_channels * self.ref_channels,
                                 self.ref_channels, 1)
        self.tanh = nn.Tanh()

        # feature transformation
        # self.conv_f = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.conv_f = nn.Linear(self.in_channels, self.out_channels)

        self.init_parameter()

    def init_parameter(self):
        # stdv = 1. / math.sqrt(self.madj.size(1))
        # self.madj.data.uniform_(-stdv, stdv)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            # elif isinstance(m, nn.BatchNorm2d):
            #     bn_init(m, 1)

    def forward(self, x, A=None, alpha_m=1):
        # xm1, xm2 = self.conv_m1(x), self.conv_m2(x)
        # xf = self.conv_f(x)
        if self.mode == "spatial":
            # xc = self.conv_c(torch.mean(x, -2, True))
            # b, t, v, c = x.shape
            # x = x.permute(0, 1, 3, 2).contiguous()
            xp = x.permute(0, 3, 1, 2).contiguous()
            xf = self.conv_f(x)
            xm1, xm2 = self.conv_m1(xp), self.conv_m2(xp)
            n, c, t, v = xm1.shape
            xm = self.tanh(
                xm1.view(n, c * t, v).unsqueeze(-1) -
                xm2.view(n, c * t, v).unsqueeze(-2))
            xm = self.conv_rm(xm) * alpha_m + A.unsqueeze(0)
            # xfm = torch.einsum("ncvt,ntvw->ncwt", xf, xm)
            # (n, t, v, c)
            xfm = torch.matmul(xm, xf)

            # xc1, xc2 = self.conv_c1(xf).mean(-2), self.conv_c2(xf).mean(-2)
            # xc = self.tanh(xc1.unsqueeze(-1) - xc2.unsqueeze(-2))
            # xc = self.conv_rc(xc) * alpha_c
            # xfc = torch.einsum("nctv,ncvw->nctw", xfm, xc)
            # x = xfm + xfc
        else:
            # xc = self.conv_c(torch.mean(x, -1, True))
            xp = x.permute(0, 3, 2, 1).contiguous()
            x = x.permute(0, 2, 1, 3).contiguous()
            xf = self.conv_f(x)
            xm1, xm2 = self.conv_m1(xp), self.conv_m2(xp)
            n, c, v, t = xm1.shape
            xm = self.tanh(
                xm1.view(n, c * v, t).unsqueeze(-1) -
                xm2.view(n, c * v, t).unsqueeze(-2))
            xm = self.conv_rm(xm) * alpha_m + A.unsqueeze(0)
            # xfm = torch.einsum("ncvt,nvtu->ncvu", xf, xm)

            xfm = torch.matmul(xm, xf)
            xfm = xfm.permute(0, 2, 1, 3).contiguous()
            # x = xc * x

            # xc1, xc2 = self.conv_c1(xf).mean(-1), self.conv_c2(xf).mean(-1)
            # xc = self.tanh(xc1.unsqueeze(-1) - xc2.unsqueeze(-2))
            # xc = self.conv_rc(xc) * alpha_c
            # xfc = torch.einsum("nctv,nctu->ncuv", xfm, xc)
            # x = xfm + xfc
        x = xfm
        return x


class DSTDGCB(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_dim,
                 joint_dim,
                 layout="h36m"):
        super(DSTDGCB, self).__init__()

        # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        graph_gen = Graph(layout)
        time_gen = Time(time_dim)
        A_s = graph_gen.get_all_adjacency()
        A_t = time_gen.get_all_adjacency()

        # spatial modeling
        self.A_s = nn.Parameter(torch.FloatTensor(A_s))
        # temporal modeling
        self.A_t = nn.Parameter(torch.FloatTensor(A_t), False)
        self.R_t = nn.Parameter(torch.zeros_like(self.A_t))

        self.conv_s = nn.ModuleList()
        self.conv_t = nn.ModuleList()

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                BatchNorm(out_channels, joint_dim, time_dim))
        else:
            self.residual = lambda x: x

        for _ in range(A_s.shape[0]):
            self.conv_s.append(
                DSTDGC(in_channels,
                       out_channels,
                       time_dim,
                       joint_dim,
                       mode="spatial"))
        self.alpha_sm = nn.Parameter(torch.zeros(1))
        # self.alpha_sc = nn.Parameter(torch.zeros(1))
        self.bn = BatchNorm(out_channels, joint_dim, time_dim)
        # self.bn_s = nn.BatchNorm2d(out_dim)

        # temporal modeling
        # if A_t is not None:
        #     self.T_fixed = nn.Parameter(torch.FloatTensor(A_t[np.newaxis, ]),
        #                                 requires_grad=False)
        # else:
        #     self.T_fixed = None
        # self.T_rand = nn.Parameter(torch.FloatTensor(1, time_dim, time_dim))
        # stdv = 1. / math.sqrt(self.T_rand.size(1))
        # self.T_rand.data.uniform_(-stdv, stdv)
        # self.conv_t = DSTDGC(out_channels,
        #                      out_channels,
        #                      joint_dim,
        #                      time_dim,
        #                      mode="temporal")
        for _ in range(A_t.shape[0]):
            self.conv_t.append(
                DSTDGC(out_channels,
                       out_channels,
                       joint_dim,
                       time_dim,
                       mode="temporal"))
        self.alpha_tm = nn.Parameter(torch.zeros(1))
        # self.alpha_tc = nn.Parameter(torch.zeros(1))

        self.prelu = nn.PReLU()
        self.do = nn.Dropout(0.1)
        '''
        self.prelu = nn.PReLU()

        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
        # self.init_parameter()

    def init_parameter(self):
        # stds = 1. / math.sqrt(self.R_s.size(1))
        # self.R_s.data.uniform_(-stds, stds)
        stdt = 1. / math.sqrt(self.R_t.size(1))
        self.R_t.data.uniform_(-stdt, stdt)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         conv_init(m)
        # elif isinstance(m, nn.BatchNorm2d):
        #     bn_init(m, 1)

    def forward(self, x):
        # there is no bn activation at the last
        # also the residual connection is removed
        # a = self.A_rand + self.A_fixed if self.A_fixed is not None else self.A_rand
        # x = self.conv_s(x, a, self.alpha_s)
        r = self.residual(x)
        y = None
        for i, conv in enumerate(self.conv_s):
            a_s = self.A_s[i:i + 1]
            z = conv(x, a_s, self.alpha_sm)
            y = y + z if y is not None else z
        x = y
        x = self.bn(x)
        x += r
        x = self.prelu(x)

        # t = self.T_rand + self.T_fixed if self.T_fixed is not None else self.T_rand
        # x = self.conv_t(x, t, self.alpha_tm, self.alpha_tc)
        y = None
        for i, conv in enumerate(self.conv_t):
            a_t = self.A_t[i:i + 1]
            r_t = self.R_t[i:i + 1]
            z = conv(x, a_t + r_t, self.alpha_tm)
            y = y + z if y is not None else z
        x = y
        # x = self.bn_t(x)
        # x = self.act(x)
        return x.contiguous()


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, time_dim, joints_dim, layout="h36m"):
        super(ConvTemporalGraphical, self).__init__()

        self.A = nn.Parameter(
            torch.FloatTensor(time_dim, joints_dim, joints_dim)
        )  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim,
                                                time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

        graph_gen = Graph(layout)
        adj = graph_gen.get_adjacency()[np.newaxis, :]
        self.A_fixed = nn.Parameter(torch.FloatTensor(adj),
                                    requires_grad=False)
        '''
        self.prelu = nn.PReLU()

        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''

    def forward(self, x):
        # update spatial and temporal separately
        # x - (n, c, t, v)
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        x = torch.einsum('nctv,tvw->nctw', (x, self.A + self.A_fixed))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous()


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

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
        padding = ((self.kernel_size[0] - 1) // 2,
                   (self.kernel_size[1] - 1) // 2)

        # self.gcn = ConvRefineTemporalGraphical(in_channels, out_channels,
        #                                        time_dim, joints_dim)

        if refine:
            self.stgcn = nn.ModuleList()
            self.stgcn.append(
                nn.Sequential(*nn.ModuleList([
                    DSTDGCB(in_channels, out_channels, time_dim, joints_dim,
                            layout),
                    # nn.Conv2d(in_channels, out_channels, (self.kernel_size[0],
                    #                                       self.kernel_size[1]),
                    #           (stride, stride), padding),
                    # BatchNorm(out_channels, joints_dim, time_dim),
                ])))
            # A_connect = graph_gen.get_adjacency_type("connect")
            # self.stgcn.append(
            #     nn.Sequential(*nn.ModuleList([
            #         ConvRefineTemporalGraphical(in_channels, time_dim,
            #                                     joints_dim, A_connect),
            #         nn.BatchNorm2d(out_channels),
            #         nn.Dropout(dropout, inplace=True),
            #     ])))
            # A_part = graph_gen.get_adjacency_type("part")
            # random map
            # self.stgcn.append(
            #     nn.Sequential(*nn.ModuleList([
            #         ConvRefineTemporalGraphical(in_channels, time_dim,
            #                                     joints_dim),
            #         nn.BatchNorm2d(out_channels),
            #         nn.Dropout(dropout, inplace=True),
            #     ])))
        else:
            self.stgcn = nn.Sequential(*nn.ModuleList([
                ConvTemporalGraphical(time_dim, joints_dim, layout),
                nn.Conv2d(in_channels, out_channels, (
                    self.kernel_size[0],
                    self.kernel_size[1]), (stride, stride), padding),
            ]))

        # residual connection
        if not residual:
            self.residual = None
        # else:
        #     self.residual = nn.Conv2d(in_channels,
        #                               out_channels,
        #                               kernel_size=1,
        #                               stride=1)
        elif stride != 1 or in_channels != out_channels:
            # conv-based residual
            self.residual = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      stride=1)
        else:
            self.residual = nn.Identity()

        self.apply(weights_init)

    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
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

        # residual connection
        if self.residual is not None:
            x = x + res
        # x = self.prelu(x)
        return x


class DSTDGCN(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

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
        # self.st_gcnns = nn.ModuleList()
        # self.txcnns = nn.ModuleList()

        self.encoders = nn.ModuleList()
        # self.decoders = nn.ModuleList()

        all_time_frame = input_time_frame + output_time_frame
        self.conv_st_in = ST_GCNN_layer(input_channels, num_feature, [1, 1], 1,
                                        all_time_frame, joints_to_consider,
                                        True, True, False, layout)

        # self.bn_in = nn.BatchNorm1d(num_feature * all_time_frame *
        #                             joints_to_consider)
        # self.bn_in = nn.BatchNorm2d(num_feature)
        # self.bn_in = nn.BatchNorm1d(num_feature * joints_to_consider)
        self.bn_in = BatchNorm(num_feature, joints_to_consider, all_time_frame)
        self.do_in = nn.Dropout(st_gcnn_dropout)

        # encoder
        for _ in range(num_layers):
            self.encoders.append(
                nn.Sequential(*nn.ModuleList([
                    ST_GCNN_layer(
                        num_feature, num_feature, [1, 1], 1, input_time_frame +
                        output_time_frame, joints_to_consider, False, True,
                        True, layout),
                    BatchNorm(num_feature, joints_to_consider, all_time_frame),
                    nn.PReLU(),
                    # nn.Dropout(st_gcnn_dropout, inplace=True)
                    # CNN_layer(input_time_frame +
                    #           output_time_frame, input_time_frame +
                    #           output_time_frame, txc_kernel_size, txc_dropout)
                ])))

        # transition
        # self.conv_st_in_out = ST_GCNN_layer(64, 64, [1, 1], 1,
        #                                     input_time_frame,
        #                                     joints_to_consider,
        #                                     st_gcnn_dropout, False, True)
        # self.bn_in_out = nn.BatchNorm2d(64)
        # self.tcns_in_out = CNN_layer(input_time_frame + output_time_frame,
        #                              input_time_frame + output_time_frame,
        #                              txc_kernel_size, txc_dropout)

        # decoder
        # for _ in range(num_layers):
        #     self.decoders.append(
        #         nn.Sequential(*nn.ModuleList([
        #             ST_GCNN_layer(64, 64, [1, 1], 1, output_time_frame,
        #                           joints_to_consider, st_gcnn_dropout, False,
        #                           True),
        #             nn.BatchNorm2d(64),
        #             nn.PReLU()
        #         ])))

        self.conv_st_out = ST_GCNN_layer(num_feature, input_channels // 2,
                                         [1, 1], 1,
                                         input_time_frame + output_time_frame,
                                         joints_to_consider, True, True, False,
                                         layout)

        # for i in range(1, n_txcnn_layers):
        #     self.txcnns.append(
        #         CNN_layer(output_time_frame, output_time_frame,
        #                   txc_kernel_size, txc_dropout))

        self.prelu = nn.PReLU()

    def forward(self, x):
        # input
        # (n, t, v, c)
        n, t, v, c = x.shape
        assert t == self.input_time_frame + self.output_time_frame

        x_orig = x.clone()
        residual = x[:, -1:]
        # x = x[:, :self.input_time_frame]  # filter last joints
        # use motion input
        x = x - residual
        x = torch.cat((x_orig, x), dim=-1)
        # (n, t, v, c) -> (n, c, t, v)
        # x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv_st_in(x)
        # test different bn mechanism
        # n, c, t, v = x.shape
        # x = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        # x = self.bn_in(x)
        # x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()
        # x = self.bn_in(x.view(n, -1)).view(n, c, t, v)
        x = self.bn_in(x)
        x = self.prelu(x)
        x = self.do_in(x)

        # (n, t, v, c) -> (n, c, t, v)
        # x = x.permute(0, 2, 1, 3).contiguous()

        # x = self.tcns_in_out(x)
        # # x = self.prelu(x)
        # x = x.permute(0, 2, 1, 3).contiguous()

        # encoder
        # gc_ins = []
        for gcn in (self.encoders):
            x = gcn(x)
            # gc_ins.append(x)

        # feature transformation
        # x = self.conv_st_in_out(x)
        # x = self.bn_in_out(x)
        # x = self.prelu(x)
        # x = x.permute(0, 2, 1, 3).contiguous()
        # x = self.tcns_in_out(x)
        # x = self.prelu(x)
        # x = x.permute(0, 2, 1, 3).contiguous()

        # decoder
        # for i, gcn in enumerate(self.decoders):
        #     pdb.set_trace()
        #     x = gcn(x)
        # output
        x = self.conv_st_out(x)

        # (n, c, t, v) -> (n, t, v, c)
        # x = x.permute(0, 2, 3, 1).contiguous()

        # only consider the input frames
        # x = x[:, -25:]

        # motion to position
        x = x + residual
        # concatenate the input channel
        # x = torch.cat((x_in, x), dim=1)

        return x

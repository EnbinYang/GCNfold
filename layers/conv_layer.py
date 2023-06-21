import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv

class ListReadoutLayer(nn.Module):
    """List Readout Layer"""
    def __init__(self, in_channel, window_size=501):
        super().__init__()
        self.in_channel = in_channel
        self.window_size = window_size

    def forward(self, g, feat):
        output = torch.reshape(feat, (feat.shape[0]//self.window_size, self.window_size, feat.shape[1]))
        output =  torch.flatten(output, start_dim=1)
        # start, output = 0, 0
        # first_flag = 0
        # for batch_num in g.batch_num_nodes:
        #     if first_flag == 0:
        #         output = torch.flatten(feat[start:start + batch_num].unsqueeze(0), start_dim=1)
        #         first_flag = 1
        #         continue
        #     output = torch.cat([output, torch.flatten(feat[start:start + batch_num].unsqueeze(0), start_dim=1)], dim=0)
        return output


class GraphConvLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_channel, out_channel, kernel_size, activation, batch_norm,
                 dropout=False, residual=False, window_size=501):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.residual = residual
        self.window_size = window_size

        if in_channel != out_channel:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_channel)
        self.activation = activation
        # self.dropout = nn.Dropout(dropout)

        self.conv_layer = ConvReadoutLayer(in_channel, out_channel, kernel_size, padding=kernel_size[0] // 2, stride=1)
        self.graph_conv_layer = GraphConv(out_channel, out_channel)

    def forward(self, g, feature):
        h_in = feature  # to be used for residual connection

        h = self.conv_layer(g, feature)
        h = self.graph_conv_layer(g, h)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        # h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channel,
                                                                         self.out_channel, self.residual)


class ConvLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_channel, out_channel, kernel_size, activation, batch_norm, padding=0, residual=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.residual = residual
        self.padding = padding

        if in_channel != out_channel:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm2d(out_channel)
        self.activation = activation
        # self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)

    def forward(self, feature):
        h_in = feature  # to be used for residual connection

        h = self.conv(feature)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        # h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channel,
                                                                         self.out_channel, self.residual)


class MAXPoolLayer(nn.Module):
    """MAXPool layer."""
    def __init__(self, kernel_size, stride, padding=0, **kwargs):
        super().__init__()

        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding

        self.pooling = nn.MaxPool2d(kernel_size, stride, padding=padding)

    def forward(self, inputs):
        output = self.pooling(inputs)
        return output

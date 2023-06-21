import torch
import dgl
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import GraphConv

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""

# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
# msg = fn.copy_src(src='h', out='m')
#
# def reduce(nodes):
#     accum = torch.mean(nodes.mailbox['m'], 1)
#     return {'h': accum}

# def msg_func(edges):
#     return {'m': torch.mul(edges.data['feat'], edges.src['h'])}

msg_func = fn.u_mul_e('h', 'feat', 'm')
reduce_mean = fn.mean('m', 'h')
reduce_sum = fn.sum('m', 'h')
reduce_max = fn.max('m', 'h')


# def reduce(nodes):
#     accum = torch.sum(nodes.mailbox['m'], 1)
#     return {'h': accum}


class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}


class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False, dgl_builtin=True):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin

        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.dgl_builtin == False:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        else:
            self.conv = GraphConv(in_dim, out_dim)

    def forward(self, g, feature):
        h_in = feature  # to be used for residual connection

        if self.dgl_builtin == False:
            g.ndata['h'] = feature
            # g.update_all(msg, reduce)
            g.update_all(message_func=msg_func, reduce_func=reduce_mean)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h']  # result of graph convolution
        else:
            h = self.conv(g, feature)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h  # torch.Size([64128, 32])

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.out_channels, self.residual)


class GNNPoolLayer(nn.Module):
    def __init__(self, stride=2, batch_size=128, node_num=501):
        super().__init__()
        self.stride = stride
        self.node_num = node_num
        self.ind_graph = [i*2 for i in range((node_num + 1)//stride)]
        self.ind_feat = [j*2 + i*node_num for i in range(batch_size) for j in range((node_num + 1)//2)]
        self.conv_readout_layer = ConvReadoutLayer()

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(message_func=msg_func, reduce_func=reduce_mean)

        sub_graph_list = dgl.unbatch(g)

        new_graph_list = []
        for i in range(len(sub_graph_list)):
            new_graph = sub_graph_list[i].subgraph(self.ind_graph)
            new_graph.copy_from_parent()
            new_graph_list.append(new_graph)

        new_graph = dgl.batch(new_graph_list)
        # new_graph = dgl.add_self_loop(new_graph)
        del g, sub_graph_list, feature

        return new_graph, new_graph.ndata['h'], new_graph.edata['feat']

    def __repr__(self):
        return '{}(stride={}, node_num={})'.format(self.__class__.__name__, self.stride, self.node_num)


class WeightCrossLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.relu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.encoder = nn.Linear(in_dim, 32)
        self.decoder = nn.Linear(32, out_dim)

    def forward(self, feature):
        output = self.encoder(feature)
        output = self.activation(output)
        output = self.decoder(output)

        return torch.sigmoid(output)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)


class ConvReadoutLayerBACK(nn.Module):
    """Conv Readout layer."""
    def __init__(self, mode="mean", **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, g, feat):
        start, output = 0, 0

        for i, batch_num in enumerate(g.batch_num_nodes):
            if i == 0:
                readout = feat[start:start + batch_num]
                readout += feat[start + batch_num:start + 2 * batch_num]
                output = torch.transpose(readout/2, 1, 0).unsqueeze(0)
            elif i == len(g.batch_num_nodes) - 1:
                readout = feat[start - batch_num:start]
                readout += feat[start:start + batch_num]
                output = torch.cat([output, torch.transpose(readout/2, 1, 0).unsqueeze(0)], dim=0)
            else:
                readout = feat[start - batch_num:start]
                readout += feat[start:start + batch_num]
                readout += feat[start + batch_num:start + 2*batch_num]
                output = torch.cat([output, torch.transpose(readout/3, 1, 0).unsqueeze(0)], dim=0)
            start += batch_num

        return output.unsqueeze(-1)


class ConvReadoutLayer(nn.Module):
    """Conv Readout layer."""
    def __init__(self, mode="mean", **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, g, feat):
        # feat: torch.Size([64128, 32])
        start, output = 0, 0
        first_flag = 0

        for batch_num in g.batch_num_nodes:
            if first_flag == 0:
                # g.batch_num_nodes: 501
                output = torch.transpose(feat[start: start+batch_num], 1, 0).unsqueeze(0)
                # output: torch.Size([1, 32, 501])
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(feat[start: start+batch_num], 1, 0).unsqueeze(0)], dim=0)
                # final output: torch.Size([128, 32, 501])
            start += batch_num

        return output.unsqueeze(-1)  # torch.Size([128, 32, 501, 1])






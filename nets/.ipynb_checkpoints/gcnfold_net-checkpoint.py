import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl
from layers.gcn_layer import GCNLayer, ConvReadoutLayer, GNNPoolLayer, WeightCrossLayer
from layers.mlp_readout_layer import MLPReadout
from layers.conv_layer import ConvLayer, MAXPoolLayer

CH_FOLD2 = 1

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class GCNFoldNet(nn.Module):
    def __init__(self, d, L, device, net_params):
        super().__init__()
        in_dim = net_params['in_dim']  # in_dim=4
        hidden_dim = net_params['hidden_dim']  # 32
        out_dim = net_params['out_dim']  # 32
        in_feat_dropout = net_params['in_feat_dropout']  # 0.25
        dropout = net_params['dropout']  # 0.25

        self.d = d
        self.L = L
        self.device = net_params['device']
        self.n_layers = net_params['L']  # 2
        self.readout = net_params['readout']  # mean
        self.batch_norm = net_params['batch_norm']  # ture
        self.residual = net_params['residual']  # ture
        self.pre_gnn, self.pre_cnn = None, None
        self.base_weight = None
        self.node_weight = None
        self.sequence = None
        self.filter_out = None

        window_size = 600
        conv_kernel1, conv_kernel2 = [9, 4], [9, 1]
        conv_padding, conv_stride = [conv_kernel1[0] // 2, 0], 1  # //除完求商再向下取整
        pooling_kernel = [3, 1]
        pooling_padding, pooling_stride = [pooling_kernel[0] // 2, 0], 2

        # math.ceil() 向上取整
        width_o1 = math.ceil((window_size - conv_kernel1[0] + 2 * conv_padding[0] + 1) / conv_stride)  # 501
        width_o1 = math.ceil((width_o1 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)  # 251
        width_o2 = math.ceil((width_o1 - conv_kernel2[0] + 2 * conv_padding[0] + 1) / conv_stride)  # 251
        width_o2 = math.ceil((width_o2 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)  # 126

        # GNN start
        self.embedding_h = nn.Linear(in_dim, hidden_dim)  # Linear层 (4, 32)
        self.embedding_hg = nn.Linear(1, d)  # Linear层 (1, 10)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)  # dropout一般用在nn.Linear后 (部分神经元有0.25的概率不被激活)

        # four layers: GCN
        self.layers_gnn = nn.ModuleList()
        self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        for _ in range(self.n_layers * 2 - 2):
            self.layers_gnn.append(
                GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        self.layers_gnn.append(GCNLayer(hidden_dim, out_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        # GNN end

        # CNN start
        self.conv_readout_layer = ConvReadoutLayer(self.readout)  # readout
        self.layers_cnn = nn.ModuleList()
        self.layers_cnn.append(
            ConvLayer(1, 32, conv_kernel1, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))
        self.conv1d1 = nn.Conv1d(in_channels=4, out_channels=d, kernel_size=9, padding=8, dilation=2)

        self.conv_test_1 = nn.Conv2d(in_channels=8 * d, out_channels=d, kernel_size=1)
        self.bn_conv_1 = nn.BatchNorm2d(d)
        self.conv_test_2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1)
        self.bn_conv_2 = nn.BatchNorm2d(d)
        self.conv_test_3 = nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1)

        self.batchnorm_weight = nn.BatchNorm1d(window_size)  # 501
        self.bn1 = nn.BatchNorm1d(d)  # 10

        self.PE_net = nn.Sequential(nn.Linear(111, 5 * d), nn.ReLU(), nn.Linear(5 * d, 5 * d), nn.ReLU(),
                                    nn.Linear(5 * d, d))  # MLP

        # transformer block
        self.encoder_layer = nn.TransformerEncoderLayer(3 * d, 2)  # (d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)  # three layers Transformer

    def forward(self, g, h, e, pe, seq, state):  # graph
        batch_size = len(g.batch_num_nodes)
        window_size = g.batch_num_nodes[0]
        similar_loss = 0
        gcn_layers = 4
        cnn_node_weight = 0
        weight2gnn_list = []
        weight2cnn_list = []

        h2 = self._graph2feature(g)  # _graph2feature函数 torch.Size([128, 1, 501, 4])
        self.sequence = h2
        h2 = h2.to(self.device)  # 后续进入CNN操作

        h1 = self.embedding_h(h)  # in -> hidden; torch.Size([64128, 4]) -> torch.Size([64128, 32])
        h1 = self.in_feat_dropout(h1)  # dropout

        # GNN
        for i in range(gcn_layers):
            h1 = self.layers_gnn[i](g, h1)  # 0, 1, 3, 4 torch.Size([64128, 32])

        # CNN
        h2 = self.layers_cnn[0](h2)  # 0: [128, 32, 501, 1]
        self.filter_out = h2
        cnn_node_weight = torch.mean(h2, dim=1).squeeze(-1)  # Average pool
        self.base_weight = self.batchnorm_weight(cnn_node_weight)  # batchnorm
        cnn_node_weight = torch.sigmoid(self.batchnorm_weight(cnn_node_weight))  # sigmoid [128, 501]

        # 经过四层GraphConv得到h1
        g.ndata['h'] = h1  # torch.Size([64128, 32])

        # 最终hg是GCN和CNN组合的结果
        hg = self.conv_readout_layer(g, h1)  # torch.Size([128, 32, 501, 1])
        hg = torch.mul(hg, cnn_node_weight.unsqueeze(1).unsqueeze(-1))  # H(4)*CNN_weight torch.Size([128, 32, 501, 1])
        hg = self.embedding_hg(hg)  # torch.Size([128, 32, 501, 10])
        hg = self.in_feat_dropout(hg)
        hg = torch.mean(hg, dim=1)  # torch.Size([128, 501, 10])
        hg = hg.permute(0, 2, 1)  # torch.Size([128, 10, 501])

        # load sequence data [128, 600, 4]
        seq = seq.permute(0, 2, 1)  # [128, 600, 4] -> [128, 4, 600]
        seq = F.relu(self.bn1(self.conv1d1(seq)))  # [128, 4, 600] -> [128, 10, 600]

        # load position embedding and combind with the hg, seq, and position_embeds information
        position_embeds = self.PE_net(pe.view(-1, 111)).view(-1, self.L, self.d)  # N*L*111 -> N*L*d
        position_embeds = position_embeds.permute(0, 2, 1)  # N*d*L
        seq = torch.cat([hg, seq, position_embeds], 1)  # N*3d*L

        # input to the transformer block
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))  # N*3d*L -> L*N*3d
        seq = seq.permute(1, 2, 0)  # L*N*3d -> N*3d*L

        seq_mat = self.matrix_rep(seq)  # N*3d*L -> N*6d*L*L
        p_mat = self.matrix_rep(position_embeds)  # N*d*L -> N*2d*L*L
        infor = torch.cat([seq_mat, p_mat], 1)  # N*8d*L*L

        contact = F.relu(self.bn_conv_1(self.conv_test_1(infor)))  # N*8d*L*L -> N*d*L*L
        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))  # N*d*L*L -> N*d*L*L
        contact = self.conv_test_3(contact)  # N*1*L*L

        contact = contact.view(-1, self.L, self.L)  # N*1*L*L -> N*L*L
        contact = (contact + torch.transpose(contact, -1, -2)) / 2  # Symmetrization

        return contact.view(-1, self.L, self.L)

    def matrix_rep(self, x):
        x = x.permute(0, 2, 1)  # N*d*l -> N*L*d
        L = x.shape[1]  # 600
        x2 = x  # N*L*d
        x = x.unsqueeze(1)  # N*L*d -> N*1*L*d
        x2 = x2.unsqueeze(2)  # N*L*d -> N*L*1*d
        x = x.repeat(1, L, 1, 1)  # N*1*L*d -> N*L*L*d
        x2 = x2.repeat(1, 1, L, 1)  # N*L*1*d -> N*L*L*d
        mat = torch.cat([x, x2], -1)  # N*L*L*d -> N*L*L*2d

        # return a symmetric matrix (mat)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2))  # N*2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def _graph2feature(self, g):
        feat = g.ndata['feat']  # 提取节点特征 torch.Size([64128, 4])
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes:
            # g.batch_num_nodes: 此批次中每个图的节点数(501)
            if first_flag == 0:
                output = torch.transpose(feat[start: start + batch_num], 1, 0).unsqueeze(0)  # 转置
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(feat[start: start + batch_num], 1, 0).unsqueeze(0)], dim=0)
            start += batch_num
        output = torch.transpose(output, 1, 2)
        output = output.unsqueeze(1)
        return output

class GCNFoldNet_UNet(nn.Module):
    def __init__(self, d, L, device, net_params):
        super().__init__()
        in_dim = net_params['in_dim']  # in_dim=4
        hidden_dim = net_params['hidden_dim']  # 32
        out_dim = net_params['out_dim']  # 32
        in_feat_dropout = net_params['in_feat_dropout']  # 0.25
        dropout = net_params['dropout']  # 0.25

        self.d = d
        self.L = L
        self.device = net_params['device']
        self.n_layers = net_params['L']  # 2
        self.readout = net_params['readout']  # mean
        self.batch_norm = net_params['batch_norm']  # ture
        self.residual = net_params['residual']  # ture
        self.pre_gnn, self.pre_cnn = None, None
        self.base_weight = None
        self.node_weight = None
        self.sequence = None
        self.filter_out = None

        window_size = 600
        conv_kernel1, conv_kernel2 = [9, 4], [9, 1]
        conv_padding, conv_stride = [conv_kernel1[0] // 2, 0], 1  # //除完求商再向下取整
        pooling_kernel = [3, 1]
        pooling_padding, pooling_stride = [pooling_kernel[0] // 2, 0], 2

        # math.ceil() 向上取整
        width_o1 = math.ceil((window_size - conv_kernel1[0] + 2 * conv_padding[0] + 1) / conv_stride)  # 501
        width_o1 = math.ceil((width_o1 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)  # 251
        width_o2 = math.ceil((width_o1 - conv_kernel2[0] + 2 * conv_padding[0] + 1) / conv_stride)  # 251
        width_o2 = math.ceil((width_o2 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)  # 126

        # GNN start
        self.embedding_h = nn.Linear(in_dim, hidden_dim)  # Linear层 (4, 32)
        self.embedding_hg = nn.Linear(1, d)  # Linear层 (1, 10)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)  # dropout一般用在nn.Linear后 (部分神经元有0.25的概率不被激活)

        # four layers: GCN
        self.layers_gnn = nn.ModuleList()
        self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        self.layers_gnn.append(GCNLayer(hidden_dim, out_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        # GNN end

        # CNN start
        self.conv_readout_layer = ConvReadoutLayer(self.readout)  # readout
        self.layers_cnn = nn.ModuleList()
        self.layers_cnn.append(
            ConvLayer(1, 32, conv_kernel1, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))
        self.conv1d1 = nn.Conv1d(in_channels=4, out_channels=d, kernel_size=9, padding=8, dilation=2)

        self.conv_test_1 = nn.Conv2d(in_channels=32, out_channels=d, kernel_size=1)
        self.bn_conv_1 = nn.BatchNorm2d(d)
        self.conv_test_2 = nn.Conv2d(in_channels=9 * d, out_channels=d, kernel_size=1)
        self.bn_conv_2 = nn.BatchNorm2d(d)
        self.conv_test_3 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1)
        self.bn_conv_3 = nn.BatchNorm2d(d)
        self.conv_test_4 = nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1, stride=1)

        self.batchnorm_weight = nn.BatchNorm1d(window_size)  # 501
        self.bn1 = nn.BatchNorm1d(d)  # 10

        self.PE_net = nn.Sequential(nn.Linear(111, 5 * d), nn.ReLU(), nn.Linear(5 * d, 5 * d), nn.ReLU(),
                                    nn.Linear(5 * d, d))  # MLP

        # transformer block
        self.encoder_layer = nn.TransformerEncoderLayer(3 * d, 2)  # (d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)  # three layers Transformer

        # 给定输入ch_in和输出ch_out, CH_FOLD2 = 1
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=2 * d, ch_out=int(32 * CH_FOLD2))  # (2d, 32)
        self.Conv2 = conv_block(ch_in=int(32 * CH_FOLD2), ch_out=int(64 * CH_FOLD2))  # (32, 64)
        self.Conv3 = conv_block(ch_in=int(64 * CH_FOLD2), ch_out=int(128 * CH_FOLD2))  # (64, 128)
        self.Conv4 = conv_block(ch_in=int(128 * CH_FOLD2), ch_out=int(256 * CH_FOLD2))  # (128, 256)

        self.Up4 = up_conv(ch_in=int(256 * CH_FOLD2), ch_out=int(128 * CH_FOLD2))  # (256, 128) up_conv
        self.Up_conv4 = conv_block(ch_in=int(256 * CH_FOLD2), ch_out=int(128 * CH_FOLD2))  # (256, 128) conv_block

        self.Up3 = up_conv(ch_in=int(128 * CH_FOLD2), ch_out=int(64 * CH_FOLD2))  # (128, 64) up_conv
        self.Up_conv3 = conv_block(ch_in=int(128 * CH_FOLD2), ch_out=int(64 * CH_FOLD2))  # (128, 64) up_conv

        self.Up2 = up_conv(ch_in=int(64 * CH_FOLD2), ch_out=int(32 * CH_FOLD2))  # (64, 32) up_conv
        self.Up_conv2 = conv_block(ch_in=int(64 * CH_FOLD2), ch_out=int(32 * CH_FOLD2))  # (64, 32) conv_block

    def forward(self, g, h, e, pe, seq, state):  # graph
        batch_size = len(g.batch_num_nodes)
        window_size = g.batch_num_nodes[0]
        similar_loss = 0
        gcn_layers = 3
        cnn_node_weight = 0
        weight2gnn_list = []
        weight2cnn_list = []

        h2 = self._graph2feature(g)  # _graph2feature函数 torch.Size([128, 1, 501, 4])
        self.sequence = h2
        h2 = h2.to(self.device)  # 后续进入CNN操作

        h1 = self.embedding_h(h)  # in -> hidden; torch.Size([64128, 4]) -> torch.Size([64128, 32])
        h1 = self.in_feat_dropout(h1)  # dropout

        # GNN
        for i in range(gcn_layers):
            h1 = self.layers_gnn[i](g, h1)  # 0, 1, 3, 4 torch.Size([64128, 32])

        # CNN
        h2 = self.layers_cnn[0](h2)  # 0: [128, 32, 501, 1]
        self.filter_out = h2
        cnn_node_weight = torch.mean(h2, dim=1).squeeze(-1)  # Average pool
        self.base_weight = self.batchnorm_weight(cnn_node_weight)  # batchnorm
        cnn_node_weight = torch.sigmoid(self.batchnorm_weight(cnn_node_weight))  # sigmoid [128, 501]
        
        # tmp
        cnn_node_weight_mean = cnn_node_weight.squeeze(0) 
        cnn_node_weight_mean_np = np.array(cnn_node_weight_mean.cpu())

        # 经过四层GraphConv得到h1
        g.ndata['h'] = h1  # torch.Size([64128, 32])

        # 最终hg是GCN和CNN组合的结果
        hg = self.conv_readout_layer(g, h1)  # torch.Size([128, 32, 501, 1])
        hg = torch.mul(hg, cnn_node_weight.unsqueeze(1).unsqueeze(-1))  # H(4)*CNN_weight torch.Size([128, 32, 501, 1])
        
        # tmp
        hg_mean = torch.mean(hg, dim=1)
        hg_mean = hg_mean.squeeze(0)
        hg_mean = hg_mean.squeeze(1)
        hg_mean_np = np.array(hg_mean.cpu())
        
        hg = self.embedding_hg(hg)  # torch.Size([128, 32, 501, 10])
        hg = self.in_feat_dropout(hg)
        hg = torch.mean(hg, dim=1)  # torch.Size([128, 501, 10])
        hg = hg.permute(0, 2, 1)  # torch.Size([128, 10, 501])

        # load sequence data [128, 600, 4]
        seq_signal = seq.permute(0, 2, 1)  # [128, 600, 4] -> [128, 4, 600]
        seq_signal = F.relu(self.bn1(self.conv1d1(seq_signal)))  # [128, 4, 600] -> [128, 10, 600]

        # load position embedding and combind with the hg, seq, and position_embeds information
        position_embeds = self.PE_net(pe.view(-1, 111)).view(-1, self.L, self.d)  # N*L*111 -> N*L*d
        position_embeds = position_embeds.permute(0, 2, 1)  # N*d*L
        seq = torch.cat([hg, seq_signal, position_embeds], 1)  # N*3d*L

        # input to the transformer block
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))  # N*3d*L -> L*N*3d
        seq = seq.permute(1, 2, 0)  # L*N*3d -> N*3d*L
        seq_mat = self.matrix_rep(seq)  # N*3d*L -> N*6d*L*L

        seq_signal_mat = self.matrix_rep(seq_signal)  # N*d*L -> N*2d*L*L
        p_mat = self.matrix_rep(position_embeds)  # N*d*L -> N*2d*L*L

        # encoding path
        x1 = self.Conv1(seq_signal_mat)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # (32, 64)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        contact = F.relu(self.bn_conv_1(self.conv_test_1(d2)))  # N*32*L*L -> N*d*L*L
        contact = torch.cat([seq_mat, p_mat, contact], 1)  # N*9d*L*L

        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))  # N*9d*L*L -> # N*d*L*L
        contact = F.relu(self.bn_conv_3(self.conv_test_3(contact)))  # N*d*L*L -> N*d*L*L
        contact = self.conv_test_4(contact)  # N*d*L*L -> N*1*L*L

        contact = contact.view(-1, self.L, self.L)  # N*1*L*L -> N*L*L
        contact = (contact + torch.transpose(contact, -1, -2)) / 2  # Symmetrization

        return contact.view(-1, self.L, self.L)

    def matrix_rep(self, x):
        x = x.permute(0, 2, 1)  # N*d*l -> N*L*d
        L = x.shape[1]  # 600
        x2 = x  # N*L*d
        x = x.unsqueeze(1)  # N*L*d -> N*1*L*d
        x2 = x2.unsqueeze(2)  # N*L*d -> N*L*1*d
        x = x.repeat(1, L, 1, 1)  # N*1*L*d -> N*L*L*d
        x2 = x2.repeat(1, 1, L, 1)  # N*L*1*d -> N*L*L*d
        mat = torch.cat([x, x2], -1)  # N*L*L*d -> N*L*L*2d

        # return a symmetric matrix (mat)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2))  # N*2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def _graph2feature(self, g):
        feat = g.ndata['feat']  # 提取节点特征 torch.Size([64128, 4])
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes:
            # g.batch_num_nodes: 此批次中每个图的节点数(501)
            if first_flag == 0:
                output = torch.transpose(feat[start: start + batch_num], 1, 0).unsqueeze(0)  # 转置
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(feat[start: start + batch_num], 1, 0).unsqueeze(0)], dim=0)
            start += batch_num
        output = torch.transpose(output, 1, 2)
        output = output.unsqueeze(1)
        return output
    
    
class GCNFoldNet_UNet_small(nn.Module):
    def __init__(self, d, L, device, net_params):
        super().__init__()
        in_dim = net_params['in_dim']  # in_dim=4
        hidden_dim = net_params['hidden_dim']  # 32
        out_dim = net_params['out_dim']  # 32
        in_feat_dropout = net_params['in_feat_dropout']  # 0.25
        dropout = net_params['dropout']  # 0.25

        self.d = d
        self.L = L
        self.device = net_params['device']
        self.n_layers = net_params['L']  # 2
        self.readout = net_params['readout']  # mean
        self.batch_norm = net_params['batch_norm']  # ture
        self.residual = net_params['residual']  # ture
        self.pre_gnn, self.pre_cnn = None, None
        self.base_weight = None
        self.node_weight = None
        self.sequence = None
        self.filter_out = None

        window_size = 600
        conv_kernel1, conv_kernel2 = [9, 4], [9, 1]
        conv_padding, conv_stride = [conv_kernel1[0] // 2, 0], 1  # //除完求商再向下取整
        pooling_kernel = [3, 1]
        pooling_padding, pooling_stride = [pooling_kernel[0] // 2, 0], 2

        # math.ceil() 向上取整
        width_o1 = math.ceil((window_size - conv_kernel1[0] + 2 * conv_padding[0] + 1) / conv_stride)  # 501
        width_o1 = math.ceil((width_o1 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)  # 251
        width_o2 = math.ceil((width_o1 - conv_kernel2[0] + 2 * conv_padding[0] + 1) / conv_stride)  # 251
        width_o2 = math.ceil((width_o2 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)  # 126

        # GNN start
        self.embedding_h = nn.Linear(in_dim, hidden_dim)  # Linear层 (4, 32)
        self.embedding_hg = nn.Linear(1, d)  # Linear层 (1, 10)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)  # dropout一般用在nn.Linear后 (部分神经元有0.25的概率不被激活)

        # four layers: GCN
        self.layers_gnn = nn.ModuleList()
        self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        self.layers_gnn.append(GCNLayer(hidden_dim, out_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        # GNN end

        # CNN start
        self.conv_readout_layer = ConvReadoutLayer(self.readout)  # readout
        self.layers_cnn = nn.ModuleList()
        self.layers_cnn.append(
            ConvLayer(1, 32, conv_kernel1, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))
        self.conv1d1 = nn.Conv1d(in_channels=4, out_channels=d, kernel_size=9, padding=8, dilation=2)

        self.conv_test_1 = nn.Conv2d(in_channels=32, out_channels=d, kernel_size=1)
        self.bn_conv_1 = nn.BatchNorm2d(d)
        self.conv_test_2 = nn.Conv2d(in_channels=9 * d, out_channels=d, kernel_size=1)
        self.bn_conv_2 = nn.BatchNorm2d(d)
        self.conv_test_3 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1)
        self.bn_conv_3 = nn.BatchNorm2d(d)
        self.conv_test_4 = nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1, stride=1)

        self.batchnorm_weight = nn.BatchNorm1d(window_size)  # 501
        self.bn1 = nn.BatchNorm1d(d)  # 10

        self.PE_net = nn.Sequential(nn.Linear(111, 5 * d), nn.ReLU(), nn.Linear(5 * d, 5 * d), nn.ReLU(),
                                    nn.Linear(5 * d, d))  # MLP

        # transformer block
        self.encoder_layer = nn.TransformerEncoderLayer(3 * d, 2)  # (d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)  # three layers Transformer

        # 给定输入ch_in和输出ch_out, CH_FOLD2 = 1
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=2 * d, ch_out=int(32 * CH_FOLD2))  # (2d, 32)
        self.Conv2 = conv_block(ch_in=int(32 * CH_FOLD2), ch_out=int(64 * CH_FOLD2))  # (32, 64)
        self.Conv3 = conv_block(ch_in=int(64 * CH_FOLD2), ch_out=int(128 * CH_FOLD2))  # (64, 128)

        self.Up3 = up_conv(ch_in=int(128 * CH_FOLD2), ch_out=int(64 * CH_FOLD2))  # (128, 64) up_conv
        self.Up_conv3 = conv_block(ch_in=int(128 * CH_FOLD2), ch_out=int(64 * CH_FOLD2))  # (128, 64) up_conv

        self.Up2 = up_conv(ch_in=int(64 * CH_FOLD2), ch_out=int(32 * CH_FOLD2))  # (64, 32) up_conv
        self.Up_conv2 = conv_block(ch_in=int(64 * CH_FOLD2), ch_out=int(32 * CH_FOLD2))  # (64, 32) conv_block

    def forward(self, g, h, e, pe, seq, state):  # graph
        batch_size = len(g.batch_num_nodes)
        window_size = g.batch_num_nodes[0]
        similar_loss = 0
        gcn_layers = 3
        cnn_node_weight = 0
        weight2gnn_list = []
        weight2cnn_list = []

        h2 = self._graph2feature(g)  # _graph2feature函数 torch.Size([128, 1, 501, 4])
        self.sequence = h2
        h2 = h2.to(self.device)  # 后续进入CNN操作

        h1 = self.embedding_h(h)  # in -> hidden; torch.Size([64128, 4]) -> torch.Size([64128, 32])
        h1 = self.in_feat_dropout(h1)  # dropout

        # GNN
        for i in range(gcn_layers):
            h1 = self.layers_gnn[i](g, h1)  # 0, 1, 3, 4 torch.Size([64128, 32])

        # CNN
        h2 = self.layers_cnn[0](h2)  # 0: [128, 32, 501, 1]
        self.filter_out = h2
        cnn_node_weight = torch.mean(h2, dim=1).squeeze(-1)  # Average pool
        self.base_weight = self.batchnorm_weight(cnn_node_weight)  # batchnorm
        cnn_node_weight = torch.sigmoid(self.batchnorm_weight(cnn_node_weight))  # sigmoid [128, 501]

        # 经过四层GraphConv得到h1
        g.ndata['h'] = h1  # torch.Size([64128, 32])

        # 最终hg是GCN和CNN组合的结果
        hg = self.conv_readout_layer(g, h1)  # torch.Size([128, 32, 501, 1])
        hg = torch.mul(hg, cnn_node_weight.unsqueeze(1).unsqueeze(-1))  # H(4)*CNN_weight torch.Size([128, 32, 501, 1])
        hg = self.embedding_hg(hg)  # torch.Size([128, 32, 501, 10])
        hg = self.in_feat_dropout(hg)
        hg = torch.mean(hg, dim=1)  # torch.Size([128, 501, 10])
        hg = hg.permute(0, 2, 1)  # torch.Size([128, 10, 501])

        # load sequence data [128, 600, 4]
        seq_signal = seq.permute(0, 2, 1)  # [128, 600, 4] -> [128, 4, 600]
        seq_signal = F.relu(self.bn1(self.conv1d1(seq_signal)))  # [128, 4, 600] -> [128, 10, 600]

        # load position embedding and combind with the hg, seq, and position_embeds information
        position_embeds = self.PE_net(pe.view(-1, 111)).view(-1, self.L, self.d)  # N*L*111 -> N*L*d
        position_embeds = position_embeds.permute(0, 2, 1)  # N*d*L
        seq = torch.cat([hg, seq_signal, position_embeds], 1)  # N*3d*L

        # input to the transformer block
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))  # N*3d*L -> L*N*3d
        seq = seq.permute(1, 2, 0)  # L*N*3d -> N*3d*L
        seq_mat = self.matrix_rep(seq)  # N*3d*L -> N*6d*L*L

        seq_signal_mat = self.matrix_rep(seq_signal)  # N*d*L -> N*2d*L*L
        p_mat = self.matrix_rep(position_embeds)  # N*d*L -> N*2d*L*L

        # encoding path
        x1 = self.Conv1(seq_signal_mat)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # (32, 64)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        contact = F.relu(self.bn_conv_1(self.conv_test_1(d2)))  # N*32*L*L -> N*d*L*L
        contact = torch.cat([seq_mat, p_mat, contact], 1)  # N*9d*L*L

        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))  # N*9d*L*L -> # N*d*L*L
        contact = F.relu(self.bn_conv_3(self.conv_test_3(contact)))  # N*d*L*L -> N*d*L*L
        contact = self.conv_test_4(contact)  # N*d*L*L -> N*1*L*L

        contact = contact.view(-1, self.L, self.L)  # N*1*L*L -> N*L*L
        contact = (contact + torch.transpose(contact, -1, -2)) / 2  # Symmetrization

        return contact.view(-1, self.L, self.L)

    def matrix_rep(self, x):
        x = x.permute(0, 2, 1)  # N*d*l -> N*L*d
        L = x.shape[1]  # 600
        x2 = x  # N*L*d
        x = x.unsqueeze(1)  # N*L*d -> N*1*L*d
        x2 = x2.unsqueeze(2)  # N*L*d -> N*L*1*d
        x = x.repeat(1, L, 1, 1)  # N*1*L*d -> N*L*L*d
        x2 = x2.repeat(1, 1, L, 1)  # N*L*1*d -> N*L*L*d
        mat = torch.cat([x, x2], -1)  # N*L*L*d -> N*L*L*2d

        # return a symmetric matrix (mat)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2))  # N*2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def _graph2feature(self, g):
        feat = g.ndata['feat']  # 提取节点特征 torch.Size([64128, 4])
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes:
            # g.batch_num_nodes: 此批次中每个图的节点数(501)
            if first_flag == 0:
                output = torch.transpose(feat[start: start + batch_num], 1, 0).unsqueeze(0)  # 转置
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(feat[start: start + batch_num], 1, 0).unsqueeze(0)], dim=0)
            start += batch_num
        output = torch.transpose(output, 1, 2)
        output = output.unsqueeze(1)
        return output
    

class RNA_SS_e2e(nn.Module):
    def __init__(self, model_score, model_pp):
        super(RNA_SS_e2e, self).__init__()
        self.model_score = model_score
        self.model_pp = model_pp

    def forward(self, g, h, e, pe, seq, state):
        u = self.model_score.forward(g, h, e, pe, seq, state)
        map_list = self.model_pp(u, seq)
        return u, map_list


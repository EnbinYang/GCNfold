import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import multiprocessing as mp
import itertools
import utils
from utils.general_utils import Pool
from utils.rna_utils import load_mat, load_seq

import dgl
import torch
import torch.utils.data
from tqdm import tqdm
import time

import csv
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from random import shuffle
        

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        self.lists = lists
        self.graph_lists = lists[0]
        self.seq = lists[1]
        self.ss_label = lists[2]
        self.length = lists[3]
        self.name = lists[4]
        self.pair = lists[5]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


class RNAGraphDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_list, p=None):
        
        print("read data")
        self.graph_lists = []
        self.seq_list = seq_list

        if p is None:
            pool = Pool(min(int(mp.cpu_count() * 2 / 3), 12))
        else:
            pool = p

        print("convert seq to graph")
        filepath = os.path.join(data_dir, 'data/archiveII')
        matrix = load_mat(filepath, self.seq_list, pool, load_dense=False, probabilistic=True)
        adjacency_matrix, probability_matrix = matrix

        print("convert graph to dglgraph")
        self.graph_lists = self._convert2dglgraph(self.seq_list, probability_matrix)

        self.n_samples = len(self.graph_lists)  # dglgraph数量
        print("prepare data")
        self._prepare()

    def _prepare(self):
        window_size = 600
        for graph in tqdm(self.graph_lists):
            graph.add_nodes(window_size - graph.ndata['feat'].shape[0])

    def _convert2dglgraph(self, seq_list, csr_matrixs):
        dgl_graph_list = []
        for i in tqdm(range(len(csr_matrixs))):
            dgl_graph_list.append(self._constructGraph(seq_list[i], csr_matrixs[i]))  # csr行压缩

        return dgl_graph_list

    def _constructGraph(self, seq, csr_matrix):
        seq_upper = seq.upper()
        d = {'A': torch.tensor([[1., 0., 0., 0.]]),
             'U': torch.tensor([[0., 1., 0., 0.]]),
             'C': torch.tensor([[0., 0., 1., 0.]]),
             'G': torch.tensor([[0., 0., 0., 1.]]),
             'T': torch.tensor([[0., 1., 0., 0.]]),
             'N': torch.tensor([[0., 0., 0., 0.]])}

        grh = dgl.DGLGraph(csr_matrix)

        grh.ndata['feat'] = torch.zeros((grh.number_of_nodes(), 4))

        for i in range(len(seq)):
            grh.ndata['feat'][i] = d[seq_upper[i]]  # 添加AUGC为图节点

        grh.edata['feat'] = csr_matrix.data  # 提取配对概率作为边特征
        grh.edata['feat'] = grh.edata['feat'].unsqueeze(1)  # 增加一个维度 e.g. torch.Size([7058, 1])

        for i in range(len(seq)):
            for j in range(-2, 3):
                if j == 0 or j == -1 or j == 1:
                    continue
                if i + j < 0 or i + j > len(seq) - 1:
                    continue
                if grh.has_edge_between(i, i + j):
                    continue
                grh.add_edges(i, i + j)
                grh.edges[i, i + j].data['feat'] = torch.tensor([[1/j]], dtype=torch.float64)

        return grh

    def __getitem__(self, idx):
        return self.graph_lists[idx]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples
        
    def __setitem__(self, idx, v):
        self.graph_lists[idx], self.graph_labels[idx] = v
        pass


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, name, config, train_upsampling=False):

        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        
        data_dir = base_dir + '/data/' + str(name) + '/'

        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
        
        if train_upsampling:
            self.train = self.upsampling_data()
            inds = np.random.permutation(np.arange(0, int(len(self.train))))
            train_graphs = [self.train[ind][0] for ind in inds]
            train_seqs = np.array([self.train[ind][1] for ind in inds])
            train_labels = np.array([self.train[ind][2] for ind in inds])
            train_length = np.array([self.train[ind][3] for ind in inds])
            train_names = [self.train[ind][4] for ind in inds]
            train_pairs = np.array([self.train[ind][5] for ind in inds])
            
            self.train = DGLFormDataset(train_graphs, train_seqs, train_labels, train_length, train_names, train_pairs)
        
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))
        
    def upsampling_data(self):
        name = [instance[4] for instance in self.train]
        d_type = np.array(list(map(lambda x: x.split('/')[6], name)))
        data = np.array(self.train)
        max_num = max(Counter(list(d_type)).values())

        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type==t)[0]
            data_list.append(data[index]) 

        final_d_list= list()
        for i in [0, 1, 5, 7]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num)
            final_d_list += list(d[index])

        for i in [2, 3, 4]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num*2)
            final_d_list += list(d[index])
        
        d = data_list[6]
        index = np.random.choice(d.shape[0], int(max_num/2))
        final_d_list += list(d[index])

        shuffle(final_d_list)
        return final_d_list

    # form a mini batch from a given list of samples
    def collate(self, samples):
        graphs, seqs, ss_labels, length, names, pairs = map(list, zip(*samples))

        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
            
        contacts = []
        matrix_reps = []
        
        for idx in range(len(graphs)):
            contact = 0
            contact = self.pairs2map(pairs[idx])
            contacts.append(contact)
            matrix_rep = 0
            matrix_rep = np.zeros(contact.shape)
            matrix_reps.append(matrix_rep)
            
        # list to np.array
        contacts = np.array(contacts)
        seqs = np.array(seqs)
        matrix_reps = np.array(matrix_reps)
        length = np.array(length)
        return graphs, contacts, seqs, matrix_reps, length
    
    def pairs2map(self, pairs):
        seq_len = self.train.seq.shape[1]  # 600
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact


class RNADatasetSingle(torch.utils.data.Dataset):
    def __init__(self, base_dir, dataset_name, filename, config):

        start = time.time()
        print("[I] Loading dataset %s..." % (filename))
        data_dir = base_dir + '/data/' + str(dataset_name) + '/'

        with open(data_dir + filename + '.pkl', "rb") as f:
            self.data = pickle.load(f)
        
        print('test no redundant sizes :', len(self.data))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples
    def collate(self, samples):
        graphs, seqs, ss_labels, length, names, pairs = map(list, zip(*samples))

        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
            
        contacts = []
        matrix_reps = []
        
        for idx in range(len(graphs)):
            contact = 0
            contact = self.pairs2map(pairs[idx])
            contacts.append(contact)
            matrix_rep = 0
            matrix_rep = np.zeros(contact.shape)
            matrix_reps.append(matrix_rep)
            
        # list to np.array
        contacts = np.array(contacts)
        seqs = np.array(seqs)
        matrix_reps = np.array(matrix_reps)
        length = np.array(length)

        return graphs, contacts, seqs, matrix_reps, length, names
    
    def pairs2map(self, pairs):
        seq_len = self.data.seq.shape[1]  # 600
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact 


class RNAGraphDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_list, seq_encoding, ss_label, length, name, pair):
        """
            Takes input standard image dataset name (MNIST/CIFAR10) and returns the superpixels graph.

            This class uses results from the above SuperPix class.
            which contains the steps for the generation of the Superpixels graph from a superpixel .pkl file that has been given by
            https://github.com/bknyaz/graph_attention_pool

            Please refer the SuperPix class for details.
        """
        t_data = time.time()

        print("processing data")
        self.data_ = RNAGraphDGL(data_dir, seq_list)
        
        inds = np.random.permutation(np.arange(0, int(len(self.data_))))  # 37
        
        print("train, val, and test data shuffle and split")
        _train_graphs = [self.data_.graph_lists[ind] for ind in inds[:int(len(self.data_)*0.8)]]  # :29
        _train_seqs = np.array([seq_encoding[ind] for ind in inds[:int(len(self.data_)*0.8)]])
        _train_labels = np.array([ss_label[ind] for ind in inds[:int(len(self.data_)*0.8)]])
        _train_length = np.array([length[ind] for ind in inds[:int(len(self.data_)*0.8)]])
        _train_names = [name[ind] for ind in inds[:int(len(self.data_)*0.8)]]
        _train_pairs = np.array([pair[ind] for ind in inds[:int(len(self.data_)*0.8)]])
        
        _test_val_graphs = [self.data_.graph_lists[ind] for ind in inds[int(len(self.data_)*0.8):]]  # 29:
        _test_val_seqs = np.array([seq_encoding[ind] for ind in inds[int(len(self.data_)*0.8):]])
        _test_val_labels = np.array([ss_label[ind] for ind in inds[int(len(self.data_)*0.8):]]) 
        _test_val_length = np.array([length[ind] for ind in inds[int(len(self.data_)*0.8):]]) 
        _test_val_names = [name[ind] for ind in inds[int(len(self.data_)*0.8):]]
        _test_val_pairs = np.array([pair[ind] for ind in inds[int(len(self.data_)*0.8):]])
        
        inds = np.random.permutation(np.arange(0, int(len(_test_val_graphs))))
        
        print("val and test data shuffle and split")
        _test_graphs = [_test_val_graphs[ind] for ind in inds[:int(len(_test_val_graphs)*0.5)]]
        _test_seqs = np.array([_test_val_seqs[ind] for ind in inds[:int(len(_test_val_graphs)*0.5)]])
        _test_labels = np.array([_test_val_labels[ind] for ind in inds[:int(len(_test_val_graphs)*0.5)]])
        _test_length = np.array([_test_val_length[ind] for ind in inds[:int(len(_test_val_graphs)*0.5)]])
        _test_names = [_test_val_names[ind] for ind in inds[:int(len(_test_val_graphs)*0.5)]]
        _test_pairs = np.array([_test_val_pairs[ind] for ind in inds[:int(len(_test_val_graphs)*0.5)]])
        
        _val_graphs = [_test_val_graphs[ind] for ind in inds[int(len(_test_val_graphs)*0.5):]]
        _val_seqs = np.array([_test_val_seqs[ind] for ind in inds[int(len(_test_val_graphs)*0.5):]])
        _val_labels = np.array([_test_val_labels[ind] for ind in inds[int(len(_test_val_graphs)*0.5):]])
        _val_length = np.array([_test_val_length[ind] for ind in inds[int(len(_test_val_graphs)*0.5):]])
        _val_names = [_test_val_names[ind] for ind in inds[int(len(_test_val_graphs)*0.5):]]
        _val_pairs = np.array([_test_val_pairs[ind] for ind in inds[int(len(_test_val_graphs)*0.5):]])
        
        print("data convert to DGLFormDataset")
        self.train = DGLFormDataset(_train_graphs, _train_seqs, _train_labels, _train_length, _train_names, _train_pairs)
        self.val = DGLFormDataset(_val_graphs, _val_seqs, _val_labels, _val_length, _val_names, _val_pairs)
        self.test = DGLFormDataset(_test_graphs, _test_seqs, _test_labels, _test_length, _test_names, _test_pairs)

        print("[I] Data load time: {:.4f}s".format(time.time() - t_data))


class RNAGraphDatasetDGLTest(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_list, seq_encoding, ss_label, length, name, pair):
        """
            Takes input standard image dataset name (MNIST/CIFAR10) and returns the superpixels graph.

            This class uses results from the above SuperPix class.
            which contains the steps for the generation of the Superpixels graph from a superpixel .pkl file that has been given by
            https://github.com/bknyaz/graph_attention_pool

            Please refer the SuperPix class for details.
        """
        t_data = time.time()

        print("processing data")
        self.data_ = RNAGraphDGL(data_dir, seq_list)
        
        seqs = np.array(seq_encoding)
        labels = np.array(ss_label)
        length = np.array(length)
        pairs = np.array(pair)
        
        print("data convert to DGLFormDataset")
        self.data = DGLFormDataset(self.data_.graph_lists, seqs, labels, length, name, pairs)

        print("[I] Data load time: {:.4f}s".format(time.time() - t_data))


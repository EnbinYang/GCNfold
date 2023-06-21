"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dgl
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score
from train_utils.metrics import accuracy_MNIST_CIFAR as accuracy


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    parts = 1
    for iter, (rnagraphs, contacts, seq_embeddings, matrix_reps, seq_lens) in enumerate(train_loader):
        batch_size = labels.shape[0]
        for i in range(parts):
            batch_graphs = dgl.batch(graphs[i*batch_size//parts:(i+1)*batch_size//parts])
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['feat'] = batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.ndata['feat'] # num x feat
            batch_e = batch_graphs.edata['feat']
            batch_labels = labels[i*batch_size//parts:(i+1)*batch_size//parts].to(device)
            optimizer.zero_grad()

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # batch_scores = model.forward(batch_graphs, batch_feature, batch_x, batch_e)

            loss = model.loss(batch_scores, batch_labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_train_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)*parts
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


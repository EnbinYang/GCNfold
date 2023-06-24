import pickle
import numpy as np
import os
from os import walk
import pandas as pd
import collections
from collections import defaultdict
from GCNfold.common.utils import get_pairings
import seaborn as sns
import matplotlib.pyplot as plt
from data.RNAGraph import DGLFormDataset, RNAGraphDatasetDGL

dataset = 'gcn_plot_data'
data_dir = os.getcwd()  # /content/drive/MyDrive/GCNfold
datapath = os.path.join(data_dir, 'data', dataset)

# check the testing data
with open(datapath + '/%s.pkl' % dataset, 'rb') as f1:
    f1 = pickle.load(f1)
    test_all_600_used = f1

# list convert to DGLFormDataset object 
inds_test = range(len(test_all_600_used))
test_graphs = [test_all_600_used[ind][0] for ind in inds_test]
test_seqs = np.array([test_all_600_used[ind][1] for ind in inds_test])
test_labels = np.array([test_all_600_used[ind][2] for ind in inds_test])
test_length = np.array([test_all_600_used[ind][3] for ind in inds_test])
test_names = [test_all_600_used[ind][4] for ind in inds_test]
test_pairs = np.array([test_all_600_used[ind][5] for ind in inds_test])
test_all_600_used_obj = DGLFormDataset(test_graphs, test_seqs, test_labels, test_length, test_names, test_pairs)

with open(datapath + '/test_no_redundant.pkl', 'wb') as f:
    pickle.dump(test_all_600_used_obj, f)
print('file save finished: ' + str(datapath) + '/test_no_redundant.pkl')


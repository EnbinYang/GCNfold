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

dataset = 'rnastralign_tmRNA'
# rna_types = ['tmRNA', 'tRNA', 'telomerase', 'RNaseP', 'SRP', '16S_rRNA', '5S_rRNA', 'group_I_intron']
rna_types = ['tmRNA']

data_dir = os.getcwd()  # /content/drive/MyDrive/GCNfold
datapath = os.path.join(data_dir, 'data', dataset)
raw_datapath = os.path.join(data_dir, 'data/raw_data/RNAStrAlign')
seed = 0

# select all files within the preferred rna_type
file_list = list()
for rna_type in rna_types:
    type_dir = os.path.join(raw_datapath, rna_type+'_database')
    for r, d, f in walk(type_dir):
        for file in f:
            if file.endswith(".ct"):
                file_list.append(os.path.join(r, file))

# load data
data_list = list(map(lambda x: pd.read_csv(x, sep='\s+', skiprows=1, header=None), file_list))
seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])), data_list))
seq_file_pair_list = list(zip(seq_list, file_list))
d = defaultdict(list)

for k,v in seq_file_pair_list:
	d[k].append(v)
	
unique_seqs = list()
seq_files = list()
for k,v in d.items():
	unique_seqs.append(k)
	seq_files.append(v)

original_seq_len = list(map(len, seq_list))
unique_seq_len = list(map(len, unique_seqs))
cluster_size = list(map(len, seq_files))
used_files = list(map(lambda x: x[0], seq_files))
used_files_rna_type = list(map(lambda x: x.split('/')[6], used_files))

# check the testing data
with open(datapath + '/%s.pkl' % dataset, 'rb') as f:
    f = pickle.load(f)
    train_all_600 = f[0]
    test_all_600 = f[2]

file_seq_d = dict()
for k,v in seq_file_pair_list:
	file_seq_d[v] = k

train_files = [instance[4] for instance in train_all_600]
train_seqs = [file_seq_d[file] for file in train_files]
train_in_files = list()
for seq in train_seqs:
	files_tmp = d[seq]
	train_in_files += files_tmp
train_in_files = list(set(train_in_files))

test_files = [instance[4] for instance in test_all_600]
test_set = list(set(test_files) - set(test_files).intersection(train_in_files))
test_seqs = [file_seq_d[file] for file in test_set]
test_seq_file_pair_list = zip(test_seqs, test_set)
test_seq_file_d = defaultdict(list)
for k,v in test_seq_file_pair_list:
	test_seq_file_d[k].append(v)
test_files_used = [test_seq_file_d[seq][0] for seq in test_seqs]
test_rna_type = list(map(lambda x: x.split('/')[6], test_files_used))

# use the test_files_used to filter the test files
test_all_600_used = list()
for instance in test_all_600:
	if instance[4] in test_files_used:
		test_all_600_used.append(instance)

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

test_16s = list()
for instance in test_all_600_used:
	if '16S_rRNA' in instance[4]:
		test_16s.append(instance)

# list convert to DGLFormDataset object 
inds_test_16s = range(len(test_16s))
test_graphs_16s = [test_16s[ind][0] for ind in inds_test_16s]
test_seqs_16s = np.array([test_16s[ind][1] for ind in inds_test_16s])
test_labels_16s = np.array([test_16s[ind][2] for ind in inds_test_16s])
test_length_16s = np.array([test_16s[ind][3] for ind in inds_test_16s])
test_names_16s = [test_16s[ind][4] for ind in inds_test_16s]
test_pairs_16s = np.array([test_16s[ind][5] for ind in inds_test_16s])
test_16s_obj = DGLFormDataset(test_graphs_16s, test_seqs_16s, test_labels_16s, test_length_16s, test_names_16s, test_pairs_16s)

with open(datapath + '/test_16s.pkl', 'wb') as f:
	pickle.dump(test_16s_obj, f)
print('file save finished: ' + str(datapath) + '/test_16s.pkl')


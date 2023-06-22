import pickle
import numpy as np
import os
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from GCNfold.common.utils import get_pairings
from data.RNAGraph import RNAGraphDatasetDGL

dataset = 'archiveII'
rna_type = ['5s', '16s', '23s', 'grp1', 'grp2', 'RNaseP', 'srp', 'telomerase', 'tmRNA', 'tRNA']

data_dir = os.getcwd()  # /content/drive/MyDrive/GCNfold
datapath = os.path.join(data_dir, 'data/raw_data/archiveII')
seed = 0

# select all the 5s files
file_list = os.listdir(datapath)
file_list = list(filter(lambda x: x.startswith(tuple(rna_type)) and x.endswith(".ct"), file_list))

# load data, 5s do not have pseudoknot so we do not have to delete them
data_list = list()
for file in file_list:
    df = pd.read_csv(os.path.join(datapath, file), sep='\s+', skiprows=1, header=None)
    data_list.append(df)

# for 5s, the sequence length is from 102 to 135
seq_len_list= list(map(len, data_list))

print(min(seq_len_list))  # 28
print(max(seq_len_list))  # 2968
print(len(seq_len_list)) # 3975

# cut the sequence length to 600
seq_len_list_600 = list()
data_list_600 = list()
file_list_600 = list()

for i in range(len(seq_len_list)):
    if seq_len_list[i] > 600:
        continue
    else:
        seq_len_list_600.append(seq_len_list[i])
        data_list_600.append(data_list[i])
        file_list_600.append(file_list[i])

print('Cut Sequence Information:')
print(min(seq_len_list_600))  # 28
print(max(seq_len_list_600))  # 595
print(len(seq_len_list_600))  # 3911

def generate_label(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] == 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)  # string (.....)

# generate the ".()" labeling for each position and the sequence
structure_list = list(map(generate_label, data_list_600))  # string (.....)
seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])), data_list_600))  # 'AUUCG'
pairs_list = list(map(get_pairings, data_list_600))  # '[1, 5]'

label_dict = {
    '.': np.array([1, 0, 0]), 
    '(': np.array([0, 1, 0]), 
    ')': np.array([0, 0, 1])
}
seq_dict = {
    'A':np.array([1, 0, 0, 0]),
    'U':np.array([0, 1, 0, 0]),
    'C':np.array([0, 0, 1, 0]),
    'G':np.array([0, 0, 0, 1])
}

def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)

def stru_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: label_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)

def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,0)), 'constant')

# label and sequence encoding, plus padding to the maximum length
# max_len = max(seq_len_list_600)
max_len = 600
seq_encoding_list = list(map(seq_encoding, seq_list))
stru_encoding_list = list(map(stru_encoding, structure_list))

seq_encoding_list_padded = list(map(lambda x: padding(x, max_len), seq_encoding_list))
stru_encoding_list_padded = list(map(lambda x: padding(x, max_len), stru_encoding_list))

# dglgraph
rna_datset = RNAGraphDatasetDGL(data_dir, seq_list, seq_encoding_list_padded, stru_encoding_list_padded, 
                                seq_len_list_600, file_list_600, pairs_list)

savepath = dataset + "_" + "_".join(rna_type)
os.mkdir(savepath)

with open(savepath + '/%s.pkl' % dataset, 'wb') as f:
    pickle.dump([rna_datset.train, rna_datset.val, rna_datset.test], f)


import pickle
import numpy as np
import os
from os import walk
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from GCNfold.common.utils import get_pairings
from data.RNAGraph import RNAGraphDatasetDGL

dataset = 'rnastralign'
# rna_types = ['tmRNA', 'tRNA', 'telomerase', 'RNaseP', 'SRP', '16S_rRNA', '5S_rRNA', 'group_I_intron']
# rna_types = ['telomerase']
rna_types = ['tmRNA']

data_dir = os.getcwd()  # /content/drive/MyDrive/GCNfold
datapath = os.path.join(data_dir, 'data/raw_data/RNAStrAlign')
seed = 0
length_limit = 600

# select all files within the preferred rna_type
file_list = list()

for rna_type in rna_types:
    type_dir = os.path.join(datapath, rna_type+'_database')
    for r, d, f in walk(type_dir):
        for file in f:
            if file.endswith(".ct"):
                file_list.append(os.path.join(r,file))

# load data
data_list = list(map(lambda x: pd.read_csv(x, sep='\s+', skiprows=1, header=None), file_list))

# for 5s, the sequence length is from 102 to 135
seq_len_list= list(map(len, data_list))

file_length_dict = dict()
for i in range(len(seq_len_list)):
    file_length_dict[file_list[i]] = seq_len_list[i]

print(min(seq_len_list))
print(max(seq_len_list))
print(len(seq_len_list))

data_list = list(filter(lambda x: len(x)<=length_limit, data_list))
seq_len_list = list(map(len, data_list))
file_list = list(filter(lambda x: file_length_dict[x]<=length_limit, file_list))
pairs_list = list(map(get_pairings, data_list))

print('Cut Sequence Information:')
print(min(seq_len_list))
print(max(seq_len_list))
print(len(seq_len_list))

def generate_label(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)

def find_pseudoknot(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                flag = True
                break
    return flag

# generate the ".()" labeling for each position and the sequence
structure_list = list(map(generate_label, data_list))
seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])), data_list))


label_dict = {
    '.': np.array([1,0,0]), 
    '(': np.array([0,1,0]), 
    ')': np.array([0,0,1])
}
seq_dict = {
    'A':np.array([1,0,0,0]),
    'U':np.array([0,1,0,0]),
    'C':np.array([0,0,1,0]),
    'G':np.array([0,0,0,1]),
    'N':np.array([0,0,0,0])
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
max_len = 600
seq_encoding_list = list(map(seq_encoding, seq_list))
stru_encoding_list = list(map(stru_encoding, structure_list))

seq_encoding_list_padded = list(map(lambda x: padding(x, max_len), seq_encoding_list))
stru_encoding_list_padded = list(map(lambda x: padding(x, max_len), stru_encoding_list))

# dglgraph
rna_datset = RNAGraphDatasetDGL(data_dir, seq_list, seq_encoding_list_padded, stru_encoding_list_padded, 
                                seq_len_list, file_list, pairs_list)

savepath = dataset + "_" + "_".join(rna_type)
os.mkdir(savepath)

with open(savepath + '/%s.pkl' % dataset, 'wb') as f:
    pickle.dump([rna_datset.train, rna_datset.val, rna_datset.test], f)


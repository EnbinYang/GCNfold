import re
import os
import sys
import rna
import torch
import gzip
import pickle
import dgl
from utils.seq_motifs import get_motif
import random
import subprocess
import numpy as np
import scipy.sparse as sp
from functools import partial
import forgi.graph.bulge_graph as fgb

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.general_utils import Pool

# >>>> equilibrium probability using RNAplfold >>>>>>>
# when on compute canada, make sure this is happening on the compute nodes

def fold_seq_rnaplfold(seq, w, l, cutoff, no_lonely_bps):
    np.random.seed(random.seed())
    name = str(np.random.rand())

    # 启用cmd调用rnaplfold
    no_lonely_bps_str = ""
    if no_lonely_bps:
        no_lonely_bps_str = "--noLP"
    
    # adopt different window size
    if len(seq) <=70:
        cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f --id-prefix %s %s' % (seq, 25, 25, cutoff, name, no_lonely_bps_str)
    elif len(seq) <= 150:
        cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f --id-prefix %s %s' % (seq, 70, 70, cutoff, name, no_lonely_bps_str)
    else:
        cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f --id-prefix %s %s' % (seq, 150, 150, cutoff, name, no_lonely_bps_str)

    ret = subprocess.call(cmd, shell=True)

    # assemble adjacency matrix
    row_col, link, prob = [], [], []
    length = len(seq)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            link.append(1)
            prob.append(1.)
        if i != 0:
            row_col.append((i, i - 1))
            link.append(2)
            prob.append(1.)

    # Extract base pair information.
    name += '_0001_dp.ps'
    start_flag = False
    with open(name) as f:
        for line in f:
            if start_flag:
                values = line.split()
                if len(values) == 4:
                    source_id = int(values[0]) - 1
                    dest_id = int(values[1]) - 1
                    avg_prob = float(values[2])

                    # source_id < dest_id
                    row_col.append((source_id, dest_id))
                    link.append(3)
                    prob.append(avg_prob ** 2)
                    row_col.append((dest_id, source_id))
                    link.append(4)
                    prob.append(avg_prob ** 2)
            if 'start of base pair probability data' in line:
                start_flag = True
    # delete RNAplfold output file.
    os.remove(name)
    # placeholder for dot-bracket structure

    if length == 28:
        prob_matrix_np = sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)).toarray()
        print(prob_matrix_np)
        print(prob_matrix_np.shape)
        
        np.savetxt("pl_heatmap_data.txt", prob_matrix_np)
    
    return (sp.csr_matrix((link, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
            sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),)


# >>>> MFE structure using RNAfold >>>>>>>

def fold_seq_rnafold(seq):
    '''fold sequence using RNAfold'''
    struct = RNA.fold(seq)[0]
    matrix = adj_mat(struct)
    return structural_content([struct]), matrix


def adj_mat(struct):
    # create sparse matrix
    row_col, data = [], []
    length = len(struct)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            data.append(1)
        if i != 0:
            row_col.append((i, i - 1))
            data.append(2)
    bg = fgb.BulgeGraph.from_dotbracket(struct)
    for i, ele in enumerate(struct):
        if ele == '(':
            row_col.append((i, bg.pairing_partner(i + 1) - 1))
            data.append(3)
        elif ele == ')':
            row_col.append((i, bg.pairing_partner(i + 1) - 1))
            data.append(4)
    return sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])),
                         shape=(length, length))
    

# 导入rnaplfold产生的邻接矩阵
def load_mat(filepath, seq_list, pool=None, fold_algo='rnaplfold', load_dense=False, probabilistic=False, window_size=70):
    prefix = '%s_%s_%s_' % (fold_algo, probabilistic, window_size)  # rnaplfold_True_20_

    if not os.path.exists(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix))) or probabilistic and \
                          not os.path.exists(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix))):
        print('adj mat or prob mat is missing. Begin folding from scratch.')
    fold_rna_from_file(filepath, seq_list, pool, fold_algo, probabilistic)

    sp_rel_matrix = pickle.load(open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'rb'))
    
    if load_dense:
        adjacency_matrix = np.array([mat.toarray() for mat in sp_rel_matrix])
    else:
        adjacency_matrix = np.array(sp_rel_matrix)

    if probabilistic:
        sp_prob_matrix = pickle.load(open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'rb'))
        if load_dense:
            probability_matrix = np.array([mat.toarray() for mat in sp_prob_matrix])
        else:
            probability_matrix = np.array(sp_prob_matrix)
        matrix = (adjacency_matrix, probability_matrix)
    else:
        matrix = adjacency_matrix

    return matrix


def load_seq(filepath):
    if filepath.endswith('.fa'):
        file = open(filepath, 'r')
    else:
        file = gzip.open(filepath, 'rb')

    all_id, all_seq = load_fasta_format(file)
    for i in range(len(all_seq)):
        seq = all_seq[i]
        # seq = seq[:-1].upper()
        all_seq[i] = seq.replace('T', 'U')
        all_seq[i] = seq.replace('t', 'u')
    return all_id, all_seq


def matrix2seq(one_hot_matrices):
    d = {'A': torch.tensor([[1., 0., 0., 0.]]),
         'U': torch.tensor([[0., 1., 0., 0.]]),
         'C': torch.tensor([[0., 0., 1., 0.]]),
         'G': torch.tensor([[0., 0., 0., 1.]])}
    seq_list = []
    for i in range(one_hot_matrices.shape[0]):
        one_hot_matrice = one_hot_matrices[i, 0, :]
        seq = ""
        for loc in range(one_hot_matrice.shape[0]):
            if one_hot_matrice[loc, 0] == 1:
                seq += 'A'
            elif one_hot_matrice[loc, 1] == 1:
                seq += 'G'
            elif one_hot_matrice[loc, 2] == 1:
                seq += 'C'
            elif one_hot_matrice[loc, 3] == 1:
                seq += 'U'
            else:
                seq += 'N'
        seq_list.append(seq)

    return seq_list


def fold_rna_from_file(filepath, all_seq, p=None, fold_algo='rnaplfold', probabilistic=False, window_size=70):
    print('Parsing', filepath)

    # compatible with already computed structures with RNAfold
    prefix = '%s_%s_%s_' % (fold_algo, probabilistic, window_size)

    if p is None:
        pool = Pool(int(os.cpu_count() * 2 / 3))
    else:
        pool = p

    print('running rnaplfold with winsize %d' % (window_size))
    fold_func = partial(fold_seq_rnaplfold, w=window_size, l=min(window_size, 150), cutoff=1e-4, no_lonely_bps=True)
    res = list(pool.imap(fold_func, all_seq))
    sp_rel_matrix = []
    sp_prob_matrix = []
    for rel_mat, prob_mat in res:
        sp_rel_matrix.append(rel_mat)
        sp_prob_matrix.append(prob_mat)

    pickle.dump(sp_rel_matrix, open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
    pickle.dump(sp_prob_matrix, open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'wb'))

    if p is None:
        pool.close()
        pool.join()

    print('Parsing', filepath, 'finished')


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000, )

    # res = fold_seq_rnaplfold(
    #     'cgcgggacgcggcccgaggccgtgcgcgagccggggcaccgggcggcggcggcggcggcgcgcgccatgtcgttcagtgaaatgaaccgcaggacgctggcgttccgaggaggcgggttggtcaccgctagcggcggcggctccacgaacAATAACGCTGGCGGGGAGGCCTCAGcttggcctccgcagccccagccgagacagcccccgccgccagcgccgcccgcgcttcagccgcctaatgggcggggggccgacgaggaagtggaattggagggcctggagccccaagacctggaggcctccgccgggccggccgccggcg',
    #     150, 150, 0.0001, True)
    # print(res[1].todense().sum(axis=-1))
    res = fold_seq_rnashapes(
        'cgcgggacgcggcccgaggccgtgcgcgagccggggcaccgggcggcggcggcggcggcgcgcgccatgtcgttcagtgaaatgaaccgcaggacgctggcgttccgaggaggcgggttggtcaccgctagcggcggcggctccacgaacAATAACGCTGGCGGGGAGGCCTCAGcttggcctccgcagccccagccgagacagcccccgccgccagcgccgcccgcgcttcagccgcctaatgggcggggggccgacgaggaagtggaattggagggcctggagccccaagacctggaggcctccgccgggccggccgccggcg',
        150, iterations=100)
    # print(res[0].todense())
    print(res[1].todense().sum(axis=-1))

    # with open('boltzmann-sampling-acc.txt', 'w') as file:
    #     for amount in [5, 10, 100, 1000, 5000, 10000]:
    #         rel_diff, prob_diff = [], []
    #         for replcate in range(100):
    #             _, res = fold_seq_subopt(
    #                 'TGTGAAGCGCGGCTAGCTGCCGGGGTTCGAGGTGGGTCCCAGGGTTAAAATCCCTTGTTGTCTTACTGGTGGCAGCAAGCTAGGACTATACTCCTCGGTCG',
    #                 'rnafold', True, amount)
    #             _, new_res = fold_seq_subopt(
    #                 'TGTGAAGCGCGGCTAGCTGCCGGGGTTCGAGGTGGGTCCCAGGGTTAAAATCCCTTGTTGTCTTACTGGTGGCAGCAAGCTAGGACTATACTCCTCGGTCG',
    #                 'rnafold', True, amount)
    #
    #             diff = (res[0].todense() != new_res[0].todense()).astype(np.int32)
    #             rel_diff.append(np.sum(diff))
    #
    #             diff = np.abs(res[1].todense() - new_res[1].todense())
    #             prob_diff.append(np.mean(np.max(diff, axis=-1)))
    #         file.writelines('sampling amount %d, relation difference: %.4f\u00b1%.4f, probability difference: %.4f\u00b1%.4f' %
    #               (amount, np.mean(rel_diff), np.std(rel_diff), np.mean(prob_diff), np.std(prob_diff)))
    #         print('sampling amount %d, relation difference: %.4f\u00b1%.4f, probability difference: %.4f\u00b1%.4f' %
    #               (amount, np.mean(rel_diff), np.std(rel_diff), np.mean(prob_diff), np.std(prob_diff)))

    # annotation for the multiloop elements
    # all_seqs, adjacency_matrix, all_labels, _ = generate_element_dataset(80000, 101, 'i', return_label=False)
    # print(all_labels.shape)
    # print(np.where(np.count_nonzero(all_labels, axis=-1) > 0)[0].__len__())
    #
    # all_seqs, adjacency_matrix, all_labels, _ = generate_hairpin_dataset(80000, 101, 'm', return_label=False)
    # print(all_labels.shape)
    # print(np.where(np.count_nonzero(all_labels, axis=-1) > 0)[0].__len__())

# from e2efold
# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:,0].values)
    rnadata2 = list(data.loc[:,4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))  # 组成二元组 (a, b)
    rna_pairs = list(filter(lambda x: x[1]>0, rna_pairs))  # 只保留配对的碱基对二元组
    rna_pairs = (np.array(rna_pairs)-1).tolist()  # 二元组各值-1
    return rna_pairs
    

import os
import sys
import pickle
import time
from tqdm import tqdm
import warnings

import dgl

import torch
import torch.optim as optim
from torch.utils import data

from GCNfold.common.utils import *
from GCNfold.common.config import process_config
from GCNfold.postprocess import postprocess
from data.RNAGraph import RNADataset, RNADatasetSingle
from GCNfold.models import Lag_PP_mixed
from nets.gcnfold_net import GCNFoldNet, RNA_SS_e2e

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

def load_model(dataset_name, base_dir, net_params, model_type, epoch):
    save_dir = os.path.join(base_dir, 'model_save/')
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + dataset_name) is False:
        os.makedirs(save_dir + dataset_name)

    model = GCNFoldNet(d=d, L=seq_len, device=device, net_params=net_params)

    PATH = save_dir + dataset_name + '/model_{}_{}.pth'.format(model_type, epoch)
    model.load_state_dict(torch.load(PATH))
    return model


def view_model_param(net_params):
    model = GCNFoldNet(d=d, L=seq_len, device=device, net_params=net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
        
    print('GCNfold Score Net/Total parameters:', total_param)
    return total_param


# set the base directory path
base_dir = os.getcwd()  # /content/drive/MyDrive/GCNfold

# load config
args = get_args()
config_file = args.config
config = process_config(config_file)
net_params = config['net_params']

print('Here is the configuration of this run:')
print(config)

# setup device
os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialization
d = config.gcn_net_d  # 10
BATCH_SIZE = config.BATCH_SIZE  # 2
out_step = config.OUT_STEP  # 100
data_type = config.data_type  # archiveII
model_type = config.model_type  # pretrained
epochs = config.epochs  # 100
step_gamma = config.step_gamma  # 1
k = config.k  # 1
pp_steps = config.pp_steps  # 20
pp_loss = config.pp_loss  # f1
rho_per_position = config.rho_per_position  # matrix
pp_model_path = os.path.join(base_dir, 'model_save', model_type, 'model_pp_{}.pth'.format(config['best_epoch']))
e2e_model_path = os.path.join(base_dir, 'model_save', model_type, 'model_e2e_{}.pth'.format(config['best_epoch']))

seed_torch()

# Load and generator data
print('Load train data')
dataset = RNADataset(base_dir, data_type, config, True)
trainset, valset = dataset.train, dataset.val
dataset_test = RNADatasetSingle(base_dir, data_type, 'test_no_redundant', config)
testset = dataset_test.data
names = np.array(list(map(lambda x: x.split('/')[-1], testset.name)))

drop_last = True
train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
val_loader = data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=dataset.collate)
test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset_test.collate)
print('Data Loading Done!!!')

seq_len = trainset.seq.shape[1]  # (29, 600, 4)
net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)  # 4
net_params['device'] = device  # 'cpu'
print('Max seq length: ', seq_len)  # 600

# load Net and put it to device
print('load GCNfold Net')
best_epoch = 0
net_params['total_param'] = view_model_param(net_params)

contact_net = load_model(data_type, base_dir, net_params, 'score', config['best_epoch'])
contact_net.to(device)
print('Net Loading Done!!!')
lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position)
rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)
print(rna_ss_e2e)

pp_total_param = 0
for param in rna_ss_e2e.parameters():
    pp_total_param += np.prod(list(param.data.size()))
print('GCNfold PP Net/Total parameters:', pp_total_param)

lag_pp_net.load_state_dict(torch.load(pp_model_path))
lag_pp_net.to(device)
print('PP Net Loading Done!!!')
rna_ss_e2e.load_state_dict(torch.load(e2e_model_path))
rna_ss_e2e.to(device)
print('E2E Net Loading Done!!!')

# define optimizer and loss function
gcn_optimizer = optim.Adam(rna_ss_e2e.parameters())
pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_mse = torch.nn.MSELoss(reduction='sum')


def all_data_test(contact_net, lag_pp_net, device, test_loader):
    contact_net.eval()
    lag_pp_net.eval()
    auc_test_all_list = list()
    exact_test_all_list = list()
    shift_test_all_list = list()
    ct_pred_list = list()
    epoch_test_loss = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_graphs, contacts, seq_embeddings, matrix_reps, seq_lens) in enumerate(test_loader): 
            batch_graphs = dgl.batch(batch_graphs)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['feat'] = batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            
            # convert to tensor
            contacts_batch = torch.Tensor(contacts.astype(float)).to(device)  # torch.Size([2, 600, 600])
            seq_embedding_batch = torch.Tensor(seq_embeddings.astype(float)).to(device)  # torch.Size([2, 600, 4])
            matrix_reps_batch = torch.unsqueeze(torch.Tensor(matrix_reps.astype(float)).to(device), -1)  # torch.Size([2, 600, 600, 1])
            state_pad = torch.zeros([matrix_reps_batch.shape[0], seq_len, seq_len]).to(device)  # torch.Size([2, 600, 600])
            seq_lens = torch.Tensor(seq_lens).int()  # torch.Size([2])
            
            PE_batch = get_pe(seq_lens, seq_len).float().to(device)  # utils, torch.Size([2, 600, 111])
            contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
            
            pred_contacts = contact_net(batch_graphs, batch_x, batch_e, PE_batch, seq_embedding_batch, state_pad)
            a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)
            final_pred = (a_pred_list[-1].cpu()>0.5).float()
            
            for i in range(final_pred.shape[0]):
                ct_tmp = contact2ct(final_pred[i].cpu().numpy(), seq_embeddings[i], seq_lens.numpy()[i])
                ct_pred_list.append(ct_tmp)
            
            result_exact = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            result_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            result_auc = list(map(lambda i: calculate_auc(final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            auc_test_all_list += result_auc
            exact_test_all_list += result_exact
            shift_test_all_list += result_shift
        
        model_auc = np.average(auc_test_all_list)
        
        exact_p, exact_r, exact_f1, exact_mcc = zip(*exact_test_all_list)
        shift_p, shift_r, shift_f1 = zip(*shift_test_all_list)
        
        exact_f1, exact_p, exact_r, exact_mcc = np.average(exact_f1), np.average(exact_p), np.average(exact_r), np.average(exact_mcc)
        shift_f1, shift_p, shift_r = np.average(shift_f1), np.average(shift_p), np.average(shift_r)

    return model_auc, exact_f1, exact_p, exact_r, exact_mcc, shift_f1, shift_p, shift_r, ct_pred_list

# test all data 
auc, f1, p, r, mcc, f1_shift, p_shift, r_shift, ct_list = all_data_test(contact_net, lag_pp_net, device, test_loader)
print('test results, auc: {:.3f}, f1: {:.3f}, p: {:.3f}, r: {:.3f}, mcc: {:.3f}'.format(auc, f1, p, r, mcc))
print('f1_shift: {:.3f}, p_shift: {:.3f}, r_shift: {:.3f}'.format(f1_shift, p_shift, r_shift))

# for saving the results
save_path = config.save_folder
if not os.path.exists(save_path):
    os.makedirs(save_path)

def save_file(folder, file, ct_contact):
    file_path = os.path.join(folder, file)
    print(file_path)
    first_line = str(len(ct_contact)) + '\t' + file + '\n'
    content = ct_contact.to_csv(header=None, index=None, sep='\t')
    with open(file_path, 'w') as f:
        f.write(first_line + content)

for i in range(len(names)):
    save_file(save_path, names[i], ct_list[i])

print(save_path)
print(names)


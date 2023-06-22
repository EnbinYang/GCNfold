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
# from nets.gcnfold_net import GCNFoldNet
from nets.gcnfold_net import GCNFoldNet_UNet

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

def load_model(dataset_name, base_dir, net_params, epoch):
    save_dir = os.path.join(base_dir, 'model_save/')
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + dataset_name) is False:
        os.makedirs(save_dir + dataset_name)

    model = GCNFoldNet_UNet(d=d, L=seq_len, device=device, net_params=net_params)

    PATH = save_dir + dataset_name + '/model_unet_' + str(epoch) + '.pth'
    print(PATH)
    model.load_state_dict(torch.load(PATH))
    return model


def view_model_param(net_params):
    model = GCNFoldNet_UNet(d=d, L=seq_len, device=device, net_params=net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
        
    print('MODEL/Total parameters:', total_param)
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
init_lr = config.init_lr  # 0.0005

seed_torch()

# Load and generator data
print('Load train data')
# dataset = RNADataset(base_dir, data_type, config, True)
dataset = RNADataset(base_dir, data_type, config)
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

contact_net = load_model(data_type, base_dir, net_params, config['best_epoch'])
contact_net.to(device)
print('Net Loading Done!!!')

# define optimizer and loss function
gcn_optimizer = optim.Adam(contact_net.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(gcn_optimizer, mode='min', factor=config.lr_reduce_factor,
                                                 patience=config.lr_schedule_patience, verbose=True)
pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def all_data_test(model, device, val_loader):
    model.eval()
    auc_test_all_list = list()
    exact_test_all_list = list()
    shift_test_all_list = list()
    ct_pred_list = list()
    
    with torch.no_grad():
        for iter, (batch_graphs, contacts, seq_embeddings, matrix_reps, seq_lens, names) in enumerate(val_loader): 
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
            
            pred_contacts = model.forward(batch_graphs, batch_x, batch_e, PE_batch, seq_embedding_batch, state_pad)
            
            u_no_train = postprocess(pred_contacts, seq_embedding_batch, 0.01, 0.1, 100, 1.6, True, 1.5)
            map_no_train = (u_no_train > 0.5).float()

            for i in range(map_no_train.shape[0]):
                ct_tmp = contact2ct(map_no_train[i].cpu().numpy(), seq_embeddings[i], seq_lens.numpy()[i])
                ct_pred_list.append(ct_tmp)
            
            result_exact = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            result_shift = list(map(lambda i: evaluate_shifted(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            result_auc = list(map(lambda i: calculate_auc(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
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
test_auc, test_f1, test_p, test_r, test_mcc, test_f1_shift, test_p_shift, test_r_shift, ct_list = all_data_test(contact_net, device, test_loader)
print('test results, auc: {:.3f}, f1: {:.3f}, p: {:.3f}, r: {:.3f}, mcc: {:.3f}'.format(test_auc, test_f1, test_p, test_r, test_mcc))
print('f1_shift: {:.3f}, p_shift: {:.3f}, r_shift: {:.3f}'.format(test_f1_shift, test_p_shift, test_r_shift))

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


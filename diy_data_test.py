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
# from nets.gcnfold_net import GCNFoldNet_UNet_small

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")


def load_test_model(model_type, base_dir, net_params, epoch):
    save_dir = os.path.join(base_dir, 'model_save/')
    print(save_dir)
    
    model = GCNFoldNet_UNet(d=d, L=seq_len, device=device, net_params=net_params)
    PATH = save_dir + model_type + '/model_unet_' + str(epoch) + '.pth'
    
    # model = GCNFoldNet_UNet_small(d=d, L=seq_len, device=device, net_params=net_params)
    # PATH = save_dir + model_type + '/model_unet_small_' + str(epoch) + '.pth'
    
    print(PATH)
    model.load_state_dict(torch.load(PATH, map_location='cpu'))
    return model


def view_model_param(net_params):
    model = GCNFoldNet_UNet(d=d, L=seq_len, device=device, net_params=net_params)
    # model = GCNFoldNet_UNet_small(d=d, L=seq_len, device=device, net_params=net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
        
    print('MODEL/Total parameters:', total_param)
    return total_param


def save_to_csv(data, file_name, save_format = 'csv', save_type = 'col'):
    
    name = []
    times = 0
 
    if save_type == 'col':
        for data_name, data_list in data.items():
            name.append(data_name)
            if times == 0:
                data = np.array(data_list).reshape(-1,1)
            else:
                data = np.hstack((data, np.array(data_list).reshape(-1,1)))
                
            times += 1
            
        pd_data = pd.DataFrame(columns=name, data=data) 
        
    else:
        for data_name, data_list in data.items():
            name.append(data_name)
            if times == 0:
                data = np.array(data_list)
            else:
                data = np.vstack((data, np.array(data_list)))
        
            times += 1
    
        pd_data = pd.DataFrame(index=name, data=data)  
    
    if save_format == 'csv':
        pd_data.to_csv('./'+ file_name +'.csv',encoding='utf-8')
    else:
        pd_data.to_excel('./'+ file_name +'.xls',encoding='utf-8')


# set the base directory path
base_dir = os.getcwd()  # /content/drive/MyDrive/GCNfold

# load config
args = get_args()
config_file = args.config
config = process_config(config_file)

print('Here is the configuration of this run:')
print(config)

# setup device
os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialization
d = config.gcn_net_d  # 10
net_params = config['net_params']
BATCH_SIZE = config.BATCH_SIZE  # 2
out_step = config.OUT_STEP  # 100
data_type = config.data_type  # archiveII
test_data_type = config.test_data_type  # archiveII
model_type = config.model_type  # archiveII_all

seed_torch()

# Load and generator data
print('Load train data')
dataset_test = RNADatasetSingle(base_dir, data_type, 'test_no_redundant', config)
testset = dataset_test.data
names = np.array(list(map(lambda x: x.split('/')[-1], testset.name)))

drop_last = True
test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset_test.collate)
print('Data Loading Done!!!')

seq_len = testset.seq.shape[1]  # (29, 600, 4)
net_params['device'] = device  # 'cpu'
print('Max seq length: ', seq_len)  # 600

# load Net and put it to device
print('load GCNfold Net')
best_epoch = 0
net_params['in_dim'] = testset.seq.shape[2]
net_params['total_param'] = view_model_param(net_params) 

contact_net = load_test_model(model_type, base_dir, net_params, config['best_epoch'])
contact_net.to(device)
print('Net Loading Done!!!')


def model_all_test(model, device, test_loader):
    model.eval()
    ct_pred_list = list()
    auc_test_all_list = list()
    exact_test_all_list = list()
    shift_test_all_list = list()
    ct_pred_list = list()
    name_list = list()
    seq_lens_list = list()
    exact_f1_list, exact_p_list, exact_r_list, exact_mcc_list = list(), list(), list(), list()
    
    with torch.no_grad():
        for iter, (batch_graphs, contacts, seq_embeddings, matrix_reps, seq_lens, names) in enumerate(test_loader):
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
            
            name_list.append(names[0])
            seq_lens_list.append(seq_lens)
            
        model_auc = np.average(auc_test_all_list)
        
        exact_p, exact_r, exact_f1, exact_mcc = zip(*exact_test_all_list)
        shift_p, shift_r, shift_f1 = zip(*shift_test_all_list)
        
        """
        exact_r = [0 if math.isnan(x) else x for x in exact_r]
        exact_mcc = [0 if math.isnan(x) else x for x in exact_mcc]
        shift_r = [0 if math.isnan(x) else x for x in shift_r]
        """
        
        for i in range(len(exact_test_all_list)):
            exact_p_list.append(exact_p[i])
            exact_r_list.append(exact_r[i])
            exact_f1_list.append(exact_f1[i])
            exact_mcc_list.append(exact_mcc[i])
        
        exact_f1, exact_p, exact_r, exact_mcc = np.average(exact_f1), np.average(exact_p), np.average(exact_r), np.average(exact_mcc)
        shift_f1, shift_p, shift_r = np.average(shift_f1), np.average(shift_p), np.average(shift_r)

    return model_auc, exact_f1, exact_p, exact_r, exact_mcc, shift_f1, shift_p, shift_r, ct_pred_list, name_list, seq_lens_list, \
           exact_f1_list, exact_p_list, exact_r_list, exact_mcc_list


# test all data
time_start = time.time()
auc, f1, p, r, mcc, f1_shift, p_shift, r_shift, ct_list, name_list, seq_lens_list, \
f1_list, p_list, p_list, mcc_list = model_all_test(contact_net, device, test_loader)

print('test results, auc: {:.3f}, f1: {:.3f}, p: {:.3f}, r: {:.3f}, mcc: {:.3f}'.format(auc, f1, p, r, mcc))
print('f1_shift: {:.3f}, p_shift: {:.3f}, r_shift: {:.3f}'.format(f1_shift, p_shift, r_shift))
print('testing time: {:.2f}'.format(time.time() - time_start))

# save all data list to csv
all_data = {'name': name_list, 'length': seq_lens_list, 'f1': f1_list, 'p': p_list, 'r': p_list, 'mcc': mcc_list}
save_to_csv(data=all_data, file_name='archiveII_test_unet', save_format = 'xls', save_type = 'col')

# for saving the results
save_path = config.save_folder
if not os.path.exists(save_path):
    os.makedirs(save_path)

def save_file(folder, file, ct_contact):
    file_path = os.path.join(folder, file)
    first_line = str(len(ct_contact)) + '\t' + file + '\n'
    content = ct_contact.to_csv(header=None, index=None, sep='\t')
    with open(file_path, 'w') as f:
        f.write(first_line + content)

for i in range(len(names)):
    save_file(save_path, names[i], ct_list[i])


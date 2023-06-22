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
from data.RNAGraph import RNADataset
from nets.gcnfold_net import GCNFoldNet

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")


def save_model(dataset_name, base_dir, model, epoch):
    save_dir = os.path.join(base_dir, 'model_save/')
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + dataset_name) is False:
        os.makedirs(save_dir + dataset_name)

    torch.save(model.state_dict(), save_dir + dataset_name + '/model_' + str(epoch) + '.pth')


def load_model(dataset_name, base_dir, net_params, epoch):
    save_dir = os.path.join(base_dir, 'model_save/')
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + dataset_name) is False:
        os.makedirs(save_dir + dataset_name)

    model = GCNFoldNet(d=d, L=seq_len, device=device, net_params=net_params)

    PATH = save_dir + dataset_name + '/model_' + str(epoch) + '.pth'
    model.load_state_dict(torch.load(PATH))
    return model


def view_model_param(net_params):
    model = GCNFoldNet(d=d, L=seq_len, device=device, net_params=net_params)
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
        pd_data.to_csv('./'+ file_name +'.csv', encoding='utf-8')
    else:
        pd_data.to_excel('./'+ file_name +'.xls', encoding='utf-8')


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

seed_torch()

# Load and generator data
print('Load train data')
dataset = RNADataset(base_dir, data_type, config)
trainset, valset, testset = dataset.train, dataset.val, dataset.test

drop_last = True
train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
val_loader = data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=dataset.collate)
test_loader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=dataset.collate)
print('Data Loading Done!!!')

seq_len = trainset.seq.shape[1]  # (29, 600, 4)
net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)  # 4
net_params['device'] = device  # 'cpu'
print('Max seq length: ', seq_len)  # 600

# load Net and put it to device
print('load GCNfold Net')
best_epoch = 0
net_params['total_param'] = view_model_param(net_params) 

contact_net = GCNFoldNet(d=d, L=seq_len, device=device, net_params=net_params)
save_model(data_type, base_dir, contact_net, best_epoch)

contact_net = load_model(data_type, base_dir, net_params, config['best_epoch'])
contact_net.to(device)
print('Net Loading Done!!!')

# define optimizer and loss function
gcn_optimizer = optim.Adam(contact_net.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(gcn_optimizer, mode='min', factor=config.lr_reduce_factor,
                                                 patience=config.lr_schedule_patience, verbose=True)
pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def evaluate(model, device, val_loader, epoch):
    model.eval()
    auc_val_list = list()
    epoch_val_loss = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_graphs, contacts, seq_embeddings, matrix_reps, seq_lens) in enumerate(val_loader): 
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
            
            loss = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
            epoch_val_loss += loss.detach().item()
            
            u_no_train = postprocess(pred_contacts, seq_embedding_batch, 0.01, 0.1, 100, 1.6, True, 1.5)
            map_no_train = (u_no_train > 0.5).float()
            
            result_auc = list(map(lambda i: calculate_auc(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            auc_val_list += result_auc
                
        epoch_val_loss /= (iter + 1)
        model_auc = np.average(auc_val_list)

        return epoch_val_loss, model_auc


def evaluate_test(model, device, val_loader, epoch):
    model.eval()
    auc_val_all_list = list()
    exact_val_all_list = list()
    shift_val_all_list = list()
    epoch_val_loss = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_graphs, contacts, seq_embeddings, matrix_reps, seq_lens) in enumerate(val_loader): 
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
            
            loss = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
            epoch_val_loss += loss.detach().item()
            
            u_no_train = postprocess(pred_contacts, seq_embedding_batch, 0.01, 0.1, 100, 1.6, True, 1.5)
            map_no_train = (u_no_train > 0.5).float()
            
            result_auc = list(map(lambda i: calculate_auc(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            result_exact = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            result_shift = list(map(lambda i: evaluate_shifted(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            auc_val_all_list += result_auc
            exact_val_all_list += result_exact
            shift_val_all_list += result_shift
                
        epoch_val_loss /= (iter + 1)
        model_auc = np.average(auc_val_all_list)
        
        exact_p, exact_r, exact_f1, exact_mcc = zip(*exact_val_all_list)
        shift_p, shift_r, shift_f1 = zip(*shift_val_all_list)
        exact_f1, exact_p, exact_r, exact_mcc = np.average(exact_f1), np.average(exact_p), np.average(exact_r), np.average(exact_mcc)
        shift_f1, shift_p, shift_r = np.average(shift_f1), np.average(shift_p), np.average(shift_r)

    return epoch_val_loss, model_auc, exact_f1, exact_p, exact_r, exact_mcc, shift_f1, shift_p, shift_r


def all_data_test(model, device, val_loader, epoch):
    model.eval()
    auc_test_all_list = list()
    exact_test_all_list = list()
    shift_test_all_list = list()
    epoch_test_loss = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_graphs, contacts, seq_embeddings, matrix_reps, seq_lens) in enumerate(val_loader): 
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

    return model_auc, exact_f1, exact_p, exact_r, exact_mcc, shift_f1, shift_p, shift_r


def train(model, dataset_name, optimizer, device, data_loader, epoch):
    model.train()
    steps_done = 0
    nb_data = 0
    parts = 1
    epoch_loss = 0
    evaluate_epi = 0
    
    for iter, (rnagraphs, contacts, seq_embeddings, matrix_reps, seq_lens) in enumerate(train_loader): 
        batch_size = seq_lens.shape[0]  # 2
        
        for i in range(parts):
            batch_graphs = dgl.batch(rnagraphs[i*batch_size//parts:(i+1)*batch_size//parts])
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['feat'] = batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.ndata['feat']  # num x feat
            batch_e = batch_graphs.edata['feat']
            
            # mini batch
            contacts_batch = contacts[i*batch_size//parts:(i+1)*batch_size//parts]
            seq_embedding_batch = seq_embeddings[i*batch_size//parts:(i+1)*batch_size//parts]
            matrix_reps_batch = matrix_reps[i*batch_size//parts:(i+1)*batch_size//parts]
            seq_lens = seq_lens[i*batch_size//parts:(i+1)*batch_size//parts]
            
            # convert to tensor
            contacts_batch = torch.Tensor(contacts.astype(float)).to(device)  # torch.Size([2, 600, 600])
            seq_embedding_batch = torch.Tensor(seq_embeddings.astype(float)).to(device)  # torch.Size([2, 600, 4])
            matrix_reps_batch = torch.unsqueeze(torch.Tensor(matrix_reps.astype(float)).to(device), -1)  # torch.Size([2, 600, 600, 1])
            state_pad = torch.zeros([matrix_reps_batch.shape[0], seq_len, seq_len]).to(device)  # torch.Size([2, 600, 600])
            seq_lens = torch.Tensor(seq_lens).int()  # torch.Size([2])
            
            PE_batch = get_pe(seq_lens, seq_len).float().to(device)  # utils, torch.Size([2, 600, 111])
            contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)  # torch.Size([2, 600, 600])
            
            pred_contacts = model.forward(batch_graphs, batch_x, batch_e, PE_batch, seq_embedding_batch, state_pad)  # torch.Size([2, 600, 600])
            
            # compute loss
            loss = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
            
            if steps_done % out_step ==0:
                print('epoch: {}, step: {}, loss: {:.4f}'.format(epoch, steps_done, loss))
            
            # optimize the model
            gcn_optimizer.zero_grad()
            loss.backward()
            gcn_optimizer.step()
            
            steps_done = steps_done + 1
            epoch_loss += loss.detach().item()
            nb_data += contacts_batch.size(0)
    
    epoch_loss /= (iter + 1) * parts
    
    return epoch_loss, gcn_optimizer


best_val_auc = 0
val_epi = 5
epoch_train_losses, epoch_val_losses = [], []
auc_val_lists = []

for epoch in range(epochs):
        
    start = time.time()
        
    epoch_train_loss, gcn_optimizer = train(contact_net, data_type, gcn_optimizer, device, train_loader, epoch)
    
    if (epoch+1) % val_epi == 0:
        epoch_val_loss, model_val_auc, val_f1, val_p, val_r, val_mcc, \
        val_f1_shift, val_p_shift, val_r_shift = evaluate_test(contact_net, device, val_loader, epoch)
    else:
        epoch_val_loss, model_val_auc = evaluate(contact_net, device, val_loader, epoch)
    
    # save value  
    epoch_train_losses.append(epoch_train_loss)
    epoch_val_losses.append(epoch_val_loss)
    auc_val_lists.append(model_val_auc)
    
    # evaluate per 5 epochs
    print('Epoch Information in Each Dataset: ')
    print('epoch: {}, time: {:.2f}, lr: {}'.format(epoch, time.time()-start, gcn_optimizer.param_groups[0]['lr']))
    print('epoch: {}, train_loss: {:.4f}'.format(epoch, epoch_train_loss))
    
    # 
    if (epoch+1) % val_epi == 0:
        print('epoch: {}, val_loss: {:.4f}, auc: {:.3f}, f1: {:.3f}, p: {:.3f}, r: {:.3f}, mcc: {:.3f}'.format(epoch, epoch_val_loss,
                                                                                                               model_val_auc, val_f1, 
                                                                                                               val_p, val_r, val_mcc))
        print('epoch: {}, f1_shift: {:.3f}, p_shift: {:.3f}, r_shift: {:.3f} \n'.format(epoch, val_f1_shift, val_p_shift, val_r_shift))
    else:
        print('epoch: {}, val_loss: {:.4f}, auc: {:.3f} \n'.format(epoch, epoch_val_loss, model_val_auc))
        
    scheduler.step(epoch_val_loss)
    
    if best_val_auc <= model_val_auc:
        best_val_auc = model_val_auc
        best_epoch = epoch
        save_model(data_type, base_dir, contact_net, best_epoch)

# save and print list
epoch = np.arange(0, epochs).astype(int)
train_loss = np.array(epoch_train_losses)
val_loss = np.array(epoch_val_losses)

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(epoch, train_loss, marker='o', label='train')
plt.plot(epoch, val_loss, marker='o', label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(base_dir + '/loss.pdf', dpi=300)

# save all data list to csv
all_data = {'Epoch': list(epoch), 'Train Loss': epoch_train_losses, 'Val Loss': epoch_val_losses, 'Val AUC': auc_val_lists}
save_to_csv(data=all_data, file_name='GCNFold Exper', save_format = 'csv', save_type = 'col')

# test all data 
test_auc, test_f1, test_p, test_r, test_mcc, test_f1_shift, test_p_shift, test_r_shift = all_data_test(contact_net, device, test_loader, epoch)
print('test results, auc: {:.3f}, f1: {:.3f}, p: {:.3f}, r: {:.3f}, mcc: {:.3f}'.format(test_auc, test_f1, test_p, test_r, test_mcc))
print('f1_shift: {:.3f}, p_shift: {:.3f}, r_shift: {:.3f}'.format(test_f1_shift, test_p_shift, test_r_shift))


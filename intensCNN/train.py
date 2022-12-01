import sys
import time
import configparser

import numpy as np
import h5py

import torch
from torch import nn
import torch.nn.functional as F

import network
import preprocessing

def _q_rotation(x, y, z, quat):
    '''Get Rotation Matrix from quaternions.
    Input: x, y, z --> detector pixel coordinates
           qw, qx, qy, qz --> quaternions'''
    qw, qx, qy, qz = quat
    matrx = [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
             [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
             [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]]
    return np.transpose(np.transpose(np.array([x, y, z]), (1,2,0))@matrx, (2,0,1))

def slice_planes(orien):
    '''Get slicing planes for corresponding rotation matrices'''
    n_slices, imgsize = all_intens.shape[:2]
    slices_np = np.zeros((n_slices, imgsize, imgsize, 3))
    for i in range(n_slices):
        slices_np[i] = _q_rotation(qx_d, qy_d, qz_d, orien[i]).T
    return torch.from_numpy(slices_np).to(device)

def best_projection_slice(recon_intens_3d, slices, ind):
    '''Slicing the reconstructed volume'''
    select_plane =  slices[ind]
    size = recon_intens_3d.shape[2]
    grid = select_plane.float()
    return F.grid_sample(recon_intens_3d.view(1, 1, size, size, size),
            grid.view(1, grid.shape[0], grid.shape[0], 1, 3), mode='bilinear',
            padding_mode='zeros',
            align_corners=True)[0][0][:,:].reshape(grid.shape[0], grid.shape[0])

def loss_function(recon_intens_3d, slices, images, bnum):
    '''Loss for NN'''
    recon_images = torch.zeros_like(images)
    for i in range(BATCH_SIZE):
        # Applying Symmetrization : Friedel Symmetry or Icosahedra Symmetry
        arrth = recon_intens_3d[i]
        symarrth = friedel_symm(arrth)
        recon_images[i] = best_projection_slice(torch.Tensor.permute(symarrth, (0,3,2,1)),
                                                slices, bnum*BATCH_SIZE + i)
        recon_images[i] = torch.Tensor.permute(recon_images[i].clone(), (0,2,1))
        # No Symmetrization
        #recon_images[i] = best_projection_slice(torch.Tensor.permute(arrth, (0,3,2,1)),
        #                                        slices, bnum*BATCH_SIZE + i)
        #recon_images[i] = torch.Tensor.permute(recon_images[i], (0,2,1))
    loss = nnLoss(recon_images, images)
    return loss, recon_images, symarrth

def friedel_symm(arr):
    '''Apply friedel symmetry to the reconstructed volume'''
    return (arr + torch.flip(arr, dims = (1,2,3))) / 2

def run_network(train=True, save=False, input_intens=None,
                orientation=None, objsize=None, slices=None):
    '''Steps in training/validation of NN'''
    recon_vol = []
    true_objsize = []
    true_intens = []
    pred_intens = []

    epochloss=0
    for i in range(len(orientation)//BATCH_SIZE):
        intens_batch = input_intens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        images = torch.from_numpy(intens_batch).view((BATCH_SIZE, 1) + input_intens.shape[1:])
        images = images.float().to(device)

        ori_batch = orientation[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        oris = torch.from_numpy(ori_batch).view(BATCH_SIZE, 4)
        oris = oris.float().to(device)

        objsize_batch = objsize[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        objsizes = torch.from_numpy(objsize_batch).view(BATCH_SIZE,1)
        objsizes = objsizes.float().to(device)

        output = model.forward(objsizes)

        loss, recon_slice, sym_intens = loss_function(output, slices, images, i)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epochloss += loss.data.item()

        if not train and save:
            recon_vol.append(sym_intens.detach().cpu().clone().numpy().reshape((VOL_SIZE,) * 3))
            true_objsize.append(objsizes.detach().cpu().clone().numpy())
            true_intens.append(images.detach().cpu().clone().numpy())
            pred_intens.append(recon_slice.detach().cpu().clone().numpy())
    if not save:
        return epochloss / len(input_intens)
    return epochloss / len(input_intens), true_intens, pred_intens, true_objsize, recon_vol

# HYPERPARMETERS
config = configparser.ConfigParser()
config.read('config.ini')

LR = config.getfloat('decoder', 'learning_rate')
N_INTENS = config.getint('decoder', 'n_intens')
BATCH_SIZE = config.getint('decoder', 'batch_size')
N_EPOCHS = config.getint('decoder', 'n_epochs')
SPLIT_RATIO = config.getfloat('decoder', 'split_ratio')
VOL_SIZE = config.getint('decoder', 'vol_size')
DEVICE_IDS = [int(devnum) for devnum in config.get('decoder', 'device_ids').split()]
OUTDICT_FNAME = config.get('decoder', 'outdict_fname'))
OUTDATA_FNAME = config.get('decoder', 'outdata_fname'))
DecoderCNN = getattr(network, config.get('decoder', 'network_class'))

# Loading data and detector File
all_intens, all_ori, all_objsize = preprocessing.load_data(N_INTENS)
qx_d, qy_d, qz_d = preprocessing.get_detector()

print('Total Dataset Points:', N_INTENS)
print('Total # of Training Epochs:', N_EPOCHS)
print('Batch Size:', BATCH_SIZE)
sys.stdout.flush()

# Splitting the Dataset into Train/Valid sets
train_data, valid_data = preprocessing.split_data(all_intens, all_ori, all_objsize, SPLIT_RATIO)

# Load Network
model = DecoderCNN()
# Optimizaer for NN
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)
# Choose Device to Train on
device = torch.device('cuda:%d'%DEVICE_IDS[0] if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model, device_ids=DEVICE_IDS)
model.to(device)

slices_s = slice_planes(all_ori)
# Split Sliced planes into Train/Valid
train_data['slices'] = slices_s[:len(train_data['orientation'])]
valid_data['slices'] = slices_s[len(train_data['orientation']):]

nnLoss = nn.MSELoss()

train_loss = []
valid_loss = []
stime = time.time()
for epoch in range(N_EPOCHS):
    t_loss = run_network(**train_data)
    if epoch < N_EPOCHS - 1:
        valid_data['save'] = False
        v_loss = run_network(**valid_data)
    else:
        valid_data['save'] = True
        v_loss, v_true_intens, v_pred_intens, v_objsize, v_recon_vol = run_network(**valid_data)
    sys.stderr.write('\rEPOCH %d/%d: '%(epoch+1, N_EPOCHS))
    sys.stderr.write('Training loss: %e, '%t_loss)
    sys.stderr.write('Validation loss: %e, '%v_loss)
    sys.stderr.write('%.3f s/iteration   ' % ((time.time() - stime) / (epoch+1)))
    sys.stderr.flush()
    train_loss.append(t_loss)
    valid_loss.append(v_loss)
sys.stderr.write('\n')
sys.stderr.flush()

torch.save(model.module.state_dict(), OUTDICT_FNAME)

with h5py.File(OUTDATA_FNAME, "w") as f:
    f['true_intens'] = v_true_intens
    f['pred_intens'] = v_pred_intens
    f['objsize'] = v_objsize
    f['validation_loss'] = valid_loss
    f['training_loss'] = train_loss
    f['recon_vol'] = v_recon_vol

print('Training & Validation Done, Data Saved')

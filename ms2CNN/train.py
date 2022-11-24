import sys
import time

import numpy as np
import h5py

import torch
from torch import nn
import torch.nn.functional as F

from network import DecoderCNN
import preprocessing

OUT_PREFIX = '/home/ayyerkar/acads/ms2_deepl/'

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
        recon_images[i] = torch.Tensor.permute(recon_images[i], (0,2,1))
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
                orientation=None, z1_pca=None, z2_pca=None, slices=None):
    '''Steps in training/validation of NN'''
    recon_vol = []
    true_z1_pca = []
    true_z2_pca = []
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

        z1_pca_batch = z1_pca[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        z1_pcas = torch.from_numpy(z1_pca_batch).view(BATCH_SIZE,1)
        z1_pcas = z1_pcas.float().to(device)

        z2_pca_batch = z2_pca[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        z2_pcas = torch.from_numpy(z2_pca_batch).view(BATCH_SIZE,1)
        z2_pcas = z2_pcas.float().to(device)

        output = model.forward(z1_pcas, z2_pcas)

        loss, recon_slice, sym_intens = loss_function(output, slices, images, i)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epochloss += loss.data.item()

        if not train and save:
            recon_vol.append(sym_intens.detach().cpu().clone().numpy().reshape((VOL_SIZE,) * 3))
            true_z1_pca.append(z1_pcas.detach().cpu().clone().numpy())
            true_z2_pca.append(z2_pcas.detach().cpu().clone().numpy())
            true_intens.append(images.detach().cpu().clone().numpy())
            pred_intens.append(recon_slice.detach().cpu().clone().numpy())
    if not save:
        return epochloss / len(input_intens)
    return epochloss / len(input_intens), true_intens, pred_intens, true_z1_pca, true_z2_pca, recon_vol

# HYPERPARMETERS
LR = 1e-4 # Learning Rate
N_INTENS = 100
BATCH_SIZE = 4
N_EPOCHS = 50
SPLIT_RATIO = 0.80
VOL_SIZE = 267
DEVICE_IDS = [1,2,3]

# Loading data and detector File
all_intens, all_ori, all_z1, all_z2 = preprocessing.load_data(N_INTENS)
qx_d, qy_d, qz_d = preprocessing.get_detector()

print('Total Dataset Points:', N_INTENS)
print('Total # of Training Epochs:', N_EPOCHS)
print('Batch Size:', BATCH_SIZE)

# Splitting the Dataset into Train/Valid sets
train_data, valid_data = preprocessing.split_data(all_intens, all_ori, all_z1, all_z2, SPLIT_RATIO)

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
for epoch in np.arange(N_EPOCHS)+1:
    t_loss = run_network(**train_data)
    if epoch < N_EPOCHS:
        valid_data['save'] = False
        v_loss = run_network(**valid_data)
    else:
        valid_data['save'] = True
        v_loss, v_true_intens, v_pred_intens, v_z1_pca, v_z2_pca, v_recon_vol = run_network(**valid_data)
    sys.stderr.write('\rEPOCH %d/%d: '%(epoch, N_EPOCHS))
    sys.stderr.write('Training loss: %e, '%t_loss)
    sys.stderr.write('Validation loss: %e, '%v_loss)
    sys.stderr.write('%.3f s/iteration   ' % ((time.time() - stime) / (epoch+1)))
    train_loss.append(t_loss)
    valid_loss.append(v_loss)
sys.stderr.write('\n')

torch.save(model.module.state_dict(), OUT_PREFIX + 'ms2CNN_dict')

with h5py.File(OUT_PREFIX + 'ms2_data.h5', "w") as f:
    f['true_intens'] = v_true_intens
    f['pred_intens'] = v_pred_intens
    f['z1_pca'] = v_z1_pca
    f['z2_pca'] = v_z2_pca
    f['validation_loss'] = valid_loss
    f['training_loss'] = train_loss
    f['recon_vol'] = v_recon_vol

print('Training & Validation Done, Data Saved')
import sys
import time

import numpy as np
import h5py

import torch
from torch import nn
import torch.nn.functional as F

from network import VAE
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
    sq_err_loss =  ((recon_images - images)**2).sum()
    kl_div_loss = model.module.encoder.kl
    loss = sq_err_loss + kl_div_loss
    return loss, recon_images, symarrth

def friedel_symm(arr):
    '''Apply friedel symmetry to the reconstructed volume'''
    return (arr + torch.flip(arr, dims = (1,2,3))) / 2

def train_network(input_intens, orientation, slices, save=False):
    '''Steps in training/validation of NN'''
    recon_vol_list = []
    true_intens_list = []
    pred_intens_list = []
    mu_list = []
    sigma_list = []

    epochloss=0
    for i in range(len(orientation)//BATCH_SIZE):
        intens_batch = input_intens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        images = torch.from_numpy(intens_batch).view((BATCH_SIZE, 1) + input_intens.shape[1:])
        images = images.float().to(device)

        ori_batch = orientation[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        oris = torch.from_numpy(ori_batch).view(BATCH_SIZE, 4)
        oris = oris.float().to(device)

        output, mu, sigma = model.forward(images, oris)

        loss, recon_slice, sym_intens = loss_function(output, slices, images, i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochloss += loss.data.item()

        if save:
            recon_vol_list.append(sym_intens.detach().cpu().clone().numpy().reshape((VOL_SIZE,) * 3))
            true_intens_list.append(images.detach().cpu().clone().numpy())
            pred_intens_list.append(recon_slice.detach().cpu().clone().numpy())
            mu_list.append(mu.detach().cpu().clone().numpy())
            sigma_list.append(sigma.detach().cpu().clone().numpy())
    if not save:
        return epochloss / len(input_intens)
    out_dict = {'true_intens': true_intens_list,
                'pred_intens': pred_intens_list,
                'recon_vol': recon_vol_list,
                'mu': mu_list,
                'sigma': sigma_list,
    }
    return epochloss / len(input_intens), out_dict

# HYPERPARMETERS
LR = 1e-4 # Learning Rate
N_INTENS = 100
BATCH_SIZE = 8
N_EPOCHS = 50
VOL_SIZE = 165
DEVICE_IDS = [1,2,3]
LATENT_DIMS = 1

# Loading data and detector File
all_intens, all_ori = preprocessing.load_data(N_INTENS)
qx_d, qy_d, qz_d = preprocessing.get_detector()

print('Total Dataset Points:', N_INTENS)
print('Total # of Training Epochs:', N_EPOCHS)
print('Batch Size:', BATCH_SIZE)

# Load Network
model = VAE(LATENT_DIMS)
# Optimizaer for NN
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)
# Choose Device to Train on
device = torch.device('cuda:%d'%DEVICE_IDS[0] if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model, device_ids=DEVICE_IDS)
model.to(device)

slices_s = slice_planes(all_ori)

nnLoss = nn.MSELoss()

train_loss = []
stime = time.time()
for epoch in np.arange(N_EPOCHS) + 1:
    if epoch == N_EPOCHS:
        t_loss, data_dict = train_network(all_intens, all_ori, slices_s, save=True)
    else:
        t_loss = train_network(all_intens, all_ori, slices_s)
    sys.stderr.write('\rEPOCH %d/%d: '%(epoch, N_EPOCHS))
    sys.stderr.write('Training loss: %f, '%t_loss)
    sys.stderr.write('%.3f s/iteration   ' % ((time.time() - stime) / (epoch+1)))
    train_loss.append(t_loss)
sys.stderr.write('\n')

torch.save(model.module.state_dict(), OUT_PREFIX + 'sim_vae_dict')

with h5py.File(OUT_PREFIX + 'sim_vae_data.h5', "w") as f:
    f['loss'] = train_loss
    for key in data_dict.keys():
        f[key] = data_dict[key]

print('Training & Validation Done, Data Saved')

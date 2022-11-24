import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms 

import numpy as np

from network import VAE
from preprocessing import load_data
from preprocessing import get_detector
import os
import sys
import h5py
import time
import neoemc as ne
import quaternion as quat

'''Loading Data and Deetecor File'''
input_intens, orientation = load_data()
qx_d, qy_d, qz_d = get_detector()

DATA_POINTS = len(orientation)
print('Total Dataset Points:', DATA_POINTS)
 
'''HYPERPARMETERS'''   
'''Learning Rate'''      
LR = 1e-4
'''Batch Size''' 
BATCH_SIZE = 8
'''# of EPOCHS'''
N_EPOCHS = 500

print('Total # of Training Epochs:', N_EPOCHS)
print('Batch Size:', BATCH_SIZE)

'''Load Network'''
latent_dims=1
model = VAE(latent_dims)
'''Optimizaer for NN'''
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)
'''Choose Device to Train on'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print('# of devices:', torch.cuda.device_count())
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])

'''Auxiliary Functions'''

def _q_rotation(x, y, z, i, qw, qx, qy, qz):
    '''Get Rotation Matrix from quaternions.
        Input :x, y, z --> detector pixel coordinates
              qw, qx, qy, qz --> quaternions'''
    matrx = [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]]
    return np.transpose(np.transpose(np.array([x, y, z]), (1,2,0))@matrx, (2,0,1))

def slice_planes(orien):
    '''Get slicing planes for corresponding rotation matrices'''
    imgsize = input_intens.shape[-1]
    n_slices = DATA_POINTS 
    slices_s = np.zeros((n_slices, imgsize, imgsize, 3))
    for i in range(n_slices):
        q0, q1, q2, q3 = orien[i]
        slices_s[i] = _q_rotation(qx_d, qy_d, qz_d, i, q0, q1, q2, q3).T
    return torch.from_numpy(slices_s).to(device)

slices_s = slice_planes(orientation)

def best_projection_slice(recon_intens_3d, slices, ind):
    '''Slicing the reconstructed volume'''
    select_plane =  slices[ind]
    size = recon_intens_3d.shape[2]
    grid = select_plane.float()
    return F.grid_sample(recon_intens_3d.view(1, 1, size, size, size), 
            grid.view(1, grid.shape[0], grid.shape[0], 1, 3), mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True)[0][0][:,:].reshape(grid.shape[0], grid.shape[0])

def loss_function(epoch, recon_intens_3d, slices, images, bnum):
    recon_images = torch.zeros_like(images)
    for i in range(BATCH_SIZE):
        '''Applying Symmetrization : Friedal Symmetry or Icosahedra Symmetry'''
        arrth = recon_intens_3d[i]
        symarrth = friedel_symm(arrth)
        recon_intens_3d_sym = symarrth
        recon_images[i] = best_projection_slice(torch.Tensor.permute(recon_intens_3d_sym, (0,3,2,1)), slices, bnum*BATCH_SIZE + i)
        recon_images[i] = torch.Tensor.permute(recon_images[i], (0,2,1))
        '''No Symmetrization'''
        #else:
        #recon_images[i] = best_projection_slice(torch.Tensor.permute(recon_intens_3d[i], (0,3,2,1)), slices, bnum*BATCH_SIZE + i)
        #recon_images[i] = torch.Tensor.permute(recon_images[i], (0,2,1))
    SEloss =  ((recon_images - images)**2).sum() 
    KLDloss = model.module.encoder.kl
    loss = SEloss + KLDloss
    return loss, SEloss, KLDloss, recon_images, recon_intens_3d_sym


def friedel_symm(recon_intens_3d):
    '''Apply friedel symmetry to the reconstructed volume'''
    a = recon_intens_3d
    return (a + torch.flip(a, dims = (1,2,3))) / 2  

def trainNN(epoch, input_intens, orientation):
    epochloss=0
    vsize=165
    mu_ = []
    sigma_ = []
    recon_3dVol = np.zeros((0,vsize,vsize,vsize))
    true_intens = np.zeros((BATCH_SIZE, input_intens.shape[1], input_intens.shape[1]))
    pred_intens = np.zeros((BATCH_SIZE, input_intens.shape[1], input_intens.shape[1]))
    for i in range(len(orientation)//BATCH_SIZE):
        intens_batch = input_intens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        images = torch.from_numpy(intens_batch).view(BATCH_SIZE, 1, input_intens.shape[1], input_intens.shape[1])
        images = images.float().to(device)
        
        ori_batch = orientation[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        oris = torch.from_numpy(ori_batch).view(BATCH_SIZE, 4)
        oris = oris.float().to(device)
        
        output, mu, sigma = model.forward(images, oris) 
        loss, seloss, kldloss, recon_2D_x, recon_intens_3d_sym = loss_function(epoch, output, slices_s, images, i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epochloss += loss.data.item()
        if epoch == (N_EPOCHS-1):
            recon_3dVol = np.concatenate((recon_3dVol, recon_intens_3d_sym.detach().cpu().clone().numpy().reshape(1,vsize,vsize,vsize)), axis=0)
            true_intens = np.concatenate((true_intens, images.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])), axis=0)
            pred_intens = np.concatenate((pred_intens, recon_2D_x.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])), axis=0)
            mu_.append(mu.detach().cpu().numpy())
            sigma_.append(sigma.detach().cpu().numpy())
    return true_intens, pred_intens, (epochloss/len(orientation)), recon_3dVol, mu_, sigma_

training_loss = []
stime = time.time()
for epoch in range(N_EPOCHS):
    true_intens, pred_intens, loss, recon_vol, mu, sigma = trainNN(epoch, input_intens, orientation)
    sys.stderr.write('\rEPOCH %d/%d: '%(epoch+1, N_EPOCHS))
    sys.stderr.write('Training loss: %e, '%loss)
    sys.stderr.write('%.3f s/iteration   ' % ((time.time() - stime) / (epoch+1)))
    training_loss.append(loss)
sys.stderr.write('\n')

training_loss = np.array(training_loss)

torch.save(model.module.state_dict(), '/home/mallabhi/StrucNN/vae/output/vae_dict_2')

with h5py.File('/home/mallabhi/StrucNN/vae/output/valid_data_2.h5', "w") as f:
                f['true_intens'] = true_intens
                f['pred_intens'] = pred_intens
                f['loss'] = training_loss
                f['mu'] = mu
                f['sigma'] = sigma
                f['recon_vol'] = recon_vol
print('Training & Validation Done, Data Saved') 

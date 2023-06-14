import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms 

import neoemc as ne
import quaternion as quat
import numpy as np

from network import VAE
from preprocessing import load_data
from preprocessing import get_detector
import os
import sys
import h5py
import time
from random import randint
from sklearn.utils import shuffle 

#torch.manual_seed(42)

'''Loading Data and Detector File'''
input_intens, orientation = load_data()
qx_d, qy_d, qz_d = get_detector()

DATA_POINTS = len(orientation)
print('Total Dataset Points:', DATA_POINTS)
 
'''HYPERPARMETERS'''  
BETA=0.5
'''Learning Rate'''      
LR = 1e-4
'''Batch Size''' 
BATCH_SIZE = 32
'''# of EPOCHS'''
N_EPOCHS = 400
print('Total # of Training Epochs:', N_EPOCHS)
print('Batch Size:', BATCH_SIZE)

'''Load Network'''
LATENT_DIMS=2
print('Latent Dims:', LATENT_DIMS)
model = VAE(LATENT_DIMS)
'''Choose Device to Train on'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_model=True
if load_model is True:
    model.load_state_dict(torch.load('/u/mallabhi/StrucNN/ms2vae/output/vae_ms2_sel_l2_01', map_location=device))
    print('Model loaded')
'''Optimizaer for NN'''
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)
model.to(device)
print('# of devices:', torch.cuda.device_count())
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
sys.stdout.flush()

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

def loss_function(epoch, recon_intens_3d, slices, images, bnum, mu, logvar, beta):
    recon_images = torch.zeros_like(images)
    for i in range(BATCH_SIZE):
        '''Applying Symmetrization : Friedal Symmetry'''
        arrth = recon_intens_3d[i]
        symarrth = friedel_symm(arrth)
        recon_intens_3d_sym = symarrth
        recon_images[i] = best_projection_slice(torch.Tensor.permute(recon_intens_3d_sym, (0,3,2,1)), slices, bnum*BATCH_SIZE + i)
        recon_images[i] = torch.Tensor.permute(recon_images[i].clone(), (0,2,1))
    BSE =  ((recon_images - images)**2).sum() 
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    LOSS = BSE + beta * KLD
    return LOSS, BSE, KLD, recon_images


def friedel_symm(recon_intens_3d):
    '''Apply friedel symmetry to the reconstructed volume'''
    a = recon_intens_3d
    return (a + torch.flip(a, dims = (1,2,3))) / 2  

def ico_symm(recon_intens_3d):
    '''Apply Icosahedral Symmetry'''
    sym_models = recon_intens_3d.clone()
    q_ind = np.array([randint(0,59) for i in range(10)])
    quats = ne.math.sym_group_quat(sym_type='IL', qs=quat.quaternion(2**-0.5, 0, 2**-0.5, 0))
    quats_n = quats[q_ind]
    a_matrx = torch.from_numpy(np.array([np.append(quat.as_rotation_matrix(q).T, np.zeros((3,1)), axis=1) for q in quats_n]))
    grid = F.affine_grid(a_matrx,torch.Size((len(a_matrx),)+recon_intens_3d.shape), align_corners=True)
    for i in range(len(quats)):  
        sym_models += F.grid_sample(
            recon_intens_3d.view((1,)+recon_intens_3d.shape),grid[[i]].float().to(device),align_corners=True).reshape(recon_intens_3d.shape)
    return sym_models/len(quats)



def trainNN(epoch, input_intens, orientation, beta):
    epochloss = 0
    bseloss = 0
    kldloss=0
    mu_ = np.zeros((len(orientation), LATENT_DIMS))
    logvar_ = np.zeros((len(orientation), LATENT_DIMS))
    true_intens = np.zeros((len(orientation), input_intens.shape[1], input_intens.shape[1]))
    pred_intens = np.zeros((len(orientation), input_intens.shape[1], input_intens.shape[1]))
    idx=0
    for i in range(len(orientation)//BATCH_SIZE):
        intens_batch = input_intens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        images = torch.from_numpy(intens_batch).view(BATCH_SIZE, 1, input_intens.shape[1], input_intens.shape[1])
        images = images.float().to(device)
        
        ori_batch = orientation[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        oris = torch.from_numpy(ori_batch).view(BATCH_SIZE, 4)
        oris = oris.float().to(device)
        
        output, mu, logvar = model.forward(images, oris) 
        loss, bse, kld, recon_2D_x = loss_function(epoch, output, slices_s, images, i, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        bseloss += bse.data.item()
        kldloss += kld.data.item()
        epochloss += loss.data.item()
        if epoch % 5==0:
            true_intens[idx:idx+BATCH_SIZE,:,:] = images.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])
            pred_intens[idx:idx+BATCH_SIZE,:,:] = recon_2D_x.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])
            mu_[idx:idx+BATCH_SIZE,:] = mu.detach().cpu().numpy()
            logvar_[idx:idx+BATCH_SIZE,:] = logvar.detach().cpu().numpy()
            idx += BATCH_SIZE
    return true_intens, pred_intens, mu_, logvar_, epochloss/len(orientation), bseloss/len(orientation), kldloss/len(orientation)


quats = ne.math.sym_group_quat(sym_type='IL', qs=quat.quaternion(2**-0.5, 0, 2**-0.5, 0))
quaternions = quat.as_float_array(quats)
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def _get_orient(orientation, quaternions):
    orient_n = np.copy(orientation)
    pick = np.arange(0, 60, 1)
    for j in range(len(orientation)):
        q_ind = np.random.choice(pick)
        orient_n[j] = quaternion_multiply(orientation[j], quaternions[q_ind])
    return orient_n

training_loss = []
bse_loss = []
kld_loss = []
stime = time.time()

for epoch in np.arange(N_EPOCHS)+1:
    orientation_n = _get_orient(orientation, quaternions)
    true_intens, pred_intens, mu, logvar, loss, bseloss, kldloss = trainNN(epoch, input_intens, orientation_n, BETA)
    training_loss.append(loss)
    bse_loss.append(bseloss)
    kld_loss.append(kldloss)

    sys.stderr.write('\rEPOCH %d/%d: '%(epoch, N_EPOCHS))
    sys.stderr.write('Training loss: %e, '%loss)
    sys.stderr.write('SE loss: %e, '%bseloss)
    sys.stderr.write('KLD loss: %e, '%kldloss)
    sys.stderr.write('%.3f s/iteration   ' % ((time.time() - stime) / (epoch+1)))
    sys.stderr.flush()

    if epoch %5==0:
        torch.save(model.module.state_dict(), '/u/mallabhi/StrucNN/ms2vae/output/vae_ms2_sel_l2_01')
        with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/ms2_sel_l2_01.h5', "w") as f:
                        f['true_intens'] = true_intens
                        f['pred_intens'] = pred_intens
                        f['loss'] = training_loss
                        f['bseloss'] = bse_loss
                        f['kldloss'] = kld_loss
                        f['mu'] = mu
                        f['logvar'] = logvar

sys.stderr.write('\n')
sys.stderr.flush()

print('Training & Validation Done, Data Saved') 

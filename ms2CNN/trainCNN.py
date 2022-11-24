import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms 

import numpy as np

from network import ms2CNN
from preprocessing import load_data
from preprocessing import get_detector
from preprocessing import split_data
import os
import sys
import h5py
import time
import neoemc as ne
import quaternion as quat

'''Loading Data and Deetecor File'''
input_intens, orientation, z1_pca, z2_pca = load_data()
qx_d, qy_d, qz_d = get_detector()

DATA_POINTS = len(z1_pca)
print('Total Dataset Points:', DATA_POINTS)
 
'''HYPERPARMETERS'''   
'''Learning Rate'''      
LR = 1e-4
'''Batch Size''' 
BATCH_SIZE = 32
'''# of EPOCHS'''
N_EPOCHS = 300

print('Total # of Training Epochs:', N_EPOCHS)
print('Batch Size:', BATCH_SIZE)

'''Splitting the Dataset into Train/Valid sets'''
SPLIT_RATIO = 0.80
train_intens, train_ori, train_z1_pca, train_z2_pca, valid_intens, valid_ori, valid_z1_pca, valid_z2_pca = split_data(input_intens, orientation, z1_pca, z2_pca, DATA_POINTS, SPLIT_RATIO)
'''Load Network'''
model = ms2CNN()
'''Optimizaer for NN'''
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)
'''Choose Device to Train on'''
device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])

model.to(device)

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
    imgsize = train_intens.shape[-1]
    n_slices = DATA_POINTS 
    slices_s = np.zeros((n_slices, imgsize, imgsize, 3))
    for i in range(n_slices):
        q0, q1, q2, q3 = orien[i]
        slices_s[i] = _q_rotation(qx_d, qy_d, qz_d, i, q0, q1, q2, q3).T
    return torch.from_numpy(slices_s).to(device)

slices_s = slice_planes(orientation)
'''Split Sliced planes into Train/Valid'''
train_slices_s = slices_s[:int(DATA_POINTS*SPLIT_RATIO), :,:,:]
valid_slices_s = slices_s[int(DATA_POINTS*SPLIT_RATIO):, :,:,:]

def best_projection_slice(recon_intens_3d, slices, ind):
    '''Slicing the reconstructed volume'''
    select_plane =  slices[ind]
    size = recon_intens_3d.shape[2]
    grid = select_plane.float()
    return F.grid_sample(recon_intens_3d.view(1, 1, size, size, size), 
            grid.view(1, grid.shape[0], grid.shape[0], 1, 3), mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True)[0][0][:,:].reshape(grid.shape[0], grid.shape[0])

'''Loss for NN'''
nnLoss= nn.MSELoss()
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
    loss = nnLoss(recon_images, images)
    return loss, recon_images, recon_intens_3d_sym


def friedel_symm(recon_intens_3d):
    '''Apply friedel symmetry to the reconstructed volume'''
    a = recon_intens_3d
    return (a + torch.flip(a, dims = (1,2,3))) / 2  

def ico_symm(recon_intens_3d):
    '''Apply Icosahedral Symmetry'''
    sym_models = recon_intens_3d.clone()
    quats = ne.math.sym_group_quat(sym_type='IL')
    a_matrx = torch.from_numpy(np.array([np.append(quat.as_rotation_matrix(q).T, np.zeros((3,1)), axis=1) for q in quats]))
    grid = F.affine_grid(a_matrx,torch.Size((len(a_matrx),)+recon_intens_3d.shape), align_corners=True)
    for i in range(len(quats)):  
        sym_models += F.grid_sample(
            recon_intens_3d.view((1,)+recon_intens_3d.shape),grid[[i]].float().to(device),align_corners=True).reshape(recon_intens_3d.shape)
    return sym_models/len(quats)


'''Steps in training of NN'''
def trainNN(epoch, input_intens, orientation, z1_pca, z2_pca):
    epochloss=0
    for i in range(len(orientation)//BATCH_SIZE):
        intens_batch = input_intens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        images = torch.from_numpy(intens_batch).view(BATCH_SIZE, 1, input_intens.shape[1], input_intens.shape[1])
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
        loss, recon_2D_x, recon_intens_3d_sym = loss_function(epoch, output, train_slices_s, images, i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epochloss += loss.data.item()
    return (epochloss/len(train_ori)), output.shape[-1]


'''Steps in Validation of NN'''
def validateNN(epoch, input_intens, orientation, z1_pca, z2_pca, vol_size):
    epochloss=0
    vsize=vol_size
    recon_3dVol = np.zeros((0,vsize,vsize,vsize))
    true_z1_pca = np.zeros((BATCH_SIZE,1))
    true_z2_pca = np.zeros((BATCH_SIZE,1))
    true_intens = np.zeros((BATCH_SIZE, input_intens.shape[1], input_intens.shape[1]))
    pred_intens = np.zeros((BATCH_SIZE, input_intens.shape[1], input_intens.shape[1]))
    for i in range(len(orientation)//BATCH_SIZE):
        intens_batch = input_intens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        images = torch.from_numpy(intens_batch).view(BATCH_SIZE, 1, input_intens.shape[1], input_intens.shape[1])
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
        loss, recon_2D_x, recon_intens_3d_sym = loss_function(epoch, output, valid_slices_s, images, i)

        epochloss += loss.data.item()
        #if epoch == (N_EPOCHS-1):
        if epoch % 10 == 0:
            recon_3dVol = np.concatenate((recon_3dVol, recon_intens_3d_sym.detach().cpu().clone().numpy().reshape(1,vsize,vsize,vsize)), axis=0)
            true_z1_pca = np.concatenate((true_z1_pca, z1_pcas.detach().cpu().clone().numpy()), axis=0)
            true_z2_pca = np.concatenate((true_z2_pca, z2_pcas.detach().cpu().clone().numpy()), axis=0)
            true_intens = np.concatenate((true_intens, images.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])), axis=0)
            pred_intens = np.concatenate((pred_intens, recon_2D_x.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])), axis=0)
    return true_intens, pred_intens, (epochloss/len(valid_ori)), true_z1_pca, true_z2_pca, recon_3dVol

train_loss = []
valid_loss = []
stime = time.time()
for epoch in np.arange(N_EPOCHS)+1:
    t_loss, vol_size = trainNN(epoch, train_intens, train_ori, train_z1_pca, train_z2_pca)
    v_true_intens, v_pred_intens, v_loss, v_z1_pca, v_z2_pca, v_recon_vol = validateNN(epoch, valid_intens, valid_ori, valid_z1_pca, valid_z2_pca, vol_size)
    if epoch %10 ==0:
        torch.save(model.module.state_dict(), '/home/mallabhi/StrucNN/ms2CNN/output/model_dict_MS2_8')

        with h5py.File('/home/mallabhi/StrucNN/ms2CNN/output/valid_data_MS2_8.h5', "w") as f:
                        f['true_intens'] = v_true_intens
                        f['pred_intens'] = v_pred_intens
                        f['tloss'] = train_loss
                        f['vloss'] = valid_loss
                        f['recon_vol'] = v_recon_vol
    sys.stderr.write('\rEPOCH %d/%d: '%(epoch, N_EPOCHS))
    sys.stderr.write('Training loss: %e, '%t_loss)
    sys.stderr.write('Validation loss: %e, '%v_loss)
    sys.stderr.write('%.3f s/iteration   ' % ((time.time() - stime) / (epoch+1)))
    train_loss.append(t_loss)
    valid_loss.append(v_loss)
sys.stderr.write('\n')

train_loss = np.array(train_loss)
valid_loss = np.array(valid_loss)

torch.save(model.module.state_dict(), '/home/mallabhi/StrucNN/ms2CNN/output/model_dict_MS2_8')

with h5py.File('/home/mallabhi/StrucNN/ms2CNN/output/valid_data_MS2_8.h5', "w") as f:
                f['true_intens'] = v_true_intens
                f['pred_intens'] = v_pred_intens
                f['tloss'] = train_loss
                f['vloss'] = valid_loss
                f['recon_vol'] = v_recon_vol
print('Training & Validation Done, Data Saved') 

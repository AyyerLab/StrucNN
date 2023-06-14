import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mrcfile
import neoemc as ne
import quaternion as quat
import h5py
from scipy import ndimage
from scipy.signal import argrelextrema
import network
import sys
import os


def radial_average(intens):
    size = intens.shape[-1]
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    r = np.sqrt((x - size/2)**2 + (y - size/2)**2)
    bins = np.arange(r.max() + 1)
    radial_sum = ndimage.sum(intens, labels=np.round(r), index=bins)
    radial_count = ndimage.sum(np.ones_like(intens), labels=np.round(r), index=bins)
    radial_count[radial_count==0] = 1e-4
    radial_avg = radial_sum / radial_count
    return radial_avg, bins


def ave_fun(q, a,b,c):
    '''Scaling 2D Intensitis with mean of radial average of all 2D-Intensities in datatset'''
    return a*q**(-b) + c


def _mask_reconVol(recon_vol):
    '''mask for generated volume'''
    r_min=12
    r_max=80
    volsize = recon_vol.shape[-1]
    ind = np.arange(volsize) - volsize//2
    x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
    rad = np.sqrt(x**2 + y**2 + z**2)
    recon_vol[:,rad>=r_max] = 0
    recon_vol[:,rad<=r_min] = -1.
    return recon_vol

def bg_subtract(vol, bgrad=8.,r_min=13, r_max=80):
    ini = vol.copy()
    size = ini.shape[-1]
    ind = np.arange(size) - size//2
    x, y, z = np.meshgrid(ind,ind,ind, indexing='ij')
    rad = np.sqrt(x**2+y**2+z**2)
    mask = (rad<=r_min) | (rad>=r_max)
    ini[mask] = 1e20
    ini_min = np.array(ndimage.minimum_filter(ini, bgrad, mode='constant'))
    ini -= ini_min
    ini[rad<=r_min] = -1.
    ini[rad>=r_max] = 0
    outmask = (rad<=(r_max+1)) & (rad>=(r_max-1))
    ini[outmask] = -1
    return ini.astype('f4')

def _scale_reconVol(recon_vol):
    '''scaling back data using ave_3d function'''
    scale = 171
    half_scale = scale//2
    factor = 503/scale
    x0,y0,z0 = np.indices((scale, scale, scale)); x0-=half_scale; y0-=half_scale; z0-=half_scale
    rad_float = np.sqrt(x0**2 + y0**2 +z0**2)*factor
    rad_float[rad_float==0] = 1e-3
    ave_3d = np.array([ave_fun(i,2.74e8,3.25,9.34e-1) for i in rad_float.ravel()]).reshape(scale,scale,scale)
    #zero padding to match the shape of RECON VOL & sampled down INTENS
    ave_3d_pad = np.pad(ave_3d, 48, mode='constant')
    return ave_3d_pad*recon_vol

def ico_symm(recon_intens_3d):
    '''Apply Icosahedral Symmetry'''
    sym_models = recon_intens_3d.clone()
    quats = ne.math.sym_group_quat(sym_type='IL', qs=quat.quaternion(2**-0.5, 0, 2**-0.5, 0))
    a_matrx = torch.from_numpy(np.array([np.append(quat.as_rotation_matrix(q).T, np.zeros((3,1)), axis=1) for q in quats]))
    grid = F.affine_grid(a_matrx,torch.Size((len(a_matrx),)+recon_intens_3d.shape), align_corners=True)
    for i in range(len(quats)):
        sym_models += F.grid_sample(recon_intens_3d.view((1,)+recon_intens_3d.shape),grid[[i]].float().to(device),align_corners=True).reshape(recon_intens_3d.shape)
    return sym_models/len(quats)


with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/ms2_sel_l2_01.h5', 'r') as f:
    mu = f['mu'][:]
m1 = mu[:,0].ravel()[1344*3:]
m2 = mu[:,1].ravel()[1344*3:]

BATCH_SIZE=8

LATENT_DIMS=2
print('Latent Dims:', LATENT_DIMS)
model = network.VAE(LATENT_DIMS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
model.load_state_dict(torch.load('/u/mallabhi/StrucNN/ms2vae/output/vae_ms2_sel_l2_01', map_location=device))
print('Model Loaded')
sys.stdout.flush()

'''Latent coordinated'''
#straight trajectory
#x1, y1 = -1.8, -0.2
#x2, y2 = 0.2, 1.23
#slope = (y2-y1)/(x2-x1)
#y_intercept = y1 - slope*x1
#x_values = np.linspace(x1, x2, num=20)
#y_values = np.array([slope*x + y_intercept for x in x_values])
#coordsx = torch.Tensor(x_values.tolist())
#coordsy = torch.Tensor(y_values.tolist())

#curved trajectory
#coordsx = torch.Tensor(np.linspace(1.6,-1.8, 10).tolist())
#coordsy = torch.Tensor((-0.4*(coordsx)**2+1).tolist())

#Volumes for all mu's
coordsx = torch.Tensor(m1.tolist())
coordsy = torch.Tensor(m2.tolist())
coords = np.vstack((coordsx, coordsy)).T

num_coords = len(coords)
recon_vol = np.zeros((len(coords), 267,267,267))
for i in range(len(coords)//BATCH_SIZE):
    batch_coords = coords[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    batch_recon_vol = np.array(model.decoder(torch.Tensor(batch_coords)).detach().numpy())
    recon_vol[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = batch_recon_vol[:,0,:,:,:]
    print('which vol batch generated:',i+1)
    sys.stdout.flush()
print('Volume Generated & Masked')
sys.stdout.flush()

recon_vol = _mask_reconVol(recon_vol)

recon_vol_sym = np.zeros((recon_vol.shape))
recon_vol_scaled = np.zeros((recon_vol.shape))
for i in range(len(recon_vol)//BATCH_SIZE):
    recon_vol_sym[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = ico_symm(torch.Tensor(recon_vol[i*BATCH_SIZE:(i+1)*BATCH_SIZE]).to(device)).cpu().numpy()
    recon_vol_scaled[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = _scale_reconVol(recon_vol_sym[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    recon_vol_scaled[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = _mask_reconVol(recon_vol_scaled[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    print('which batch:',i+1)
    sys.stdout.flush()
print('Symmetrization Done')
sys.stdout.flush()



recon_vol_bgsbt = np.zeros((recon_vol_scaled.shape))
for i in range(len(recon_vol_scaled)):
    print('which vol done:', i+1)
    sys.stdout.flush()
    recon_vol_bgsbt[i] = bg_subtract(recon_vol_scaled[i],bgrad=8.)

print('Background Subtracted')
sys.stdout.flush()

with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/volumes_vae_l2_01_d.h5', 'w') as f:
     f['recon_vol'] = recon_vol_sym[:,53:214,53:214,53:214]
     f['recon_vol_scaled'] = recon_vol_scaled[:,53:214,53:214,53:214]
     f['recon_vol_bgsbt'] = recon_vol_bgsbt[:,53:214,53:214,53:214]

#for l in range(len(coords)):
#    with mrcfile.new('/u/mallabhi/StrucNN/ms2vae/output/volumes/vol_bgsubt_vae_25_2_b_%.3d.ccp4'%l, overwrite=True) as f:
#         f.set_data(recon_vol_bgsbt[l,53:214,53:214,53:214].astype('f4'))

print('Volume Saved')
sys.stderr.flush()


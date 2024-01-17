import configparser
import h5py
from scipy import ndimage
from scipy.signal import argrelextrema
import network
import sys
import os
import torch
import numpy as np

import torch.nn as nn
import torch

config = configparser.ConfigParser()
config.read('recon_config.ini')

NUM_START = int(config['Hyperparameters']['NUM_START'])
NUM_END = int(config['Hyperparameters']['NUM_END'])
BATCH_SIZE = int(config['Hyperparameters']['BATCH_SIZE'])
LATENT_DIMS = int(config['Hyperparameters']['LATENT_DIMS'])
ICO_SYMMETRIZATION = config['Hyperparameters'].getboolean('ICO_SYMMETRIZATION')

DEFINED_COORDINATES = config['Hyperparameters'].getboolean('DEFINED_COORDINATES')

OUTPUT_FILE = config['Files']['OUTPUT_FILE']
DEFINED_COORDINATES_FILE = config['Files']['DEFINED_COORDINATES_FILE']

VAE_MODEL_FILE = config['Files']['VAE_MODEL_FILE']
VOLUMES_FILE = config['Files']['VOLUMES_FILE']


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


def ave_fun(q, a,b,c):
    '''Scaling 2D Intensitis with mean of radial average of all 2D-Intensities in datatset'''
    return a*q**(-b) + c


def ico_symm(recon_intens_3d):
    '''Apply Icosahedral Symmetry'''
    sym_models = recon_intens_3d.clone()
    quats = ne.math.sym_group_quat(sym_type='IL', qs=quat.quaternion(2**-0.5, 0, 2**-0.5, 0))
    a_matrx = torch.from_numpy(np.array([np.append(quat.as_rotation_matrix(q).T, np.zeros((3,1)), axis=1) for q in quats]))
    grid = F.affine_grid(a_matrx,torch.Size((len(a_matrx),)+recon_intens_3d.shape), align_corners=True)
    for i in range(len(quats)):
        sym_models += F.grid_sample(recon_intens_3d.view((1,)+recon_intens_3d.shape),grid[[i]].float().to(device),align_corners=True).reshape(recon_intens_3d.shape)
    return sym_models/len(quats)


def friedel_symm(recon_intens_3d):
    '''Apply friedel symmetry to the reconstructed volume'''
    a = recon_intens_3d
    return (a + torch.flip(a, dims = (0,1,2))) / 2


def sampling(mu, logvar):
    std = np.exp(0.5 * logvar)
    eps = np.random.randn(*std.shape)
    z = eps * std + mu
    return z

if DEFINED_COORDINATES:
    with h5py.File(DEFINED_COORDINATES_FILE, 'r') as f:
        mu = f['mu'][:]
else:
    with h5py.File(OUTPUT_FILE, 'r') as f:
        mu = f['mu'][NUM_START:NUM_END]


model = network.VAE(LATENT_DIMS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
model.load_state_dict(torch.load(VAE_MODEL_FILE, map_location=device))
print('Model Loaded')
sys.stdout.flush()
model.to(device)
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
sys.stdout.flush()


num_coords = len(mu)
recon_vol = np.zeros((len(mu), 267,267,267))
for i in range(len(mu)//BATCH_SIZE):
    batch_mu = mu[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    #batch_recon_vol = np.array(model.decoder(torch.Tensor(batch_z)).detach().numpy())
    batch_recon_vol = model.module.decoder(torch.Tensor(batch_mu).to(device)).detach().cpu().numpy()
    recon_vol[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = batch_recon_vol[:,0,:,:,:]
    sys.stderr.write('\rVolume Generated, Batch %d/%d: '%(i+1, (len(mu)//BATCH_SIZE)))
    sys.stdout.flush()
print('Volume Generation : Done')
sys.stdout.flush()

recon_vol = _mask_reconVol(recon_vol)

recon_vol_sym = np.zeros((recon_vol.shape))
recon_vol_scaled = np.zeros((recon_vol.shape))
for i in range(len(mu)//BATCH_SIZE):
    
    batch_recon_vol = recon_vol[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    for j in range(len(batch_recon_vol)):
            if ICO_SYMMETRIZATION:
                recon_vol_sym[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = ico_symm(torch.Tensor(recon_vol[i*BATCH_SIZE:(i+1)*BATCH_SIZE]).to(device)).cpu().numpy()
            else:
                recon_vol_sym[i*BATCH_SIZE + j] = friedel_symm(torch.Tensor(batch_recon_vol[j]).to(device)).cpu().numpy()

    recon_vol_scaled[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = _scale_reconVol(recon_vol_sym[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    recon_vol_scaled[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = _mask_reconVol(recon_vol_scaled[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    sys.stderr.write('\rVolume Scaled, Batch %d/%d: '%(i+1, (len(mu)//BATCH_SIZE)))
    sys.stdout.flush()
print('Volume Scale : Done')
sys.stdout.flush()


recon_vol_bgsbt = np.zeros((recon_vol_scaled.shape))
for i in range(len(recon_vol_scaled)):
    recon_vol_bgsbt[i] = bg_subtract(recon_vol_scaled[i], bgrad=8.)
    sys.stderr.write('\rBackground Subtracted Volume: %d/%d: '%(i+1, len(recon_vol_scaled)))
    sys.stdout.flush()

print('Background Subtracted : Done')
sys.stdout.flush()

with h5py.File(VOLUMES_FILE, 'w') as f:
     f['recon_vol'] = recon_vol_sym
     f['recon_vol_scaled'] = recon_vol_scaled
     f['recon_vol_bgsbt'] = recon_vol_bgsbt
     f['recon_vol_crop'] = recon_vol_sym[:,53:214,53:214,53:214]
     f['recon_vol_scaled_crop'] = recon_vol_scaled[:,53:214,53:214,53:214]
     f['recon_vol_bgsbt_crop'] = recon_vol_bgsbt[:,53:214,53:214,53:214]


print('Volume Saved')
sys.stderr.flush()



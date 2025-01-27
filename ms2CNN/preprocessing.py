import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import shuffle
import os
import h5py
from scipy import interpolate

def ave_fun(q, a,b,c):
    '''Scaling 2D Intensitis with average of radial avgs. 
                        of all 2D-Intensities in datatset'''
    return a*q**(-b) + c

def load_data():
    '''Load object's size, 2D Intensity avgs. and corresponding orientations'''
    with h5py.File('/home/mallabhi/StrucNN/data/MS2_10k_subset.h5', 'r') as f:
        intens = f['intens'][:]
    orientation = np.load('/home/mallabhi/StrucNN/data/orient_10k_subet.npy')[:]
    orien_new = np.zeros((len(orientation), 4))
    orien_new[:,0] = orientation[:,0]
    orien_new[:,1] = -orientation[:,1]
    orien_new[:,2] = -orientation[:,2]
    orien_new[:,3] = -orientation[:,3]
    orientation = orien_new

    '''Load z1 and z2 pca components of cc-matrix(calculated for 2d intens)'''
    with h5py.File('/home/mallabhi/StrucNN/data/MS2_10k_subset.h5', 'r') as f:
        z1_pca = f['z1_pca'][:]
        z2_pca = f['z2_pca'][:]

    '''Normalizing Intensity vals to range: 0-1'''
    sz = intens.shape[1]
    x0, y0 = np.indices((sz,sz))
    x0 -= sz//2
    y0 -= sz//2
    rad_float = np.sqrt(x0**2 + y0**2)
    rad_float[rad_float==0] = 1e-3
    ave_2d = np.array([ave_fun(i,1.36e8,3.56,6.89) for i in rad_float.ravel()]).reshape(sz,sz)

    norm_intens = intens/ave_2d
    input_intens = norm_intens/np.max(norm_intens) * 0.99
    input_intens = sample_down_intens(input_intens)

    print('Data Processed.')
    return input_intens, orientation, z1_pca, z2_pca

def mask_circle(intens_input):
    '''Masking outer region (rad>imagesize//2) to zero'''
    print('Input Image Size:', intens_input.shape)
    imagesize = intens_input.shape[-1]
    ind = np.arange(imagesize) - imagesize//2
    x, y = np.meshgrid(ind, ind, indexing='ij')
    intrad_2d = np.sqrt(x**2 + y**2)
    intens_input[:, intrad_2d>imagesize//2] = 0.0
    return intens_input

def sample_down_intens(intens_input):
    '''Scale down 2D average intensity  size'''
    num_l = intens_input.shape[0]
    size = intens_input.shape[1]//2
    x_orig = np.arange(-size, size+1, 1.) / size
    y_orig = np.arange(-size, size+1, 1.) / size
    lmax =50 
    step_pix = 1.33
    x_new = np.arange(-lmax, lmax+1, 1) / size * step_pix
    y_new = np.arange(-lmax, lmax+1, 1) / size * step_pix
    intens_input_c = np.zeros([num_l, 2*lmax+1, 2*lmax+1])
    for i in range(num_l):
        input_map = intens_input[i]
        interpf = interpolate.interp2d(x_orig, y_orig, input_map, kind='cubic')
        intens_input_c[i] = interpf(x_new,y_new)
    intens_input_c = mask_circle(intens_input_c)
    return intens_input_c

def split_data(input_intens, orientation, z1_pca, z2_pca, data_points, split_ratio):
    '''Splitting the data into training and validation dataset'''
    t_intens = input_intens[:int(data_points*split_ratio),:,:]
    t_ori = orientation[:int(data_points*split_ratio),:]
    t_z1_pca = z1_pca[:int(data_points*split_ratio)]
    t_z2_pca = z2_pca[:int(data_points*split_ratio)]

    v_intens = input_intens[int(data_points*split_ratio):,:,:]
    v_ori = orientation[int(data_points*split_ratio):,:]
    v_z1_pca = z1_pca[int(data_points*split_ratio):]
    v_z2_pca = z2_pca[int(data_points*split_ratio):]
    print('Total Training Datapoints:', len(t_ori))
    print('Total Validation Datapoints:', len(v_ori))
    return t_intens, t_ori, t_z1_pca, t_z2_pca, v_intens, v_ori, v_z1_pca, v_z2_pca


def sample_down_plane(input_plane):
    '''Scaled down sliced plane'''
    size= 503//2
    x_orig = np.arange(-size, size+1, 1) / size
    y_orig = np.arange(-size, size+1, 1) / size
    interpf = interpolate.interp2d(x_orig, y_orig, input_plane, kind='cubic')
    lmax = 50
    frac_vox = 1
    x_new = np.arange(-lmax, lmax+1, 1) / lmax * frac_vox
    y_new = np.arange(-lmax, lmax+1, 1) / lmax * frac_vox
    return interpf(x_new, y_new)
    

def get_detector():
    ''''Get Detector Pixel Coordinates'''
    with h5py.File('/home/mallabhi/SPIEncoder/data/det_vae.h5', 'r') as f:
        qx1 = f['qx'][:].reshape(503, 503)
        qy1 = f['qy'][:].reshape(503, 503)
        qz1 = f['qz'][:].reshape(503, 503)
  
    '''Normalizing Detector qx, qy and qz coordinates'''
    factor = np.sqrt(qx1**2 + qy1**2 + qz1**2).max()
    qx_d = qx1/factor
    qy_d = qy1/factor
    qz_d= qz1/factor
    qx_ds = sample_down_plane(qx_d)
    qy_ds = sample_down_plane(qy_d)
    qz_ds = sample_down_plane(qz_d)
    return qx_ds, qy_ds, qz_ds



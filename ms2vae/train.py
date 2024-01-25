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
from auxiliary_functions import *
#####from estimate_best_orientation import LocalCCOrinetationOptimizer as LCC
from estimate_best_orientation_quat import LocalCCOrinetationOptimizer as LCC

import os
import sys
import h5py
import time
from random import randint
from sklearn.utils import shuffle 

import configparser
import logging


config = configparser.ConfigParser()
config.read('config.ini')


NUM_SAMPLES = int(config['Hyperparameters']['NUM_SAMPLES'])
BETA = float(config['Hyperparameters']['BETA'])
LR = float(config['Hyperparameters']['LR'])
WEIGHT_DECAY = float(config['Hyperparameters']['WEIGHT_DECAY'])
BATCH_SIZE = int(config['Hyperparameters']['BATCH_SIZE'])
N_EPOCHS = int(config['Hyperparameters']['N_EPOCHS'])
LATENT_DIMS = int(config['Hyperparameters']['LATENT_DIMS'])
LOAD_MODEL = config['Hyperparameters'].getboolean('LOAD_MODEL')
SAVE_AT_EPOCH = int(config['Hyperparameters']['SAVE_AT_EPOCH'])
ICO_SYMMETRIZATION = config['Hyperparameters'].getboolean('ICO_SYMMETRIZATION')
OPTIMIZE_ORIENTATIONS = config['Hyperparameters'].getboolean('OPTIMIZE_ORIENTATIONS')
SIGMA_KERNEL = float(config['Hyperparameters']['SIGMA_KERNEL'])

DATA_FILE = config['Files']['DATA_FILE']
ORIENTATION_FILE = config['Files']['ORIENTATION_FILE']
DETECTOR_FILE = config['Files']['DETECTOR_FILE']
START_MODEL_FILE = config['Files']['START_MODEL_FILE']
OUTPUT_FILE = config['Files']['OUTPUT_FILE']
VAE_MODEL_FILE = config['Files']['VAE_MODEL_FILE']

LOG_FILENAME = config['Files']['LOG_FILE']
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Hyperparameters:")
for key, value in config['Hyperparameters'].items():
    logging.info(f"{key}: {value}")

logging.info("\nFile paths:")
for key, value in config['Files'].items():
    logging.info(f"{key}: {value}")


'''Loading Data and Detector File'''
input_intens, orientation = load_data(DATA_FILE, ORIENTATION_FILE, NUM_SAMPLES)
qx_d, qy_d, qz_d = get_detector(DETECTOR_FILE)

DATA_POINTS = len(orientation)

print('Total Dataset Points:', DATA_POINTS)
print('Total # of Training Epochs:', N_EPOCHS)
print('Batch Size:', BATCH_SIZE)
print('Latent Dims:', LATENT_DIMS)

model = VAE(LATENT_DIMS)
'''Choose Device to Train on'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if LOAD_MODEL:
    model.load_state_dict(torch.load(START_MODEL_FILE, map_location=device))
    print('Model loaded')
else:
    torch.manual_seed(42)

'''Optimizaer for NN'''
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
model.to(device)
print('# of devices:', torch.cuda.device_count())
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
sys.stdout.flush()


slices_s = slice_planes(orientation, input_intens.shape[-1], DATA_POINTS, qx_d, qy_d, qz_d, device)
orient_optimizer = LCC(SIGMA_KERNEL, 256, qx_d, qy_d, qz_d)


def trainNN(epoch, input_intens, orientation, beta):
    epochloss = 0
    bseloss = 0
    kldloss=0
    true_intens = np.zeros((len(orientation), input_intens.shape[1], input_intens.shape[1]))
    recon_vol = np.zeros((len(orientation), 267,267,267))
    mu_ = np.zeros((len(orientation), LATENT_DIMS))
    logvar_ = np.zeros((len(orientation), LATENT_DIMS))
    pred_intens = np.zeros((len(orientation), input_intens.shape[1], input_intens.shape[1]))
    idx=0
    for i in range(len(orientation)//BATCH_SIZE):
        intens_batch = input_intens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        images = torch.from_numpy(intens_batch).view(BATCH_SIZE, 1, input_intens.shape[1], input_intens.shape[1])
        images = images.float().to(device)
        
        ori_batch = orientation[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        oris = torch.from_numpy(ori_batch).view(BATCH_SIZE, 4)
        oris = oris.float().to(device)
        
        output, z, mu, logvar = model.forward(images, oris)
        loss, bse, kld, recon_2D_x = loss_function(epoch, output, slices_s, images, i, mu, logvar, beta, BATCH_SIZE)
        
        bseloss += bse.data.item()
        kldloss += kld.data.item()
        epochloss += loss.data.item()
        if epoch % SAVE_AT_EPOCH == 0:
            true_intens[idx:idx+BATCH_SIZE,:,:] = images.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])
            pred_intens[idx:idx+BATCH_SIZE,:,:] = recon_2D_x.detach().cpu().clone().numpy().reshape(BATCH_SIZE, input_intens.shape[1], input_intens.shape[1])
            recon_vol[idx:idx+BATCH_SIZE, :,:,:] = output.detach().cpu().clone().numpy().reshape(BATCH_SIZE, 267,267,267)
            mu_[idx:idx+BATCH_SIZE,:] = mu.detach().cpu().numpy()
            logvar_[idx:idx+BATCH_SIZE,:] = logvar.detach().cpu().numpy()
            idx += BATCH_SIZE


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return true_intens, pred_intens, recon_vol, mu_, logvar_, epochloss/len(orientation), bseloss/len(orientation), kldloss/len(orientation)


quats = ne.math.sym_group_quat(sym_type='IL', qs=quat.quaternion(2**-0.5, 0, 2**-0.5, 0))
quaternions = quat.as_float_array(quats)
def _get_orient(orientation, quaternions):
    orient_n = np.copy(orientation)
    pick = np.arange(0, 60, 1)
    for j in range(len(orientation)):
        q_ind = np.random.choice(pick)
        orient_n[j] = quaternion_multiply(orientation[j], quaternions[q_ind])
    return orient_n

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


training_loss = []
bse_loss = []
kld_loss = []
stime = time.time()

for epoch in np.arange(N_EPOCHS)+1:
    if ICO_SYMMETRIZATION:
        orientation_n = _get_orient(orientation, quaternions)
    else:
        orientation_n = orientation

    true_intens, pred_intens, recon_vol, mu, logvar, loss, bseloss, kldloss = trainNN(epoch, input_intens, orientation_n, BETA)

    if epoch % SAVE_AT_EPOCH == 0:
        if OPTIMIZE_ORIENTATIONS:
            for q in range(len(orientation_n)):
                q0s_oris = quat.quaternion(*orientation_n[q])
                updated_orientation = orient_optimizer(pred_intens[q], recon_vol[q], q0s_oris)
                updated_orientation  = np.array([updated_orientation.w, updated_orientation.x, updated_orientation.y, updated_orientation.z])
                orientation_n[q] = updated_orientation


    training_loss.append(loss)
    bse_loss.append(bseloss)
    kld_loss.append(kldloss)

    sys.stderr.write('\rEPOCH %d/%d: '%(epoch, N_EPOCHS))
    sys.stderr.write('Training loss: %e, '%loss)
    sys.stderr.write('SE loss: %e, '%bseloss)
    sys.stderr.write('KLD loss: %e, '%kldloss)
    sys.stderr.write('%.3f s/iteration   ' % ((time.time() - stime) / (epoch+1)))
    sys.stderr.flush()

    if epoch % SAVE_AT_EPOCH == 0:
        MODEL_FNAME = f'{VAE_MODEL_FILE}_epoch{epoch}'
        torch.save(model.module.state_dict(), MODEL_FNAME)
        OUTPUT_FNAME = f'{OUTPUT_FILE}_epoch{epoch}.h5'
        with h5py.File(OUTPUT_FNAME, "w") as f:
                        f['pred_intens'] = pred_intens
                        f['loss'] = training_loss
                        f['bseloss'] = bse_loss
                        f['kldloss'] = kld_loss
                        f['mu'] = mu
                        f['logvar'] = logvar
                        f['coors'] = orientation_n

sys.stderr.write('\n')
sys.stderr.flush()

print('Training & Validation Done, Data Saved') 

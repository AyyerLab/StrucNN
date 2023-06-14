import numpy as np
import h5py
from scipy import optimize
from scipy import ndimage

import sys
import os

def sphere(qvals, a, dia):
    s = np.pi*qvals*dia
    s[s<1e-6] = 1e-5
    return ndimage.gaussian_filter1d(a * (dia**3) * ((np.sin(s) - s*np.cos(s)) / s**3)**2, 0.9)

qvals = (726/526)*np.arange(18,68)*6/(1.23984*2000)
diams = np.arange(1, 100, 0.1)
msphere = np.array([sphere(qvals, 1, d) for d in diams])


with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/radavg_volumes_l2_01.h5', 'r') as f:
    radavg_vol = f['radavg_vol'][:]
    vol = f['vol'][:]

print('Data Loaded')
sys.stdout.flush()


popts = []
fits = []
radavgs = []
for i in range(len(vol)):
    cc = np.corrcoef(radavg_vol[0][18:68], msphere)[0,1:]
    popt, pcov = optimize.curve_fit(sphere, qvals,radavg_vol[i][18:68], p0=(3e4, diams[cc.argmax()]), maxfev=20000)
    popts.append(popt)
    fits.append(sphere(qvals, popt[0], popt[1]))
    radavgs.append(radavg_vol[i][18:68])
    sys.stderr.write('\rVolume Fitted %d/%d: '%(i, len(vol)))
sys.stderr.write('\n')
sys.stdout.flush()


radavgs = np.array(radavgs)
fits = np.array(fits)
popts = np.array(popts)


with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/fit_volumes_l2_01.h5', 'w') as f:
    f['popts'] = popts
    f['fits'] = fits
    f['radavgs'] = radavgs
print('Data saved')
sys.stdout.flush()


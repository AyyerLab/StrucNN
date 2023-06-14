import numpy as np
from scipy import ndimage
from scipy import optimize
import sys
import os
import h5py


def radavg_vol(vol):
    center = np.array(vol.shape) / 2
    indices = np.indices(vol.shape)
    distances = np.sqrt(np.sum((indices - center.reshape(-1, 1, 1, 1))**2, axis=0))
    max_distance = int(np.ceil(distances.max()))
    bins = np.arange(0, max_distance + 1)
    bin_indices = np.digitize(distances.flat, bins)
    radial_sum = ndimage.sum(vol.flat, bin_indices, index=bins)
    radial_count = ndimage.sum(np.ones_like(vol).flat, bin_indices, index=bins)
    radial_average = radial_sum / radial_count
    return radial_average, bins


with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/volumes_vae_l2_01_a.h5', 'r') as f:
    a_recon_vol_bgsbt = f['recon_vol_bgsbt'][:]

with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/volumes_vae_l2_01_b.h5', 'r') as f:
    b_recon_vol_bgsbt = f['recon_vol_bgsbt'][:]

with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/volumes_vae_l2_01_c.h5', 'r') as f:
    c_recon_vol_bgsbt = f['recon_vol_bgsbt'][:]

with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/volumes_vae_l2_01_d.h5', 'r') as f:
    d_recon_vol_bgsbt = f['recon_vol_bgsbt'][:]

volumes = np.concatenate([a_recon_vol_bgsbt, b_recon_vol_bgsbt,c_recon_vol_bgsbt, d_recon_vol_bgsbt], axis=0)
print('Volume Loaded')
print('Volume Shape:', volumes.shape)
sys.stdout.flush()

np.seterr(divide='ignore', invalid='ignore')
all_radavg_vol = []
for p in range(len(volumes)):
    vol_radavg, vol_rad = radavg_vol(volumes[p])
    sys.stderr.write('\rRadavg %d/%d: '%(p, len(volumes)))
    all_radavg_vol.append(vol_radavg)
all_radavg_vol = np.array(all_radavg_vol)
sys.stderr.write('\n')
print('Radavg Calculated')
sys.stdout.flush()


with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/radavg_volumes_l2_01.h5', 'w') as f:
    f['radavg_vol'] = all_radavg_vol
    f['vol'] = volumes

print('Data Saved')
sys.stdout.flush()


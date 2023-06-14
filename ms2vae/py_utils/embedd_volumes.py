import numpy as np
import h5py
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import sys
import os


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
sys.stdout.flush()

X = volumes.reshape(volumes.shape[0], -1)
##--MDS
embedding = MDS(n_components=2, n_init=10, random_state=1)
em = embedding.fit_transform(X)

print('MDS done')
sys.stdout.flush()

with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/embeding_volumes_l2_01.h5', 'w') as f:
    f['embed_comps'] = em
print('Data saved')
sys.stdout.flush()




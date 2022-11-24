'''Functions to load and process data before training'''
import numpy as np
import h5py
from scipy import interpolate

PREFIX = '/home/mallabhi/StrucNN/data/'

def load_data(nintens=None):
    '''Load object's size, 2D Intensity avgs. and corresponding orientations'''
    with h5py.File(PREFIX + 'MS2_10k_subset.h5', 'r') as fptr:
        intens = fptr['intens'][:nintens]
        # Load z1 and z2 pca components of cc-matrix(calculated for 2d intens)
        z1_pca = fptr['z1_pca'][:nintens]
        z2_pca = fptr['z2_pca'][:nintens]
    orientation = np.load(PREFIX + 'orient_10k_subet.npy')[:nintens] # sic
    orientation[:,1:] *= -1.

    # Calculate pixel radii
    ind = np.arange(intens.shape[-1]) - intens.shape[-1]//2
    x, y = np.meshgrid(ind, ind, indexing='ij')
    rad = np.sqrt(x**2 + y**2)
    rad[rad==0] = 1e-4
    # Hard coded function to get flat radial profile
    ave_2d = 1.36e8 * rad**-3.56 + 6.89

    # Normalizing Intensity vals to range: 0-1
    norm_intens = intens / ave_2d
    norm_intens *= 0.99 / norm_intens.max()

    # Resample input intensities
    input_intens = sample_down_intens(norm_intens)

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

def split_data(input_intens, orientation, z1_pca, z2_pca, split_ratio):
    '''Splitting the data into training and validation dataset'''
    n_train = int(len(input_intens) * split_ratio)
    print('Total Training Datapoints:', n_train)
    print('Total Validation Datapoints:', len(input_intens) - n_train)
    train_dict = {'input_intens': input_intens[:n_train],
                  'orientation': orientation[:n_train],
                  'z1_pca': z1_pca[:n_train],
                  'z2_pca': z2_pca[:n_train],
                  'train': True,
                  'save': False}
    valid_dict = {'input_intens': input_intens[n_train:],
                  'orientation': orientation[n_train:],
                  'z1_pca': z1_pca[n_train:],
                  'z2_pca': z2_pca[n_train:],
                  'train': False,
                  'save': False}
    return train_dict, valid_dict

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
    with h5py.File(PREFIX + 'det_vae.h5', 'r') as fptr:
        qx1 = fptr['qx'][:].reshape(503, 503)
        qy1 = fptr['qy'][:].reshape(503, 503)
        qz1 = fptr['qz'][:].reshape(503, 503)

    # Normalizing Detector qx, qy and qz coordinates
    factor = np.sqrt(qx1**2 + qy1**2 + qz1**2).max()
    qx_d = qx1/factor
    qy_d = qy1/factor
    qz_d= qz1/factor
    qx_ds = sample_down_plane(qx_d)
    qy_ds = sample_down_plane(qy_d)
    qz_ds = sample_down_plane(qz_d)
    return qx_ds, qy_ds, qz_ds

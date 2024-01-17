import numpy as np
import torch
import torch.nn.functional as F
import neoemc as ne
import quaternion as quat


def _q_rotation(x, y, z, i, qw, qx, qy, qz):
    '''Get Rotation Matrix from quaternions.
        Input :x, y, z --> detector pixel coordinates
              qw, qx, qy, qz --> quaternions'''
    matrx = [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]]
    return np.transpose(np.transpose(np.array([x, y, z]), (1,2,0))@matrx, (2,0,1))


def slice_planes(orien, imgsize, n_slices, qx_d, qy_d, qz_d, device):
    '''Get slicing planes for corresponding rotation matrices'''
    slices_s = np.zeros((n_slices, imgsize, imgsize, 3))
    for i in range(n_slices):
        q0, q1, q2, q3 = orien[i]
        slices_s[i] = _q_rotation(qx_d, qy_d, qz_d, i, q0, q1, q2, q3).T
    return torch.from_numpy(slices_s).to(device)


def best_projection_slice(recon_intens_3d, slices, ind):
    '''Slicing the reconstructed volume'''
    select_plane =  slices[ind]
    size = recon_intens_3d.shape[2]
    grid = select_plane.float()
    return F.grid_sample(recon_intens_3d.view(1, 1, size, size, size), 
            grid.view(1, grid.shape[0], grid.shape[0], 1, 3), mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True)[0][0][:,:].reshape(grid.shape[0], grid.shape[0])


def loss_function(epoch, recon_intens_3d, slices, images, bnum, mu, logvar, beta, BATCH_SIZE):
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



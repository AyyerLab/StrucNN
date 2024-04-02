import time
from pathlib import Path

import cupy as cp
import emcfile as ef
import h5py
import matplotlib.pylab as plt
import neoemc as ne
import neoemc.cuda as nc
import numpy as np
import quaternion as quat

def cc(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    cc = a @ b.T
    cc /= (a**2).sum(axis=1, keepdims=True)
    cc /= (b**2).sum(axis=1)
    return cc


class LocalCCOrinetationOptimizer:
    def __init__(
        self,
        sigma: float,
        num_quat: int,
        det_qx,
        det_qy,
        det_qz,
        q_min=10,
        q_max=110,
        volume_size=267,
        apply_ico=False,
    ):
        self.sigma = sigma
        self.num_quat = num_quat
        self.q_min = q_min
        self.q_max = q_max
        self.kernel = self.init_kernel(apply_ico)
        det = self.init_det(
            volume_size / 2 * np.array([det_qx, det_qy, det_qz]).T.reshape(-1, 3)
        )
        self._pixel_mask = det.mask == ef.PixelType.GOOD
        self._det = det[self._pixel_mask]

    def init_kernel(self, apply_ico):
        k = ne.rand_normal_quatws(self.num_quat, self.sigma)
        k.coorws_[0, :4] = [1, 0, 0, 0.0]
        if apply_ico:
            k = k @ ne.quat_distribution(
                ne.math.sym_group_quat(
                    sym_type="IL", qs=quat.quaternion(2**-0.5, 0, 2**-0.5, 0)
                ),
                weight=1,
            )
        return k

    def init_det(self, coor):
        mask = np.zeros(coor.shape[0], dtype=int)
        r = np.linalg.norm(coor, axis=1)
        print(coor.shape, r.min(), r.max())
        mask[r < self.q_min] = 2
        mask[r > self.q_max] = 2
        factor = np.ones(coor.shape[0])
        det = ef.detector(
            coor=coor,
            mask=mask,
            factor=factor,
            detd=-1,
            ewald_rad=-1,
            check_consistency=False,
        )
        return det

    def __call__(
        self, pattern, vol: cp.ndarray, q0: quat.quaternion
    ) -> quat.quaternion:
        latent = self.kernel @ q0
        t0 = time.time()
        expander = nc.expander(vol, latent, self._det)
        p_d = cp.array(pattern.ravel()[self._pixel_mask])
        slices = expander[:, :]
        c = cc(p_d.reshape(1, -1), slices.T)
        return latent.quats[np.argmax(c.get())]

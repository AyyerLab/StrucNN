from pathlib import Path
import sys
import numpy as np
import emcfile as ef
import neoemc as ne
import neoemc.cuda as nc
import neoemc.simulation as sim
import os
import logging
from rich.logging import RichHandler
logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[RichHandler()])

cfg_path = Path(sys.argv[1])
cfg = sim.read_config(cfg_path)

logging.info(cfg['make_densities']['in_pdb_file'])

os.system(f"ne make_detector -c {cfg_path} --overwrite")
os.system(f"ne fetch_pdb -c {cfg_path}")
os.system(f"ne make_densities -c {cfg_path} --overwrite")
os.system(f"ne make_intensities -c {cfg_path} --overwrite")

det = ef.detector(cfg["make_detector"]["out_detector_file"])
inten = ef.read_array(cfg["make_intensities"]["out_intensity_file"]).astype("f4")

latent = ne.rand_quatws(10000)
logging.info(latent)
expander = nc.expander(inten, latent, det=det)
slices = np.concatenate(
    [expander[:, i : i + 1024].get() for i in range(0, expander.shape[1], 1024)], axis=1
)
slices[det.mask != ef.PixelType.GOOD] = 0

logging.info("start writing")
latent.write(cfg["make_slices"]["out_orientation_file"], overwrite=True)
detsize = cfg["parameters"]["detsize"]
ef.write_array(cfg["make_slices"]["out_slice_file"], slices.T.reshape(-1, detsize, detsize), overwrite=True)
# ef.write_array("data/orien_slice.h5::size", np.ones(len(latent), 'f4'), overwrite=True)

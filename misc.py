import numpy as np
import matplotlib.pyplot as plt
import argparse

from omegaconf import OmegaConf
from matplotlib import cm
from tqdm import tqdm


def configurator(config_base=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--configs', type=str, nargs='*')
    parser.add_argument('--params', type=str, nargs='*')

    args = parser.parse_args()

    cfg = OmegaConf.create()
    if args.configs is None and config_base is not None:
        cfg_config = OmegaConf.load(config_base)
        cfg = OmegaConf.merge(cfg, cfg_config)
    else:
        for config in args.configs:
            cfg_config = OmegaConf.load(config)
            cfg = OmegaConf.merge(cfg, cfg_config)

    if args.params is not None:
        params = []
        # for param in args.params:
        #     # params.append(OmegaConf.decode)
        cfg_params = OmegaConf.from_dotlist(args.params)
        cfg = OmegaConf.merge(cfg, cfg_params)

    return cfg


def color_depthmap(grey, maxval=None, minval=None):

    if minval is None:
        minval = np.amin(grey)
    if maxval is None:
        maxval = np.amax(grey)

    grey -= minval
    grey[grey < 0] = 0
    grey /= maxval

    rgb = cm.get_cmap(plt.get_cmap('jet'))(grey)[:, :, :3]

    return rgb
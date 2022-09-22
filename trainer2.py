import numpy as np
import torch
import torchmetrics
import time
import math as m
import matplotlib.pyplot as plt

from tqdm import tqdm


class NeRFTrainer(object):
    def __init__(
        self,
        models,
        dataloader,
        logger,
        optimizer,
        scheduler,
    ):

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import time

from tqdm import tqdm

from nerf_benchmark.nets import NeRFNetwork


class Trainer(object):
    def __init__(self):
        self.max_iters = 10000

        self.n_rays = 16384
        self.n_samples = 128

        self.model = NeRFNetwork().to("cuda")

        # print(self.model.parameters())
        # for i in self.model.parameters():
        #     print(i.shape)
        # exit()

        self.optimizer = torch.optim.Adam([
            # {'name': 'encoding', 'params': list(self.model.encoder.parameters()), 'lr': 2e-2},
            # {'name': 'net', 'params': list(self.model.network.parameters()), 'lr': 1e-3},
            {'name': 'net', 'params': list(self.model.parameters()), 'lr': 1e-3},
            ])

        self.scaler = torch.cuda.amp.GradScaler()

        self.iter = 0

    def train(self):
        self.t0 = time.time()

        x = torch.rand((self.n_rays, 3), device="cuda")
        y = torch.rand((self.n_rays, 4), device="cuda")
        z = torch.rand((self.n_samples), device="cuda")

        for i in tqdm(range(self.max_iters)):
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = self.train_step(x, y, z)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.iter += 1

    @torch.no_grad()
    def inference(self):
        self.t0 = time.time()

        x = torch.rand((self.n_rays, 3), device="cuda")
        y = torch.rand((self.n_rays, 4), device="cuda")
        z = torch.rand((self.n_samples), device="cuda")

        for i in tqdm(range(self.max_iters)):
            with torch.cuda.amp.autocast():
                loss = self.train_step(x, y, z)

            self.iter += 1

    def train_step(self, x, y, z):

        xz = x[:, None, :] * z[None, :, None]
        xzf = xz.reshape(-1, 3)

        yzf_ = self.model(xzf)
        yz_ = yzf_.reshape(self.n_rays, self.n_samples, 4)

        y_ = torch.mean(yz_, dim=1)

        loss = F.huber_loss(y_, y, delta=0.1)

        return loss


if __name__ == "__main__":
    trainer = Trainer()
    # trainer.train()
    trainer.inference()
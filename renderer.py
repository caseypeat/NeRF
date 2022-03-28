import torch
import torch.nn as nn
import torch.nn.functional as F

from raymarching import raymarching

def near_far_from_bound(rays_o, rays_d, bound, type='cube'):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=0.05)

    return near, far

class NerfRenderer(nn.Module):
    def __init__(self):
        super().__init__()

        # density grid
        density_grid_inner = torch.zeros([128] * 3)
        self.register_buffer('density_grid_inner', density_grid_inner)
        self.mean_density_inner = 0

        # self.mask_inner = torch.zeros([256] * 3, dtype=bool)
        # dist = torch.linspace(-1, 1, )
        # for i in range(256):
        #     for j in range(256):
        #         for k in range(256):
        #             if 

        density_grid_outer = torch.zeros([128] * 3)
        self.register_buffer('density_grid_outer', density_grid_outer)
        self.mean_density_outer = 0

        self.iter_density = 0
        # step counter
        step_counter = torch.zeros(64, 2, dtype=torch.int32) # 64 is hardcoded for averaging...
        self.register_buffer('step_counter', step_counter)
        self.mean_count = 0
        self.local_step = 0

    def forward(self, x, d, bound):
        raise NotImplementedError()

    def density(self, x, d, bound):
        raise NotImplementedError()

    # def reset_extra_state(self):
    #     # density grid
    #     self.density_grid.zero_()
    #     self.mean_density = 0
    #     self.iter_density = 0
    #     # step counter
    #     self.step_counter.zero_()
    #     self.mean_count = 0
    #     self.local_step = 0


    def render(self, rays_o, rays_d, bound, bg_color, perturb, force_all_rays):

        if self.training:
            counter = self.step_counter[self.local_step % 64]
            counter.zero_() # set to 0
            self.local_step += 1

        xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid_inner, self.mean_density_inner, self.density_grid_outer, self.mean_density_outer, self.iter_density, None, self.mean_count, perturb, 128, force_all_rays)
        # print(xyzs.shape)
        sigmas, rgbs = self(xyzs, dirs, bound)
        weights_sum, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, bound)

        # composite bg (shade_kernel_nerf)
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        depth = image[..., 0]  # placeholder

        image = image[None, ...]
        depth = depth[None, ...]

        return image, depth

    def update_extra_state(self, bound, decay=0.95):
        
        # call before each epoch to update extra states.
        
        ### update density grid

        ## Inner
        resolution_inner = self.density_grid_inner.shape[0]

        half_grid_size_inner = 1 / resolution_inner
        
        X = torch.linspace(-1 + half_grid_size_inner, 1 - half_grid_size_inner, resolution_inner).split(128)
        Y = torch.linspace(-1 + half_grid_size_inner, 1 - half_grid_size_inner, resolution_inner).split(128)
        Z = torch.linspace(-1 + half_grid_size_inner, 1 - half_grid_size_inner, resolution_inner).split(128)

        tmp_grid = torch.zeros_like(self.density_grid_inner)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        lx, ly, lz = len(xs), len(ys), len(zs)
                        # construct points
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                        # add noise in [-hgs, hgs]
                        pts += (torch.rand_like(pts) * 2 - 1) * half_grid_size_inner
                        # manual padding for ffmlp
                        n = pts.shape[0]
                        pad_n = 128 - (n % 128)
                        if pad_n != 0:
                            pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
                        # query density
                        density_inner = self.density(pts.to(tmp_grid.device), bound)[:n].reshape(lx, ly, lz).detach()
                        tmp_grid[xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density_inner

        d_inner = torch.linspace(-1 + half_grid_size_inner, 1 - half_grid_size_inner, resolution_inner)
        D_inner = torch.stack(torch.meshgrid(d_inner, d_inner, d_inner), dim=-1)
        dist_inner = torch.linalg.norm(D_inner, dim=-1)
        mask_inner = torch.zeros(dist_inner.shape, dtype=bool)
        mask_inner[dist_inner < 1] = True
        
        # ema update
        self.density_grid_inner = torch.maximum(self.density_grid_inner * decay, tmp_grid)
        self.mean_density_inner = torch.mean(self.density_grid_inner[mask_inner]).item()


        # ## Outer
        # resolution_outer = self.density_grid_outer.shape[0]

        # half_grid_size_outer = 1 / resolution_outer
        
        # X = torch.linspace(-2 + half_grid_size_outer, 2 - half_grid_size_outer, resolution_outer).split(128)
        # Y = torch.linspace(-2 + half_grid_size_outer, 2 - half_grid_size_outer, resolution_outer).split(128)
        # Z = torch.linspace(-2 + half_grid_size_outer, 2 - half_grid_size_outer, resolution_outer).split(128)

        # tmp_grid = torch.zeros_like(self.density_grid_outer)
        # with torch.no_grad():
        #     for xi, xs in enumerate(X):
        #         for yi, ys in enumerate(Y):
        #             for zi, zs in enumerate(Z):
        #                 lx, ly, lz = len(xs), len(ys), len(zs)
        #                 # construct points
        #                 xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
        #                 pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
        #                 # add noise in [-hgs, hgs]
        #                 pts += (torch.rand_like(pts) * 2 - 1) * half_grid_size_outer
        #                 # manual padding for ffmlp
        #                 n = pts.shape[0]
        #                 pad_n = 128 - (n % 128)
        #                 if pad_n != 0:
        #                     pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
        #                 # query density
        #                 density_outer = self.density(pts.to(tmp_grid.device), bound)[:n].reshape(lx, ly, lz).detach()
        #                 tmp_grid[xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density_outer

        # d_outer = torch.linspace(-2 + half_grid_size_outer, 2 - half_grid_size_outer, resolution_outer)
        # D_outer = torch.stack(torch.meshgrid(d_outer, d_outer, d_outer), dim=-1)
        # dist_outer = torch.linalg.norm(D_outer, dim=-1)
        # mask_outer = torch.zeros(dist_outer.shape, dtype=bool)
        # mask_outer[((dist_outer > 1) & (dist_outer < 2))] = True
        
        # # ema update
        # self.density_grid_outer = torch.maximum(self.density_grid_outer * decay, tmp_grid)
        # self.mean_density_outer = torch.mean(self.density_grid_outer[mask_outer]).item()




        self.iter_density += 1

        ### update step counter
        total_step = min(64, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0


    # def update_extra_state(self, bound, decay=0.95):
    #     # call before each epoch to update extra states.
        
    #     ### update density grid
    #     resolution = self.density_grid.shape[0]

    #     half_grid_size = bound / resolution
        
    #     X = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)
    #     Y = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)
    #     Z = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)

    #     tmp_grid = torch.zeros_like(self.density_grid)
    #     with torch.no_grad():
    #         for xi, xs in enumerate(X):
    #             for yi, ys in enumerate(Y):
    #                 for zi, zs in enumerate(Z):
    #                     lx, ly, lz = len(xs), len(ys), len(zs)
    #                     # construct points
    #                     xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
    #                     pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
    #                     # add noise in [-hgs, hgs]
    #                     pts += (torch.rand_like(pts) * 2 - 1) * half_grid_size
    #                     # manual padding for ffmlp
    #                     n = pts.shape[0]
    #                     pad_n = 128 - (n % 128)
    #                     if pad_n != 0:
    #                         pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
    #                     # query density
    #                     density = self.density(pts.to(tmp_grid.device), bound)[:n].reshape(lx, ly, lz).detach()
    #                     tmp_grid[xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density
        
    #     # ema update
    #     self.density_grid = torch.maximum(self.density_grid * decay, tmp_grid)
    #     # print(self.density_grid[128:131, 128:131, 128:131])
    #     self.mean_density_inner = torch.mean(self.density_grid[64:192, 64:192, 64:192]).item()
    #     outer_idx = torch.cat([torch.arange(0, 64, dtype=self.density_grid.dtype), torch.arange(192, 256, dtype=self.density_grid.dtype)], dim=0).to(torch.long)
    #     oi_x, oi_y, oi_z = torch.meshgrid(outer_idx, outer_idx, outer_idx)
    #     self.mean_density_outer = torch.mean(self.density_grid[oi_x, oi_y, oi_z]).item()
    #     self.iter_density += 1

    #     ### update step counter
    #     total_step = min(64, self.local_step)
    #     if total_step > 0:
    #         self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
    #     self.local_step = 0
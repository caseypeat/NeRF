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
        density_grid = torch.zeros([256] * 3)
        self.register_buffer('density_grid', density_grid)
        self.mean_density = 0
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

    def reset_extra_state(self):
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    
    def render_eval(self, rays_o, rays_d, bound, bg_color, perturb):

        # counter = self.step_counter[self.local_step % 64]
        # counter.zero_() # set to 0
        # self.local_step += 1

        xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, None, self.mean_count, perturb, 128, True)
        # with torch.cuda.amp.autocast():
        sigmas, rgbs = self(xyzs, dirs, bound)
        weights_sum, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, bound)

        # composite bg (shade_kernel_nerf)
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        # depth = None # currently training do not requires depth
        depth = image[..., 0]  # placeholder

        image = image[None, ...]
        depth = depth[None, ...]

        return image, depth

    def render(self, rays_o, rays_d, bound, bg_color, perturb):

        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if bg_color is None:
            bg_color = 1

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 64]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, perturb, 128, False)
            # print(deltas)
            # print(xyzs.shape, dirs.shape, deltas.shape, rays.shape)
            # exit()
            # with torch.cuda.amp.autocast():
            sigmas, rgbs = self(xyzs, dirs, bound)
            weights_sum, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, bound)

            # composite bg (shade_kernel_nerf)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = None # currently training do not requires depth

        else:
            # xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, False, 128, True)
            # sigmas, rgbs = self(xyzs, dirs, bound=bound)
            # depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, bound)

            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(B * N, dtype=dtype, device=device)
            depth = torch.zeros(B * N, dtype=dtype, device=device)
            image = torch.zeros(B * N, 3, dtype=dtype, device=device)
            
            n_alive = B * N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device) # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            # pre-calculate near far
            near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')
            near = near.view(B * N)
            far = far.view(B * N)

            step = 0
            i = 0
            while step < 2048: # max step  # default value: 1024
                # count alive rays 
                if step == 0:
                    # init rays at first step.
                    torch.arange(n_alive, out=rays_alive[0])
                    rays_t[0] = near
                else:
                    alive_counter.zero_()
                    raymarching.compact_rays(n_alive, rays_alive[i % 2], rays_alive[(i + 1) % 2], rays_t[i % 2], rays_t[(i + 1) % 2], alive_counter)
                    n_alive = alive_counter.item() # must invoke D2H copy here
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(B * N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o, rays_d, bound, self.density_grid, self.mean_density, near, far, 128, perturb)
                sigmas, rgbs = self(xyzs, dirs, bound)
                raymarching.composite_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], sigmas, rgbs, deltas, weights_sum, depth, image)

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}')

                step += n_step
                i += 1

            # composite bg & rectify depth (shade_kernel_nerf)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - near, min=0) / (far - near)


        image = image.reshape(B, N, 3)
        if depth is not None:
            depth = depth.reshape(B, N)

        # return depth, image
        return image, depth

    def update_extra_state(self, bound, decay=0.95):
        # call before each epoch to update extra states.
        
        ### update density grid
        resolution = self.density_grid.shape[0]

        half_grid_size = bound / resolution
        
        X = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)
        Y = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)
        Z = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)

        tmp_grid = torch.zeros_like(self.density_grid)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        lx, ly, lz = len(xs), len(ys), len(zs)
                        # construct points
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                        # add noise in [-hgs, hgs]
                        pts += (torch.rand_like(pts) * 2 - 1) * half_grid_size
                        # manual padding for ffmlp
                        n = pts.shape[0]
                        pad_n = 128 - (n % 128)
                        if pad_n != 0:
                            pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
                        # query density
                        density = self.density(pts.to(tmp_grid.device), bound)[:n].reshape(lx, ly, lz).detach()
                        tmp_grid[xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density
        
        # ema update
        self.density_grid = torch.maximum(self.density_grid * decay, tmp_grid)
        self.mean_density = torch.mean(self.density_grid).item()
        self.iter_density += 1

        ### update step counter
        total_step = min(64, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0
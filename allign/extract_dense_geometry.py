import numpy as np
import torch
import open3d as o3d

from tqdm import tqdm

class ExtractDenseGeometry():
    def __init__(
        self,
        model,
        dataloader,
        mask_res,
        voxel_res,
        batch_size,
        thresh,
        outer_bound,
        ):

        self.model = model
        self.dataloader = dataloader
        self.mask_res = mask_res
        self.voxel_res = voxel_res
        self.batch_size = batch_size
        self.thresh = thresh
        self.outer_bound = outer_bound


    @torch.no_grad()
    def get_valid_positions(self, N, H, W, K, E, res):

        mask_full = torch.zeros((res, res, res), dtype=bool, device='cuda')

        for i in tqdm(range(res)):
            d = torch.linspace(-1, 1, res, device='cuda')
            D = torch.stack(torch.meshgrid(d[i], d, d, indexing='ij'), dim=-1)
            dist = torch.linalg.norm(D, dim=-1)[:, :, :, None].expand(-1, -1, -1, 3)
            mask = torch.zeros(dist.shape, dtype=bool, device='cuda')
            mask[dist < 1] = True

            # mask out parts outside camera coverage
            rays_d = D - E[:, None, None, :3, -1]
            dirs_ = torch.inverse(E[:, None, None, :3, :3]) @ rays_d[..., None]
            dirs_ = K[:, None, None, ...] @ dirs_
            dirs = dirs_ / dirs_[:, :, :, 2, None, :]
            mask_dirs = torch.zeros((N, res, res), dtype=int, device='cuda')
            mask_dirs[((dirs[:, :, :, 0, 0] > 0) & (dirs[:, :, :, 0, 0] < H) & (dirs[:, :, :, 1, 0] > 0) & (dirs[:, :, :, 1, 0] < W) & (dirs_[:, :, :, 2, 0] > 0))] = 1
            mask_dirs = torch.sum(mask_dirs, dim=0)
            mask_dirs[mask_dirs > 0] = 1
            mask_dirs = mask_dirs.to(bool)
            mask_dirs = mask_dirs[None, :, :, None].expand(-1, -1, -1, 3)
            mask = torch.logical_and(mask, mask_dirs)

            mask_full[i, :, :] = mask[..., 0]

        return mask_full


    @torch.no_grad()
    def extract_dense_geometry(self, N, H, W, K, E):

        mask = self.get_valid_positions(N, H, W, K, E, res=self.mask_res)

        voxels = torch.linspace(-1+1/self.voxel_res, 1-1/self.voxel_res, self.voxel_res, device='cpu')

        num_samples = self.voxel_res**3

        points = torch.zeros((0, 3), device='cpu')
        colors = torch.zeros((0, 3), device='cpu')

        for a in tqdm(range(0, num_samples, self.batch_size)):
            b = min(num_samples, a+self.batch_size)

            n = torch.arange(a, b)

            x = voxels[torch.div(n, self.voxel_res**2, rounding_mode='floor')]
            y = voxels[torch.div(n, self.voxel_res, rounding_mode='floor') % self.voxel_res]
            z = voxels[n % self.voxel_res]

            xyz = torch.stack((x, y, z), dim=-1).cuda()

            x_i = ((x+1)/2*mask.shape[0]).to(int)
            y_i = ((y+1)/2*mask.shape[1]).to(int)
            z_i = ((z+1)/2*mask.shape[2]).to(int)

            xyz = xyz[mask[x_i, y_i, z_i]].view(-1, 3)
            
            dirs = torch.Tensor(np.array([0, 0, 1]))[None, ...].expand(xyz.shape[0], 3).cuda()
            n_i = torch.zeros((xyz.shape[0]), dtype=int).cuda()

            if xyz.shape[0] > 0:
                sigmas, rgbs, _ = self.model(xyz, dirs, n_i, self.outer_bound)
                new_points = xyz[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3).cpu()
                points = torch.cat((points, new_points))
                new_colors = rgbs[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3).cpu()
                colors = torch.cat((colors, new_colors))

        pointcloud = {}
        pointcloud['points'] = points.numpy()
        pointcloud['colors'] = colors.numpy()

        return points, colors


    @torch.no_grad()
    def generate_dense_pointcloud(self):
        N, H, W, K, E = self.dataloader.get_calibration()
        points, _ = self.extract_dense_geometry(N, H, W, K, E)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
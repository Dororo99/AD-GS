import torch, os
import numpy as np
import torch.nn as nn
from utils.graphics_utils import theta_to_vector, vector_to_theta
from torch.nn.functional import grid_sample, normalize
from utils.system_utils import searchForMaxIteration
from scene.cameras import Camera
from utils.graphics_utils import fov2focal
import open3d as o3d

def get_image_cam_rays(
    focal,
    height,
    width,
    fy=None,
    cx=None,
    cy=None,
    pixel_center_offset=0.0,
    device='cuda',
):
    """Return pinhole rays, retaining the legacy centered-camera defaults."""
    fx = float(focal)
    fy = fx if fy is None else float(fy)
    cx = width / 2.0 if cx is None else float(cx)
    cy = height / 2.0 if cy is None else float(cy)
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, width, dtype=torch.float32, device=device),
        torch.arange(0, height, dtype=torch.float32, device=device),
        indexing='xy',
    ), dim=-1)  # H, W, 2
    pix_cam_ray = torch.stack(
        [
            (grid[..., 0] + pixel_center_offset - cx) / fx,
            (grid[..., 1] + pixel_center_offset - cy) / fy,
            torch.ones_like(grid[..., 0]),
        ],
        dim=-1,
    )
    pix_cam_ray = normalize(pix_cam_ray, p=2, dim=-1)
    return pix_cam_ray, grid


class EnvironmentMap:
    def __init__(self, resolution, num_channel=3, use_cache=True):
        self.resolution = resolution
        grid_map = (torch.rand((1, num_channel, resolution, resolution), dtype=torch.float32, device='cuda') * 2.0 - 1.0) * 1e-4
        self.grid_map = nn.Parameter(grid_map.requires_grad_(True))
        self.scale = torch.tensor([1.0 / torch.pi, 2.0 / torch.pi], dtype=torch.float32, device='cuda')

        self.optimizer = None
        self.image_cam_rays = dict()
        self.grid = dict()
        self.use_cache = use_cache
    
    def set_image_cam_ray(self, focal, height, width):
        self.image_cam_rays, self.grid = get_image_cam_rays(focal, height, width)

    def get_image_background(self, cam: Camera, use_cache=True, return_grid=False):
        use_cache = use_cache and self.use_cache
        if cam.intrinsics is None:
            ray_args = (
                fov2focal(cam.FoVx, cam.image_width),
                cam.image_height,
                cam.image_width,
            )
            ray_kwargs = {}
        else:
            ray_args = (cam.fx, cam.image_height, cam.image_width)
            ray_kwargs = {
                'fy': cam.fy,
                'cx': cam.cx,
                'cy': cam.cy,
                'pixel_center_offset': 0.5,
            }
        cache_key = (
            cam.cam_id,
            cam.image_height,
            cam.image_width,
            cam.fx,
            cam.fy,
            cam.cx,
            cam.cy,
        )
        if not use_cache:
            image_cam_rays, grid = get_image_cam_rays(*ray_args, **ray_kwargs)
        else:
            try:
                image_cam_rays = self.image_cam_rays[cache_key]
                grid = self.grid[cache_key]
            except KeyError:
                image_cam_rays, grid = get_image_cam_rays(*ray_args, **ray_kwargs)
                self.image_cam_rays[cache_key] = image_cam_rays  # H, W, 3
                self.grid[cache_key] = grid
        # image_cam_rays = (cam.world_view_transform[:3, :3].cuda().transpose(0, 1) @ image_cam_rays[..., None]).squeeze(-1)
        image_cam_rays = (cam.world_view_transform[:3, :3].cuda() @ image_cam_rays[..., None]).squeeze(-1)  # the matrix has already been rotated.

        background_image = self.get_env_color(image_cam_rays)

        if return_grid:
            return background_image, grid
        return background_image

    def get_env_color(self, view, input_angle=False):
        if not input_angle:
            view = normalize(view, p=2, dim=-1)
            angle = vector_to_theta(view) # H, W, 2
        else:
            angle = view  # H, W, 2
        angle = angle * self.scale
        rgb = grid_sample(self.grid_map, grid=angle[None, ...], align_corners=True)  # 1, C, H, W
        rgb = torch.sigmoid(rgb).squeeze(0)  # 3, H, W
        return rgb
    
    def training_setup(self, training_args):
        l = [
            {'params': [self.grid_map], 'lr': training_args.env_lr, "name": "env"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def save_weights(self, weights_path):
        torch.save(self.grid_map, weights_path)

    def load_weights(self, weights_path):
        grid_map = torch.load(weights_path, map_location='cuda')
        self.grid_map = nn.Parameter(grid_map.requires_grad_(True))

    def extract_env_map(self, path, num_pts=50_0000):
        pts = torch.cat([
            (torch.rand((num_pts, 1), device="cuda") * 2.0 - 1.0) * torch.pi,
            (torch.rand((num_pts, 1), device="cuda") * 2.0 - 1.0) * torch.pi / 2.0,
        ], dim=-1)
        rgb = self.get_env_color(pts[None], input_angle=True).squeeze(1).transpose(1, 0)
        pcd = o3d.geometry.PointCloud()
        pts = theta_to_vector(pts)
        pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        o3d.io.write_point_cloud(path, pcd)

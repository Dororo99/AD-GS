import torch

from scene.env import get_image_cam_rays
from utils.graphics_utils import (
    focal2fov,
    getProjectionMatrix,
    getProjectionMatrixFromIntrinsics,
)


def _rasterizer_pixel(matrix, point, width, height):
    point_h = torch.tensor([*point, 1.0], dtype=torch.float32)
    clip = matrix @ point_h
    ndc = clip[:2] / clip[3]
    return torch.stack(
        [
            ((ndc[0] + 1.0) * width - 1.0) * 0.5,
            ((ndc[1] + 1.0) * height - 1.0) * 0.5,
        ]
    )


def test_centered_intrinsics_projection_matches_legacy_fov_projection():
    width, height = 1920, 1280
    fx, fy = 2050.0, 1995.0
    legacy = getProjectionMatrix(
        0.01,
        100.0,
        focal2fov(fx, width),
        focal2fov(fy, height),
    )
    calibrated = getProjectionMatrixFromIntrinsics(
        0.01, 100.0, fx, fy, width / 2.0, height / 2.0, width, height
    )
    torch.testing.assert_close(calibrated, legacy, rtol=1e-6, atol=1e-7)


def test_off_center_projection_matches_splatad_pixel_convention():
    width, height = 1600, 820
    fx, fy, cx, cy = 1260.0, 1255.0, 827.25, 470.75
    projection = getProjectionMatrixFromIntrinsics(
        0.01, 100.0, fx, fy, cx, cy, width, height
    )
    point = (1.25, -0.4, 8.0)
    actual = _rasterizer_pixel(projection, point, width, height)
    expected = torch.tensor(
        [fx * point[0] / point[2] + cx - 0.5,
         fy * point[1] / point[2] + cy - 0.5]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-4)


def test_calibrated_rays_use_full_intrinsics_and_pixel_centers():
    fx, fy, cx, cy = 8.0, 10.0, 2.25, 1.75
    rays, grid = get_image_cam_rays(
        fx,
        height=3,
        width=4,
        fy=fy,
        cx=cx,
        cy=cy,
        pixel_center_offset=0.5,
        device='cpu',
    )
    expected = torch.tensor(
        [(3.0 + 0.5 - cx) / fx, (2.0 + 0.5 - cy) / fy, 1.0]
    )
    expected = expected / torch.linalg.vector_norm(expected)
    torch.testing.assert_close(rays[2, 3], expected)
    torch.testing.assert_close(grid[2, 3], torch.tensor([3.0, 2.0]))


def test_legacy_ray_call_retains_integer_centered_behavior():
    rays, _ = get_image_cam_rays(8.0, height=4, width=6, device='cpu')
    expected = torch.tensor([(0.0 - 3.0) / 8.0, (0.0 - 2.0) / 8.0, 1.0])
    expected = expected / torch.linalg.vector_norm(expected)
    torch.testing.assert_close(rays[0, 0], expected)

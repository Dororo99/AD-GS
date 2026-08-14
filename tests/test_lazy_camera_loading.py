from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import pytest

from scene.dataset_readers import CameraInfo
from scripts.prior_storage import (
    save_depth_prior,
    save_flow_prior,
    save_mask_prior,
)
from utils.camera_utils import camera_to_JSON, loadCam
from utils.graphics_utils import focal2fov
from scripts.validate_splatad_scene import _validate_flow_package


def _camera_info(root, image=None, eager=False):
    root = Path(root)
    width, height = 8, 6
    intrinsics = np.asarray([12.0, 10.0, 3.25, 2.75], dtype=np.float32)
    depth = np.linspace(0.0, 1.0, width * height, dtype=np.float32).reshape(
        height, width
    )
    semantic = np.zeros((height, width), dtype=np.int32)
    semantic[1:5, 2:7] = 3
    sky = np.zeros((height, width), dtype=np.bool_)
    sky[:2] = True
    flow_grid = np.stack(
        np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        ),
        axis=0,
    )
    flow_record = [
        15.0,
        np.eye(3, dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        flow_grid,
        np.ones((height, width), dtype=np.float32),
    ]

    image_path = root / "image.png"
    depth_path = root / "depth.npy"
    semantic_path = root / "semantic.npy"
    sky_path = root / "sky.npy"
    flow_path = root / "flow.npz"
    if not eager:
        rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(
            height, width, 3
        )
        Image.fromarray(rgb).save(str(image_path))
        save_depth_prior(depth_path.with_suffix(".npz"), depth)
        save_mask_prior(semantic_path.with_suffix(".npz"), semantic, "semantic")
        save_mask_prior(sky_path.with_suffix(".npz"), sky, "sky")
        save_flow_prior(flow_path, [flow_record])

    return CameraInfo(
        uid=7,
        cam_id=2,
        fid=12.0,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
        FovX=focal2fov(intrinsics[0], width),
        FovY=focal2fov(intrinsics[1], height),
        width=width,
        height=height,
        image_path=str(image_path),
        depth=depth if eager else None,
        image_name=image_path.name,
        image=image,
        time=0.2,
        semantic=semantic if eager else None,
        sky=sky if eager else None,
        flow=[[0.5] + flow_record[1:]] if eager else None,
        intrinsics=intrinsics,
        depth_path=None if eager else str(depth_path),
        semantic_path=None if eager else str(semantic_path),
        sky_path=None if eager else str(sky_path),
        flow_path=None if eager else str(flow_path),
        sensor_time_min=None if eager else 10.0,
        sensor_time_max=None if eager else 20.0,
    )


def _args(resolution=2):
    return SimpleNamespace(
        resolution=resolution,
        data_device="cpu",
        lazy_load_to_gpu=False,
    )


def test_lazy_compact_priors_are_loaded_one_camera_at_a_time(tmp_path):
    info = _camera_info(tmp_path)
    camera = loadCam(_args(), 0, info, 1.0)

    assert info.image is None
    assert info.depth is None
    assert info.semantic is None
    assert info.sky is None
    assert info.flow is None
    assert tuple(camera.original_image.shape) == (3, 3, 4)
    assert tuple(camera.depth.shape) == (3, 4)
    assert tuple(camera.semantic.shape) == (3, 4)
    assert tuple(camera.sky.shape) == (3, 4)
    assert camera.semantic.dtype == torch.bool
    assert camera.sky.dtype == torch.bool
    assert len(camera.flow) == 1
    assert camera.flow[0][0] == 0.5
    assert all(torch.is_tensor(value) for value in camera.flow[0][1:])
    assert camera.flow[0][5].dtype == torch.bool
    assert torch.all(camera.flow[0][5])
    np.testing.assert_allclose(
        camera.intrinsics,
        np.asarray([6.0, 5.0, 1.625, 1.375], dtype=np.float32),
    )


def test_lazy_image_file_handle_is_closed(tmp_path, monkeypatch):
    info = _camera_info(tmp_path)
    real_open = Image.open
    opened = []

    def tracked_open(*args, **kwargs):
        image = real_open(*args, **kwargs)
        opened.append(image)
        return image

    monkeypatch.setattr("utils.camera_utils.Image.open", tracked_open)
    loadCam(_args(), 0, info, 1.0)
    assert len(opened) == 1
    assert getattr(opened[0], "fp", None) is None


def test_eager_legacy_payload_keeps_existing_semantics(tmp_path):
    rgb = np.full((6, 8, 3), 127, dtype=np.uint8)
    info = _camera_info(tmp_path, image=Image.fromarray(rgb), eager=True)
    camera = loadCam(_args(), 0, info, 1.0)

    assert tuple(camera.original_image.shape) == (3, 3, 4)
    assert tuple(camera.depth.shape) == (3, 4)
    assert tuple(camera.semantic.shape) == (3, 4)
    assert tuple(camera.sky.shape) == (3, 4)
    assert camera.semantic.dtype == torch.bool
    assert camera.sky.dtype == torch.bool
    assert camera.flow[0][0] == 0.5


def test_lazy_legacy_npy_priors_use_the_same_resize_path(tmp_path):
    info = _camera_info(tmp_path)
    for compact_path in (
        Path(info.depth_path).with_suffix(".npz"),
        Path(info.semantic_path).with_suffix(".npz"),
        Path(info.sky_path).with_suffix(".npz"),
    ):
        compact_path.unlink()

    depth = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(6, 8, 1)
    semantic = np.arange(48, dtype=np.int32).reshape(6, 8)
    sky = np.arange(48).reshape(6, 8) % 2
    np.save(info.depth_path, depth)
    np.save(info.semantic_path, semantic)
    np.save(info.sky_path, sky)

    lazy_camera = loadCam(_args(), 0, info, 1.0)
    with Image.open(info.image_path) as image:
        eager_image = image.copy()
    eager_info = info._replace(
        image=eager_image,
        depth=depth.squeeze(-1),
        semantic=semantic,
        sky=sky != 0,
        depth_path=None,
        semantic_path=None,
        sky_path=None,
    )
    eager_camera = loadCam(_args(), 0, eager_info, 1.0)
    eager_image.close()

    torch.testing.assert_close(lazy_camera.depth, eager_camera.depth)
    torch.testing.assert_close(lazy_camera.semantic, eager_camera.semantic)
    torch.testing.assert_close(lazy_camera.sky, eager_camera.sky)


def test_empty_lazy_flow_becomes_none(tmp_path):
    info = _camera_info(tmp_path)
    save_flow_prior(info.flow_path, np.asarray([], dtype=object))
    camera = loadCam(_args(), 0, info, 1.0)
    assert camera.flow is None


def test_camera_json_supports_raw_camera_info_and_resized_camera(tmp_path):
    info = _camera_info(tmp_path)
    raw = camera_to_JSON(1, info)
    assert raw["width"] == 8
    assert raw["height"] == 6
    assert (raw["fx"], raw["fy"], raw["cx"], raw["cy"]) == (
        12.0,
        10.0,
        3.25,
        2.75,
    )

    camera = loadCam(_args(), 0, info, 1.0)
    resized = camera_to_JSON(2, camera)
    assert resized["width"] == 4
    assert resized["height"] == 3
    assert (resized["fx"], resized["fy"], resized["cx"], resized["cy"]) == (
        6.0,
        5.0,
        1.625,
        1.375,
    )

    legacy_camera = loadCam(_args(), 0, info._replace(intrinsics=None), 1.0)
    legacy = camera_to_JSON(3, legacy_camera)
    assert legacy["width"] == 4
    assert legacy["height"] == 3
    assert legacy["cx"] == 2.0
    assert legacy["cy"] == 1.5


def test_strict_flow_package_shape_and_finiteness_validation():
    height, width = 6, 8
    package = [
        15.0,
        np.eye(3, dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros((2, height, width), dtype=np.float32),
        np.ones((height, width), dtype=np.float32),
    ]
    _validate_flow_package(package, (height, width), 10.0, 20.0, "frame.png")

    invalid_shape = list(package)
    invalid_shape[4] = np.zeros((2, height - 1, width), dtype=np.float32)
    with pytest.raises(ValueError, match="flow shape"):
        _validate_flow_package(
            invalid_shape, (height, width), 10.0, 20.0, "frame.png"
        )

    invalid_values = list(package)
    invalid_values[5] = np.full((height, width), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="visibility contains"):
        _validate_flow_package(
            invalid_values, (height, width), 10.0, 20.0, "frame.png"
        )

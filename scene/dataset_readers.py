#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import cv2
import torch
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from scripts.splatad_split import (
    normalize_sensor_times,
    normalized_train_frame_gap,
    sensor_time_bounds,
    splatad_is_val_mask,
)
from scripts.prior_storage import (
    load_depth_prior,
    load_flow_prior,
    load_mask_prior,
    prior_exists,
)


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: int
    time: float
    semantic: np.array
    sky: np.array
    cam_id: int = 0
    flow: list = None
    intrinsics: np.array = None
    depth_path: str = None
    semantic_path: str = None
    sky_path: str = None
    flow_path: str = None
    sensor_time_min: float = None
    sensor_time_max: float = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    frame_gap: float = None
    bound: list = None
    others: dict = None

def get_val_frames(num_frames, test_every=None, train_every=None):
    assert train_every is None or test_every is None
    if train_every is None:
        val_frames = set(np.arange(test_every, num_frames, test_every))
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else train_frames

    return list(val_frames)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path, return_tuple=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = None

    try:
        t = np.array(vertices['t'])[..., None]
    except:
        t = None

    try:
        obj_mask = np.array(vertices['obj'])[..., None]
    except:
        obj_mask = None


    if return_tuple:
        return BasicPointCloud(points=positions, colors=colors, normals=normals, time=t, obj_mask=obj_mask)
    return positions, colors, normals, t, obj_mask

def storePly(path, xyz, rgb, t=None, obj_id=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if t is not None:
        dtype.append(('t', 'f4'))
    if obj_id is not None:
        dtype.append(('obj', 'f4'))
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    if t is not None:
        attributes = np.concatenate([attributes, t], axis=-1)
    if obj_id is not None:
        attributes = np.concatenate([attributes, obj_id], axis=-1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readKITTIInfo(path, use_colmap, split_mode='nvs-75', num_cam: int = 2):
    meta = np.load(os.path.join(path, "poses.npz"), allow_pickle=True)
    time_stamp = meta['time_stamp']
    R = meta['R']
    T = meta['T']
    height = int(meta['height'])
    width = int(meta['width'])
    focal = float(meta['focal'])
    FovX=focal2fov(focal, width)
    FovY=focal2fov(focal, height)
    frame_gap = num_cam / time_stamp.shape[0]
    max_fid = np.max(time_stamp)
    min_fid = np.min(time_stamp)
    time_scale_func = lambda x: ((x - min_fid) / (max_fid - min_fid))
    if split_mode == 'nvs-25':
        i_test = get_val_frames(time_stamp.shape[0] // num_cam, train_every=4)
        frame_gap *= 4
    elif split_mode == 'nvs-50':
        i_test = get_val_frames(time_stamp.shape[0] // num_cam, test_every=2)
        frame_gap *= 2
    elif split_mode == 'nvs-75':
        i_test = get_val_frames(time_stamp.shape[0] // num_cam, test_every=4)
    else:
        raise ValueError("No such split method: " + split_mode)
    
    # print(sorted(i_test))
    train_cameras = []
    test_cameras = []
    for idx, (img_path, fid) in enumerate(zip(sorted(os.listdir(os.path.join(path, "image"))), time_stamp)):
        depth_path = os.path.join(path, "depth", img_path.split(".")[0] + ".npy")
        flow_path = os.path.join(path, "flow", split_mode, img_path.split(".")[0] + ".npz")
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, "image", img_path)
        img = Image.open(img_path)
        assert img.size[0] == width and img.size[1] == height
        flow = load_flow_prior(flow_path) if prior_exists(flow_path) else None
        if flow is not None:
            for i in range(len(flow)):
                flow[i][0] = time_scale_func(flow[i][0])
        cam = CameraInfo(
            uid=idx,
            cam_id=idx % num_cam,
            fid=fid,
            R=R[idx, :3, :3],
            T=T[idx, :3],
            FovX=FovX,
            FovY=FovY,
            width=width,
            height=height,
            image_path=img_path,
            depth=load_depth_prior(depth_path).squeeze(-1),
            image_name=img_path.split("/")[-1],
            image=img,
            time=time_scale_func(fid),
            semantic=load_mask_prior(semantic_path, "semantic").astype(np.int32),
            sky=load_mask_prior(sky_path, "sky") != 0,
            flow=flow,
        )
        if idx // num_cam in i_test:
            test_cameras.append(cam)
        else:
            if not prior_exists(flow_path):
                print(f'[WARNING] Frame {fid} has no flow data. Image {img_path} might have no object, or fail to run prepare-flow.sh')
            # assert prior_exists(flow_path)
            train_cameras.append(cam)
        
    assert len(test_cameras) == len(i_test) * num_cam, "Wrong Test Cam Number: find {}, but need {}".format(len(test_cameras), len(i_test) * 2)
    nerf_normalization = getNerfppNorm(train_cameras)

    ply_path = os.path.join(path, "points3d-{}.ply".format(split_mode[-2:]))
    assert os.path.join(ply_path), 'Cannot Find PCD for initialization: {}'.format(ply_path)
    xyz, rgb, _, tim, obj_id = fetchPly(ply_path)
    bound = [np.min(xyz, axis=0), np.max(xyz, axis=0)]
    print("Load PCD:", ply_path)
    tim = time_scale_func(tim)
    if use_colmap:
        colmap_ply_path = os.path.join(path, 'colmap-{}.ply'.format(split_mode[-2:]))
        assert os.path.exists(colmap_ply_path), 'Cannot find SfM point cloud: ' + colmap_ply_path
        colmap_xyz, colmap_rgb, _, _, _ = fetchPly(colmap_ply_path)
        obj_id = np.concatenate([obj_id, np.zeros((colmap_xyz.shape[0], 1), dtype=np.float32)], axis=0)
        tim = np.concatenate([tim, np.full((colmap_xyz.shape[0], 1), fill_value=-1, dtype=np.float32)], axis=0)
        xyz = np.concatenate([xyz, colmap_xyz], axis=0)
        rgb = np.concatenate([rgb, colmap_rgb], axis=0)
        print("Load SfM PCD:", colmap_ply_path)
    
    scene_mask = obj_id[..., 0] <= 0.5
    obj_mask = np.bitwise_not(scene_mask)
    scene_xyz, scene_rgb = xyz[scene_mask], rgb[scene_mask]
    obj_xyz, obj_rgb, obj_tim, obj_id = xyz[obj_mask], rgb[obj_mask], tim[obj_mask], obj_id[obj_mask]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_rgb)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.5)
    # scene_pcd, _ = scene_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    scene_xyz = np.asarray(scene_pcd.points, dtype=np.float32)
    scene_rgb = np.asarray(scene_pcd.colors, dtype=np.float32)
    
    obj_pts_num = int(obj_xyz.shape[0] * 0.1)
    rand_choice = np.random.permutation(obj_xyz.shape[0])[: obj_pts_num]
    obj_xyz, obj_rgb, obj_tim, obj_id = obj_xyz[rand_choice], obj_rgb[rand_choice], obj_tim[rand_choice], obj_id[rand_choice]
    
    xyz = np.concatenate([scene_xyz, obj_xyz], axis=0)
    rgb = np.concatenate([scene_rgb, obj_rgb], axis=0)
    tim = np.concatenate([np.full((scene_xyz.shape[0], 1), fill_value=-1, dtype=np.float32), obj_tim], axis=0)
    obj_id = np.concatenate([np.zeros((scene_xyz.shape[0], 1), dtype=np.float32), obj_id], axis=0)
    pcd = BasicPointCloud(points=xyz, colors=rgb, time=tim, obj_id=obj_id)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        nerf_normalization=nerf_normalization,
        frame_gap=frame_gap,
        bound=bound,
    )
    return scene_info

def _read_splatad_metadata(meta, image_names, num_cam, meta_path):
    """Validate and return the sensor-aware fields shared by AD datasets."""
    required = {
        'K', 'R', 'T', 'time_stamps', 'is_val_list', 'camera_ids',
        'lidar_time_stamps', 'lidar_sensor_ids', 'lidar_names',
        'lidar_is_val_list', 'split_type',
        'train_split_fraction', 'frame_gap', 'sensor_time_min',
        'sensor_time_max', 'sensor_time_duration',
        'time_normalization_scope', 'image_widths', 'image_heights',
    }
    missing = sorted(required.difference(meta.files))
    if missing:
        raise ValueError("{} is missing keys: {}".format(meta_path, missing))

    count = len(image_names)
    for key in (
        'K', 'R', 'T', 'time_stamps', 'is_val_list',
        'image_widths', 'image_heights',
    ):
        if len(meta[key]) != count:
            raise ValueError(
                "{}: {} has {} entries but image/ has {}".format(
                    meta_path, key, len(meta[key]), count
                )
            )

    image_widths = np.asarray(meta['image_widths'], dtype=np.int64)
    image_heights = np.asarray(meta['image_heights'], dtype=np.int64)
    if (
        image_widths.shape != (count,)
        or image_heights.shape != (count,)
        or np.any(image_widths <= 0)
        or np.any(image_heights <= 0)
    ):
        raise ValueError(
            "image_widths/image_heights must be positive arrays of shape ({},)".format(
                count
            )
        )

    camera_ids = np.asarray(meta['camera_ids'], dtype=np.int64)
    if camera_ids.shape != (count,):
        raise ValueError("camera_ids must have shape ({},)".format(count))
    unique_camera_ids = np.unique(camera_ids)
    if len(unique_camera_ids) != num_cam:
        raise ValueError(
            "Config expects {} cameras, metadata contains {} ({})".format(
                num_cam, len(unique_camera_ids), unique_camera_ids.tolist()
            )
        )

    time_stamps = np.asarray(meta['time_stamps'], dtype=np.float64)
    if not np.isfinite(time_stamps).all() or time_stamps.max() <= time_stamps.min():
        raise ValueError("time_stamps must be finite and span a positive duration")
    for camera_id in unique_camera_ids:
        sensor_times = time_stamps[camera_ids == camera_id]
        if len(sensor_times) < 2 or np.any(np.diff(sensor_times) <= 0):
            raise ValueError(
                "Camera {} timestamps must be strictly increasing".format(camera_id)
            )

    is_val_list = np.asarray(meta['is_val_list'], dtype=np.bool_)
    split_type = str(np.asarray(meta['split_type']).item()).lower()
    if split_type != 'linspace':
        raise ValueError("Expected LINSPACE split, got {}".format(split_type))
    fraction = float(np.asarray(meta['train_split_fraction']).item())
    expected_is_val = splatad_is_val_mask(camera_ids, fraction)
    if not np.array_equal(is_val_list, expected_is_val):
        raise ValueError(
            "is_val_list does not match SplatAD sensor-wise LINSPACE split"
        )

    if is_val_list.all() or (~is_val_list).all():
        raise ValueError("Both train and validation cameras are required")

    lidar_time_stamps = np.asarray(meta['lidar_time_stamps'], dtype=np.float64)
    lidar_sensor_ids = np.asarray(meta['lidar_sensor_ids'], dtype=np.int64)
    lidar_is_val = np.asarray(meta['lidar_is_val_list'], dtype=np.bool_)
    if (
        lidar_time_stamps.ndim != 1
        or lidar_time_stamps.size < 2
        or not np.isfinite(lidar_time_stamps).all()
    ):
        raise ValueError("lidar_time_stamps must be a finite non-trivial 1-D array")
    if not (
        lidar_sensor_ids.shape
        == lidar_is_val.shape
        == lidar_time_stamps.shape
    ):
        raise ValueError("{}: LiDAR metadata arrays are misaligned".format(meta_path))
    unique_lidar_ids = np.unique(lidar_sensor_ids)
    lidar_names = tuple(str(name) for name in meta['lidar_names'].tolist())
    if not np.array_equal(
        unique_lidar_ids, np.arange(len(lidar_names), dtype=np.int64)
    ):
        raise ValueError(
            "{}: LiDAR sensor IDs {} do not match names {}".format(
                meta_path, unique_lidar_ids.tolist(), lidar_names
            )
        )
    for lidar_sensor_id in unique_lidar_ids:
        lidar_sensor_times = lidar_time_stamps[
            lidar_sensor_ids == lidar_sensor_id
        ]
        if (
            len(lidar_sensor_times) < 2
            or np.any(np.diff(lidar_sensor_times) <= 0)
        ):
            raise ValueError(
                "LiDAR {} timestamps must be strictly increasing".format(
                    lidar_names[int(lidar_sensor_id)]
                )
            )
    expected_lidar_is_val = splatad_is_val_mask(
        lidar_sensor_ids, fraction
    )
    if not np.array_equal(lidar_is_val, expected_lidar_is_val):
        raise ValueError(
            "lidar_is_val_list does not match SplatAD sensor-wise LINSPACE split"
        )
    sensor_time_min, sensor_time_max = sensor_time_bounds(
        time_stamps, lidar_time_stamps
    )
    scope = str(np.asarray(meta['time_normalization_scope']).item())
    if scope != 'all_cameras_all_lidars':
        raise ValueError(
            "Expected all-camera/all-lidar time normalization, got {}".format(scope)
        )
    stored_bounds = (
        float(np.asarray(meta['sensor_time_min']).item()),
        float(np.asarray(meta['sensor_time_max']).item()),
        float(np.asarray(meta['sensor_time_duration']).item()),
    )
    expected_bounds = (
        sensor_time_min,
        sensor_time_max,
        sensor_time_max - sensor_time_min,
    )
    if not np.allclose(stored_bounds, expected_bounds, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "{}: stored sensor time bounds {} do not match {}".format(
                meta_path, stored_bounds, expected_bounds
            )
        )

    frame_gap = float(np.asarray(meta['frame_gap']).item())
    if not np.isfinite(frame_gap) or frame_gap <= 0.0:
        raise ValueError("frame_gap must be positive and finite")
    expected_frame_gap = normalized_train_frame_gap(
        time_stamps,
        camera_ids,
        is_val_list,
        normalization_time_stamps=np.concatenate(
            [time_stamps, lidar_time_stamps]
        ),
    )
    if not np.isclose(frame_gap, expected_frame_gap, rtol=1e-5, atol=1e-7):
        raise ValueError(
            "{}: frame_gap {} does not match all-sensor duration formula {}".format(
                meta_path, frame_gap, expected_frame_gap
            )
        )

    return (
        camera_ids,
        time_stamps,
        is_val_list,
        frame_gap,
        sensor_time_min,
        sensor_time_max,
        image_widths,
        image_heights,
    )


def _validate_train_only_lidar(meta, point_times, meta_path):
    """Reject initialization points originating from held-out LiDAR sweeps."""
    lidar_keys = {
        'lidar_time_stamps', 'lidar_sensor_ids', 'lidar_is_val_list'
    }
    if not lidar_keys.issubset(meta.files):
        return
    lidar_times = np.asarray(meta['lidar_time_stamps'], dtype=np.float64)
    lidar_sensor_ids = np.asarray(meta['lidar_sensor_ids'], dtype=np.int64)
    lidar_is_val = np.asarray(meta['lidar_is_val_list'], dtype=np.bool_)
    if not (
        lidar_times.shape == lidar_sensor_ids.shape == lidar_is_val.shape
    ):
        raise ValueError("{}: LiDAR metadata arrays are misaligned".format(meta_path))
    if 'train_split_fraction' in meta.files:
        fraction = float(np.asarray(meta['train_split_fraction']).item())
        expected = splatad_is_val_mask(lidar_sensor_ids, fraction)
        if not np.array_equal(lidar_is_val, expected):
            raise ValueError("LiDAR split is not SplatAD LINSPACE")
    if point_times is None:
        raise ValueError("points3d.ply must contain per-point LiDAR timestamps")
    allowed = lidar_times[~lidar_is_val]
    observed = np.asarray(point_times, dtype=np.float64).reshape(-1)
    if allowed.size == 0 or observed.size == 0:
        raise ValueError("Train-only LiDAR point cloud is empty")
    # Converter timestamps are float32 in PLY, hence a small absolute tolerance.
    allowed = np.sort(allowed)
    insertion = np.searchsorted(allowed, observed)
    left = allowed[np.clip(insertion - 1, 0, len(allowed) - 1)]
    right = allowed[np.clip(insertion, 0, len(allowed) - 1)]
    distance = np.minimum(np.abs(observed - left), np.abs(observed - right))
    if np.any(distance > 1e-4):
        raise ValueError(
            "{}: points3d.ply contains timestamps outside train LiDAR split".format(
                meta_path
            )
        )


def _pinhole_intrinsics(K):
    """Return ``fx, fy, cx, cy`` from an AD converter calibration entry."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (3, 3):
        intrinsics = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    elif K.ndim == 1 and K.size >= 4:
        # Waymo stores [fx, fy, cx, cy, distortion...].
        intrinsics = K[:4]
    else:
        raise ValueError("Unsupported pinhole calibration shape: {}".format(K.shape))
    if not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
        raise ValueError("Invalid pinhole intrinsics: {}".format(intrinsics.tolist()))
    return intrinsics.astype(np.float32)


def readWaymoInfo(path, use_colmap=False, num_cam: int = 3):
    train_cam_infos, test_cam_infos = [], []
    meta_path = os.path.join(path, "cameras.npz")
    image_names = sorted(os.listdir(os.path.join(path, "image")))
    with np.load(meta_path, allow_pickle=True) as meta:
        K = np.asarray(meta['K'])
        R = np.asarray(meta['R'])
        T = np.asarray(meta['T'])
        (
            camera_ids,
            time_stamps,
            is_val_list,
            frame_gap,
            sensor_time_min,
            sensor_time_max,
            image_widths,
            image_heights,
        ) = _read_splatad_metadata(
            meta, image_names, num_cam, meta_path
        )
    time_scale_func = lambda x: normalize_sensor_times(
        x, sensor_time_min, sensor_time_max
    )
    
    for idx, (image_name, fid) in enumerate(zip(image_names, time_stamps)):
        stem = os.path.splitext(image_name)[0]
        depth_path = os.path.join(path, "depth", stem + ".npy")
        flow_path = os.path.join(path, "flow", stem + ".npz")
        semantic_path = os.path.join(path, "semantic", 'mask_' + stem + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + stem + ".npy")
        img_path = os.path.join(path, "image", image_name)
        width = int(image_widths[idx])
        height = int(image_heights[idx])
        intrinsics = _pinhole_intrinsics(K[idx])
        cam = CameraInfo(
            uid=idx,
            cam_id=int(camera_ids[idx]),
            fid=fid,
            R=R[idx, :3, :3],
            T=T[idx, :3],
            FovX=focal2fov(intrinsics[0], width),
            FovY=focal2fov(intrinsics[1], height),
            width=width,
            height=height,
            image_path=img_path,
            depth=None,
            image_name=image_name,
            image=None,
            time=time_scale_func(fid),
            semantic=None,
            sky=None,
            flow=None,
            intrinsics=intrinsics,
            depth_path=depth_path,
            semantic_path=semantic_path,
            sky_path=sky_path,
            flow_path=flow_path,
            sensor_time_min=sensor_time_min,
            sensor_time_max=sensor_time_max,
        )
        if is_val_list[idx]:
            test_cam_infos.append(cam)
        else:
            if not prior_exists(flow_path):
                print(f'[WARNING] Frame {fid} has no flow data. Image {img_path} might have no object, or fail to run prepare-flow.sh')
            # assert prior_exists(flow_path)
            train_cam_infos.append(cam)
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    assert os.path.exists(ply_path), 'Cannot Find PCD for initialization: {}'.format(ply_path)
    xyz, rgb, _, tim, obj_id = fetchPly(ply_path)
    if obj_id is None:
        raise ValueError(
            "points3d.ply has no obj field; run scripts/segment_pcd.py first"
        )
    with np.load(meta_path, allow_pickle=True) as meta:
        _validate_train_only_lidar(meta, tim, meta_path)
    bound = [np.min(xyz, axis=0), np.max(xyz, axis=0)]
    print("Load PCD:", ply_path)
    tim = time_scale_func(tim)
    if use_colmap:
        colmap_ply_path = os.path.join(path, 'colmap.ply')
        assert os.path.exists(colmap_ply_path), 'Cannot find SfM point cloud: ' + colmap_ply_path
        colmap_xyz, colmap_rgb, _, _, _ = fetchPly(colmap_ply_path)
        obj_id = np.concatenate([obj_id, np.zeros((colmap_xyz.shape[0], 1), dtype=np.float32)], axis=0)
        tim = np.concatenate([tim, np.full((colmap_xyz.shape[0], 1), fill_value=-1, dtype=np.float32)], axis=0)
        xyz = np.concatenate([xyz, colmap_xyz], axis=0)
        rgb = np.concatenate([rgb, colmap_rgb], axis=0)
        print("Load SfM PCD:", colmap_ply_path)
    
    scene_mask = obj_id[..., 0] <= 0.5
    obj_mask = np.bitwise_not(scene_mask)
    scene_xyz, scene_rgb = xyz[scene_mask], rgb[scene_mask]
    obj_xyz, obj_rgb, obj_tim, obj_id = xyz[obj_mask], rgb[obj_mask], tim[obj_mask], obj_id[obj_mask]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_rgb)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.2)
    # scene_pcd, _ = scene_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    scene_xyz = np.asarray(scene_pcd.points, dtype=np.float32)
    scene_rgb = np.asarray(scene_pcd.colors, dtype=np.float32)
    
    obj_pts_num = int(obj_xyz.shape[0] * 0.3)
    rand_choice = np.random.permutation(obj_xyz.shape[0])[: obj_pts_num]
    obj_xyz, obj_rgb, obj_tim, obj_id = obj_xyz[rand_choice], obj_rgb[rand_choice], obj_tim[rand_choice], obj_id[rand_choice]
    
    xyz = np.concatenate([scene_xyz, obj_xyz], axis=0)
    rgb = np.concatenate([scene_rgb, obj_rgb], axis=0)
    tim = np.concatenate([np.full((scene_xyz.shape[0], 1), fill_value=-1, dtype=np.float32), obj_tim], axis=0)
    obj_id = np.concatenate([np.zeros((scene_xyz.shape[0], 1), dtype=np.float32), obj_id], axis=0)
    pcd = BasicPointCloud(points=xyz, colors=rgb, time=tim, obj_id=obj_id)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        frame_gap=frame_gap,
        bound=bound,
    )
    return scene_info

def readnuScenesInfo(path, use_colmap=False, num_cam: int = 6):
    train_cam_infos, test_cam_infos = [], []
    meta_path = os.path.join(path, "meta.npz")
    image_names = sorted(os.listdir(os.path.join(path, "image")))
    with np.load(meta_path, allow_pickle=True) as meta:
        K = np.asarray(meta['K'])
        R = np.asarray(meta['R'])
        T = np.asarray(meta['T'])
        (
            camera_ids,
            time_stamps,
            is_val_list,
            frame_gap,
            sensor_time_min,
            sensor_time_max,
            image_widths,
            image_heights,
        ) = _read_splatad_metadata(
            meta, image_names, num_cam, meta_path
        )
    time_scale_func = lambda x: normalize_sensor_times(
        x, sensor_time_min, sensor_time_max
    )

    for idx, (image_name, fid) in enumerate(zip(image_names, time_stamps)):
        stem = os.path.splitext(image_name)[0]
        depth_path = os.path.join(path, "depth", stem + ".npy")
        flow_path = os.path.join(path, "flow", stem + ".npz")
        semantic_path = os.path.join(path, "semantic", 'mask_' + stem + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + stem + ".npy")
        img_path = os.path.join(path, "image", image_name)
        width = int(image_widths[idx])
        height = int(image_heights[idx])
        intrinsics = _pinhole_intrinsics(K[idx])
        cam = CameraInfo(
            uid=idx,
            cam_id=int(camera_ids[idx]),
            fid=fid,
            R=R[idx, :3, :3],
            T=T[idx, :3],
            FovX=focal2fov(intrinsics[0], width),
            FovY=focal2fov(intrinsics[1], height),
            width=width,
            height=height,
            image_path=img_path,
            depth=None,
            image_name=image_name,
            image=None,
            time=time_scale_func(fid),
            semantic=None,
            sky=None,
            flow=None,
            intrinsics=intrinsics,
            depth_path=depth_path,
            semantic_path=semantic_path,
            sky_path=sky_path,
            flow_path=flow_path,
            sensor_time_min=sensor_time_min,
            sensor_time_max=sensor_time_max,
        )
        if is_val_list[idx]:
            test_cam_infos.append(cam)
        else:
            if not prior_exists(flow_path):
                print(f'[WARNING] Frame {fid} has no flow data. Image {img_path} might have no object, or fail to run prepare-flow.sh')
            # assert prior_exists(flow_path)
            train_cam_infos.append(cam)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    assert os.path.exists(ply_path), 'Cannot Find PCD for initialization: {}'.format(ply_path)
    xyz, rgb, _, tim, obj_id = fetchPly(ply_path)
    if obj_id is None:
        raise ValueError(
            "points3d.ply has no obj field; run scripts/segment_pcd.py first"
        )
    with np.load(meta_path, allow_pickle=True) as meta:
        _validate_train_only_lidar(meta, tim, meta_path)
    bound = [np.min(xyz, axis=0), np.max(xyz, axis=0)]
    print("Load PCD:", ply_path)
    tim = time_scale_func(tim)
    if use_colmap:
        colmap_ply_path = os.path.join(path, 'colmap.ply')
        assert os.path.exists(colmap_ply_path), 'Cannot find SfM point cloud: ' + colmap_ply_path
        colmap_xyz, colmap_rgb, _, _, _ = fetchPly(colmap_ply_path)
        obj_id = np.concatenate([obj_id, np.zeros((colmap_xyz.shape[0], 1), dtype=np.float32)], axis=0)
        tim = np.concatenate([tim, np.full((colmap_xyz.shape[0], 1), fill_value=-1, dtype=np.float32)], axis=0)
        xyz = np.concatenate([xyz, colmap_xyz], axis=0)
        rgb = np.concatenate([rgb, colmap_rgb], axis=0)
        print("Load SfM PCD:", colmap_ply_path)
    
    scene_mask = obj_id[..., 0] <= 0.5
    obj_mask = np.bitwise_not(scene_mask)
    scene_xyz, scene_rgb = xyz[scene_mask], rgb[scene_mask]
    obj_xyz, obj_rgb, obj_tim, obj_id = xyz[obj_mask], rgb[obj_mask], tim[obj_mask], obj_id[obj_mask]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_rgb)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.15)
    # scene_pcd, _ = scene_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    scene_xyz = np.asarray(scene_pcd.points, dtype=np.float32)
    scene_rgb = np.asarray(scene_pcd.colors, dtype=np.float32)
    
    obj_pts_num = int(obj_xyz.shape[0] * 0.5)
    rand_choice = np.random.permutation(obj_xyz.shape[0])[: obj_pts_num]
    obj_xyz, obj_rgb, obj_tim, obj_id = obj_xyz[rand_choice], obj_rgb[rand_choice], obj_tim[rand_choice], obj_id[rand_choice]
    
    xyz = np.concatenate([scene_xyz, obj_xyz], axis=0)
    rgb = np.concatenate([scene_rgb, obj_rgb], axis=0)
    tim = np.concatenate([np.full((scene_xyz.shape[0], 1), fill_value=-1, dtype=np.float32), obj_tim], axis=0)
    obj_id = np.concatenate([np.zeros((scene_xyz.shape[0], 1), dtype=np.float32), obj_id], axis=0)
    pcd = BasicPointCloud(points=xyz, colors=rgb, time=tim, obj_id=obj_id)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        frame_gap=frame_gap,
        bound=bound,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    'KITTI': readKITTIInfo,
    'Waymo': readWaymoInfo,
    'nuScenes': readnuScenesInfo,
}

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

from scene.cameras import Camera
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from torch.nn.functional import interpolate
from scene.dataset_readers import CameraInfo
from scripts.prior_storage import (
    load_depth_prior,
    load_flow_prior,
    load_mask_prior,
    prior_exists,
)
from scripts.splatad_split import normalize_sensor_times

WARNED = False


def _resized_camera_image(cam_info, resolution, expected_size):
    """Load one image without retaining an open PIL handle in CameraInfo."""
    if cam_info.image is not None:
        if cam_info.image.size != expected_size:
            raise ValueError(
                "CameraInfo image size {} does not match metadata {} for {}".format(
                    cam_info.image.size, expected_size, cam_info.image_path
                )
            )
        return PILtoTorch(cam_info.image, resolution)

    with Image.open(cam_info.image_path) as source_image:
        if source_image.size != expected_size:
            raise ValueError(
                "Image size {} does not match metadata {} for {}".format(
                    source_image.size, expected_size, cam_info.image_path
                )
            )
        image = source_image.copy()
    try:
        return PILtoTorch(image, resolution)
    finally:
        image.close()


def _lazy_array(payload, path, loader):
    if payload is not None:
        return payload
    if path is None:
        return None
    return loader(path)


def _flow_to_tensors(cam_info):
    flow_payload = cam_info.flow
    if flow_payload is None and cam_info.flow_path is not None:
        if not prior_exists(cam_info.flow_path):
            return None
        flow_payload = load_flow_prior(cam_info.flow_path)
    if flow_payload is None or len(flow_payload) == 0:
        return None

    normalize_target_time = cam_info.flow_path is not None
    if normalize_target_time and (
        cam_info.sensor_time_min is None or cam_info.sensor_time_max is None
    ):
        raise ValueError(
            "Lazy AD flow requires shared sensor_time_min/sensor_time_max"
        )

    flow = []
    for record in flow_payload:
        if len(record) != 6:
            raise ValueError(
                "Flow records must contain time, K, R, T, flow, visibility"
            )
        target_time = record[0]
        if normalize_target_time:
            target_time = normalize_sensor_times(
                target_time,
                cam_info.sensor_time_min,
                cam_info.sensor_time_max,
            )
        tensor_values = []
        for value_index, value in enumerate(record[1:], start=1):
            if value_index == 5:
                # CoTracker visibility is binary and every consumer thresholds
                # it before use, so bool is lossless and substantially smaller.
                tensor_values.append(
                    torch.tensor(np.asarray(value) > 0.5, dtype=torch.bool)
                )
            else:
                tensor_values.append(torch.tensor(value, dtype=torch.float32))
        flow.append([target_time] + tensor_values)
    return flow


def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    if cam_info.image is None:
        orig_w, orig_h = int(cam_info.width), int(cam_info.height)
    else:
        orig_w, orig_h = cam_info.image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Camera dimensions must be positive")

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = _resized_camera_image(
        cam_info, resolution, (orig_w, orig_h)
    )

    intrinsics = None
    if cam_info.intrinsics is not None:
        intrinsics = np.asarray(cam_info.intrinsics, dtype=np.float64).reshape(-1)
        if intrinsics.shape != (4,):
            raise ValueError("Camera intrinsics must contain fx, fy, cx, cy")
        scale_x = float(resolution[0]) / float(orig_w)
        scale_y = float(resolution[1]) / float(orig_h)
        intrinsics = np.asarray(
            [
                intrinsics[0] * scale_x,
                intrinsics[1] * scale_y,
                intrinsics[2] * scale_x,
                intrinsics[3] * scale_y,
            ],
            dtype=np.float32,
        )

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask, depth, semantic, sky, lidar_depth, flow = None, None, None, None, None, None

    depth_array = _lazy_array(
        cam_info.depth, cam_info.depth_path, load_depth_prior
    )
    if depth_array is not None:
        depth_array = np.asarray(depth_array)
        if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
            depth_array = depth_array.squeeze(-1)
        if depth_array.ndim != 2:
            raise ValueError(
                "Depth prior must be HxW or HxWx1, got {} for {}".format(
                    depth_array.shape, cam_info.image_name
                )
            )
        depth = interpolate(
            torch.tensor(depth_array, dtype=torch.float32)[None, None, ...],
            [resolution[1], resolution[0]],
            mode='bilinear',
        ).squeeze()
    del depth_array

    semantic_array = _lazy_array(
        cam_info.semantic,
        cam_info.semantic_path,
        lambda path: load_mask_prior(path, "semantic"),
    )
    if semantic_array is not None:
        # AD-GS supervises only foreground-vs-background (semantic > 0).
        # Keeping that lossless binary value as bool avoids an 8-byte int64
        # tensor for every pixel of every camera.
        semantic = torch.tensor(
            np.asarray(semantic_array) != 0, dtype=torch.bool
        )
        if semantic.shape[0] != resolution[1] or semantic.shape[1] != resolution[0]:
            idxh = torch.linspace(
                0, semantic.shape[0] - 1, resolution[1], dtype=torch.int64
            )
            idxw = torch.linspace(
                0, semantic.shape[1] - 1, resolution[0], dtype=torch.int64
            )
            semantic = semantic[idxh[:, None], idxw]
    del semantic_array

    sky_array = _lazy_array(
        cam_info.sky,
        cam_info.sky_path,
        lambda path: load_mask_prior(path, "sky"),
    )
    if sky_array is not None:
        sky = interpolate(
            torch.tensor(np.asarray(sky_array) != 0, dtype=torch.float32)[
                None, None, ...
            ],
            [resolution[1], resolution[0]],
            mode='bilinear',
        ).squeeze()
        sky = sky > 0.5
    del sky_array
    
    flow = _flow_to_tensors(cam_info)

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        cam_id=cam_info.cam_id,
        R=cam_info.R, 
        T=cam_info.T, 
        FoVx=cam_info.FovX, 
        FoVy=cam_info.FovY, 
        image=gt_image, 
        gt_alpha_mask=loaded_mask, 
        fid=cam_info.fid, 
        time=cam_info.time,
        image_name=cam_info.image_name, 
        uid=id, 
        data_device=args.data_device if not args.lazy_load_to_gpu else 'cpu',
        depth=depth, 
        semantic=semantic,
        sky=sky,
        flow=flow,
        intrinsics=intrinsics,
    )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    width = getattr(camera, 'width', None)
    height = getattr(camera, 'height', None)
    if width is None:
        width = camera.image_width
    if height is None:
        height = camera.image_height
    width, height = int(width), int(height)

    if camera.intrinsics is None:
        fov_x = getattr(camera, 'FovX', None)
        fov_y = getattr(camera, 'FovY', None)
        if fov_x is None:
            fov_x = camera.FoVx
        if fov_y is None:
            fov_y = camera.FoVy
        fx = fov2focal(fov_x, width)
        fy = fov2focal(fov_y, height)
        cx = width / 2.0
        cy = height / 2.0
    else:
        fx, fy, cx, cy = [float(value) for value in camera.intrinsics]

    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : width,
        'height' : height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fy,
        'fx' : fx,
        'cx' : cx,
        'cy' : cy,
    }
    return camera_entry

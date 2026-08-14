"""Convert one nuScenes scene to AD-GS without changing SplatAD's split.

SplatAD (through NeurAD's ``ADDataParser``) treats every sensor stream as an
independent, time-ordered sequence.  For a stream with N samples and split
fraction f, the training samples are exactly::

    np.linspace(0, N - 1, ceil(N * f), dtype=np.int64)

and evaluation is the complement.  This converter deliberately keeps the six
camera chains and the LIDAR_TOP chain independent; nuScenes does not guarantee
that their sweep counts are equal.
"""

import argparse
import functools
import json
import math
import os
import shutil
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splatad_split import (
    normalized_train_frame_gap,
    sensor_time_bounds,
    splatad_is_val_mask,
)


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAMERA_CROP_BOTTOM = {
    "CAM_FRONT": 0,
    "CAM_FRONT_LEFT": 0,
    "CAM_FRONT_RIGHT": 0,
    "CAM_BACK": 80,
    "CAM_BACK_LEFT": 0,
    "CAM_BACK_RIGHT": 0,
}
LIDAR_NAME = "LIDAR_TOP"
SPLIT_REFERENCE = "SplatAD/NeurAD ADDataParser._get_linspaced_indices"


def build_rotation(quaternion):
    """Return a 3x3 rotation for a nuScenes [w, x, y, z] quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_pose(rotation, translation):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = build_rotation(rotation)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def collect_full_chain(nusc, sample_data_token):
    """Rewind a sample_data token, then return its complete ordered chain."""
    sample_data = nusc.get("sample_data", sample_data_token)
    while sample_data["prev"]:
        sample_data = nusc.get("sample_data", sample_data["prev"])

    chain = []
    seen = set()
    while True:
        if sample_data["token"] in seen:
            raise RuntimeError("sample_data chain contains a cycle")
        seen.add(sample_data["token"])
        chain.append(sample_data)
        if not sample_data["next"]:
            break
        sample_data = nusc.get("sample_data", sample_data["next"])

    timestamps = np.asarray([item["timestamp"] for item in chain], dtype=np.int64)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("sample_data timestamps must be strictly increasing")
    return chain


def select_chain(chain, first_frame, last_frame, sensor_name):
    """Apply an inclusive local range independently to one sensor chain.

    ``last_frame=-1`` means the end of that particular sensor stream.  An
    explicit last index beyond a shorter stream is clamped to its final sample,
    so unequal camera counts remain valid and are never padded or synchronized.
    """
    if first_frame < 0:
        raise ValueError("first_frame must be >= 0")
    if not chain or first_frame >= len(chain):
        raise ValueError(
            "{} has {} samples; first_frame={} is out of range".format(
                sensor_name, len(chain), first_frame
            )
        )
    if last_frame != -1 and last_frame < first_frame:
        raise ValueError("last_frame must be -1 or >= first_frame")
    inclusive_last = len(chain) - 1 if last_frame == -1 else min(last_frame, len(chain) - 1)
    selected = chain[first_frame : inclusive_last + 1]
    source_indices = np.arange(first_frame, inclusive_last + 1, dtype=np.int64)
    return selected, source_indices


def exact_is_val_mask(num_samples, train_split_fraction):
    sensor_ids = np.zeros(num_samples, dtype=np.int64)
    is_val = splatad_is_val_mask(sensor_ids, train_split_fraction)
    expected_train = int(math.ceil(num_samples * train_split_fraction))
    if int((~is_val).sum()) != expected_train:
        raise AssertionError("SplatAD train count mismatch")
    # np.linspace can include both endpoints only when it selects at least two
    # samples (e.g. N=2, f=.5 correctly selects local index 0 only).
    if train_split_fraction < 1.0 and expected_train >= 2:
        if is_val[0] or is_val[-1]:
            raise AssertionError("SplatAD LINSPACE must include both endpoints in train")
    return is_val


def raw_path(dataroot, sample_data):
    return os.path.join(dataroot, sample_data["filename"])


def validate_image(path, crop_bottom):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if height <= crop_bottom:
        raise ValueError("crop removes entire image: {}".format(path))
    return width, height - crop_bottom


def validate_lidar(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    byte_size = os.path.getsize(path)
    if byte_size == 0 or byte_size % (5 * np.dtype(np.float32).itemsize) != 0:
        raise ValueError("invalid nuScenes lidar file: {}".format(path))


def sensor_to_world(nusc, sample_data, world_from_global):
    calibrated = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    global_from_ego = make_pose(ego_pose["rotation"], ego_pose["translation"])
    ego_from_sensor = make_pose(calibrated["rotation"], calibrated["translation"])
    return world_from_global @ global_from_ego @ ego_from_sensor, calibrated


def build_records(nusc, dataroot, scene, args):
    first_sample = nusc.get("sample", scene["first_sample_token"])
    camera_chains = {}
    camera_source_indices = {}
    for camera_name in CAMERA_NAMES:
        full_chain = collect_full_chain(nusc, first_sample["data"][camera_name])
        selected, source_indices = select_chain(
            full_chain, args.first_frame, args.last_frame, camera_name
        )
        camera_chains[camera_name] = selected
        camera_source_indices[camera_name] = source_indices

    full_lidar_chain = collect_full_chain(nusc, first_sample["data"][LIDAR_NAME])
    lidar_chain, lidar_source_indices = select_chain(
        full_lidar_chain, args.first_frame, args.last_frame, LIDAR_NAME
    )

    all_selected = [item for chain in camera_chains.values() for item in chain] + list(lidar_chain)
    origin_sample = min(all_selected, key=lambda item: item["timestamp"])
    origin_ego = nusc.get("ego_pose", origin_sample["ego_pose_token"])
    global_from_origin_ego = make_pose(origin_ego["rotation"], origin_ego["translation"])
    world_from_global = np.linalg.inv(global_from_origin_ego)
    time_origin_us = int(min(item["timestamp"] for item in all_selected))

    camera_records = []
    camera_summary = {}
    for camera_id, camera_name in enumerate(CAMERA_NAMES):
        chain = camera_chains[camera_name]
        source_indices = camera_source_indices[camera_name]
        is_val = exact_is_val_mask(len(chain), args.train_split_fraction)
        crop_bottom = CAMERA_CROP_BOTTOM[camera_name]
        camera_summary[camera_name] = {
            "total": len(chain),
            "train": int((~is_val).sum()),
            "eval": int(is_val.sum()),
            "crop_bottom": crop_bottom,
        }
        for local_frame_id, (sample_data, source_frame_id) in enumerate(zip(chain, source_indices)):
            path = raw_path(dataroot, sample_data)
            width, height = validate_image(path, crop_bottom)
            camera_to_world, calibrated = sensor_to_world(nusc, sample_data, world_from_global)
            intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=np.float64)
            if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
                raise ValueError("invalid camera intrinsic for {}".format(sample_data["token"]))
            camera_records.append(
                {
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "local_frame_id": local_frame_id,
                    "source_frame_id": int(source_frame_id),
                    "sample_data": sample_data,
                    "raw_path": path,
                    "crop_bottom": crop_bottom,
                    "width": width,
                    "height": height,
                    "K": intrinsic,
                    "world_to_camera": np.linalg.inv(camera_to_world),
                    "timestamp": (sample_data["timestamp"] - time_origin_us) / 1e6,
                    "timestamp_us": int(sample_data["timestamp"]),
                    "is_val": bool(is_val[local_frame_id]),
                }
            )

    lidar_is_val = exact_is_val_mask(len(lidar_chain), args.train_split_fraction)
    lidar_records = []
    for local_frame_id, (sample_data, source_frame_id) in enumerate(
        zip(lidar_chain, lidar_source_indices)
    ):
        path = raw_path(dataroot, sample_data)
        validate_lidar(path)
        lidar_to_world, _ = sensor_to_world(nusc, sample_data, world_from_global)
        lidar_records.append(
            {
                "local_frame_id": local_frame_id,
                "source_frame_id": int(source_frame_id),
                "sample_data": sample_data,
                "raw_path": path,
                "lidar_to_world": lidar_to_world,
                "timestamp": (sample_data["timestamp"] - time_origin_us) / 1e6,
                "timestamp_us": int(sample_data["timestamp"]),
                "is_val": bool(lidar_is_val[local_frame_id]),
            }
        )

    summary = {
        "scene": scene["name"],
        "range": {
            "first_frame_inclusive_per_sensor": args.first_frame,
            "last_frame_inclusive_per_sensor": args.last_frame,
        },
        "split": {
            "type": "linspace",
            "scope": "per_sensor",
            "train_fraction": args.train_split_fraction,
            "reference": SPLIT_REFERENCE,
        },
        "cameras": camera_summary,
        "lidar": {
            "name": LIDAR_NAME,
            "total": len(lidar_records),
            "train": int((~lidar_is_val).sum()),
            "eval": int(lidar_is_val.sum()),
        },
    }
    return camera_records, lidar_records, time_origin_us, summary


def load_cropped_rgb(path, crop_bottom):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if crop_bottom:
            image = image.crop((0, 0, image.width, image.height - crop_bottom))
        return np.asarray(image, dtype=np.uint8)


def load_world_points(lidar_record):
    points = np.fromfile(lidar_record["raw_path"], dtype=np.float32).reshape(-1, 5)[:, :3]
    rotation = lidar_record["lidar_to_world"][:3, :3]
    translation = lidar_record["lidar_to_world"][:3, 3]
    return points.astype(np.float64) @ rotation.T + translation


def nearest_record(records, timestamp):
    timestamps = np.asarray([record["timestamp"] for record in records], dtype=np.float64)
    insertion = int(np.searchsorted(timestamps, timestamp))
    candidates = []
    if insertion < len(records):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    best = min(candidates, key=lambda idx: abs(timestamps[idx] - timestamp))
    return records[best]


def project_points(points_world, camera_record):
    world_to_camera = camera_record["world_to_camera"]
    camera_points = points_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera_points[:, 2]
    valid = depth > 1e-6
    uvw = camera_points @ camera_record["K"].T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] <= camera_record["width"] - 1
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] <= camera_record["height"] - 1
    return uv, depth, valid


def colorize_points(points_world, lidar_timestamp, train_cameras_by_id, rng):
    colors_sum = np.zeros((len(points_world), 3), dtype=np.float64)
    observations = np.zeros(len(points_world), dtype=np.int32)
    for camera_id in range(len(CAMERA_NAMES)):
        camera_record = nearest_record(train_cameras_by_id[camera_id], lidar_timestamp)
        if camera_record["is_val"]:
            raise AssertionError("point colors must never use an evaluation image")
        uv, _, valid = project_points(points_world, camera_record)
        if not np.any(valid):
            continue
        image = load_cropped_rgb(camera_record["raw_path"], camera_record["crop_bottom"])
        pixels = np.rint(uv[valid]).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, image.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, image.shape[0] - 1)
        colors_sum[valid] += image[pixels[:, 1], pixels[:, 0]]
        observations[valid] += 1

    colors = rng.uniform(0.0, 255.0, size=(len(points_world), 3))
    observed = observations > 0
    colors[observed] = colors_sum[observed] / observations[observed, None]
    return colors


def store_ply(path, xyz, rgb, timestamps):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("t", "f4"),
    ]
    vertices = np.empty(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    vertices["red"], vertices["green"], vertices["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertices["t"] = timestamps
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def write_images(camera_records, image_dir):
    for image_id, record in enumerate(tqdm(camera_records, desc="Writing camera images")):
        with Image.open(record["raw_path"]) as image:
            image = image.convert("RGB")
            if record["crop_bottom"]:
                image = image.crop((0, 0, image.width, image.height - record["crop_bottom"]))
            image.save(os.path.join(image_dir, "{:06d}.png".format(image_id)))


def write_point_cloud(lidar_records, camera_records, dst_path, args):
    rng = np.random.RandomState(args.seed)
    train_cameras_by_id = {
        camera_id: [
            record
            for record in camera_records
            if record["camera_id"] == camera_id and not record["is_val"]
        ]
        for camera_id in range(len(CAMERA_NAMES))
    }
    if any(not records for records in train_cameras_by_id.values()):
        raise ValueError("every camera must have at least one training image")

    point_chunks = []
    color_chunks = []
    time_chunks = []
    train_lidar_records = [record for record in lidar_records if not record["is_val"]]
    for lidar_record in tqdm(train_lidar_records, desc="Building train-only point cloud"):
        if lidar_record["is_val"]:
            raise AssertionError("evaluation lidar reached point-cloud construction")
        points_world = load_world_points(lidar_record)
        if args.use_color:
            colors = colorize_points(
                points_world,
                lidar_record["timestamp"],
                train_cameras_by_id,
                rng,
            )
        else:
            colors = rng.uniform(0.0, 255.0, size=(len(points_world), 3))

        if args.downsample_ratio < 1.0:
            keep_count = max(1, int(len(points_world) * args.downsample_ratio))
            keep = rng.permutation(len(points_world))[:keep_count]
            points_world = points_world[keep]
            colors = colors[keep]
        point_chunks.append(points_world.astype(np.float32))
        color_chunks.append(colors.astype(np.float32))
        time_chunks.append(
            np.full(len(points_world), lidar_record["timestamp"], dtype=np.float32)
        )

    if not point_chunks:
        raise ValueError("no training lidar sweeps selected")
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    timestamps = np.concatenate(time_chunks, axis=0)
    store_ply(os.path.join(dst_path, "points3d.ply"), points, colors, timestamps)
    return points.shape


def write_depth(camera_records, lidar_records, depth_dir):
    train_lidars = [record for record in lidar_records if not record["is_val"]]
    record_to_index = {record["sample_data"]["token"]: idx for idx, record in enumerate(train_lidars)}

    @functools.lru_cache(maxsize=24)
    def cached_world_points(train_lidar_idx):
        return load_world_points(train_lidars[train_lidar_idx])

    for image_id, camera_record in enumerate(tqdm(camera_records, desc="Writing train-lidar depth")):
        lidar_record = nearest_record(train_lidars, camera_record["timestamp"])
        if lidar_record["is_val"]:
            raise AssertionError("depth must never use an evaluation lidar sweep")
        lidar_idx = record_to_index[lidar_record["sample_data"]["token"]]
        points_world = cached_world_points(lidar_idx)
        uv, depth, valid = project_points(points_world, camera_record)
        pixels = np.rint(uv[valid]).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, camera_record["width"] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, camera_record["height"] - 1)

        depth_map = np.full(
            (camera_record["height"], camera_record["width"]),
            np.inf,
            dtype=np.float32,
        )
        np.minimum.at(depth_map, (pixels[:, 1], pixels[:, 0]), depth[valid].astype(np.float32))
        depth_mask = np.isfinite(depth_map)
        depth_map[~depth_mask] = 0.0
        np.savez(
            os.path.join(depth_dir, "{:06d}.npz".format(image_id)),
            depth=depth_map,
            mask=depth_mask,
            source_lidar_frame_id=np.int64(lidar_record["local_frame_id"]),
            source_lidar_is_train=np.bool_(True),
        )

        preview = np.zeros_like(depth_map, dtype=np.uint8)
        if np.any(depth_mask):
            valid_depth = depth_map[depth_mask]
            depth_range = float(valid_depth.max() - valid_depth.min())
            if depth_range > 0:
                preview[depth_mask] = np.clip(
                    (valid_depth - valid_depth.min()) / depth_range * 255.0, 0, 255
                ).astype(np.uint8)
            else:
                preview[depth_mask] = 255
        Image.fromarray(np.repeat(preview[:, :, None], 3, axis=2)).save(
            os.path.join(depth_dir, "{:06d}.png".format(image_id))
        )


def save_metadata(path, camera_records, lidar_records, time_origin_us, args):
    world_to_camera = np.stack([record["world_to_camera"] for record in camera_records])
    intrinsics = np.stack([record["K"] for record in camera_records])
    time_stamps = np.asarray([record["timestamp"] for record in camera_records], dtype=np.float64)
    camera_ids = np.asarray([record["camera_id"] for record in camera_records], dtype=np.int64)
    frame_ids = np.asarray([record["local_frame_id"] for record in camera_records], dtype=np.int64)
    source_frame_ids = np.asarray(
        [record["source_frame_id"] for record in camera_records], dtype=np.int64
    )
    is_val_list = np.asarray([record["is_val"] for record in camera_records], dtype=np.bool_)
    lidar_time_stamps = np.asarray(
        [record["timestamp"] for record in lidar_records], dtype=np.float64
    )
    normalization_time_stamps = np.concatenate(
        [time_stamps, lidar_time_stamps]
    )
    sensor_time_min, sensor_time_max = sensor_time_bounds(
        normalization_time_stamps
    )
    frame_gap = normalized_train_frame_gap(
        time_stamps,
        camera_ids,
        is_val_list,
        normalization_time_stamps=normalization_time_stamps,
    )

    lidar_to_world = np.stack([record["lidar_to_world"] for record in lidar_records])
    metadata = {
        "R": world_to_camera[:, :3, :3].astype(np.float32),
        "T": world_to_camera[:, :3, 3].astype(np.float32),
        "K": intrinsics.astype(np.float32),
        "time_stamps": time_stamps,
        "frame_ids": frame_ids,
        "source_frame_ids": source_frame_ids,
        "camera_ids": camera_ids,
        "camera_names": np.asarray(CAMERA_NAMES),
        "image_file_names": np.asarray(
            ["{:06d}.png".format(idx) for idx in range(len(camera_records))]
        ),
        "image_heights": np.asarray(
            [record["height"] for record in camera_records], dtype=np.int64
        ),
        "image_widths": np.asarray(
            [record["width"] for record in camera_records], dtype=np.int64
        ),
        "camera_crop_bottom": np.asarray(
            [CAMERA_CROP_BOTTOM[name] for name in CAMERA_NAMES], dtype=np.int64
        ),
        "camera_sample_counts": np.asarray(
            [sum(record["camera_id"] == idx for record in camera_records) for idx in range(len(CAMERA_NAMES))],
            dtype=np.int64,
        ),
        "camera_train_counts": np.asarray(
            [
                sum(record["camera_id"] == idx and not record["is_val"] for record in camera_records)
                for idx in range(len(CAMERA_NAMES))
            ],
            dtype=np.int64,
        ),
        "is_val_list": is_val_list,
        "lidar_names": np.asarray((LIDAR_NAME,)),
        "lidar_sensor_ids": np.zeros(
            len(lidar_records), dtype=np.int64
        ),
        "lidar_time_stamps": lidar_time_stamps,
        "lidar_frame_ids": np.asarray(
            [record["local_frame_id"] for record in lidar_records], dtype=np.int64
        ),
        "lidar_source_frame_ids": np.asarray(
            [record["source_frame_id"] for record in lidar_records], dtype=np.int64
        ),
        "lidar_is_val_list": np.asarray(
            [record["is_val"] for record in lidar_records], dtype=np.bool_
        ),
        "lidar_to_world_R": lidar_to_world[:, :3, :3].astype(np.float32),
        "lidar_to_world_T": lidar_to_world[:, :3, 3].astype(np.float32),
        "dataset_type": np.asarray("nuscenes"),
        "camera_layout": np.asarray("camera_major"),
        "split_type": np.asarray("linspace"),
        "split_scope": np.asarray("per_sensor"),
        "split_reference": np.asarray(SPLIT_REFERENCE),
        "train_split_fraction": np.asarray(args.train_split_fraction, dtype=np.float32),
        "first_frame_inclusive_per_sensor": np.asarray(args.first_frame, dtype=np.int64),
        "last_frame_inclusive_per_sensor": np.asarray(args.last_frame, dtype=np.int64),
        "frame_gap": np.asarray(frame_gap, dtype=np.float32),
        "sensor_time_min": np.asarray(sensor_time_min, dtype=np.float64),
        "sensor_time_max": np.asarray(sensor_time_max, dtype=np.float64),
        "sensor_time_duration": np.asarray(
            sensor_time_max - sensor_time_min, dtype=np.float64
        ),
        "time_normalization_scope": np.asarray("all_cameras_all_lidars"),
        "time_origin_us": np.asarray(time_origin_us, dtype=np.int64),
        "pcd_uses_train_lidar_only": np.asarray(True),
        "pcd_color_uses_train_cameras_only": np.asarray(bool(args.use_color)),
        "depth_uses_train_lidar_only": np.asarray(bool(args.use_depth)),
        "pointcloud_downsample_ratio": np.asarray(args.downsample_ratio, dtype=np.float32),
    }
    np.savez(path, **metadata)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert nuScenes using SplatAD's per-sensor LINSPACE split."
    )
    parser.add_argument("src", help="nuScenes dataroot (contains v1.0-trainval and samples)")
    parser.add_argument("dst", help="processed dataset root; output is dst/SCENE")
    parser.add_argument("scene", help="scene name, e.g. scene-0101")
    parser.add_argument(
        "--first_frame",
        default=0,
        type=int,
        help="inclusive local index applied independently to every sensor (default: 0)",
    )
    parser.add_argument(
        "--last_frame",
        default=-1,
        type=int,
        help=(
            "inclusive local index per sensor; -1 means each stream's end; an explicit "
            "index beyond a shorter stream is clamped (default: -1/full)"
        ),
    )
    parser.add_argument(
        "--downsample_ratio",
        "-r",
        default=1.0,
        type=float,
        help="per-train-sweep point retention ratio in (0, 1]",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        choices=("v1.0-mini", "v1.0-trainval"),
    )
    parser.add_argument("--train_split_fraction", default=0.5, type=float)
    parser.add_argument("--use_color", action="store_true", help="color PLY only from nearest TRAIN images")
    parser.add_argument("--use_depth", action="store_true", help="write depth using nearest TRAIN lidar only")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="audit raw streams and exact split counts without writing any output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty dst/SCENE directory",
    )
    args = parser.parse_args()
    if not 0.0 < args.train_split_fraction <= 1.0:
        parser.error("--train_split_fraction must be in (0, 1]")
    if not 0.0 < args.downsample_ratio <= 1.0:
        parser.error("--downsample_ratio must be in (0, 1]")
    return args


def find_scene(nusc, scene_name):
    tokens = nusc.field2token("scene", "name", scene_name)
    if len(tokens) != 1:
        raise ValueError("expected exactly one scene named {}, got {}".format(scene_name, len(tokens)))
    return nusc.get("scene", tokens[0])


def main():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.src, verbose=False)
    scene = find_scene(nusc, args.scene)
    camera_records, lidar_records, time_origin_us, summary = build_records(
        nusc, args.src, scene, args
    )
    if args.validate_only:
        print("VALIDATION_OK " + json.dumps(summary, sort_keys=True))
        return

    dst_path = os.path.join(args.dst, args.scene)
    if os.path.isdir(dst_path) and os.listdir(dst_path):
        if not args.overwrite:
            raise FileExistsError(
                "{} is non-empty; pass --overwrite to replace it".format(dst_path)
            )
        shutil.rmtree(dst_path)
    image_dir = os.path.join(dst_path, "image")
    os.makedirs(image_dir, exist_ok=True)
    depth_dir = os.path.join(dst_path, "lidar_depth")
    if args.use_depth:
        os.makedirs(depth_dir, exist_ok=True)

    write_images(camera_records, image_dir)
    point_shape = write_point_cloud(lidar_records, camera_records, dst_path, args)
    if args.use_depth:
        write_depth(camera_records, lidar_records, depth_dir)
    save_metadata(
        os.path.join(dst_path, "meta.npz"),
        camera_records,
        lidar_records,
        time_origin_us,
        args,
    )
    print("CONVERSION_OK " + json.dumps(summary, sort_keys=True))
    print("Get PCD: {}".format(point_shape))
    print("Get Images and RTs: {}".format(len(camera_records)))


if __name__ == "__main__":
    main()

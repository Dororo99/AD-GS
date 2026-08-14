#!/usr/bin/env python3
"""Camera-safe staging and verified staged installs for AD-GS priors.

This helper deliberately does not run any model.  It builds one image directory
per physical camera from ``camera_ids``, validates staged outputs, and commits a
complete set of priors only after every expensive step has succeeded.
"""

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scripts.prior_storage import (
        compact_prior,
        is_compact_prior,
        load_depth_prior,
        load_flow_prior,
        load_mask_prior,
        resolve_prior_path,
        save_depth_prior,
        save_mask_prior,
    )
except ImportError:  # Direct ``python scripts/camera_safe_priors.py`` execution.
    from prior_storage import (
        compact_prior,
        is_compact_prior,
        load_depth_prior,
        load_flow_prior,
        load_mask_prior,
        resolve_prior_path,
        save_depth_prior,
        save_mask_prior,
    )


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_CAMERAS = {
    "waymo": ("FRONT", "FRONT_LEFT", "FRONT_RIGHT"),
    "nuscenes": (
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ),
    "av2": (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    ),
}
TOOL_NAME = "adgs-camera-safe-priors"


def _metadata_path(scene, dataset):
    return scene / ("cameras.npz" if dataset == "waymo" else "meta.npz")


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _image_names(scene):
    image_dir = scene / "image"
    if not image_dir.is_dir():
        raise FileNotFoundError("Missing image directory: {}".format(image_dir))
    names = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not names:
        raise ValueError("No images in {}".format(image_dir))
    stems = [Path(name).stem for name in names]
    if len(stems) != len(set(stems)):
        raise ValueError("Image stems must be unique across the scene")
    if any(not stem.isdigit() for stem in stems):
        raise ValueError(
            "Grounded-SAM-2 requires numeric image stems; first invalid set: {}"
            .format([stem for stem in stems if not stem.isdigit()][:3])
        )
    numeric_names = sorted(names, key=lambda name: int(Path(name).stem))
    if numeric_names != names:
        raise ValueError(
            "Lexical image order (metadata order) differs from numeric video order"
        )
    return names


def load_scene(scene, dataset):
    scene = Path(scene).resolve()
    if dataset not in EXPECTED_CAMERAS:
        raise ValueError("Unsupported dataset: {}".format(dataset))
    metadata_path = _metadata_path(scene, dataset)
    if not metadata_path.is_file():
        raise FileNotFoundError("Missing metadata: {}".format(metadata_path))
    names = _image_names(scene)

    with np.load(str(metadata_path), allow_pickle=True) as meta:
        required = ("camera_ids", "camera_names", "time_stamps", "is_val_list")
        missing = [key for key in required if key not in meta.files]
        if missing:
            raise ValueError("Missing metadata fields: {}".format(missing))
        camera_ids = np.asarray(meta["camera_ids"], dtype=np.int64).reshape(-1)
        camera_names = tuple(_as_text(value) for value in meta["camera_names"].tolist())
        time_stamps = np.asarray(meta["time_stamps"], dtype=np.float64).reshape(-1)
        is_val = np.asarray(meta["is_val_list"], dtype=np.bool_).reshape(-1)
        if "dataset_type" in meta.files:
            actual_dataset = _as_text(np.asarray(meta["dataset_type"]).item()).lower()
            if actual_dataset != dataset:
                raise ValueError(
                    "Dataset mismatch: requested {}, metadata says {}".format(
                        dataset, actual_dataset
                    )
                )

    count = len(names)
    for field, values in (
        ("camera_ids", camera_ids),
        ("time_stamps", time_stamps),
        ("is_val_list", is_val),
    ):
        if len(values) != count:
            raise ValueError(
                "{} has {} entries but image/ has {}".format(field, len(values), count)
            )

    expected_names = EXPECTED_CAMERAS[dataset]
    if camera_names != expected_names:
        raise ValueError(
            "Camera order mismatch for {}: expected {}, got {}".format(
                dataset, expected_names, camera_names
            )
        )
    expected_ids = set(range(len(expected_names)))
    actual_ids = set(camera_ids.tolist())
    if actual_ids != expected_ids:
        raise ValueError(
            "camera_ids mismatch: expected {}, got {}".format(
                sorted(expected_ids), sorted(actual_ids)
            )
        )

    camera_entries = []
    for camera_id, camera_name in enumerate(camera_names):
        indices = np.flatnonzero(camera_ids == camera_id)
        camera_times = time_stamps[indices]
        if len(camera_times) == 0:
            raise ValueError("Camera {} has no images".format(camera_name))
        if len(camera_times) > 1 and np.any(np.diff(camera_times) <= 0):
            raise ValueError(
                "Camera {} is not time-ordered by its preserved numeric stems"
                .format(camera_name)
            )
        camera_entries.append(
            {
                "camera_id": int(camera_id),
                "camera_name": camera_name,
                "indices": indices.tolist(),
                "images": [names[index] for index in indices],
                "total": int(len(indices)),
                "train": int(np.count_nonzero(~is_val[indices])),
                "val": int(np.count_nonzero(is_val[indices])),
            }
        )

    return {
        "scene": str(scene),
        "dataset": dataset,
        "metadata_name": metadata_path.name,
        "images": names,
        "camera_ids": camera_ids,
        "time_stamps": time_stamps,
        "is_val": is_val,
        "cameras": camera_entries,
    }


def public_plan(info):
    return {
        "scene": info["scene"],
        "dataset": info["dataset"],
        "metadata": info["metadata_name"],
        "image_total": len(info["images"]),
        "image_train": int(np.count_nonzero(~info["is_val"])),
        "image_val": int(np.count_nonzero(info["is_val"])),
        "cameras": [
            {
                "camera_id": item["camera_id"],
                "camera_name": item["camera_name"],
                "total": item["total"],
                "train": item["train"],
                "val": item["val"],
            }
            for item in info["cameras"]
        ],
    }


def _manifest_path(work):
    return work / "manifest.json"


def _read_manifest(work, expected_scene=None):
    path = _manifest_path(work)
    if not path.is_file():
        raise ValueError("Unmarked work directory (manifest missing): {}".format(work))
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("tool") != TOOL_NAME:
        raise ValueError("Work directory has an unknown manifest: {}".format(work))
    if expected_scene is not None:
        manifest_scene = str(Path(manifest["scene"]).resolve())
        if manifest_scene != str(Path(expected_scene).resolve()):
            raise ValueError("Work manifest belongs to another scene")
    return manifest


def _remove_marked_work(work, scene):
    _read_manifest(work, scene)
    shutil.rmtree(str(work))


def stage_camera_streams(scene, dataset, work, dry_run=False, overwrite=False):
    info = load_scene(scene, dataset)
    work = Path(work).resolve()
    scene_path = Path(info["scene"])
    if work == scene_path:
        raise ValueError("Work directory cannot be the scene directory")
    if dry_run:
        result = public_plan(info)
        result["work"] = str(work)
        result["dry_run"] = True
        return result

    if work.exists():
        entries = list(work.iterdir()) if work.is_dir() else [work]
        if entries:
            if not overwrite:
                raise FileExistsError(
                    "Work directory is non-empty: {} (use --overwrite)".format(work)
                )
            _remove_marked_work(work, scene_path)
        elif not work.is_dir():
            raise ValueError("Work path is not a directory: {}".format(work))
    work.mkdir(parents=True, exist_ok=True)

    manifest = public_plan(info)
    manifest.update({"tool": TOOL_NAME, "version": 1, "work": str(work)})
    with _manifest_path(work).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    image_root = scene_path / "image"
    for camera in info["cameras"]:
        camera_root = work / "cameras" / "camera_{:03d}".format(camera["camera_id"])
        staged_image_root = camera_root / "image"
        staged_image_root.mkdir(parents=True, exist_ok=False)
        for name in camera["images"]:
            # SAM2's video loader accepts JPEG filenames only, even though PIL
            # can decode the nuScenes PNG payload.  A same-stem ``.jpg`` alias
            # preserves every metadata/output stem without re-encoding pixels.
            staged_name = Path(name).stem + ".jpg"
            os.symlink(
                str((image_root / name).resolve()),
                str(staged_image_root / staged_name),
            )
    return manifest


def validate_staged_camera_streams(scene, dataset, work):
    """Validate a manifest-marked camera staging tree without changing it."""
    info = load_scene(scene, dataset)
    scene_path = Path(info["scene"])
    work = Path(work).resolve()
    manifest = _read_manifest(work, scene_path)
    if manifest.get("dataset") != dataset:
        raise ValueError(
            "Work manifest dataset mismatch: expected {}, got {}".format(
                dataset, manifest.get("dataset")
            )
        )

    image_root = scene_path / "image"
    for camera in info["cameras"]:
        staged_root = (
            work / "cameras" / "camera_{:03d}".format(camera["camera_id"]) / "image"
        )
        if not staged_root.is_dir():
            raise FileNotFoundError("Missing staged camera directory: {}".format(staged_root))
        expected = {Path(name).stem + ".jpg" for name in camera["images"]}
        actual = {path.name for path in staged_root.iterdir()}
        if actual != expected:
            raise ValueError(
                "Staged camera {} images differ: missing={}, extra={}".format(
                    camera["camera_id"],
                    sorted(expected - actual)[:3],
                    sorted(actual - expected)[:3],
                )
            )
        source_by_stem = {
            Path(name).stem: image_root / name for name in camera["images"]
        }
        for staged_name in expected:
            staged = staged_root / staged_name
            source = source_by_stem[Path(staged_name).stem]
            if not staged.is_symlink() or staged.resolve() != source.resolve():
                raise ValueError("Invalid staged image link: {}".format(staged))
    return manifest


def _ply_properties(path):
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    if len(vertices) == 0:
        raise ValueError("PLY has no vertices: {}".format(path))
    return set(vertices.data.dtype.names)


def preflight(scene, dataset, overwrite=False):
    info = load_scene(scene, dataset)
    scene_path = Path(info["scene"])
    points_path = scene_path / "points3d.ply"
    if not points_path.is_file():
        raise FileNotFoundError("Missing converter point cloud: {}".format(points_path))
    point_fields = _ply_properties(points_path)
    if "t" not in point_fields:
        raise ValueError("points3d.ply has no timestamp field")

    targets = [
        scene_path / "depth",
        scene_path / "semantic",
        scene_path / "sky",
        scene_path / "flow",
        scene_path / "colmap.ply",
    ]
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing priors: {}".format(existing)
        )
    if "obj" in point_fields and not overwrite:
        raise FileExistsError(
            "points3d.ply is already segmented; use --overwrite to replace priors"
        )
    result = public_plan(info)
    result["overwrite"] = bool(overwrite)
    result["existing_targets"] = existing
    return result


def _link(source, target):
    if not source.exists():
        raise FileNotFoundError(source)
    os.symlink(str(source.resolve()), str(target))


def _validate_sandbox_link(sandbox, name, expected):
    path = sandbox / name
    if not path.is_symlink() or path.resolve() != expected.resolve():
        raise ValueError(
            "Invalid reusable sandbox link {}: expected {}".format(path, expected)
        )


def prepare_sandbox(scene, dataset, work, kind, overwrite=False, reuse=False):
    info = load_scene(scene, dataset)
    scene_path = Path(info["scene"])
    work = Path(work).resolve()
    _read_manifest(work, scene_path)
    sandbox = work / (kind + "_scene")
    if overwrite and reuse:
        raise ValueError("Sandbox overwrite and reuse are mutually exclusive")
    if sandbox.exists():
        if reuse:
            if sandbox.is_symlink() or not sandbox.is_dir():
                raise ValueError(
                    "Reusable sandbox is not a directory: {}".format(sandbox)
                )
            _validate_sandbox_link(sandbox, "image", scene_path / "image")
            _validate_sandbox_link(
                sandbox, info["metadata_name"], scene_path / info["metadata_name"]
            )
            _validate_sandbox_link(
                sandbox, "semantic", work / "assembled" / "semantic"
            )
            if kind == "colmap":
                _validate_sandbox_link(
                    sandbox, "sky", work / "assembled" / "sky"
                )
            return str(sandbox)
        if not overwrite:
            raise FileExistsError("Sandbox already exists: {}".format(sandbox))
        if sandbox.is_symlink() or not sandbox.is_dir():
            raise ValueError(
                "Refusing to replace non-directory sandbox: {}".format(sandbox)
            )
        shutil.rmtree(str(sandbox))
    sandbox.mkdir(parents=True)
    _link(scene_path / "image", sandbox / "image")
    _link(scene_path / info["metadata_name"], sandbox / info["metadata_name"])

    semantic = work / "assembled" / "semantic"
    sky = work / "assembled" / "sky"
    if kind in ("flow", "segment", "colmap"):
        _link(semantic, sandbox / "semantic")
    if kind == "colmap":
        _link(sky, sandbox / "sky")
    if kind == "segment":
        # A real copy is essential: segment_pcd.py rewrites this file in place.
        shutil.copy2(str(scene_path / "points3d.ply"), str(sandbox / "points3d.ply"))
    return str(sandbox)


def _image_sizes(scene, names):
    sizes = {}
    for name in names:
        with Image.open(str(scene / "image" / name)) as image:
            sizes[Path(name).stem] = (image.height, image.width)
    return sizes


def _validate_numpy(path, expected_shape, kind):
    try:
        if kind == "flow":
            load_flow_prior(path)
            return
        if kind == "depth":
            array = load_depth_prior(path)
        else:
            array = load_mask_prior(path, kind)
    except Exception as exc:
        raise ValueError("Cannot read {} output {}: {}".format(kind, path, exc))
    if kind == "depth":
        valid_shapes = (expected_shape, expected_shape + (1,))
        if array.shape not in valid_shapes:
            raise ValueError(
                "Depth/image size mismatch for {}: {} not in {}".format(
                    path, array.shape, valid_shapes
                )
            )
        if not np.isfinite(array).all():
            raise ValueError("Depth contains non-finite values: {}".format(path))
    elif array.shape != expected_shape:
        raise ValueError(
            "Mask/image size mismatch for {}: {} != {}".format(
                path, array.shape, expected_shape
            )
        )


def _load_mask_for_collection(path, expected_shape, kind):
    """Load a mask, losslessly repairing only wrong-sized all-zero masks."""
    try:
        array = load_mask_prior(path, kind)
    except Exception as exc:
        raise ValueError("Cannot read {} output {}: {}".format(kind, path, exc))
    if array.shape == expected_shape:
        return array, False
    if np.any(array):
        raise ValueError(
            "Mask/image size mismatch for {}: {} != {}; refusing to resize a "
            "non-empty mask".format(path, array.shape, expected_shape)
        )
    return np.zeros(expected_shape, dtype=np.bool_), True


def compact_depth_outputs(scene, dataset, work):
    """Compact DPT outputs in place, deleting each raw NPY after verification."""
    info = load_scene(scene, dataset)
    scene_path = Path(info["scene"])
    work = Path(work).resolve()
    _read_manifest(work, scene_path)
    root = work / "depth"
    if not root.is_dir():
        raise FileNotFoundError("Missing staged depth: {}".format(root))
    expected = {Path(name).stem for name in info["images"]}
    candidates = [
        path for path in root.iterdir()
        if path.is_file() and path.suffix in (".npy", ".npz")
    ]
    keys = [path.stem for path in candidates]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            "Staged depth differs: missing={}, extra={}, duplicates={}".format(
                sorted(expected - set(keys))[:3],
                sorted(set(keys) - expected)[:3],
                len(keys) - len(set(keys)),
            )
        )
    sizes = _image_sizes(scene_path, info["images"])
    raw_bytes = 0
    compact_bytes = 0
    for stem in sorted(expected):
        source = resolve_prior_path(root / (stem + ".npz"))
        _validate_numpy(source, sizes[stem], "depth")
        target = root / (stem + ".npz")
        if source != target or not is_compact_prior(source, "depth"):
            source_bytes = source.stat().st_size
            depth = load_depth_prior(source)
            save_depth_prior(target, depth)
            _validate_numpy(target, sizes[stem], "depth")
            if source != target:
                source.unlink()
            raw_bytes += source_bytes
        compact_bytes += target.stat().st_size
    return {
        "root": str(root),
        "count": len(expected),
        "raw_bytes_processed": raw_bytes,
        "compact_bytes": compact_bytes,
    }


def collect_camera_mask(scene, dataset, work, kind, camera_id):
    """Pack one camera's masks, safely resuming already-collected outputs.

    Grounded-SAM historically emitted 1080x1920 masks before the first
    detection, even for differently-sized inputs. A wrong-sized all-zero mask
    has no spatial content and is losslessly canonicalized to the image shape.
    Non-empty mismatches remain fatal because resizing them would move labels.
    """
    if kind not in ("semantic", "sky"):
        raise ValueError("Only semantic and sky masks can be collected")
    info = load_scene(scene, dataset)
    scene_path = Path(info["scene"])
    work = Path(work).resolve()
    _read_manifest(work, scene_path)
    if camera_id < 0 or camera_id >= len(info["cameras"]):
        raise ValueError("Invalid camera_id {}".format(camera_id))
    camera = info["cameras"][camera_id]
    source_root = (
        work / "cameras" / "camera_{:03d}".format(camera_id) / kind
    )
    expected = {"mask_{}.npy".format(Path(name).stem) for name in camera["images"]}
    actual = (
        {path.name for path in source_root.glob("*.npy")}
        if source_root.is_dir()
        else set()
    )
    unexpected = actual - expected
    if unexpected:
        raise ValueError(
            "{} camera {} outputs contain unexpected files: {}".format(
                kind,
                camera_id,
                sorted(unexpected)[:3],
            )
        )
    target_root = work / "assembled" / kind
    target_root.mkdir(parents=True, exist_ok=True)
    sizes = _image_sizes(scene_path, camera["images"])
    missing = []
    for filename in sorted(expected):
        target = target_root / (Path(filename).stem + ".npz")
        if filename not in actual and not target.is_file():
            missing.append(filename)
    if missing:
        raise ValueError(
            "{} camera {} outputs are incomplete: missing={}".format(
                kind, camera_id, missing[:3]
            )
        )

    raw_bytes = 0
    compact_bytes = 0
    collected = 0
    reused = 0
    normalized_empty = 0
    for filename in sorted(expected):
        stem = Path(filename[len("mask_"):]).stem
        source = source_root / filename
        target = target_root / (Path(filename).stem + ".npz")
        source_exists = source.is_file()
        if target.is_file():
            _validate_numpy(target, sizes[stem], kind)
            reused += 1
            if source_exists:
                raw_mask, was_normalized = _load_mask_for_collection(
                    source, sizes[stem], kind
                )
                packed_mask = load_mask_prior(target, kind)
                if not np.array_equal(raw_mask, packed_mask):
                    raise ValueError(
                        "Raw and collected masks disagree for {}".format(source)
                    )
                raw_bytes += source.stat().st_size
                normalized_empty += int(was_normalized)
                source.unlink()
            compact_bytes += target.stat().st_size
            continue

        mask, was_normalized = _load_mask_for_collection(
            source, sizes[stem], kind
        )
        raw_bytes += source.stat().st_size
        save_mask_prior(target, mask, kind)
        _validate_numpy(target, sizes[stem], kind)
        collected += 1
        normalized_empty += int(was_normalized)
        compact_bytes += target.stat().st_size
        source.unlink()
    try:
        source_root.rmdir()
    except OSError:
        pass
    return {
        "root": str(target_root),
        "camera_id": int(camera_id),
        "kind": kind,
        "count": len(expected),
        "collected": collected,
        "reused": reused,
        "normalized_empty": normalized_empty,
        "raw_bytes_removed": raw_bytes,
        "compact_bytes": compact_bytes,
    }


def assemble_masks(scene, dataset, work, kind):
    if kind not in ("semantic", "sky"):
        raise ValueError("Only semantic and sky masks need camera assembly")
    info = load_scene(scene, dataset)
    scene_path = Path(info["scene"])
    work = Path(work).resolve()
    _read_manifest(work, scene_path)
    target = work / "assembled" / kind
    if target.exists():
        raise FileExistsError("Assembled output already exists: {}".format(target))
    target.mkdir(parents=True)
    sizes = _image_sizes(scene_path, info["images"])

    copied = set()
    for camera in info["cameras"]:
        source_root = (
            work / "cameras" / "camera_{:03d}".format(camera["camera_id"]) / kind
        )
        if not source_root.is_dir():
            raise FileNotFoundError("Missing camera output: {}".format(source_root))
        expected = {"mask_{}.npy".format(Path(name).stem) for name in camera["images"]}
        actual = {path.name for path in source_root.glob("*.npy")}
        if actual != expected:
            raise ValueError(
                "{} camera {} outputs differ: missing={}, extra={}".format(
                    kind,
                    camera["camera_id"],
                    sorted(expected - actual)[:3],
                    sorted(actual - expected)[:3],
                )
            )
        for filename in sorted(expected):
            stem = Path(filename[len("mask_"):]).stem
            target_name = Path(filename).stem + ".npz"
            if target_name in copied:
                raise ValueError("Mask collision while merging: {}".format(target_name))
            source = source_root / filename
            _validate_numpy(source, sizes[stem], kind)
            save_mask_prior(
                target / target_name, load_mask_prior(source, kind), kind
            )
            copied.add(target_name)
    if len(copied) != len(info["images"]):
        raise ValueError("Assembled {} mask count is incomplete".format(kind))
    return str(target)


def _validate_work_outputs(info, work):
    scene = Path(info["scene"])
    names = info["images"]
    sizes = _image_sizes(scene, names)
    expected_all = {Path(name).stem for name in names}
    expected_train = {
        Path(name).stem
        for name, is_val in zip(names, info["is_val"])
        if not is_val
    }
    roots = {
        "depth": work / "depth",
        "semantic": work / "assembled" / "semantic",
        "sky": work / "assembled" / "sky",
        "flow": work / "flow_scene" / "flow",
    }
    prefixes = {"depth": "", "semantic": "mask_", "sky": "mask_", "flow": ""}

    for kind, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError("Missing staged {}: {}".format(kind, root))
        stems = expected_train if kind == "flow" else expected_all
        expected_keys = {prefixes[kind] + stem for stem in stems}
        actual_paths = [
            path for path in root.iterdir()
            if path.is_file() and path.suffix in (".npy", ".npz")
        ]
        actual_keys = [path.stem for path in actual_paths]
        if len(actual_keys) != len(set(actual_keys)):
            raise ValueError("Staged {} contains duplicate NPY/NPZ priors".format(kind))
        if set(actual_keys) != expected_keys:
            raise ValueError(
                "Staged {} differs: missing={}, extra={}".format(
                    kind,
                    sorted(expected_keys - set(actual_keys))[:3],
                    sorted(set(actual_keys) - expected_keys)[:3],
                )
            )
        for stem in stems:
            path = resolve_prior_path(root / (prefixes[kind] + stem + ".npz"))
            _validate_numpy(path, sizes[stem], kind)

    segment_ply = work / "segment_scene" / "points3d.ply"
    segment_fields = _ply_properties(segment_ply)
    if not {"t", "obj"}.issubset(segment_fields):
        raise ValueError("Segmented points3d.ply must contain t and obj fields")
    colmap_ply = work / "colmap_scene" / "colmap.ply"
    _ply_properties(colmap_ply)
    return roots, segment_ply, colmap_ply


def _compact_tree(source, target, kind):
    """Install one canonical NPZ per staged prior without a raw copy."""
    target.mkdir()
    for source_path in sorted(source.iterdir()):
        if not source_path.is_file() or source_path.suffix not in (".npy", ".npz"):
            continue
        target_path = target / (source_path.stem + ".npz")
        if is_compact_prior(source_path, kind):
            try:
                os.link(str(source_path), str(target_path))
            except OSError:
                shutil.copy2(str(source_path), str(target_path))
        else:
            compact_prior(source_path, target_path, kind)


def _replace_path(incoming, target, overwrite):
    target_exists = target.exists() or target.is_symlink()
    if target_exists and not overwrite:
        raise FileExistsError("Refusing to overwrite {}".format(target))
    backup = None
    if target_exists:
        backup = target.parent / (".{}.backup-{}".format(target.name, uuid.uuid4().hex))
        os.replace(str(target), str(backup))
    try:
        os.replace(str(incoming), str(target))
    except Exception:
        if backup is not None and backup.exists():
            os.replace(str(backup), str(target))
        raise
    if backup is not None:
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(str(backup))
        else:
            backup.unlink()


def commit_all(scene, dataset, work, overwrite=False):
    info = load_scene(scene, dataset)
    scene_path = Path(info["scene"])
    work = Path(work).resolve()
    _read_manifest(work, scene_path)
    roots, segment_ply, colmap_ply = _validate_work_outputs(info, work)

    # Re-run target checks immediately before the first write.
    preflight(scene_path, dataset, overwrite=overwrite)
    transaction = scene_path / (".adgs-priors-incoming-{}".format(uuid.uuid4().hex))
    transaction.mkdir()
    try:
        for kind in ("depth", "semantic", "sky", "flow"):
            _compact_tree(roots[kind], transaction / kind, kind)
        shutil.copy2(str(segment_ply), str(transaction / "points3d.ply"))
        shutil.copy2(str(colmap_ply), str(transaction / "colmap.ply"))

        original_points = scene_path / "points3d.ply"
        original_fields = _ply_properties(original_points)
        if "obj" not in original_fields:
            backup = scene_path / "points3d.unsegmented.ply"
            if not backup.exists():
                try:
                    os.link(str(original_points), str(backup))
                except OSError:
                    shutil.copy2(str(original_points), str(backup))

        for kind in ("depth", "semantic", "sky", "flow"):
            _replace_path(transaction / kind, scene_path / kind, overwrite)
        _replace_path(
            transaction / "colmap.ply", scene_path / "colmap.ply", overwrite
        )
        # Enriching the converter point cloud is the one expected replacement;
        # its unsegmented inode/file is preserved above for recovery.
        _replace_path(transaction / "points3d.ply", original_points, True)
    finally:
        if transaction.exists():
            shutil.rmtree(str(transaction))
    return {
        "scene": str(scene_path),
        "committed": ["depth", "semantic", "sky", "flow", "colmap.ply", "points3d.ply"],
        "unsegmented_backup": str(scene_path / "points3d.unsegmented.ply"),
    }


def cleanup(work, scene):
    work = Path(work).resolve()
    _remove_marked_work(work, Path(scene).resolve())


def _print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_scene_args(command, with_work=False):
        command.add_argument("scene", type=Path)
        command.add_argument("--dataset", required=True, choices=sorted(EXPECTED_CAMERAS))
        if with_work:
            command.add_argument("--work", required=True, type=Path)

    command = subparsers.add_parser("plan", help="read-only camera stream plan")
    add_scene_args(command)

    command = subparsers.add_parser("preflight", help="fail before expensive inference")
    add_scene_args(command)
    command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("stage", help="stage one image directory per camera")
    add_scene_args(command, with_work=True)
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("validate-stage", help="validate reusable staged camera inputs")
    add_scene_args(command, with_work=True)

    command = subparsers.add_parser("assemble-masks", help="verify and merge camera masks in work")
    add_scene_args(command, with_work=True)
    command.add_argument("--kind", required=True, choices=("semantic", "sky"))

    command = subparsers.add_parser(
        "compact-depth", help="stream-compact DPT NPY outputs and delete raw files"
    )
    add_scene_args(command, with_work=True)

    command = subparsers.add_parser(
        "collect-camera-mask",
        help="pack one camera's binary nonzero masks and delete raw files",
    )
    add_scene_args(command, with_work=True)
    command.add_argument("--kind", required=True, choices=("semantic", "sky"))
    command.add_argument("--camera-id", required=True, type=int)

    command = subparsers.add_parser("prepare-sandbox", help="prepare an isolated legacy-script scene")
    add_scene_args(command, with_work=True)
    command.add_argument("--kind", required=True, choices=("flow", "segment", "colmap"))
    sandbox_mode = command.add_mutually_exclusive_group()
    sandbox_mode.add_argument("--overwrite", action="store_true")
    sandbox_mode.add_argument("--reuse", action="store_true")

    command = subparsers.add_parser("verify-work", help="validate every staged prior")
    add_scene_args(command, with_work=True)

    command = subparsers.add_parser(
        "commit-all",
        help="install verified staged priors (targets are replaced sequentially)",
    )
    add_scene_args(command, with_work=True)
    command.add_argument("--overwrite", action="store_true")

    command = subparsers.add_parser("cleanup", help="remove a manifest-marked work directory")
    command.add_argument("scene", type=Path)
    command.add_argument("--work", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "plan":
        _print_json(public_plan(load_scene(args.scene, args.dataset)))
    elif args.command == "preflight":
        _print_json(preflight(args.scene, args.dataset, args.overwrite))
    elif args.command == "stage":
        _print_json(
            stage_camera_streams(
                args.scene, args.dataset, args.work, args.dry_run, args.overwrite
            )
        )
    elif args.command == "validate-stage":
        _print_json(
            validate_staged_camera_streams(args.scene, args.dataset, args.work)
        )
    elif args.command == "assemble-masks":
        _print_json({"assembled": assemble_masks(args.scene, args.dataset, args.work, args.kind)})
    elif args.command == "compact-depth":
        _print_json(compact_depth_outputs(args.scene, args.dataset, args.work))
    elif args.command == "collect-camera-mask":
        _print_json(
            collect_camera_mask(
                args.scene, args.dataset, args.work, args.kind, args.camera_id
            )
        )
    elif args.command == "prepare-sandbox":
        _print_json(
            {"sandbox": prepare_sandbox(
                args.scene, args.dataset, args.work, args.kind,
                args.overwrite, args.reuse
            )}
        )
    elif args.command == "verify-work":
        info = load_scene(args.scene, args.dataset)
        _read_manifest(args.work, args.scene)
        _validate_work_outputs(info, Path(args.work).resolve())
        _print_json({"scene": info["scene"], "work_valid": True})
    elif args.command == "commit-all":
        _print_json(commit_all(args.scene, args.dataset, args.work, args.overwrite))
    elif args.command == "cleanup":
        cleanup(args.work, args.scene)
        _print_json({"removed": str(args.work)})
    else:
        parser.error("Unknown command")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)

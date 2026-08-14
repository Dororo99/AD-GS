#!/usr/bin/env python3
"""Synthetic, inference-free tests for camera_safe_priors.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_safe_priors as priors  # noqa: E402
import prior_storage as storage  # noqa: E402


def write_ply(path, segmented=False):
    fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("t", "f4"),
    ]
    if segmented:
        fields.append(("obj", "f4"))
    vertices = np.zeros(2, dtype=fields)
    vertices["x"] = [0.0, 1.0]
    vertices["t"] = [0.0, 0.0]
    if segmented:
        vertices["obj"] = [0.0, 1.0]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


class CameraSafePriorsTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.scene = self.root / "scene"
        image_dir = self.scene / "image"
        image_dir.mkdir(parents=True)

        # Frame-major input with two timestamps and three physical cameras.
        self.names = ["{:06d}.png".format(index) for index in range(6)]
        for index, name in enumerate(self.names):
            image = np.full((3, 4, 3), index, dtype=np.uint8)
            Image.fromarray(image).save(str(image_dir / name))

        np.savez(
            str(self.scene / "cameras.npz"),
            camera_ids=np.asarray([0, 1, 2, 0, 1, 2]),
            camera_names=np.asarray(priors.EXPECTED_CAMERAS["waymo"]),
            time_stamps=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            is_val_list=np.asarray([False, False, False, True, True, True]),
            dataset_type=np.asarray("waymo"),
        )
        write_ply(self.scene / "points3d.ply")
        self.work = self.root / "work"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_dry_run_then_camera_isolated_transaction(self):
        result = priors.stage_camera_streams(
            self.scene, "waymo", self.work, dry_run=True
        )
        self.assertTrue(result["dry_run"])
        self.assertFalse(self.work.exists(), "dry-run must not create staging state")

        priors.preflight(self.scene, "waymo")
        priors.stage_camera_streams(self.scene, "waymo", self.work)
        resumed = priors.validate_staged_camera_streams(self.scene, "waymo", self.work)
        self.assertEqual(resumed["dataset"], "waymo")
        expected_by_camera = (
            {"000000.jpg", "000003.jpg"},
            {"000001.jpg", "000004.jpg"},
            {"000002.jpg", "000005.jpg"},
        )
        for camera_id, expected in enumerate(expected_by_camera):
            image_root = self.work / "cameras" / "camera_{:03d}".format(camera_id) / "image"
            self.assertEqual({path.name for path in image_root.iterdir()}, expected)
            self.assertTrue(all(path.is_symlink() for path in image_root.iterdir()))
            self.assertTrue(
                all(path.resolve().suffix == ".png" for path in image_root.iterdir())
            )

        depth_root = self.work / "depth"
        depth_root.mkdir()
        for name in self.names:
            stem = Path(name).stem
            np.save(str(depth_root / (stem + ".npy")), np.zeros((3, 4, 1), np.float32))
        depth_result = priors.compact_depth_outputs(
            self.scene, "waymo", self.work
        )
        self.assertEqual(depth_result["count"], len(self.names))
        self.assertFalse(list(depth_root.glob("*.npy")))
        self.assertEqual(len(list(depth_root.glob("*.npz"))), len(self.names))

        for camera_id, names in enumerate(expected_by_camera):
            camera_root = self.work / "cameras" / "camera_{:03d}".format(camera_id)
            for kind in ("sky", "semantic"):
                output = camera_root / kind
                output.mkdir()
                for name in names:
                    np.save(
                        str(output / ("mask_" + Path(name).stem + ".npy")),
                        np.zeros((3, 4), np.uint16),
                    )
                result = priors.collect_camera_mask(
                    self.scene, "waymo", self.work, kind, camera_id
                )
                self.assertEqual(result["count"], len(names))
                self.assertFalse(list(output.glob("*.npy")))
        self.assertTrue(
            all(path.suffix == ".npz" for path in (self.work / "assembled" / "semantic").iterdir())
        )

        flow_scene = Path(
            priors.prepare_sandbox(self.scene, "waymo", self.work, "flow")
        )
        (flow_scene / "flow").mkdir()
        for stem in ("000000", "000001", "000002"):
            np.savez(str(flow_scene / "flow" / (stem + ".npz")), flow=np.asarray([], dtype=object))

        flow_inode = (flow_scene / "flow" / "000000.npz").stat().st_ino
        reused_flow_scene = Path(
            priors.prepare_sandbox(
                self.scene, "waymo", self.work, "flow", reuse=True
            )
        )
        self.assertEqual(reused_flow_scene, flow_scene)
        self.assertEqual(
            (reused_flow_scene / "flow" / "000000.npz").stat().st_ino,
            flow_inode,
        )
        with self.assertRaises(ValueError):
            priors.prepare_sandbox(
                self.scene, "waymo", self.work, "flow",
                overwrite=True, reuse=True,
            )

        segment_scene = Path(
            priors.prepare_sandbox(self.scene, "waymo", self.work, "segment")
        )
        write_ply(segment_scene / "points3d.ply", segmented=True)

        colmap_scene = Path(
            priors.prepare_sandbox(self.scene, "waymo", self.work, "colmap")
        )
        write_ply(colmap_scene / "colmap.ply")

        priors._validate_work_outputs(
            priors.load_scene(self.scene, "waymo"), self.work
        )
        staged_depth_inode = (self.work / "depth" / "000000.npz").stat().st_ino
        staged_mask_inode = (
            self.work / "assembled" / "semantic" / "mask_000000.npz"
        ).stat().st_ino
        result = priors.commit_all(self.scene, "waymo", self.work)
        self.assertEqual(len(result["committed"]), 6)
        self.assertTrue((self.scene / "points3d.unsegmented.ply").is_file())
        self.assertIn("obj", priors._ply_properties(self.scene / "points3d.ply"))
        for folder in ("depth", "semantic", "sky", "flow"):
            self.assertTrue((self.scene / folder).is_dir())
            files = list((self.scene / folder).iterdir())
            self.assertTrue(files)
            self.assertTrue(all(path.suffix == ".npz" for path in files))
            self.assertFalse(any(path.suffix == ".npy" for path in files))
        self.assertTrue((self.scene / "colmap.ply").is_file())
        self.assertEqual(
            (self.scene / "depth" / "000000.npz").stat().st_ino,
            staged_depth_inode,
        )
        self.assertEqual(
            (self.scene / "semantic" / "mask_000000.npz").stat().st_ino,
            staged_mask_inode,
        )
        self.assertEqual(
            storage.load_depth_prior(self.scene / "depth" / "000000.npy").dtype,
            np.float32,
        )
        self.assertEqual(
            storage.load_mask_prior(
                self.scene / "semantic" / "mask_000000.npy", "semantic"
            ).dtype,
            np.bool_,
        )
        self.assertEqual(
            storage.load_flow_prior(self.scene / "flow" / "000000.npz").size,
            0,
        )
        with zipfile.ZipFile(str(self.scene / "flow" / "000000.npz"), "r") as archive:
            self.assertTrue(
                all(
                    member.compress_type == zipfile.ZIP_DEFLATED
                    for member in archive.infolist()
                )
            )
        self.assertFalse(list(self.scene.glob(".adgs-priors-incoming-*")))

        with self.assertRaises(FileExistsError):
            priors.preflight(self.scene, "waymo", overwrite=False)
        priors.cleanup(self.work, self.scene)
        self.assertFalse(self.work.exists())

    def test_wrong_sized_empty_mask_is_losslessly_normalized_and_reusable(self):
        priors.stage_camera_streams(self.scene, "waymo", self.work)
        output = self.work / "cameras" / "camera_000" / "sky"
        output.mkdir()
        np.save(
            str(output / "mask_000000.npy"),
            np.zeros((2, 4), dtype=np.uint16),
        )
        np.save(
            str(output / "mask_000003.npy"),
            np.zeros((3, 4), dtype=np.uint16),
        )

        result = priors.collect_camera_mask(
            self.scene, "waymo", self.work, "sky", 0
        )
        self.assertEqual(result["collected"], 2)
        self.assertEqual(result["normalized_empty"], 1)
        repaired = storage.load_mask_prior(
            self.work / "assembled" / "sky" / "mask_000000.npz", "sky"
        )
        self.assertEqual(repaired.shape, (3, 4))
        self.assertFalse(np.any(repaired))

        resumed = priors.collect_camera_mask(
            self.scene, "waymo", self.work, "sky", 0
        )
        self.assertEqual(resumed["collected"], 0)
        self.assertEqual(resumed["reused"], 2)

    def test_wrong_sized_nonempty_mask_is_rejected(self):
        priors.stage_camera_streams(self.scene, "waymo", self.work)
        output = self.work / "cameras" / "camera_000" / "semantic"
        output.mkdir()
        wrong = np.zeros((2, 4), dtype=np.uint16)
        wrong[0, 0] = 1
        np.save(str(output / "mask_000000.npy"), wrong)
        np.save(
            str(output / "mask_000003.npy"),
            np.zeros((3, 4), dtype=np.uint16),
        )

        with self.assertRaisesRegex(ValueError, "refusing to resize a non-empty mask"):
            priors.collect_camera_mask(
                self.scene, "waymo", self.work, "semantic", 0
            )
        self.assertFalse(
            (self.work / "assembled" / "semantic" / "mask_000000.npz").exists()
        )
    def test_non_monotonic_camera_stream_is_rejected(self):
        metadata_path = self.scene / "cameras.npz"
        with np.load(str(metadata_path), allow_pickle=True) as meta:
            values = {key: meta[key] for key in meta.files}
        values["time_stamps"] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        np.savez(str(metadata_path), **values)
        with self.assertRaises(ValueError):
            priors.stage_camera_streams(
                self.scene, "waymo", self.work, dry_run=True
            )

    def test_shell_dry_run_is_read_only(self):
        launcher = Path(__file__).resolve().parent / "prepare_splatad_priors.sh"
        environment = os.environ.copy()
        environment.update(
            {
                "DRY_RUN": "1",
                "PRIOR_WORK_ROOT": str(self.work),
            }
        )
        result = subprocess.run(
            ["bash", str(launcher), "waymo", str(self.scene), "6"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("camera_000", result.stdout)
        self.assertIn("camera_001", result.stdout)
        self.assertIn("camera_002", result.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=6", result.stdout)
        self.assertIn("--device cuda:0", result.stdout)
        self.assertIn("QT_QPA_PLATFORM=offscreen", result.stdout)
        self.assertNotIn("--use_gpu", result.stdout)
        self.assertIn("compact-depth", result.stdout)
        self.assertIn("collect-camera-mask", result.stdout)
        self.assertNotIn("assemble-masks", result.stdout)
        self.assertFalse(self.work.exists(), "shell dry-run must not create work")


class PriorStorageTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_depth_float16_roundtrip_and_legacy_npy(self):
        height, width = 900, 1600
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        depth = (0.65 * x + 0.35 * y)[..., None].astype(np.float32)
        compact = self.root / "depth.npz"
        storage.save_depth_prior(compact, depth)
        self.assertTrue(storage.is_compact_prior(compact, "depth"))
        with np.load(str(compact), allow_pickle=False) as archive:
            self.assertLessEqual(
                float(np.asarray(archive[storage.DEPTH_ERROR_KEY]).item()),
                storage.MAX_NORMALIZED_DEPTH_ABS_ERROR,
            )
        restored = storage.load_depth_prior(self.root / "depth.npy")
        self.assertEqual(restored.shape, depth.shape)
        self.assertEqual(restored.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(restored - depth))), 5e-4)
        self.assertLess(compact.stat().st_size, depth.nbytes // 3)

        legacy = self.root / "legacy_depth.npy"
        np.save(str(legacy), depth)
        legacy_restored = storage.load_depth_prior(legacy)
        self.assertEqual(legacy_restored.dtype, np.float32)
        np.testing.assert_array_equal(legacy_restored, depth)

    def test_interrupted_atomic_write_preserves_previous_archive(self):
        compact = self.root / "depth.npz"
        original = np.zeros((32, 48, 1), dtype=np.float32)
        replacement = np.ones_like(original)
        storage.save_depth_prior(compact, original)

        def fail_after_partial_write(handle, **_arrays):
            handle.write(b"partial archive")
            raise RuntimeError("simulated interruption")

        with mock.patch.object(
            storage.np, "savez_compressed", side_effect=fail_after_partial_write
        ):
            with self.assertRaises(RuntimeError):
                storage.save_depth_prior(compact, replacement)

        np.testing.assert_array_equal(storage.load_depth_prior(compact), original)
        self.assertFalse(list(self.root.glob(".depth.npz.tmp-*")))

    def test_mask_nonzero_pattern_is_lossless_bitpacked_and_compressed(self):
        height, width = 900, 1600
        semantic = np.zeros((height, width), dtype=np.uint16)
        semantic[180:620, 320:1280] = 17
        semantic[400:450, 700:900] = 0
        sky = np.zeros((height, width), dtype=np.uint8)
        sky[:310] = 255

        for kind, source in (("semantic", semantic), ("sky", sky)):
            compact = self.root / (kind + ".npz")
            storage.save_mask_prior(compact, source, kind)
            self.assertTrue(storage.is_compact_prior(compact, kind))
            with np.load(str(compact), allow_pickle=False) as archive:
                self.assertEqual(
                    str(np.asarray(archive[storage.KIND_KEY]).item()),
                    kind + "_binary_nonzero",
                )
            restored = storage.load_mask_prior(
                self.root / (kind + ".npy"), kind
            )
            self.assertEqual(restored.shape, source.shape)
            self.assertEqual(restored.dtype, np.bool_)
            np.testing.assert_array_equal(restored, source != 0)
            self.assertLess(compact.stat().st_size, source.nbytes // 100)

        legacy = self.root / "legacy_mask.npy"
        np.save(str(legacy), semantic)
        np.testing.assert_array_equal(
            storage.load_mask_prior(legacy, "semantic"), semantic
        )

    def test_flow_keeps_legacy_api_and_uses_zip_compression(self):
        height, width = 256, 512
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        coordinates = np.stack((grid_x + 2.0, grid_y), axis=0)
        visibility = np.ones((height, width), dtype=np.float32)
        flow = np.asarray(
            [[
                0.5,
                np.asarray([1000.0, 1000.0, 256.0, 128.0], dtype=np.float32),
                np.eye(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                coordinates,
                visibility,
            ]],
            dtype=object,
        )
        compact = self.root / "flow.npz"
        legacy = self.root / "legacy_flow.npz"
        storage.save_flow_prior(compact, flow)
        np.savez(str(legacy), flow=flow)
        self.assertTrue(storage.is_compact_prior(compact, "flow"))
        self.assertFalse(storage.is_compact_prior(legacy, "flow"))
        restored = storage.load_flow_prior(compact)
        self.assertEqual(restored.shape, flow.shape)
        np.testing.assert_array_equal(restored[0, 4], coordinates)
        np.testing.assert_array_equal(restored[0, 5], visibility)
        self.assertLess(compact.stat().st_size, legacy.stat().st_size // 5)
        legacy_restored = storage.load_flow_prior(legacy)
        np.testing.assert_array_equal(legacy_restored[0, 4], coordinates)


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "splatad_split.py"
SPEC = importlib.util.spec_from_file_location("splatad_split", MODULE_PATH)
SPLIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLIT)


def test_reference_60_frame_indices():
    actual = SPLIT.linspace_train_local_indices(60, 0.5)
    expected = np.linspace(0, 59, 30, dtype=np.int64)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == 0
    assert actual[-1] == 59
    assert len(actual) == 30


def test_sensor_wise_split_with_uneven_counts():
    sensor_ids = np.asarray([0, 1, 0, 1, 0, 1, 0, 0], dtype=np.int64)
    train, val = SPLIT.split_indices_by_sensor(sensor_ids, 0.5)

    sensor0_global = np.asarray([0, 2, 4, 6, 7])
    sensor1_global = np.asarray([1, 3, 5])
    expected_train = np.concatenate(
        [
            sensor0_global[np.linspace(0, 4, 3, dtype=np.int64)],
            sensor1_global[np.linspace(0, 2, 2, dtype=np.int64)],
        ]
    )
    np.testing.assert_array_equal(train, expected_train)
    assert set(train).isdisjoint(set(val))
    assert sorted(np.concatenate([train, val]).tolist()) == list(range(8))


def test_eval_mask_and_per_sensor_train_counts():
    sensor_ids = np.repeat(np.arange(3), [1, 4, 5])
    is_val = SPLIT.splatad_is_val_mask(sensor_ids, 0.5)
    for sensor_id, count in zip(range(3), [1, 4, 5]):
        assert np.count_nonzero((sensor_ids == sensor_id) & ~is_val) == int(
            np.ceil(count * 0.5)
        )


def test_fraction_one_matches_splatad_train_semantics():
    sensor_ids = np.asarray([0, 0, 1, 1])
    train, val = SPLIT.split_indices_by_sensor(sensor_ids, 1.0)
    np.testing.assert_array_equal(train, np.arange(4))
    np.testing.assert_array_equal(val, np.arange(4))
    assert not SPLIT.splatad_is_val_mask(sensor_ids, 1.0).any()


def test_normalized_train_frame_gap():
    times = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.float64)
    sensors = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    is_val = SPLIT.splatad_is_val_mask(sensors, 0.5)
    assert SPLIT.normalized_train_frame_gap(times, sensors, is_val) == 1.0


def test_waymo_uses_camera_and_lidar_common_bounds():
    camera_times = np.asarray(
        [0.038213, 0.039, 0.040, 0.138, 0.139, 0.140],
        dtype=np.float64,
    )
    lidar_times = np.asarray([0.0, 0.100016], dtype=np.float64)

    sensor_min, sensor_max = SPLIT.sensor_time_bounds(
        camera_times, lidar_times
    )
    assert sensor_min == 0.0
    assert sensor_max == camera_times.max()

    normalized_cameras = SPLIT.normalize_sensor_times(
        camera_times, sensor_min, sensor_max
    )
    normalized_lidar = SPLIT.normalize_sensor_times(
        lidar_times, sensor_min, sensor_max
    )
    assert normalized_lidar[0] == 0.0
    assert normalized_cameras.min() > 0.0
    assert normalized_cameras.max() == 1.0
    assert np.all((normalized_lidar >= 0.0) & (normalized_lidar <= 1.0))

    # Camera-only normalization produced the original negative LiDAR time.
    old_first_lidar = (
        lidar_times[0] - camera_times.min()
    ) / (camera_times.max() - camera_times.min())
    assert old_first_lidar < 0.0


def test_frame_gap_uses_all_sensor_duration():
    camera_times = np.asarray(
        [
            0.040, 0.041, 0.042,
            0.140, 0.141, 0.142,
            0.240, 0.241, 0.242,
            0.340, 0.341, 0.342,
        ],
        dtype=np.float64,
    )
    camera_ids = np.tile(np.arange(3), 4)
    is_val = SPLIT.splatad_is_val_mask(camera_ids, 0.5)
    lidar_times = np.asarray([0.0, 0.1, 0.2, 0.36], dtype=np.float64)

    frame_gap = SPLIT.normalized_train_frame_gap(
        camera_times,
        camera_ids,
        is_val,
        normalization_time_stamps=np.concatenate(
            [camera_times, lidar_times]
        ),
    )
    np.testing.assert_allclose(frame_gap, 0.3 / 0.36)


def test_time_normalization_rejects_unrelated_origin():
    try:
        SPLIT.normalize_sensor_times(
            np.asarray([-1.0, 0.5]), 0.0, 1.0
        )
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-range timestamps must be rejected")

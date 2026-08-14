import sys
from types import SimpleNamespace

import pytest
import torch

import train


def _camera(uid, camera_id, timestamp):
    return SimpleNamespace(
        uid=uid,
        cam_id=camera_id,
        time=timestamp,
        image_name="camera-{:02d}.png".format(uid),
    )


def test_fixed_eval_cameras_are_front_facing_and_time_spaced():
    cameras = [
        _camera(8, 1, 4.0),
        _camera(4, 0, 4.0),
        _camera(2, 0, 2.0),
        _camera(0, 0, 0.0),
        _camera(6, 1, 1.0),
        _camera(1, 0, 1.0),
        _camera(3, 0, 3.0),
    ]

    selected = train._select_fixed_eval_cameras(
        cameras, camera_id=0, count=3
    )

    assert [camera.cam_id for camera in selected] == [0, 0, 0]
    assert [camera.time for camera in selected] == [0.0, 2.0, 4.0]


def test_gt_render_grid_places_gt_left_and_render_right():
    gt_large = torch.ones(3, 2, 2)
    render_large = torch.full((3, 2, 2), 2.0)
    gt_small = torch.full((3, 1, 1), 3.0)
    render_small = torch.full((3, 1, 1), 4.0)

    grid = train._build_gt_render_grid(
        ((gt_large, render_large), (gt_small, render_small))
    )

    assert tuple(grid.shape) == (3, 3, 4)
    assert torch.all(grid[:, :2, :2] == 1.0)
    assert torch.all(grid[:, :2, 2:] == 2.0)
    assert torch.all(grid[:, 2:, :1] == 3.0)
    assert torch.all(grid[:, 2:, 1:2] == 4.0)
    assert torch.all(grid[:, 2:, 2:] == 0.0)

def test_selective_wandb_writer_allows_only_fixed_preview(monkeypatch):
    class RecordingWriter:
        def __init__(self):
            self.images = []
            self.scalars = []

        def add_image(self, *args, **kwargs):
            self.images.append((args, kwargs))

        def add_scalar(self, *args, **kwargs):
            self.scalars.append((args, kwargs))

    class RecordingRun:
        def __init__(self):
            self.logs = []

        def log(self, payload):
            self.logs.append(payload)

    fake_wandb = SimpleNamespace(
        Image=lambda value: SimpleNamespace(value=value.copy())
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    underlying = RecordingWriter()
    run = RecordingRun()
    writer = train._SelectiveWandbMediaWriter(underlying, run)

    writer.add_scalar("loss", 1.0, 500)
    writer.add_image("test_view/render", torch.ones(3, 2, 4), 500)
    writer.add_image(
        train.WANDB_EVAL_MEDIA_KEY, torch.ones(3, 2, 4), 500
    )

    assert len(underlying.scalars) == 1
    assert underlying.images == []
    assert len(run.logs) == 1
    assert set(run.logs[0]) == {
        train.WANDB_EVAL_STEP_KEY,
        train.WANDB_EVAL_MEDIA_KEY,
    }
    assert run.logs[0][train.WANDB_EVAL_STEP_KEY] == 500
    assert run.logs[0][train.WANDB_EVAL_MEDIA_KEY].value.shape == (2, 4, 3)



def test_preview_report_logs_gt_render_grid_and_metrics(monkeypatch):
    monkeypatch.setenv("WANDB_ENABLED", "1")
    monkeypatch.setenv("WANDB_EVAL_INTERVAL", "500")
    monkeypatch.setenv("WANDB_EVAL_IMAGE_COUNT", "3")
    monkeypatch.setenv("WANDB_EVAL_CAMERA_ID", "0")
    monkeypatch.setenv("WANDB_EVAL_LPIPS", "1")

    cameras = [
        _camera(0, 0, 0.0),
        _camera(1, 0, 1.0),
        _camera(2, 0, 2.0),
    ]
    for index, camera in enumerate(cameras):
        camera.original_image = torch.full(
            (3, 4, 5), 0.1 * (index + 1)
        )

    scene = SimpleNamespace(
        gaussians=object(),
        env_map=object(),
        getTestCameras=lambda: cameras,
    )

    class RecordingWriter:
        def __init__(self):
            self.images = {}
            self.scalars = {}
            self.flushed = False

        def add_image(self, name, image, global_step):
            self.images[name] = (image.clone(), global_step)

        def add_scalar(self, name, value, step):
            self.scalars[name] = (value, step)

        def flush(self):
            self.flushed = True

    writer = RecordingWriter()
    lpips_inputs = []

    def fake_lpips(predicted, ground_truth):
        lpips_inputs.append((predicted.clone(), ground_truth.clone()))
        return torch.tensor(0.25)

    monkeypatch.setattr(train, "_get_lpips_metric", lambda device: fake_lpips)

    def fake_render(viewpoint, gaussians, env_map):
        del gaussians, env_map
        return {"render": 1.0 - viewpoint.original_image}

    train._report_wandb_preview(
        writer, 500, 60000, scene, fake_render, ()
    )

    grid, step = writer.images["Eval Images/fixed_front_gt_render"]
    assert tuple(grid.shape) == (3, 12, 10)
    assert step == 500
    assert torch.allclose(grid[:, :4, :5], cameras[0].original_image)
    assert torch.allclose(grid[:, :4, 5:], 1.0 - cameras[0].original_image)
    assert writer.flushed
    assert torch.allclose(
        lpips_inputs[0][0], (1.0 - cameras[0].original_image)[None] * 2.0 - 1.0
    )
    assert torch.allclose(
        lpips_inputs[0][1], cameras[0].original_image[None] * 2.0 - 1.0
    )
    for name in (
        "Eval Images Metrics/fixed_front_l1",
        "Eval Images Metrics/fixed_front_psnr",
        "Eval Images Metrics/fixed_front_ssim",
        "Eval Images Metrics/fixed_front_lpips",
        "Eval Images Metrics/fixed_front_render_fps",
    ):
        assert name in writer.scalars
        assert writer.scalars[name][1] == 500


def test_wandb_preview_interval_and_final_step(monkeypatch):
    monkeypatch.setenv("WANDB_ENABLED", "1")
    monkeypatch.setenv("WANDB_EVAL_INTERVAL", "500")

    assert not train._should_log_wandb_preview(499, 60000)
    assert train._should_log_wandb_preview(500, 60000)
    assert not train._should_log_wandb_preview(501, 60000)
    assert train._should_log_wandb_preview(60000, 60000)


def test_wandb_preview_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_ENABLED", "1")
    monkeypatch.setenv("WANDB_EVAL_INTERVAL", "0")

    assert not train._should_log_wandb_preview(500, 60000)


def test_wandb_scalar_interval_and_final_step(monkeypatch):
    monkeypatch.setenv("WANDB_SCALAR_LOG_INTERVAL", "10")

    assert train._should_log_wandb_scalars(1, 60000)
    assert not train._should_log_wandb_scalars(9, 60000)
    assert train._should_log_wandb_scalars(10, 60000)
    assert train._should_log_wandb_scalars(60000, 60000)

    monkeypatch.setenv("WANDB_SCALAR_LOG_INTERVAL", "0")
    assert not train._should_log_wandb_scalars(10, 60000)


@pytest.mark.parametrize("value", ["-1", "not-an-integer"])
def test_invalid_wandb_interval_is_rejected(monkeypatch, value):
    monkeypatch.setenv("WANDB_ENABLED", "1")
    monkeypatch.setenv("WANDB_EVAL_INTERVAL", value)

    with pytest.raises(ValueError):
        train._should_log_wandb_preview(500, 60000)

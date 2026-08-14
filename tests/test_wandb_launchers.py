import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "launcher,scene,gpu,project,dataset_type",
    [
        (
            "scripts/train_nuscenes_splatad.sh",
            "scene-0101",
            "4",
            "SplatAD_nuScenes_AD-GS",
            "nuScenes",
        ),
        (
            "scripts/train_waymo_splatad.sh",
            "4986495627634617319_2980_000_3000_000",
            "0",
            "SplatAD_Waymo_AD-GS",
            "Waymo",
        ),
        (
            "scripts/train_av2_splatad.sh",
            "a7bcdabb-f9b7-3c16-806d-3ddf1c2d49a2",
            "5",
            "SplatAD_Argoverse2_AD-GS",
            "Argoverse2",
        ),
    ],
)
def test_launcher_wandb_convention(
    launcher, scene, gpu, project, dataset_type
):
    env = os.environ.copy()
    for name in (
        "WANDB_ENABLED",
        "WANDB_MODE",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        "WANDB_SPLIT_TYPE",
        "WANDB_DATASET_TYPE",
        "WANDB_MODEL_NAME",
        "WANDB_RUN_NAME_PREFIX",
        "WANDB_EVAL_INTERVAL",
        "WANDB_EVAL_IMAGE_COUNT",
        "WANDB_EVAL_CAMERA_ID",
        "WANDB_EVAL_LPIPS",
        "WANDB_SCALAR_LOG_INTERVAL",
        "ADGS_CONSOLE_LOG_INTERVAL",
    ):
        env.pop(name, None)
    env.update(DRY_RUN="1", RUN_RENDER="0", ONLY_SCENE=scene)

    result = subprocess.run(
        ["bash", launcher],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    command = result.stdout
    assert "CUDA_VISIBLE_DEVICES={}".format(gpu) in command
    assert "WANDB_ENABLED=1" in command
    assert "WANDB_MODE=online" in command
    assert "WANDB_ENTITY=CamoSplat_ICLR_2027" in command
    assert "WANDB_PROJECT={}".format(project) in command
    assert "WANDB_NAME={}".format(scene) in command
    assert "WANDB_RUN_GROUP={}".format(project) in command
    assert "WANDB_SPLIT_TYPE=SplatAD" in command
    assert "WANDB_DATASET_TYPE={}".format(dataset_type) in command
    assert "WANDB_MODEL_NAME=AD-GS" in command
    assert "WANDB_EVAL_INTERVAL=500" in command
    assert "WANDB_EVAL_IMAGE_COUNT=3" in command
    assert "WANDB_EVAL_CAMERA_ID=0" in command
    assert "WANDB_EVAL_LPIPS=1" in command
    assert "WANDB_SCALAR_LOG_INTERVAL=10" in command
    assert "ADGS_CONSOLE_LOG_INTERVAL=100" in command
    assert "ADGS_SCENE_NAME={}".format(scene) in command
    assert "ADGS_PHYSICAL_GPU={}".format(gpu) in command


def test_waymo_launcher_runs_every_scene_on_gpu_zero_in_order():
    scenes = [
        "4986495627634617319_2980_000_3000_000",
        "4672649953433758614_2700_000_2720_000",
        "6791933003490312185_2607_000_2627_000",
        "17364342162691622478_780_000_800_000",
        "3385534893506316900_4252_000_4272_000",
        "9747453753779078631_940_000_960_000",
        "14940138913070850675_5755_330_5775_330",
        "204421859195625800_1080_000_1100_000",
        "7566697458525030390_1440_000_1460_000",
        "17159836069183024120_640_000_660_000",
    ]
    env = os.environ.copy()
    env.pop("ONLY_SCENE", None)
    env.update(DRY_RUN="1", RUN_RENDER="0")

    result = subprocess.run(
        ["bash", "scripts/train_waymo_splatad.sh"],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    command_lines = [
        line for line in result.stdout.splitlines() if " train.py " in line
    ]
    assert "GPUs:         0 (sequential)" in result.stdout
    assert len(command_lines) == len(scenes)
    assert all("CUDA_VISIBLE_DEVICES=0" in line for line in command_lines)
    assert all("ADGS_PHYSICAL_GPU=0" in line for line in command_lines)
    assert [
        next(scene for scene in scenes if scene in line)
        for line in command_lines
    ] == scenes


def test_launcher_wandb_override_is_preserved():
    env = os.environ.copy()
    env.update(
        DRY_RUN="1",
        RUN_RENDER="0",
        ONLY_SCENE="scene-0101",
        WANDB_PROJECT="custom-project",
        WANDB_MODE="offline",
        WANDB_RUN_NAME_PREFIX="debug-",
        WANDB_EVAL_INTERVAL="250",
        WANDB_EVAL_IMAGE_COUNT="1",
        ADGS_CONSOLE_LOG_INTERVAL="25",
    )
    result = subprocess.run(
        ["bash", "scripts/train_nuscenes_splatad.sh"],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    command = result.stdout
    assert "WANDB_PROJECT=custom-project" in command
    assert "WANDB_MODE=offline" in command
    assert "WANDB_NAME=debug-scene-0101" in command
    assert "WANDB_EVAL_INTERVAL=250" in command
    assert "WANDB_EVAL_IMAGE_COUNT=1" in command
    assert "ADGS_CONSOLE_LOG_INTERVAL=25" in command


def test_launcher_streams_live_status_and_persists_log(tmp_path):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "echo fake-stdout\n"
        "echo fake-stderr >&2\n"
    )
    fake_python.chmod(0o755)
    config = tmp_path / "config.py"
    config.write_text("# test config\n")
    processed = tmp_path / "processed"
    output = tmp_path / "output"

    env = os.environ.copy()
    env.pop("ONLY_SCENE", None)
    env.update(
        ADGS_PYTHON=str(fake_python),
        WANDB_ENABLED="0",
        RUN_RENDER="0",
    )
    result = subprocess.run(
        [
            "bash",
            "scripts/train_splatad_split.sh",
            "nuscenes",
            str(config),
            str(processed),
            str(output),
            "4",
            "5",
            "scene-test",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    console = result.stdout
    log = (output / "scene-test" / "launcher.log").read_text()
    for expected in (
        "[TRAIN START]",
        "[TRAIN DONE]",
        "[SCENE DONE]",
        "[BATCH DONE]",
        "fake-stdout",
        "fake-stderr",
    ):
        assert expected in console
        if expected != "[BATCH DONE]":
            assert expected in log


def test_launcher_aggregates_all_incomplete_scenes(tmp_path):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"scripts/validate_splatad_scene.py\" ]]; then\n"
        "    if [[ \"$2\" == *scene-ready ]]; then exit 0; fi\n"
        "    echo \"ERROR: incomplete $2\" >&2\n"
        "    exit 1\n"
        "fi\n"
        "echo training-must-not-start >&2\n"
        "exit 99\n"
    )
    fake_python.chmod(0o755)
    config = tmp_path / "config.py"
    config.write_text("# test config\n")
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "scene-staged" / ".adgs-priors-work").mkdir(parents=True)

    env = os.environ.copy()
    env.pop("ONLY_SCENE", None)
    env.update(ADGS_PYTHON=str(fake_python), RUN_RENDER="0")
    result = subprocess.run(
        [
            "bash",
            "scripts/train_splatad_split.sh",
            "waymo",
            str(config),
            str(processed),
            str(tmp_path / "output"),
            "0",
            "0",
            "scene-ready",
            "scene-staged",
            "scene-missing",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    assert result.returncode == 1
    assert "[READY 1/3] scene-ready" in result.stdout
    assert "[NEEDS_PRIORS 2/3] scene-staged" in result.stdout
    assert "[NEEDS_PRIORS 3/3] scene-missing" in result.stdout
    assert "[VALIDATION FAILED] 2/3 scene(s)" in result.stdout
    assert "RESUME=1 bash scripts/prepare_splatad_priors.sh" in result.stdout
    assert "never deletes it or enables OVERWRITE automatically" in result.stdout
    assert "training-must-not-start" not in result.stdout


@pytest.mark.parametrize(
    "launcher,gpu",
    [
        ("scripts/run_nuscenes_preprocess_then_train.sh", "4"),
        ("scripts/run_av2_preprocess_then_train.sh", "5"),
    ],
)
def test_preprocess_launcher_uses_gpu_scoped_prior_lock(launcher, gpu):
    env = os.environ.copy()
    for name in (
        "ADGS_PRIOR_BUILDER_LOCK",
        "NUSCENES_PIPELINE_DRY_RUN",
        "AV2_PIPELINE_DRY_RUN",
    ):
        env.pop(name, None)
    env.update(
        PIPELINE_DRY_RUN="1",
        NUSCENES_PIPELINE_GPU="4",
        AV2_PIPELINE_GPU="5",
    )

    result = subprocess.run(
        ["bash", launcher],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    expected_lock = REPO_ROOT / "output" / (
        ".adgs-prior-builder.gpu-{}.lock".format(gpu)
    )
    assert "GPU lock:     {} (not acquired)".format(
        expected_lock
    ) in result.stdout

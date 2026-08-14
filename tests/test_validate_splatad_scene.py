from pathlib import Path

from scripts.validate_splatad_scene import (
    _missing_training_artifacts,
    _training_readiness_error,
)


def test_missing_training_artifacts_reports_full_prior_pipeline(tmp_path):
    scene = Path(tmp_path)
    missing = _missing_training_artifacts(scene, {"x", "y", "z", "t"})

    assert missing == [
        "points3d.ply[obj]",
        "depth/",
        "semantic/",
        "sky/",
        "flow/",
        "colmap.ply",
    ]
    message = _training_readiness_error(scene, "waymo", missing)
    assert "full prior pipeline" in message
    assert "must not be zero-filled" in message
    assert "scripts/prepare_splatad_priors.sh waymo" in message


def test_training_readiness_error_mentions_interrupted_staging(tmp_path):
    scene = Path(tmp_path)
    (scene / ".adgs-priors-work").mkdir()

    message = _training_readiness_error(
        scene, "waymo", ["points3d.ply[obj]"]
    )

    assert "Interrupted staging exists" in message
    assert "explicit OVERWRITE=1" in message

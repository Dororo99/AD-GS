import numpy as np
import pytest

from scripts.flow import _preflight_flow_resume
from scripts.prior_storage import save_flow_prior


def _valid_flow(target_times, shape=(2, 3)):
    flow = np.empty((len(target_times), 6), dtype=object)
    for index, target_time in enumerate(target_times):
        flow[index] = [
            target_time,
            np.eye(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros((2,) + shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
        ]
    return flow


def test_preflight_accepts_valid_completed_and_empty_flows(tmp_path, capsys):
    flow_folder = tmp_path / "flow"
    flow_folder.mkdir()
    specs = {
        "000000": {"shape": (2, 3), "target_times": (1.5,)},
        "000001": {"shape": (2, 3), "target_times": (0.0, 2.0)},
        "000002": {"shape": (2, 3), "target_times": (1.0,)},
    }
    save_flow_prior(flow_folder / "000000.npz", _valid_flow((1.5,)))
    save_flow_prior(
        flow_folder / "000001.npz", np.asarray([], dtype=object)
    )

    ready = _preflight_flow_resume(str(flow_folder), specs)

    assert ready == {"000000", "000001"}
    assert (
        "[RESUME][FLOW PREFLIGHT] ready=2 missing=1 total=3"
        in capsys.readouterr().out
    )


def test_preflight_rejects_archive_without_flow_key(tmp_path):
    flow_folder = tmp_path / "flow"
    flow_folder.mkdir()
    np.savez_compressed(flow_folder / "000000.npz", wrong=np.asarray([1]))
    specs = {
        "000000": {"shape": (2, 3), "target_times": (1.5,)},
    }

    with pytest.raises(ValueError, match="Cannot resume from existing flow"):
        _preflight_flow_resume(str(flow_folder), specs)


@pytest.mark.parametrize(
    "field,value,error",
    [
        (0, 9.0, "targets"),
        (4, np.zeros((2, 4, 3), dtype=np.float32), "coordinates"),
        (5, np.zeros((4, 3), dtype=np.float32), "visibility"),
    ],
)
def test_preflight_rejects_incompatible_flow_payload(
    tmp_path, field, value, error
):
    flow_folder = tmp_path / "flow"
    flow_folder.mkdir()
    flow = _valid_flow((1.5,))
    flow[0, field] = value
    save_flow_prior(flow_folder / "000000.npz", flow)
    specs = {
        "000000": {"shape": (2, 3), "target_times": (1.5,)},
    }

    with pytest.raises(ValueError, match=error):
        _preflight_flow_resume(str(flow_folder), specs)


def test_preflight_rejects_unexpected_or_duplicate_stems(tmp_path):
    specs = {
        "000000": {"shape": (2, 3), "target_times": (1.5,)},
    }

    unexpected_folder = tmp_path / "unexpected"
    unexpected_folder.mkdir()
    save_flow_prior(
        unexpected_folder / "999999.npz", np.asarray([], dtype=object)
    )
    with pytest.raises(ValueError, match="non-training/unknown"):
        _preflight_flow_resume(str(unexpected_folder), specs)

    duplicate_folder = tmp_path / "duplicate"
    duplicate_folder.mkdir()
    save_flow_prior(
        duplicate_folder / "000000.npz", np.asarray([], dtype=object)
    )
    np.save(
        duplicate_folder / "000000.npy", np.asarray([], dtype=object)
    )
    with pytest.raises(ValueError, match="duplicate NPY/NPZ stems"):
        _preflight_flow_resume(str(duplicate_folder), specs)

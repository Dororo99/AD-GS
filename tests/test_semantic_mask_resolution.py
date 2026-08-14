import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SEMANTIC = REPO_ROOT / "Grounded-SAM-2" / "semantic.py"
TEMPLATE_SEMANTIC = REPO_ROOT / "scripts" / "semantic.py"
MASK_MODEL = (
    REPO_ROOT / "Grounded-SAM-2" / "utils" / "mask_dictionary_model.py"
)


def _load_mask_model_module(monkeypatch):
    class FakeTensor:
        def __init__(self, array):
            self._array = array

        def numpy(self):
            return self._array

    fake_torch = types.ModuleType("torch")
    fake_torch.zeros = lambda shape: FakeTensor(np.zeros(shape))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))

    spec = importlib.util.spec_from_file_location(
        "grounded_sam_mask_dictionary_model", MASK_MODEL
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_empty_mask_uses_explicit_frame_dimensions(tmp_path, monkeypatch):
    module = _load_mask_model_module(monkeypatch)
    mask_dir = tmp_path / "masks"
    json_dir = tmp_path / "json"
    mask_dir.mkdir()
    json_dir.mkdir()

    mask_model = module.MaskDictionaryModel(
        mask_name="mask_000002.npy",
        mask_height=1280,
        mask_width=1920,
    )
    mask_model.save_empty_mask_and_json(mask_dir, json_dir)

    mask = np.load(mask_dir / "mask_000002.npy")
    assert mask.shape == (1280, 1920)
    assert not mask.any()


def test_semantic_scripts_wire_actual_frame_dimensions_into_masks():
    active_source = ACTIVE_SEMANTIC.read_text()
    template_source = TEMPLATE_SEMANTIC.read_text()
    assert active_source == template_source
    compile(active_source, str(ACTIVE_SEMANTIC), "exec")

    for expected in (
        "mask_height=first_frame_height",
        "mask_height=frame_height",
        "mask_height=empty_frame_height",
        "mask_height=out_frame_height",
        "SAM tracking mask/image size mismatch",
    ):
        assert expected in active_source

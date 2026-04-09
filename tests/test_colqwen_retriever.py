from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.retrieval.colqwen_retriever import ColQwenRetriever


class _FakeModel:
    def eval(self):
        return self

    def to(self, _device):
        return self


class _FakePeftWrapper(_FakeModel):
    def __init__(self, base_model):
        self.base_model = base_model


class _FakeColQwenModel:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return _FakeModel()


class _FakeProcessor:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls()


class _FakePeftModel:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def from_pretrained(cls, base_model, adapter_path, **kwargs):
        cls.calls.append((base_model, adapter_path, kwargs))
        return _FakePeftWrapper(base_model)


class ColQwenRetrieverRuntimeTests(unittest.TestCase):
    def setUp(self):
        _FakeColQwenModel.reset()
        _FakeProcessor.reset()
        _FakePeftModel.reset()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_snapshot_loads_directly(self):
        model_dir = self._make_complete_model_dir("colqwen-complete")
        retriever = self._build_retriever(model_dir)

        with self._patch_runtime_imports():
            retriever._ensure_engine_runtime()

        self.assertEqual(_FakeColQwenModel.calls[0][0], str(model_dir))
        self.assertEqual(_FakeProcessor.calls[0][0], str(model_dir))
        self.assertEqual(_FakePeftModel.calls, [])

    def test_adapter_uses_explicit_base_model_name(self):
        adapter_dir = self._make_adapter_dir(
            "colqwen-adapter",
            base_model_name_or_path=str(self.root / "ignored-base"),
            with_processor_assets=True,
        )
        base_dir = self._make_complete_model_dir("colqwen-base-explicit")
        retriever = self._build_retriever(adapter_dir, base_model_name=str(base_dir))

        with self._patch_runtime_imports():
            retriever._ensure_engine_runtime()

        self.assertEqual(_FakeColQwenModel.calls[0][0], str(base_dir))
        self.assertEqual(_FakePeftModel.calls[0][1], str(adapter_dir))
        self.assertEqual(_FakeProcessor.calls[0][0], str(adapter_dir))

    def test_adapter_falls_back_to_adapter_config_base_model_name(self):
        base_dir = self._make_complete_model_dir("colqwen-base-fallback")
        adapter_dir = self._make_adapter_dir(
            "colqwen-adapter",
            base_model_name_or_path=str(base_dir),
            with_processor_assets=False,
        )
        retriever = self._build_retriever(adapter_dir, base_model_name=None)

        with self._patch_runtime_imports():
            retriever._ensure_engine_runtime()

        self.assertEqual(_FakeColQwenModel.calls[0][0], str(base_dir))
        self.assertEqual(_FakePeftModel.calls[0][1], str(adapter_dir))
        self.assertEqual(_FakeProcessor.calls[0][0], str(base_dir))

    def test_adapter_without_any_base_model_path_raises_clear_error(self):
        adapter_dir = self._make_adapter_dir(
            "colqwen-adapter",
            base_model_name_or_path=None,
            with_processor_assets=True,
        )
        retriever = self._build_retriever(adapter_dir, base_model_name=None)

        with self.assertRaisesRegex(ValueError, "no base model path is available"):
            retriever._resolve_runtime_paths()

    def test_adapter_with_missing_local_base_directory_raises_file_not_found(self):
        adapter_dir = self._make_adapter_dir(
            "colqwen-adapter",
            base_model_name_or_path=str(self.root / "missing-base"),
            with_processor_assets=True,
        )
        retriever = self._build_retriever(adapter_dir, base_model_name=None)

        with self.assertRaisesRegex(FileNotFoundError, "base model directory not found"):
            retriever._resolve_runtime_paths()

    def _build_retriever(self, model_dir: Path, base_model_name: str | None = None) -> ColQwenRetriever:
        cfg = types.SimpleNamespace(
            visual_colqwen_retrieval=types.SimpleNamespace(
                model_name=str(model_dir),
                base_model_name=base_model_name,
                local_files_only=True,
                device="cpu",
                batch_size=2,
                max_pages_per_doc=256,
            )
        )
        return ColQwenRetriever(cfg)

    def _make_complete_model_dir(self, name: str) -> Path:
        model_dir = self.root / name
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return model_dir

    def _make_adapter_dir(
        self,
        name: str,
        base_model_name_or_path: str | None,
        with_processor_assets: bool,
    ) -> Path:
        adapter_dir = self.root / name
        adapter_dir.mkdir(parents=True)
        adapter_config = {}
        if base_model_name_or_path is not None:
            adapter_config["base_model_name_or_path"] = base_model_name_or_path
        (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
        if with_processor_assets:
            (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return adapter_dir

    def _patch_runtime_imports(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float32 = "float32"

        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = _FakePeftModel

        fake_import_utils = types.ModuleType("transformers.utils.import_utils")
        fake_import_utils.is_flash_attn_2_available = lambda: False

        fake_transformers_utils = types.ModuleType("transformers.utils")
        fake_transformers_utils.import_utils = fake_import_utils

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.utils = fake_transformers_utils
        fake_transformers.__version__ = "4.46.0"

        fake_colpali_engine = types.ModuleType("colpali_engine")
        fake_colpali_engine.__version__ = "0.3.7"

        fake_colpali_engine_models = types.ModuleType("colpali_engine.models")
        fake_colpali_engine_models.ColQwen2_5 = _FakeColQwenModel
        fake_colpali_engine_models.ColQwen2_5_Processor = _FakeProcessor

        return mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "peft": fake_peft,
                "transformers": fake_transformers,
                "transformers.utils": fake_transformers_utils,
                "transformers.utils.import_utils": fake_import_utils,
                "colpali_engine": fake_colpali_engine,
                "colpali_engine.models": fake_colpali_engine_models,
            },
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import ssntagger as tagger


class TaggerTests(unittest.TestCase):
    def test_recursive_by_default_and_opt_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").touch()
            nested = root / "sub" / "nested"
            nested.mkdir(parents=True)
            (nested / "b.jpg").touch()
            (nested / "notes.txt").touch()
            args = tagger.parse_args([tmp])
            self.assertTrue(args.recursive)
            self.assertEqual(len(tagger.list_images(root, args.recursive)), 2)
            args = tagger.parse_args([tmp, "--no-recursive"])
            self.assertFalse(args.recursive)
            self.assertEqual(tagger.list_images(root, args.recursive), [root / "a.png"])
            self.assertTrue(tagger.parse_args([tmp, "--recursive"]).recursive)

    def test_local_runtime_options_and_single_model_load(self):
        response = {"choices": [{"message": {"content": json.dumps({
            "caption": {k: k for k in tagger.CAPTION_KEYS}})}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 20}}
        for gpu_layers in (0, -1):
            with self.subTest(gpu_layers=gpu_layers), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = root / "Gemma4.gguf"
                model.touch()
                (root / "mmproj.gguf").touch()
                (root / "a.png").touch()
                (root / "b.png").touch()
                factory = MagicMock()
                factory.return_value.create_chat_completion.return_value = response
                handler = MagicMock()
                with patch.dict("sys.modules", {
                    "llama_cpp": SimpleNamespace(Llama=factory),
                    "llama_cpp.llama_chat_format": SimpleNamespace(Gemma4ChatHandler=handler),
                }), patch.object(tagger, "encode_image", return_value="data:image/png;base64,a"):
                    self.assertEqual(tagger.main([tmp, "--model", str(model), "--n_gpu_layers", str(gpu_layers),
                                                  "--seed", "42"]), 0)
                factory.assert_called_once()
                self.assertEqual(factory.call_args.kwargs["flash_attn"], gpu_layers != 0)
                self.assertEqual(factory.call_args.kwargs["n_batch"], 1024)
                self.assertEqual(factory.call_args.kwargs["n_ubatch"], 1024)
                self.assertEqual(factory.call_args.kwargs["seed"], 42)
                self.assertEqual(factory.return_value.create_chat_completion.call_count, 2)
                factory.return_value.close.assert_called_once()
                self.assertGreaterEqual(json.loads((root / "a_caption.json").read_text(encoding="utf-8"))["elapsed_seconds"], 0)

    def test_discover_smallest_model_and_projector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = root / "Gemma4-12B.gguf"
            small.write_bytes(b"x")
            (root / "Gemma4-31B.gguf").write_bytes(b"xx")
            projector = root / "mmproj.gguf"
            projector.touch()
            args = tagger.parse_args([tmp, "--model_dir", tmp])
            tagger.resolve_local_models(args)
            self.assertEqual(args.model, str(small))
            self.assertEqual(args.mmproj, projector)
            preferred = root / "gemma-4-31b.Q4_K_S.gguf"
            preferred.write_bytes(b"xxx")
            args = tagger.parse_args([tmp, "--model_dir", tmp])
            tagger.resolve_local_models(args)
            self.assertEqual(args.model, str(preferred))

    def test_generation_skip_and_sidecar_reuse(self):
        response = {"caption": {k: k + " caption" for k in tagger.CAPTION_KEYS},
                    "watermark": [], "borders": ["none"]}
        api_response = {"choices": [{"message": {"content": json.dumps(response)},
                                     "finish_reason": "stop"}], "usage": {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image.png").write_bytes(b"image")
            model = root / "Gemma4.gguf"
            model.touch()
            (root / "mmproj.gguf").touch()
            argv = [tmp, "--model", str(model)]
            with patch.object(tagger, "encode_image", return_value="data:image/png;base64,a"), \
                 patch.object(tagger, "call_chat_api", return_value=api_response) as inference:
                self.assertEqual(tagger.main(argv), 0)
                self.assertEqual(inference.call_count, 1)
                output = (root / "image.txt").read_text(encoding="utf-8")
                self.assertIn("---===--- # long\nlong caption", output)
                self.assertEqual(tagger.main(argv), 0)
                self.assertEqual(inference.call_count, 1)
                self.assertEqual(tagger.main([tmp, "--from_sidecar", "--overwrite",
                                             "--candidates", "short"]), 0)
                self.assertEqual(inference.call_count, 1)
                self.assertEqual((root / "image.txt").read_text(encoding="utf-8"),
                                 "---===--- # short\nshort caption\n")

    def test_local_context_and_concurrency_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "Gemma4.gguf"
            model.touch()
            (root / "mmproj.gguf").touch()
            for option in (["--max_workers", "2"], ["--n_ctx", "100"],
                           ["--n_batch", "1024", "--n_ubatch", "512"],
                           ["--extra_body", '{"unsupported": true}']):
                self.assertEqual(tagger.main([tmp, "--model", str(model), *option]), 2)


if __name__ == "__main__":
    unittest.main()

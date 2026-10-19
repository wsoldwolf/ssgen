import tempfile
import unittest
from pathlib import Path, PureWindowsPath
import contextlib
import io
import sys
import subprocess
import json
from datetime import datetime, date, time, timezone
from unittest.mock import patch

import tomllib

import ssgen


class GenerationTests(unittest.TestCase):
    def write_lora(self, path, ranks=(32,), metadata_rank='999', conv=False):
        header = {'__metadata__': {'ss_network_dim': metadata_rank}}
        payload = bytearray()
        for i, rank in enumerate(ranks):
            for direction, shape in [('down', [rank, 2]), ('up', [2, rank])]:
                if conv:
                    shape += [1, 1]
                start = len(payload)
                payload.extend(b'\x00' * (rank * 2 * 4))
                header[f'lora_unet_{i}.lora_{direction}.weight'] = {
                    'dtype': 'F32', 'shape': shape, 'data_offsets': [start, len(payload)]}
        content = json.dumps(header).encode('utf-8')
        content += b' ' * (-len(content) % 8)
        path.write_bytes(len(content).to_bytes(8, 'little') + content + payload)

    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'generated'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, family='anima', *extra):
        return ssgen.make_parser().parse_args([
            '--family', family, '--name', 'テスト LoRA',
            '--destination', str(self.root / family),
            '--add-dataset', str(self.root / '画像 a'),
            '--add-dataset', str(self.root / '画像 b'),
            '--comfyui-dir', str(self.root / 'ComfyUI'), *extra,
        ])

    def test_all_templates_and_batches(self):
        for family, script in [('sdxl', 'sdxl_train_network.py'),
                               ('anima', 'anima_train_network.py'),
                               ('moss', 'moss_image_train_network.py')]:
            with self.subTest(family=family):
                args = self.args(family, '--model', 'E:/models/base.safetensors')
                files = ssgen.generate(args)
                config = ssgen.read_toml(files[0])
                dataset = ssgen.read_toml(files[1])
                self.assertEqual(config['output_name'], args.name)
                self.assertEqual(config['pretrained_model_name_or_path'], args.model)
                self.assertEqual(config['dataset_config'], str(files[1]))
                self.assertEqual(config['output_dir'], str(files[0].parent / 'LoRA'))
                subsets = dataset['datasets'][0]['subsets']
                self.assertEqual(len(subsets), 2)
                self.assertEqual([s['num_repeats'] for s in subsets], [10, 10])
                train = files[2].read_bytes()
                self.assertIn(script.encode(), train)
                self.assertIn(b'--num_cpu_threads_per_process', train)
                self.assertIn(b'\r\n', train)
                copy = files[3].read_text(encoding='utf-8')
                self.assertIn('/R:60 /W:5', copy)
                self.assertIn('if %SSGEN_EXIT_CODE% LSS 8 set "SSGEN_EXIT_CODE=0"', copy)
                for batch in files[2:]:
                    self.assertTrue(batch.read_text(encoding='utf-8').endswith(':ssgen_finish\npause\nexit /b %SSGEN_EXIT_CODE%\n'))
                self.assertNotIn('/MIR', copy.upper())
                self.assertIn(str((args.comfyui_dir / 'models' / 'loras' / family / args.name).resolve()), copy)
                with self.assertRaises(ValueError):
                    ssgen.generate(args)

    def test_append_preserves_repeats_and_rejects_duplicates(self):
        files = ssgen.generate(self.args('anima', '--repeats', '30'))
        dataset = ssgen.read_toml(files[1])
        ssgen.add_datasets(dataset, [self.root / '追加'], 10)
        subsets = dataset['datasets'][0]['subsets']
        self.assertEqual([s['num_repeats'] for s in subsets], [30, 30, 10])
        with self.assertRaises(ValueError):
            ssgen.add_datasets(dataset, [self.root / '追加'], 10)
        with self.assertRaises(ValueError):
            ssgen.add_datasets(dataset, [self.root / '別'], 10, 99)

    def test_append_cli_only_changes_dataset(self):
        files = ssgen.generate(self.args())
        before = {p: p.read_bytes() for p in files if p.name != 'dataset.toml'}
        with patch.object(sys, 'argv', ['ssgen', '--destination', str(files[0].parent),
                                       '--add-dataset', str(self.root / '追加')]), contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        self.assertEqual(len(ssgen.read_toml(files[1])['datasets'][0]['subsets']), 3)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_template_preservation_and_override(self):
        args = self.args('anima', '--set', 'network_dim=8', '--keep-template-datasets')
        source = args.templates_dir / 'anima' / 'config.toml'
        before = source.read_bytes()
        files = ssgen.generate(args)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(ssgen.read_toml(files[0])['qwen3'], ssgen.read_toml(source)['qwen3'])
        self.assertEqual(ssgen.read_toml(files[0])['network_dim'], 8)
        self.assertGreater(len(ssgen.read_toml(files[1])['datasets'][0]['subsets']), 2)

    def test_quoting_and_nested_keys(self):
        self.assertEqual(ssgen.bat_quote('E:/100%/a! b'), '"E:/100%%/a! b"')
        with self.assertRaises(ValueError):
            ssgen.bat_quote('bad"path')
        config = tomllib.loads('[save]\noutput_name="old" # retained\n')
        ssgen.update_key(config, 'output_name', 'new')
        self.assertEqual(config['save']['output_name'], 'new')
        self.assertEqual(tomllib.loads(ssgen.dump_toml(config)), config)

    def test_invalid_name_and_source_overwrite(self):
        for name in ['../escape', 'CON', 'trailing.', 'bad/name']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                ssgen.generate(self.args('anima', '--name', name))
        with self.assertRaises(ValueError):
            ssgen.generate(self.args('anima', '--destination', str(Path('templates/anima').resolve()), '--force'))

    def test_model_discovery(self):
        expected = []
        for relative in ['anima/direct.safetensors', 'models/anima/sub/nested.ckpt',
                         'models/checkpoints/anima/checkpoint.SAFETENSORS',
                         'models/diffusion_models/anima/diffusion.safetensors']:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            expected.append(path.resolve())
        for relative in ['models/loras/anima/adapter.safetensors',
                         'sdxl/wrong.safetensors', 'anima/readme.txt']:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.assertEqual(set(ssgen.discover_models(self.root, 'anima')), set(expected))
        self.assertEqual(ssgen.discover_models(self.root, 'moss'), [])

    def test_model_selection_outside_comfyui(self):
        args = self.args('moss')
        args.comfyui_dir = PureWindowsPath('C:/Software/ComfyUI')
        inside = args.comfyui_dir / 'models/diffusion_models/moss/local.safetensors'
        outside = PureWindowsPath('E:/ComfyUI/models/diffusion_models/Moss/moss_image_base_bf16.safetensors')
        with patch.object(ssgen, 'discover_models', return_value=[inside, outside]), \
                patch.object(ssgen, 'select_option', return_value=str(outside)) as select:
            ssgen.ask_model(args)
        self.assertEqual(args.model, str(outside))
        choices = select.call_args.args[1]
        self.assertEqual(choices[0], (str(inside.relative_to(args.comfyui_dir)), str(inside)))
        self.assertEqual(choices[1], (str(outside), str(outside)))

    def test_prefix_cli_recursive_and_preserves_existing(self):
        files = ssgen.generate(self.args('anima', '--repeats', '30'))
        root = files[0].parent
        old = root / 'charname_old'
        new = root / 'nested' / 'charname_new'
        for path in [old, new, root / 'other', root / 'not_charname_match']:
            path.mkdir(parents=True)
        (root / 'charname_file.txt').touch()
        dataset = ssgen.read_toml(files[1])
        ssgen.add_datasets(dataset, [old], 40)
        dataset['datasets'][0]['subsets'][-1]['caption_prefix'] = 'keep me'
        files[1].write_text(ssgen.dump_toml(dataset), encoding='utf-8')
        before = {p: p.read_bytes() for p in files if p != files[1]}
        command = ['ssgen', str(root), '--add-dataset-with-prefix', 'charname_']
        with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        subsets = ssgen.read_toml(files[1])['datasets'][0]['subsets']
        self.assertEqual(len(subsets), 4)
        self.assertEqual(subsets[2]['num_repeats'], 40)
        self.assertEqual(subsets[2]['caption_prefix'], 'keep me')
        self.assertEqual(subsets[3]['image_dir'], str(new))
        self.assertEqual(subsets[3]['num_repeats'], 10)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        # 再実行しても既存ファイルを変更しない。
        previous = files[1].read_bytes()
        with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        self.assertEqual(files[1].read_bytes(), previous)
        command[-1] = 'missing_'
        with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        self.assertEqual(files[1].read_bytes(), previous)

    def test_prefix_repeats_and_other_dataset_duplicate(self):
        files = ssgen.generate(self.args())
        root = files[0].parent
        for name in ['charname_old', 'charname_new']:
            (root / name).mkdir()
        dataset = ssgen.read_toml(files[1])
        entry = {}
        entry['resolution'] = 512
        entry['subsets'] = []
        subset = {}
        subset['image_dir'] = str(root / 'charname_old').replace('\\', '/') + '/'
        subset['num_repeats'] = 50
        entry['subsets'].append(subset)
        dataset['datasets'].append(entry)
        files[1].write_text(ssgen.dump_toml(dataset), encoding='utf-8')
        with patch.object(sys, 'argv', ['ssgen', str(root), '--add-dataset-with-prefix',
                                       'charname_', '--repeats', '20']), contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        dataset = ssgen.read_toml(files[1])
        self.assertEqual(len(dataset['datasets'][0]['subsets']), 3)
        self.assertEqual(dataset['datasets'][0]['subsets'][-1]['num_repeats'], 20)
        self.assertEqual(dataset['datasets'][1]['subsets'][0]['num_repeats'], 50)

    def test_prefix_rejects_empty_and_path(self):
        for prefix in ['', '../char', 'char/name']:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                ssgen.find_datasets_with_prefix(self.root, [prefix])

    def test_standard_library_toml_roundtrip(self):
        value = {
            'windows_path': 'C:\\日本語\\models\\a.safetensors',
            'escaped': 'quote" slash\\ newline\n tab\t control\x00 del\x7f',
            'enabled': True, 'disabled': False, 'integer': -10, 'rate': 1e-5,
            'empty': [], 'empty_table': {},
            'date': date(2026, 10, 19), 'time': time(12, 34, 56, 123456),
            'timestamp': datetime(2026, 10, 19, 1, 2, 3, tzinfo=timezone.utc),
            'quoted.key': {'space key': [1, 'text', False, {'a': 2}]},
            'datasets': [
                {'resolution': [1024, 1024], 'subsets': [
                    {'image_dir': 'a', 'num_repeats': 30, 'extra': {'caption': 'a'}},
                    {'image_dir': 'b', 'num_repeats': 10}]},
                {'resolution': 512, 'subsets': [{'image_dir': 'c', 'num_repeats': 20}]},
            ],
        }
        self.assertEqual(tomllib.loads(ssgen.dump_toml(value)), value)

    def test_all_source_templates_roundtrip(self):
        root = Path(ssgen.__file__).parent / 'templates'
        for path in root.rglob('*.toml'):
            with self.subTest(path=path):
                value = ssgen.read_toml(path)
                self.assertEqual(tomllib.loads(ssgen.dump_toml(value)), value)

    def test_arrow_selection_and_cancel(self):
        choices = [(str(i), i) for i in range(15)]
        with patch.object(ssgen, 'read_key', side_effect=['up', 'enter']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ssgen.select_option('test', choices), 14)
        with patch.object(ssgen, 'read_key', side_effect=['down'] * 12 + ['enter']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ssgen.select_option('test', choices), 12)
        with patch.object(ssgen, 'read_key', return_value='cancel'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                ssgen.select_option('test', choices)

    def test_wizard_main_generates_in_current_directory(self):
        model_dir = self.root / 'models' / 'diffusion_models' / 'anima'
        model_dir.mkdir(parents=True)
        for name in ['a.safetensors', 'b.safetensors']:
            (model_dir / name).touch()
        # ファミリ:Anima、モデル:b、用途:コンセプト、画像パス:後から追加。
        keys = ['down', 'enter', 'down', 'enter', 'down', 'down', 'enter']
        with patch.object(sys, 'argv', ['ssgen']), \
                patch.object(ssgen, 'enable_terminal'), \
                patch.object(ssgen, 'DEFAULT_COMFYUI', self.root), \
                patch.object(ssgen.Path, 'cwd', return_value=self.root), \
                patch.object(ssgen, 'read_key', side_effect=keys), \
                patch('builtins.input', side_effect=['wizard_test', '', '', '']), \
                contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        destination = self.root / 'wizard_test'
        self.assertEqual({p.name for p in destination.iterdir()},
                         {'config.toml', 'dataset.toml', 'train.bat', 'copylora.bat'})
        config = ssgen.read_toml(destination / 'config.toml')
        self.assertEqual(config['pretrained_model_name_or_path'], str(model_dir / 'b.safetensors'))
        self.assertEqual(config['output_name'], 'wizard_test')
        self.assertIn(str(ssgen.DEFAULT_SD_SCRIPTS['anima']), (destination / 'train.bat').read_text(encoding='utf-8'))
        self.assertFalse(ssgen.read_toml(destination / 'dataset.toml')['datasets'][0].get('subsets'))
        # 未登録で生成した設定にも通常の追記ができる。
        dataset = ssgen.read_toml(destination / 'dataset.toml')
        ssgen.add_datasets(dataset, [self.root / 'later'], 10)
        self.assertEqual(dataset['datasets'][0]['subsets'][0]['num_repeats'], 10)

    def test_slider_selection_requires_images(self):
        args = self.args()
        with patch.object(ssgen, 'select_option', return_value='slider'):
            ssgen.ask_purpose(args)
        self.assertEqual(args.kind, 'slider')
        with self.assertRaisesRegex(ValueError, 'a.png'):
            ssgen.generate(args)

    def slider_args(self, family='anima', *extra):
        args = self.args(family, '--kind', 'slider', '--python', sys.executable, *extra)
        args.destination.mkdir(parents=True, exist_ok=True)
        for side in ('a', 'b'):
            (args.destination / f'{side}.png').write_bytes(side.encode())
        return args

    def test_slider_two_independent_1000_step_jobs(self):
        for family in ('sdxl', 'anima', 'moss'):
            with self.subTest(family=family):
                args = self.slider_args(family, '--set', 'max_train_epochs=4', '--set', 'save_every_n_epochs=1',
                                        '--set', 'max_train_steps=20', '--set', 'save_last_n_epochs=1')
                (args.destination / 'a.txt').write_text('caption A', encoding='utf-8')
                files = ssgen.generate(args)
                for side in ('a', 'b'):
                    config = ssgen.read_toml(args.destination / f'config_{side}.toml')
                    self.assertEqual(config['output_name'], side)
                    self.assertEqual(config['max_train_epochs'], 100)
                    self.assertEqual(config['save_every_n_epochs'], 10)
                    self.assertEqual(config['gradient_accumulation_steps'], 1)
                    self.assertNotIn('max_train_steps', config)
                    self.assertNotIn('save_last_n_epochs', config)
                    self.assertEqual(config['output_dir'], str(args.destination / 'LoRA.Temp'))
                    dataset = ssgen.read_toml(Path(config['dataset_config']))
                    self.assertEqual(len(dataset['datasets']), 1)
                    self.assertEqual(dataset['datasets'][0]['batch_size'], 1)
                    subsets = dataset['datasets'][0]['subsets']
                    self.assertEqual(len(subsets), 1)
                    self.assertEqual(subsets[0]['num_repeats'], 10)
                    image_dir = Path(subsets[0]['image_dir'])
                    self.assertEqual((image_dir / f'{side}.png').read_bytes(), side.encode())
                    self.assertEqual(subsets[0]['class_tokens'], args.name)
                    self.assertIn('"--num_processes" "1"', (args.destination / f'train_{side}.bat').read_text(encoding='utf-8'))
                self.assertEqual((args.destination / 'slider_data/a/a.txt').read_text(encoding='utf-8'), 'caption A')
                self.assertFalse((args.destination / 'merge.bat').exists())
                self.assertIn('train_a.bat', (args.destination / 'train.bat').read_text(encoding='utf-8'))
                self.assertIn('train_b.bat', (args.destination / 'train.bat').read_text(encoding='utf-8'))
                copy = (args.destination / 'copylora.bat').read_text(encoding='utf-8')
                self.assertIn(f'"{args.name}.safetensors"', copy)
                self.assertNotIn('*.safetensors', copy)
                self.assertNotIn('LoRA.Temp', copy)
                with self.assertRaises(ValueError):
                    ssgen.generate(args)

    def slider_checkpoints_fixture(self):
        args = self.slider_args()
        ssgen.generate(args)
        temp = args.destination / 'LoRA.Temp'
        temp.mkdir()
        for name in ['a-000030.safetensors', 'a-000090.safetensors', 'a.safetensors',
                     'b-000060.safetensors', 'b-000080.safetensors', 'b.safetensors']:
            self.write_lora(temp / name)
        return args, temp

    def test_slider_merge_epochs_and_final_epoch(self):
        args, temp = self.slider_checkpoints_fixture()
        merge = ssgen.make_parser().parse_args(['--merge-slider', str(args.destination), '--epoch-a', '90', '--epoch-b', '60'])
        with patch.object(ssgen, 'enable_terminal') as terminal:
            paths = ssgen.generate_slider_merge(merge)
        terminal.assert_not_called()
        text = paths[0].read_text(encoding='utf-8')
        self.assertIn(str(temp / 'a-000090.safetensors'), text)
        self.assertIn(str(temp / 'b-000060.safetensors'), text)
        self.assertIn('"--ratios" "-1" "1"', text)
        self.assertIn('"--new_rank" "32" "--device" "cuda" "--save_precision" "bf16"', text)
        self.assertIn('svd_merge_lora.py', text)
        self.assertIn('\npause\n', text)
        merge.epoch_a = merge.epoch_b = 100
        paths = ssgen.generate_slider_merge(merge)
        self.assertIn(str(temp / 'a.safetensors'), paths[0].read_text(encoding='utf-8'))
        self.assertIn(str(temp / 'b.safetensors'), paths[0].read_text(encoding='utf-8'))
        merge.epoch_a = 95
        previous = paths[0].read_bytes()
        with self.assertRaisesRegex(ValueError, '95'):
            ssgen.generate_slider_merge(merge)
        self.assertEqual(paths[0].read_bytes(), previous)

    def test_slider_merge_interactive_and_partial_epoch(self):
        args, temp = self.slider_checkpoints_fixture()
        merge = ssgen.make_parser().parse_args(['--merge-slider', str(args.destination)])
        with patch.object(ssgen, 'enable_terminal'), \
                patch.object(ssgen, 'read_key', side_effect=['down', 'enter', 'enter']), \
                contextlib.redirect_stdout(io.StringIO()):
            paths = ssgen.generate_slider_merge(merge)
        text = paths[0].read_text(encoding='utf-8')
        self.assertIn(str(temp / 'a-000090.safetensors'), text)
        self.assertIn(str(temp / 'b-000060.safetensors'), text)
        merge.epoch_a = 30
        with patch.object(ssgen, 'enable_terminal'), \
                patch.object(ssgen, 'select_option', return_value=temp / 'b.safetensors') as select:
            ssgen.generate_slider_merge(merge)
        self.assertEqual(select.call_count, 1)

    def test_slider_merge_uses_larger_tensor_rank(self):
        args, temp = self.slider_checkpoints_fixture()
        merge = ssgen.make_parser().parse_args(['--merge-slider', str(args.destination), '--epoch-a', '100', '--epoch-b', '100'])
        for a_rank, b_rank, expected in [(16, 64, 64), (64, 16, 64), (16, 16, 16)]:
            with self.subTest(a=a_rank, b=b_rank):
                self.write_lora(temp / 'a.safetensors', ranks=(8, a_rank))
                self.write_lora(temp / 'b.safetensors', ranks=(b_rank,), conv=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    paths = ssgen.generate_slider_merge(merge)
                self.assertIn(f'"--new_rank" "{expected}"', paths[0].read_text(encoding='utf-8'))
        # 途中エポックでは、最終ファイルでなく選択された重みのrankを使う。
        self.write_lora(temp / 'a-000090.safetensors', ranks=(48,))
        merge.epoch_a = 90
        with contextlib.redirect_stdout(io.StringIO()):
            paths = ssgen.generate_slider_merge(merge)
        self.assertIn('"--new_rank" "48"', paths[0].read_text(encoding='utf-8'))

    def test_lora_rank_rejects_invalid_headers_and_preserves_merge(self):
        args, temp = self.slider_checkpoints_fixture()
        merge = ssgen.make_parser().parse_args(['--merge-slider', str(args.destination), '--epoch-a', '100', '--epoch-b', '100'])
        with contextlib.redirect_stdout(io.StringIO()):
            paths = ssgen.generate_slider_merge(merge)
        previous = {path: path.read_bytes() for path in paths}
        bad = temp / 'a.safetensors'
        for content in [b'bad', (99999).to_bytes(8, 'little'), (2).to_bytes(8, 'little') + b'{}',
                        (2).to_bytes(8, 'little') + b'[]', (2).to_bytes(8, 'little') + b'xx']:
            with self.subTest(content=content):
                bad.write_bytes(content)
                with self.assertRaisesRegex(ValueError, 'a.safetensors'):
                    ssgen.generate_slider_merge(merge)
                self.assertTrue(all(path.read_bytes() == value for path, value in previous.items()))
        header = {'lora_x.lora_down.weight': {'shape': [16, 2]},
                  'lora_x.lora_up.weight': {'shape': [2, 32]}}
        content = json.dumps(header).encode('utf-8')
        bad.write_bytes(len(content).to_bytes(8, 'little') + content)
        with self.assertRaisesRegex(ValueError, '一致しません'):
            ssgen.read_lora_rank(bad)

    def test_slider_wizard_uses_existing_images_without_dataset_prompt(self):
        destination = self.root / 'slider_job'
        destination.mkdir()
        for side in ('a', 'b'):
            (destination / f'{side}.png').write_bytes(side.encode())
        with patch.object(sys, 'argv', ['ssgen', '--overwrite']), \
                patch.object(ssgen, 'enable_terminal'), \
                patch.object(ssgen.Path, 'cwd', return_value=self.root), \
                patch.object(ssgen, 'discover_models', return_value=[self.root / 'base.safetensors']), \
                patch.object(ssgen, 'select_option', side_effect=['anima', str(self.root / 'base.safetensors'), 'slider']), \
                patch('builtins.input', side_effect=['slider_job', str(self.root), '']), \
                contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        self.assertTrue((destination / 'slider.toml').is_file())
        self.assertEqual(ssgen.read_toml(destination / 'config_a.toml')['max_train_epochs'], 100)
        self.assertFalse((destination / 'merge.bat').exists())

    @unittest.skipUnless(sys.platform == 'win32', 'Windows batch execution')
    def test_slider_train_runs_b_only_after_a_success(self):
        args = self.slider_args()
        ssgen.generate(args)
        root = args.destination
        (root / 'train_a.bat').write_bytes(ssgen.batch_text(['exit /b 7']))
        (root / 'train_b.bat').write_bytes(ssgen.batch_text([f'echo ran>"{root / "ran_b.txt"}"', 'exit /b 0']))
        result = subprocess.run(['cmd.exe', '/d', '/c', str(root / 'train.bat')], input=b'\n', capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 7)
        self.assertFalse((root / 'ran_b.txt').exists())
        (root / 'train_a.bat').write_bytes(ssgen.batch_text(['exit /b 0']))
        result = subprocess.run(['cmd.exe', '/d', '/c', str(root / 'train.bat')], input=b'\n', capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0)
        self.assertTrue((root / 'ran_b.txt').is_file())

    @unittest.skipUnless(sys.platform == 'win32', 'Windows batch execution')
    def test_slider_merge_batch_invokes_tool_and_copies_only_merge(self):
        args = self.slider_args('moss', '--sd-scripts-dir', str(self.root / 'sd scripts'))
        ssgen.generate(args)
        root = args.sd_scripts_dir
        (root / 'networks').mkdir(parents=True)
        # GPU処理は行わず、sd-scriptsへの実際の引数受け渡しを検証する。
        (root / 'networks/svd_merge_lora.py').write_text(
            'import argparse\nfrom pathlib import Path\np=argparse.ArgumentParser()\n'
            'p.add_argument("--save_to"); p.add_argument("--models", nargs=2); p.add_argument("--ratios", nargs=2)\n'
            'p.add_argument("--new_rank"); p.add_argument("--device"); p.add_argument("--save_precision")\n'
            'a=p.parse_args()\nassert a.ratios==["-1","1"] and a.new_rank=="32"\n'
            'assert a.device=="cuda" and a.save_precision=="bf16"\n'
            'assert all(Path(f).is_file() for f in a.models)\nPath(a.save_to).write_bytes(b"merged")\n', encoding='utf-8')
        temp = args.destination / 'LoRA.Temp'
        temp.mkdir()
        for side in ('a', 'b'):
            self.write_lora(temp / f'{side}-000090.safetensors')
        merge = ssgen.make_parser().parse_args(['--merge-slider', str(args.destination), '--epoch-a', '90', '--epoch-b', '90'])
        with patch.object(sys, 'argv', ['ssgen', '--merge-slider', str(args.destination), '--epoch-a', '90', '--epoch-b', '90']), contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        result = subprocess.run(['cmd.exe', '/d', '/c', str(args.destination / 'merge.bat')], input=b'\n', capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        (args.destination / 'LoRA' / 'unrelated.safetensors').write_bytes(b'do not copy')
        result = subprocess.run(['cmd.exe', '/d', '/c', str(args.destination / 'copylora.bat')], input=b'\n', capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        target = args.comfyui_dir / 'models/loras/moss' / args.name
        self.assertEqual({p.name for p in target.iterdir()}, {args.name + '.safetensors'})
        self.assertEqual((target / (args.name + '.safetensors')).read_bytes(), b'merged')

    def test_sd_scripts_family_defaults_in_generated_batch(self):
        expected = {
            'sdxl': Path('E:/DataSetSDXL/sd-scripts'),
            'anima': Path('E:/DataSetAnima/sd-scripts'),
            'moss': Path('E:/DataSetMoss/sd-scripts'),
        }
        for family, path in expected.items():
            with self.subTest(family=family):
                args = self.args(family)
                with patch('builtins.input', return_value='') as prompt:
                    ssgen.ask_sd_scripts(args)
                self.assertIn(str(path), prompt.call_args.args[0])
                self.assertEqual(args.sd_scripts_dir, path.absolute())
                files = ssgen.generate(args)
                train = files[2].read_text(encoding='utf-8')
                self.assertIn(f'pushd "{path.resolve()}"', train)
                self.assertIn(str(path.resolve() / 'venv' / 'Scripts' / 'python.exe'), train)
                if family == 'moss':
                    self.assertNotIn('DataSetAnima', train)

    def test_sd_scripts_custom_path(self):
        args = self.args('moss')
        path = self.root / 'custom sd-scripts'
        with patch('builtins.input', return_value=f'"{path}"'):
            ssgen.ask_sd_scripts(args)
        files = ssgen.generate(args)
        train = files[2].read_text(encoding='utf-8')
        self.assertIn(f'pushd "{path}"', train)
        self.assertIn(str(path / 'venv' / 'Scripts' / 'python.exe'), train)

    def test_moss_models_follow_comfyui_and_use_correct_schema(self):
        args = self.args('moss')
        files = ssgen.generate(args)
        config = ssgen.read_toml(files[0])
        root = args.comfyui_dir.absolute()
        self.assertEqual(config['text_encoder'], str(root / 'models/text_encoders/Moss/qwen3.5_9b_int8_convrot.safetensors'))
        self.assertEqual(config['vae'], str(root / 'models/vae/Moss/flux2-vae.safetensors'))
        self.assertEqual(config['pretrained_model_name_or_path'], str(root / 'models/diffusion_models/Moss/moss_image_base_bf16.safetensors'))
        self.assertEqual(config['network_module'], 'networks.lora_generic')
        for key in ['qwen3', 'vae_disable_cache', 'vae_chunk_side', 'debiased_estimation']:
            self.assertNotIn(key, config)
        self.assertIn('moss_image_train_network.py', files[2].read_text(encoding='utf-8'))

    def test_moss_legacy_anima_template_and_script(self):
        # 古い用途別設定・batにAnima設定があってもファミリで分岐する。
        args = self.args('moss', '--template', str(Path('templates/anima/config.toml').resolve()))
        with patch.object(ssgen, 'training_defaults', return_value=(self.root, self.root / 'python.exe', 'anima_train_network.py')):
            files = ssgen.generate(args)
        config = ssgen.read_toml(files[0])
        self.assertNotIn('qwen3', config)
        self.assertIn('qwen3.5_9b_int8_convrot', config['text_encoder'])
        self.assertIn('flux2-vae.safetensors', config['vae'])
        self.assertEqual(config['network_module'], 'networks.lora_generic')
        self.assertNotIn('anima_train_network.py', files[2].read_text(encoding='utf-8'))

    def test_moss_explicit_model_overrides_are_preserved(self):
        encoder = self.root / 'custom_encoder.safetensors'
        vae = self.root / 'custom_flux2_vae.safetensors'
        args = self.args('moss', '--set', 'text_encoder=' + ssgen.toml_string(str(encoder)),
                         '--set', 'vae=' + ssgen.toml_string(str(vae)), '--train-script', 'custom_train_network.py')
        files = ssgen.generate(args)
        config = ssgen.read_toml(files[0])
        self.assertEqual(config['text_encoder'], str(encoder))
        self.assertEqual(config['vae'], str(vae))
        self.assertIn('custom_train_network.py', files[2].read_text(encoding='utf-8'))

    def test_wizard_name_retries_and_dataset_repeat(self):
        args = self.args()
        (self.root / 'existing').mkdir()
        with patch.object(ssgen.Path, 'cwd', return_value=self.root), \
                patch('builtins.input', side_effect=['CON', 'existing', 'valid']), \
                contextlib.redirect_stdout(io.StringIO()):
            ssgen.ask_name(args)
        self.assertEqual(args.destination, self.root / 'valid')
        args.add_dataset = []
        image_dir = self.root / 'images'
        image_dir.mkdir()
        with patch('builtins.input', side_effect=[str(image_dir), str(image_dir), '', '0', '20']), \
                contextlib.redirect_stdout(io.StringIO()):
            ssgen.ask_datasets(args)
        self.assertEqual(args.add_dataset, [image_dir])
        self.assertEqual(args.repeats, 20)

    def test_interactive_overwrite_regenerates_only_four_files(self):
        destination = self.root / 'existing_lora'
        destination.mkdir()
        for name in ['config.toml', 'dataset.toml', 'train.bat', 'copylora.bat']:
            (destination / name).write_text('old content', encoding='utf-8')
        images = destination / 'charname_images'
        images.mkdir()
        (images / 'keep.txt').write_text('keep', encoding='utf-8')
        (destination / 'notes.txt').write_text('notes', encoding='utf-8')
        for options in [['--overwrite'], ['--interactive', '--overwrite'], ['--force']]:
            with self.subTest(options=options), \
                    patch.object(sys, 'argv', ['ssgen', *options]), \
                    patch.object(ssgen, 'enable_terminal'), \
                    patch.object(ssgen.Path, 'cwd', return_value=self.root), \
                    patch.object(ssgen, 'discover_models', return_value=[self.root / 'base.safetensors']), \
                    patch.object(ssgen, 'select_option', side_effect=['moss', str(self.root / 'base.safetensors'), 'character']), \
                    patch('builtins.input', side_effect=['existing_lora', str(self.root), '', str(images), '', '']), \
                    contextlib.redirect_stdout(io.StringIO()):
                ssgen.main()
            self.assertEqual(ssgen.read_toml(destination / 'config.toml')['output_name'], 'existing_lora')
            dataset = ssgen.read_toml(destination / 'dataset.toml')
            self.assertEqual(dataset['datasets'][0]['subsets'][0]['image_dir'], str(images))
            self.assertEqual((images / 'keep.txt').read_text(encoding='utf-8'), 'keep')
            self.assertEqual((destination / 'notes.txt').read_text(encoding='utf-8'), 'notes')

    def test_overwrite_cli_and_name_file_rejection(self):
        args = self.args('moss', '--overwrite')
        files = ssgen.generate(args)
        files[0].write_text('old', encoding='utf-8')
        ssgen.generate(args)
        self.assertEqual(ssgen.read_toml(files[0])['output_name'], args.name)
        (self.root / 'file_name').write_text('keep', encoding='utf-8')
        with patch.object(ssgen.Path, 'cwd', return_value=self.root), \
                patch('builtins.input', side_effect=['file_name', 'valid_name']), \
                contextlib.redirect_stdout(io.StringIO()):
            ssgen.ask_name(args)
        self.assertEqual(args.name, 'valid_name')
        self.assertEqual((self.root / 'file_name').read_text(encoding='utf-8'), 'keep')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows batch execution')
    def test_batch_pause_preserves_exit_status(self):
        args = self.args('moss', '--sd-scripts-dir', str(self.root / 'missing'), '--copy-retries', '0')
        files = ssgen.generate(args)
        for batch in files[2:]:
            result = subprocess.run(['cmd.exe', '/d', '/c', str(batch)], input=b'\n', capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 1)
        output = Path(ssgen.read_toml(files[0])['output_dir'])
        output.mkdir()
        (output / 'test.safetensors').write_bytes(b'test')
        result = subprocess.run(['cmd.exe', '/d', '/c', str(files[3])], input=b'\n', capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0)
        target = args.comfyui_dir / 'models' / 'loras' / 'moss' / args.name / 'test.safetensors'
        self.assertEqual(target.read_bytes(), b'test')


if __name__ == '__main__':
    unittest.main()

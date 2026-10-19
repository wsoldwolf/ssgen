import tempfile
import unittest
from pathlib import Path
import contextlib
import io
import sys
from datetime import datetime, date, time, timezone
from unittest.mock import patch

import tomllib

import ssgen


class GenerationTests(unittest.TestCase):
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
                               ('moss', 'anima_train_network.py')]:
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
                self.assertIn('if errorlevel 8 exit /b %ERRORLEVEL%', copy)
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
                patch('builtins.input', side_effect=['wizard_test', '', '']), \
                contextlib.redirect_stdout(io.StringIO()):
            ssgen.main()
        destination = self.root / 'wizard_test'
        self.assertEqual({p.name for p in destination.iterdir()},
                         {'config.toml', 'dataset.toml', 'train.bat', 'copylora.bat'})
        config = ssgen.read_toml(destination / 'config.toml')
        self.assertEqual(config['pretrained_model_name_or_path'], str(model_dir / 'b.safetensors'))
        self.assertEqual(config['output_name'], 'wizard_test')
        self.assertFalse(ssgen.read_toml(destination / 'dataset.toml')['datasets'][0].get('subsets'))
        # 未登録で生成した設定にも通常の追記ができる。
        dataset = ssgen.read_toml(destination / 'dataset.toml')
        ssgen.add_datasets(dataset, [self.root / 'later'], 10)
        self.assertEqual(dataset['datasets'][0]['subsets'][0]['num_repeats'], 10)

    def test_slider_is_pending_and_reselectable(self):
        args = self.args()
        with patch.object(ssgen, 'select_option', side_effect=['slider', 'concept']), contextlib.redirect_stdout(io.StringIO()):
            ssgen.ask_purpose(args)
        self.assertEqual(args.kind, 'concept')
        args.kind = 'slider'
        with self.assertRaisesRegex(ValueError, '準備中'):
            ssgen.generate(args)

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


if __name__ == '__main__':
    unittest.main()

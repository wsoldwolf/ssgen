"""sd-scripts 用の学習 TOML と Windows bat を生成する。"""

import argparse
from collections.abc import MutableMapping
from pathlib import Path, PureWindowsPath
import re
import sys
import os
import copy
import math

import tomllib
import json
from datetime import date, datetime, time


DEFAULT_COMFYUI = Path('C:/Software/ComfyUI')
DEFAULT_SD_SCRIPTS = {
    'sdxl': Path('E:/DataSetSDXL/sd-scripts'),
    'anima': Path('E:/DataSetAnima/sd-scripts'),
    'moss': Path('E:/DataSetMoss/sd-scripts'),
}
FAMILY_SETTINGS = {
    'sdxl': {'train_script': 'sdxl_train_network.py', 'network_module': 'networks.lora', 'models': {}},
    'anima': {'train_script': 'anima_train_network.py', 'network_module': 'networks.lora_anima',
              'models': {'qwen3': 'models/text_encoders/qwen_3_06b_base.safetensors',
                         'vae': 'models/vae/qwen_image_vae.safetensors'}},
    'moss': {'train_script': 'moss_image_train_network.py', 'network_module': 'networks.lora_generic',
             'models': {'pretrained_model_name_or_path': 'models/diffusion_models/Moss/moss_image_base_bf16.safetensors',
                        'text_encoder': 'models/text_encoders/Moss/qwen3.5_9b_int8_convrot.safetensors',
                        'vae': 'models/vae/Moss/flux2-vae.safetensors'}},
}
FAMILIES = [('SDXL', 'sdxl'), ('Anima', 'anima'), ('MOSS', 'moss')]
PURPOSES = [('キャラLoRA', 'character'), ('画風LoRA', 'style'),
            ('コンセプトLoRA', 'concept'), ('スライダーLoRA', 'slider')]


def validate_name(name):
    if not re.fullmatch(r'[^<>:"/\\|?*\x00-\x1f]+', name):
        raise ValueError('LoRA名にはWindowsのファイル名として使用できる文字を指定してください')
    if name.endswith((' ', '.')) or name.split('.')[0].upper() in {
        'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))
    }:
        raise ValueError('LoRA名の末尾の空白・ピリオド、Windowsの予約名は使用できません')


def read_key():
    if os.name == 'nt':
        import msvcrt
        key = msvcrt.getwch()
        if key in ('\x00', '\xe0'):
            return {'H': 'up', 'P': 'down'}.get(msvcrt.getwch(), '')
        if key == '\x03':
            raise KeyboardInterrupt
        return {'\r': 'enter', '\x1b': 'cancel'}.get(key, '')
    import termios
    import tty
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
        if key == '\x03':
            raise KeyboardInterrupt
        if key == '\x1b':
            # POSIXではEsc単独で待機しないよう、後続バイトの有無を確認する。
            import select
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key += sys.stdin.read(2)
        return {'\x1b[A': 'up', '\x1b[B': 'down', '\r': 'enter',
                '\n': 'enter', '\x1b': 'cancel'}.get(key, '')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def enable_terminal():
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError('対話モードは端末から実行してください。自動処理にはCLI引数を指定できます')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel.GetStdHandle(-11 & 0xffffffff)
        mode = wintypes.DWORD()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)) or not kernel.SetConsoleMode(handle, mode.value | 4):
            raise ValueError('カーソル選択に対応するWindows TerminalまたはPowerShellから実行してください')


def select_option(title, choices):
    """(表示名, 値)のリストを上下キーで選択。長い一覧はスクロールする。"""
    import shutil
    import unicodedata
    if not choices:
        raise ValueError(f'{title}: 選択肢がありません')
    print(f'\n{title}  [↑/↓: 移動、Enter: 決定、Esc: 中止]')
    selected = 0
    height = min(len(choices), max(1, shutil.get_terminal_size().lines - 5), 10)

    def fit(text):
        limit = max(10, shutil.get_terminal_size().columns - 2)
        result, width = '', 0
        for char in text.replace('\n', ' ').replace('\r', ' ').replace('\x1b', '?'):
            width += 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
            if width > limit - 1:
                return result + '…'
            result += char
        return result

    while True:
        start = max(0, min(selected - height // 2, len(choices) - height))
        for i in range(start, start + height):
            prefix = ' > ' if i == selected else '   '
            print('\r\x1b[2K' + fit(f'{prefix}{i + 1}/{len(choices)}  {choices[i][0]}'))
        key = read_key()
        if key == 'enter':
            print(f'選択: {choices[selected][0]}')
            return choices[selected][1]
        if key == 'cancel':
            raise KeyboardInterrupt
        selected = (selected + {'up': -1, 'down': 1}.get(key, 0)) % len(choices)
        print(f'\x1b[{height}F', end='', flush=True)


def discover_models(comfyui_dir, family):
    """ファミリー直下とComfyUIの代表的なモデル保存先を再帰探索する。"""
    roots = [comfyui_dir / family, comfyui_dir / 'models' / family,
             comfyui_dir / 'models' / 'checkpoints' / family,
             comfyui_dir / 'models' / 'diffusion_models' / family]
    models = {}
    for root in roots:
        if root.is_dir():
            for path in root.rglob('*'):
                if path.is_file() and path.suffix.lower() in {'.safetensors', '.ckpt', '.pt', '.pth', '.bin'}:
                    # 実体パスは重複判定だけに使用し、一覧・設定には入力側のパスを保持する。
                    models.setdefault(os.path.normcase(str(path.resolve())), path.absolute())
    return sorted(models.values(), key=lambda path: str(path).casefold())


def ask_family(args):
    args.family = select_option('1. ファミリ', FAMILIES)


def ask_name(args):
    if getattr(args, '_preset_project_name', False):
        validate_name(args.name)
        if args.destination.exists() and not args.destination.is_dir():
            raise ValueError('LoRA名と同名のファイルが存在します')
        print(f'\n2. LoRA名: {args.name}')
        return
    while True:
        name = input('\n2. LoRA名: ').strip()
        try:
            validate_name(name)
            destination = Path.cwd() / name
            if destination.exists() and not destination.is_dir():
                raise ValueError('同名のファイルが存在します。別のLoRA名を入力してください')
            args.name, args.destination = name, destination
            return
        except ValueError as error:
            print(error)


def ask_comfyui(args):
    while True:
        value = input(f'\n3. ComfyUIパス [{DEFAULT_COMFYUI}]: ').strip().strip('"')
        path = Path(value).expanduser() if value else DEFAULT_COMFYUI
        if path.is_dir():
            args.comfyui_dir = path.absolute()
            return
        print(f'フォルダが見つかりません: {path}')


def ask_model(args):
    while True:
        models = discover_models(args.comfyui_dir, args.family)
        if models:
            choices = []
            for path in models:
                try:
                    label = os.path.relpath(str(path), str(args.comfyui_dir))
                except ValueError:
                    # ジャンクション等をresolveすると実体がComfyUIの外にある場合がある。
                    label = str(path)
                choices.append((label, str(path)))
            args.model = select_option('4. 学習ベースモデル', choices)
            return
        print(f'{args.comfyui_dir} 以下の {args.family} モデルが見つかりません。')
        print('探索先: <ComfyUI>/<family>、models/<family>、models/checkpoints/<family>、models/diffusion_models/<family>')
        ask_comfyui(args)


def ask_purpose(args):
    args.kind = select_option('5. 用途', PURPOSES)


def ask_sd_scripts(args):
    default = DEFAULT_SD_SCRIPTS[args.family]
    value = input(f'\n6. sd-scriptsパス [{default}]: ').strip().strip('"')
    args.sd_scripts_dir = (Path(value).expanduser() if value else default).absolute()


def ask_moss_block_swap(args):
    if args.family != 'moss':
        return
    args._moss_blocks_to_swap = select_option(
        'MOSS: blocks_to_swap = 20（VRAM節約と引き換えに転送が増えます）',
        [('無効', 0), ('有効（20ブロック）', 20)],
    )
    if args._moss_blocks_to_swap:
        print('併用できないunsloth_offload_checkpointing / cpu_offload_checkpointingを無効にします。')


def ask_datasets(args):
    if args.kind == 'slider':
        print(f'\nスライダー画像: {args.destination / "a.png"} / {args.destination / "b.png"}')
        print('各画像を10リピート・100エポック（1000ステップ）で個別に学習します。')
        return
    print('\n7. 画像フォルダを1つずつ入力してください（空欄で終了、後から --add-dataset でも追加可能）。')
    while True:
        value = input('画像フォルダ: ').strip().strip('"')
        if not value:
            break
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            print('フォルダが見つかりません。')
        elif path in args.add_dataset:
            print('このフォルダは追加済みです。')
        else:
            args.add_dataset.append(path)
    args.allow_empty_dataset = True
    if args.add_dataset:
        while True:
            try:
                args.repeats = positive(input('リピート数 [10]: ').strip() or '10')
                break
            except (ValueError, argparse.ArgumentTypeError):
                print('1以上の整数を入力してください。')


# 対話項目を増やす場合は関数を定義し、このリストに追加する。
WIZARD_STEPS = [ask_family, ask_name, ask_comfyui, ask_model, ask_purpose, ask_sd_scripts,
                ask_moss_block_swap, ask_datasets]


def interactive_settings(parser, args=None):
    enable_terminal()
    if args is None:
        args = parser.parse_args([])
    args._interactive_generation = True
    print('ssgen - 学習設定ウィザード（Ctrl+C / 選択画面のEscで中止）')
    print(f'スクリプト: {Path(__file__).absolute()}')
    for step in WIZARD_STEPS:
        step(args)
    print(f'\n出力先: {args.destination}')
    if args.kind != 'slider' and not args.add_dataset and (args.force or not (args.destination / 'dataset.toml').is_file()):
        print('データセットは未登録です。学習前に --add-dataset で追加してください。')
    return args


def ask_dataset_config_path(args):
    path = args.dataset_config
    if args.project_dir or args.destination:
        if path != Path('dataset.toml'):
            raise ValueError('LoRAフォルダ指定時の --dataset-config はパス引数を省略してください')
        path = (args.project_dir or args.destination) / path
    while not path.is_file():
        if path.is_dir():
            path = path / 'dataset.toml'
            if path.is_file():
                break
        value = input(f'dataset.tomlのパス [{path}]: ').strip().strip('"')
        if value:
            path = Path(value).expanduser()
        if not path.is_file():
            print(f'ファイルが見つかりません: {path}')
    args._editing_dataset_path = path.absolute()
    args._editing_dataset = read_toml(path)
    # dataset_a.toml / dataset_b.tomlも同じ側のconfigから判定する。
    config_name = 'config' + path.stem.removeprefix('dataset') + '.toml' if path.stem.startswith('dataset') else 'config.toml'
    config_path = path.with_name(config_name)
    args._editing_training_path = config_path.absolute()
    args._editing_training = read_toml(config_path) if config_path.is_file() else {}


def ask_dataset_config_family(args):
    config = args._editing_training
    module = config.get('network_module')
    detected = {'networks.lora_anima': 'anima', 'networks.lora': 'sdxl',
                'networks.lora_generic': 'moss'}.get(module)
    slider = args._editing_dataset_path.parent / 'slider.toml'
    if detected is None and slider.is_file():
        detected = read_toml(slider).get('slider', {}).get('family')
    if args.family and detected and args.family != detected:
        raise ValueError(f'指定ファミリ {args.family} と学習設定のファミリ {detected} が一致しません')
    args.family = args.family or detected
    if args.family is None:
        ask_family(args)
    if args.family != 'moss':
        raise ValueError('このデータセット編集ウィザードのoffset設定はMOSS用です')


def ask_dataset_timestep_sampling(args):
    path = args._editing_training_path
    while not path.is_file():
        value = input(f'学習用config.tomlのパス [{path}]: ').strip().strip('"')
        if value:
            path = Path(value).expanduser().absolute()
        if not path.is_file():
            print(f'ファイルが見つかりません: {path}')
    args._editing_training_path = path
    args._editing_training = read_toml(path)
    if args._editing_training.get('network_module') in ('networks.lora', 'networks.lora_anima'):
        raise ValueError('MOSS用のconfig.tomlを指定してください')
    ask_timestep_sampling_mode(args)


def ask_timestep_sampling_mode(args):
    current = args._editing_training.get('timestep_sampling', '(未指定)')
    mode = select_option(f'MOSS: timestep_sampling（現在: {current}）', [
        ('sigmoid: 基本のシグモイド分布', 'sigmoid'),
        ('shift: discrete_flow_shiftで分布を調整', 'shift'),
        ('flux_shift: 画像解像度に応じて分布を調整', 'flux_shift'),
    ])
    args._editing_training_changed = current != mode
    args._editing_training['timestep_sampling'] = mode


def ask_timestep_offset_scope(args):
    args._editing_offset_scope = select_option('offsetの変更方法', [
        ('個別に設定（サブセットを選択して編集）', 'individual'),
        ('一括変更（すべてのサブセットに同じ値）', 'all'),
    ])


def ask_offset_value(prompt):
    while True:
        value = input(prompt).strip()
        if not value:
            return None
        try:
            offset = float(value)
            if math.isfinite(offset):
                return offset
        except ValueError:
            pass
        print('有限の数値を入力してください（例: -0.5 / 0 / 0.5）。')


def ask_timestep_sampling_offsets(args):
    print('\nMOSS: サブセットごとのtimestep sampling offset')
    print('負の値 → 低ノイズ側のタイムステップに偏り、細部・質感の学習を重視します。')
    print('正の値 → 高ノイズ側のタイムステップに偏り、全体の構造・構図の学習を重視します。')
    print('0 → 変更なし。公式の推奨範囲は -0.5 ～ 0.5 です。')
    print('offsetが効くのは sigmoid / shift / flux_shift です。sigma / uniform では無効です。')
    mode = args._editing_training.get('timestep_sampling')
    if mode in ('sigma', 'uniform'):
        print(f'注意: 現在の学習設定は {mode} です。offsetを反映するにはconfig.tomlのtimestep_samplingも変更してください。')
    subsets = []
    for dataset_index, dataset in enumerate(args._editing_dataset.get('datasets', [])):
        for subset_index, subset in enumerate(dataset.get('subsets', [])):
            custom = subset.get('custom_attributes', {})
            if not isinstance(custom, dict) or not isinstance(custom.get('timestep_sampling', {}), dict):
                raise ValueError('custom_attributes.timestep_samplingはテーブルで指定してください')
            subsets.append((dataset_index, subset_index, subset))
    if not subsets:
        args._editing_dataset_changed = False
        print('サブセットが未登録のため、offset編集をスキップします。')
        return
    args._editing_dataset_changed = False
    if args._editing_offset_scope == 'all':
        offset = ask_offset_value('すべてのサブセットに適用するoffset（空欄で全項目を維持）: ')
        if offset is not None:
            for _, _, subset in subsets:
                sampling = subset.setdefault('custom_attributes', {}).setdefault('timestep_sampling', {})
                if sampling.get('offset') != offset:
                    sampling['offset'] = offset
                    args._editing_dataset_changed = True
            print(f'{len(subsets)}件のサブセットにoffset={offset}を適用します。')
        return
    while True:
        choices = []
        for index, (dataset_index, subset_index, subset) in enumerate(subsets):
            current = subset.get('custom_attributes', {}).get('timestep_sampling', {}).get('offset', 0.0)
            label = subset.get('image_dir', subset.get('metadata_file', '(画像パスなし)'))
            choices.append((f'datasets[{dataset_index}].subsets[{subset_index}] {label}  offset={current}', index))
        choices += [('保存して終了', 'save'), ('変更を破棄して終了', 'discard')]
        selected = select_option('変更するサブセット', choices)
        if selected == 'save':
            return
        if selected == 'discard':
            args._editing_dataset_changed = False
            args._editing_training_changed = False
            return
        subset = subsets[selected][2]
        current = subset.get('custom_attributes', {}).get('timestep_sampling', {}).get('offset', 0.0)
        offset = ask_offset_value(f'offset [{current}]（空欄で維持）: ')
        if offset is not None:
            sampling = subset.setdefault('custom_attributes', {}).setdefault('timestep_sampling', {})
            if sampling.get('offset') != offset:
                sampling['offset'] = offset
                args._editing_dataset_changed = True


DATASET_CONFIG_STEPS = [ask_dataset_config_path, ask_dataset_config_family,
                        ask_dataset_timestep_sampling, ask_timestep_offset_scope, ask_timestep_sampling_offsets]


def edit_dataset_config(args):
    enable_terminal()
    print('ssgen - データセット設定ウィザード（Ctrl+C / Escで保存せず中止）')
    for step in DATASET_CONFIG_STEPS:
        step(args)
    outputs = {}
    if args._editing_dataset_changed:
        content = dump_toml(args._editing_dataset)
        tomllib.loads(content)
        outputs[args._editing_dataset_path] = content.encode('utf-8')
    if args._editing_training_changed:
        content = dump_toml(args._editing_training)
        tomllib.loads(content)
        outputs[args._editing_training_path] = content.encode('utf-8')
    if outputs:
        for path in write_outputs(outputs):
            print(f'更新しました: {path}')
    else:
        print('ファイルは変更していません。')


def update_key(document, key, value):
    """sd-scripts のセクション形式も含め、既存キーを更新する。"""
    matches = []

    def visit(table):
        for name, item in table.items():
            if name == key:
                matches.append(table)
            elif isinstance(item, MutableMapping):
                visit(item)

    visit(document)
    if len(matches) > 1:
        raise ValueError(f"キー {key!r} が複数あります。テンプレートを確認してください")
    (matches[0] if matches else document)[key] = value


def bat_quote(value):
    # cmd.exe の変数展開と引用符によるコマンドの崩れを防ぐ。
    if any(c in value for c in '"\r\n\x00'):
        raise ValueError('bat の引数には引用符・改行・NULを使用できません')
    return '"' + value.replace('%', '%%') + '"'


def read_toml(path):
    return tomllib.loads(path.read_text(encoding='utf-8-sig'))


def toml_string(value):
    # JSONの文字列エスケープはTOML基本文字列でも使用可能。DELも明示的に処理する。
    return json.dumps(value, ensure_ascii=False).replace('\x7f', '\\u007f')


def toml_key(value):
    return value if re.fullmatch(r'[A-Za-z0-9_-]+', value) else toml_string(value)


def toml_value(value):
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, list):
        return '[' + ', '.join(toml_value(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{ ' + ', '.join(f'{toml_key(key)} = {toml_value(item)}'
                                for key, item in value.items()) + ' }' if value else '{}'
    raise TypeError(f'TOMLに出力できない値です: {type(value).__name__}')


def dump_toml(document):
    """辞書をTOMLへ出力。通常テーブル・配列テーブルを再帰的に扱う。"""
    lines = []

    def array_of_tables(value):
        return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)

    def inline_table(path, key, value):
        return (path and path[-1] == 'custom_attributes' and key == 'timestep_sampling'
                and isinstance(value, dict))

    def write_table(table, path=(), array=False):
        if path:
            if lines:
                lines.append('')
            name = '.'.join(toml_key(key) for key in path)
            lines.append(f'[[{name}]]' if array else f'[{name}]')
        for key, value in table.items():
            if inline_table(path, key, value) or (not isinstance(value, dict) and not array_of_tables(value)):
                lines.append(f'{toml_key(key)} = {toml_value(value)}')
        for key, value in table.items():
            if inline_table(path, key, value):
                continue
            if isinstance(value, dict):
                write_table(value, (*path, key))
            elif array_of_tables(value):
                for item in value:
                    write_table(item, (*path, key), array=True)

    write_table(document)
    return '\n'.join(lines) + '\n'


def dataset_path_key(value):
    return os.path.normpath(str(value)).replace('\\', '/').rstrip('/').casefold()


def find_datasets_with_prefix(root, prefixes):
    if not root.is_dir():
        raise ValueError(f'フォルダが見つかりません: {root}')
    for prefix in prefixes:
        if not prefix or any(c in prefix for c in '/\\\x00'):
            raise ValueError('プレフィックスには空でないディレクトリ名の先頭文字列を指定してください')
    prefixes = tuple(prefix.casefold() for prefix in prefixes)
    return sorted({path.resolve() for path in root.rglob('*')
                   if path.is_dir() and path.name.casefold().startswith(prefixes)},
                  key=lambda path: str(path).casefold())


def add_datasets(document, paths, repeats, index=0, skip_existing=False):
    datasets = document.get('datasets', [])
    if not 0 <= index < len(datasets):
        raise ValueError('追加先の datasets がありません。--dataset-index を確認してください')
    subsets = datasets[index].get('subsets')
    if subsets is None:
        subsets = []
        datasets[index]['subsets'] = subsets
    # 一括追記では別のdatasetsに登録されたパスも重複として保持する。
    entries = [item for entry in datasets for item in entry.get('subsets', [])] if skip_existing else subsets
    existing = {dataset_path_key(item.get('image_dir', '')) for item in entries}
    added, skipped = 0, 0
    for path in paths:
        image_dir = str(path.resolve())
        normalized = dataset_path_key(image_dir)
        if normalized in existing:
            if skip_existing:
                skipped += 1
                continue
            raise ValueError(f'データセットが重複しています: {image_dir}')
        subset = {}
        subset['image_dir'] = image_dir
        subset['num_repeats'] = repeats
        subsets.append(subset)
        existing.add(normalized)
        added += 1
    return added, skipped


def training_defaults(template_dir):
    # 元batを実行せず、学習スクリプトと仮想環境の場所だけ読み取る。
    text = (template_dir / 'train.bat').read_text(encoding='utf-8-sig')
    text = re.sub(r'\\+', lambda _: '\\', text)
    activation = re.search(r'(?im)^call\s+"?(.+?)\\Scripts\\activate\.bat"?\s*$', text)
    script = re.search(r'([A-Za-z0-9_]+_train_network\.py|train_network\.py)', text)
    if not activation or not script:
        raise ValueError('train.batから学習環境を取得できません。activate.batと学習スクリプトを確認してください')
    venv = PureWindowsPath(activation.group(1))
    return Path(str(venv.parent)), Path(str(venv / 'Scripts' / 'python.exe')), script.group(1)


def batch_text(lines):
    return ('\r\n'.join(['@echo off', 'setlocal DisableDelayedExpansion', 'chcp 65001 >nul', *lines, ''])).encode('utf-8')


def apply_family_settings(document, args):
    """ファミリ別の引数とComfyUI基準の補助モデルパスを設定する。"""
    settings = FAMILY_SETTINGS[args.family]

    def tables(table):
        yield table
        for value in table.values():
            if isinstance(value, dict):
                yield from tables(value)

    def get_value(key, default=None):
        values = [table[key] for table in tables(document) if key in table]
        if len(values) > 1:
            raise ValueError(f'キー {key!r} が複数あります。テンプレートを確認してください')
        return values[0] if values else default

    if args.family == 'moss':
        # 古い用途別テンプレートにAnimaの引数が残っていても出力しない。
        for table in tables(document):
            for key in ('qwen3', 'vae_disable_cache', 'vae_chunk_side', 'debiased_estimation'):
                table.pop(key, None)
        if get_value('network_module') == 'networks.lora_anima':
            update_key(document, 'network_module', settings['network_module'])
        # 既知のAnima用VAEの混入はMOSS用既定値に置き換える。
        if PureWindowsPath(str(get_value('vae', ''))).name.casefold() == 'qwen_image_vae.safetensors':
            update_key(document, 'vae', settings['models']['vae'])
    if get_value('network_module') is None:
        update_key(document, 'network_module', settings['network_module'])
    for key, default in settings['models'].items():
        value = str(get_value(key, default))
        # 相対パスはComfyUI基準。モデル名の手動指定や外部絶対パスも維持する。
        path = Path(value)
        if not path.is_absolute():
            path = args.comfyui_dir / path
        update_key(document, key, str(path.absolute()))


def copy_batch(source, target, pattern, retries, wait):
    return batch_text([
        'set "SSGEN_EXIT_CODE=1"',
        f'if not exist {bat_quote(str(source / pattern))} goto ssgen_finish',
        f'robocopy {bat_quote(str(source))} {bat_quote(str(target))} {bat_quote(pattern)} /COPY:DAT /R:{retries} /W:{wait}',
        'set "SSGEN_EXIT_CODE=%ERRORLEVEL%"',
        'if %SSGEN_EXIT_CODE% LSS 8 set "SSGEN_EXIT_CODE=0"',
        ':ssgen_finish', 'pause', 'exit /b %SSGEN_EXIT_CODE%',
    ])


def command_batch(root, command, checks=()):
    return batch_text(['set "SSGEN_EXIT_CODE=1"', *checks, f'pushd {bat_quote(str(root))}',
                       'if errorlevel 1 goto ssgen_finish', command,
                       'set "SSGEN_EXIT_CODE=%ERRORLEVEL%"', 'popd',
                       ':ssgen_finish', 'pause', 'exit /b %SSGEN_EXIT_CODE%'])


def plan_training(args):
    validate_name(args.name)
    template_dir = args.templates_dir / args.family
    # 用途別設定があれば優先。なければファミリー共通設定をそのまま使用する。
    config_template = args.template or template_dir / args.kind / 'config.toml'
    if not args.template and not config_template.exists():
        config_template = template_dir / 'config.toml'
    dataset_template = args.dataset_config or template_dir / 'dataset.toml'
    destination = (args.destination or Path('generated') / args.family / args.name).resolve()
    suffix = '_' + args._slider_side if hasattr(args, '_slider_side') else ''
    files = [destination / f'{name}{suffix}.{extension}' for name, extension in
             [('config', 'toml'), ('dataset', 'toml'), ('train', 'bat'), ('copylora', 'bat')]]
    existing_dataset = files[1].is_file() and not args.force
    if files[0].is_file() and not args.force:
        document = read_toml(files[0])
    else:
        document = copy.deepcopy(args._config_data) if hasattr(args, '_config_data') else read_toml(config_template)
    apply_family_settings(document, args)
    if existing_dataset:
        dataset = read_toml(files[1])
    else:
        dataset = copy.deepcopy(args._dataset_data) if hasattr(args, '_dataset_data') else read_toml(dataset_template)
    sources = {config_template.resolve(), dataset_template.resolve(), (template_dir / 'train.bat').resolve(), (template_dir / 'copylora.bat').resolve()}
    if any(path in sources for path in files):
        raise ValueError('出力先をテンプレートと別の場所にしてください')
    if not existing_dataset and not args.keep_template_datasets:
        for entry in dataset.get('datasets', []):
            if 'subsets' in entry:
                del entry['subsets']
    if args.add_dataset:
        add_datasets(dataset, args.add_dataset, args.repeats, args.dataset_index, skip_existing=existing_dataset)
    if not args.allow_empty_dataset and not any(entry.get('subsets') for entry in dataset.get('datasets', [])):
        raise ValueError('--add-dataset で画像フォルダを指定してください')
    overrides = {}
    for assignment in args.set:
        key, separator, raw = assignment.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_]+', key):
            raise ValueError('--set は key=TOML値 の形式で指定してください')
        overrides[key] = tomllib.loads(f'value = {raw}')['value']
    if args.family == 'moss' and hasattr(args, '_moss_blocks_to_swap'):
        overrides['blocks_to_swap'] = args._moss_blocks_to_swap
        if args._moss_blocks_to_swap:
            overrides['unsloth_offload_checkpointing'] = False
            overrides['cpu_offload_checkpointing'] = False
    if args.model:
        overrides['pretrained_model_name_or_path'] = args.model
    output_dir = (args.output_dir or destination / 'LoRA').resolve()
    overrides.update(output_name=args.name, output_dir=str(output_dir),
                     logging_dir=str((args.logging_dir or destination / 'log').resolve()),
                     dataset_config=str(files[1]))
    for key, value in overrides.items():
        update_key(document, key, value)
    default_root, default_python, _template_script = training_defaults(template_dir)
    root = (args.sd_scripts_dir or default_root).resolve()
    python = args.python or (root / 'venv' / 'Scripts' / 'python.exe' if args.sd_scripts_dir else default_python)
    launch_options = ['--num_processes', '1', '--num_machines', '1'] if suffix else []
    command = ' '.join(map(bat_quote, [str(python.resolve()), '-m', 'accelerate.commands.launch',
                                      '--num_cpu_threads_per_process', '1', *launch_options, args.train_script or FAMILY_SETTINGS[args.family]['train_script'],
                                      '--config_file', str(files[0])]))
    train = command_batch(root, command)
    target = (args.comfyui_dir / 'models' / 'loras' / args.family / args.name).resolve()
    copy_content = copy_batch(output_dir, target, '*.safetensors', args.copy_retries, args.copy_wait)
    contents = [dump_toml(document).encode('utf-8'), dump_toml(dataset).encode('utf-8'), train, copy_content]
    for content in contents[:2]:
        tomllib.loads(content.decode('utf-8'))
    return dict(zip(files, contents))


def write_outputs(outputs):
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return list(outputs)


def generate(args):
    if args.kind == 'slider':
        return generate_slider(args)
    outputs = plan_training(args)
    ask_generated_timestep_settings(args, outputs)
    return write_outputs(outputs)


def ask_generated_timestep_settings(args, outputs):
    if args.family != 'moss' or not getattr(args, '_interactive_generation', False):
        return
    for config_path in list(outputs):
        if config_path.suffix != '.toml' or not config_path.stem.startswith('config'):
            continue
        dataset_path = config_path.with_name('dataset' + config_path.stem.removeprefix('config') + '.toml')
        if dataset_path not in outputs:
            continue
        print(f'\n最後の設定: {config_path.name} / {dataset_path.name}')
        editing = argparse.Namespace(**vars(args))
        editing._editing_training = tomllib.loads(outputs[config_path].decode('utf-8'))
        editing._editing_dataset = tomllib.loads(outputs[dataset_path].decode('utf-8'))
        ask_timestep_sampling_mode(editing)
        ask_timestep_offset_scope(editing)
        ask_timestep_sampling_offsets(editing)
        if editing._editing_training_changed:
            outputs[config_path] = dump_toml(editing._editing_training).encode('utf-8')
        if editing._editing_dataset_changed:
            outputs[dataset_path] = dump_toml(editing._editing_dataset).encode('utf-8')


def generate_slider(args):
    validate_name(args.name)
    destination = (args.destination or Path.cwd() / args.name).resolve()
    for side in ('a', 'b'):
        if not (destination / f'{side}.png').is_file():
            raise ValueError(f'スライダー画像を配置してください: {destination / (side + ".png")}')
    config_template = args.template or args.templates_dir / args.family / 'slider' / 'config.toml'
    if not args.template and not config_template.exists():
        config_template = args.templates_dir / args.family / 'config.toml'
    config = read_toml(config_template)

    def remove_schedule(table):
        for key in list(table):
            if key in {'max_train_steps', 'save_every_n_steps', 'save_n_epoch_ratio',
                       'save_last_n_epochs', 'save_last_n_steps'}:
                del table[key]
            elif isinstance(table[key], dict):
                remove_schedule(table[key])
    remove_schedule(config)
    dataset = read_toml(args.dataset_config or args.templates_dir / args.family / 'dataset.toml')
    entry = copy.deepcopy(dataset.get('datasets', [{}])[0])
    entry['batch_size'] = 1
    entry['subsets'] = []
    dataset['datasets'] = [entry]
    outputs = {}
    temp_dir = destination / 'LoRA.Temp'
    root = (args.sd_scripts_dir or DEFAULT_SD_SCRIPTS[args.family]).absolute()
    python = (args.python or root / 'venv' / 'Scripts' / 'python.exe').absolute()
    for side in ('a', 'b'):
        child = argparse.Namespace(**vars(args))
        image_dir = destination / 'slider_data' / side
        child.name, child.destination, child.output_dir = side, destination, temp_dir
        child.sd_scripts_dir, child.python = root, python
        child.logging_dir = (args.logging_dir or destination / 'log') / side
        child._slider_side, child._config_data, child._dataset_data = side, config, copy.deepcopy(dataset)
        child._dataset_data['datasets'][0]['subsets'] = [
            {'image_dir': str(image_dir), 'num_repeats': 10, 'class_tokens': args.name}]
        child.add_dataset, child.keep_template_datasets = [], True
        child.set = [assignment for assignment in args.set
                     if assignment.partition('=')[0] not in {'max_train_steps', 'save_every_n_steps',
                         'save_n_epoch_ratio', 'save_last_n_epochs', 'save_last_n_steps'}]
        child.set += ['max_train_epochs=100', 'save_every_n_epochs=10', 'gradient_accumulation_steps=1',
                      'save_model_as="safetensors"']
        plan = plan_training(child)
        outputs.update({path: value for path, value in plan.items() if not path.name.startswith('copylora_')})
        image = image_dir / f'{side}.png'
        # コピー先に別画像があると1000ステップにならないため、その場合は止める。
        if image_dir.exists() and any(path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
                                      and path != image for path in image_dir.iterdir()):
            raise ValueError(f'スライダー用フォルダには対象画像1枚だけ配置してください: {image_dir}')
        outputs[image] = (destination / f'{side}.png').read_bytes()
        caption = destination / f'{side}.txt'
        if caption.is_file():
            outputs[image.with_suffix('.txt')] = caption.read_bytes()
        elif image.with_suffix('.txt').exists():
            outputs[image.with_suffix('.txt')] = args.name.encode('utf-8')
    train = batch_text([
        'set "SSGEN_EXIT_CODE=1"', f'pushd {bat_quote(str(destination))}',
        'if errorlevel 1 goto ssgen_finish', 'call train_a.bat',
        'set "SSGEN_EXIT_CODE=%ERRORLEVEL%"', 'if not "%SSGEN_EXIT_CODE%"=="0" goto ssgen_popd',
        'call train_b.bat', 'set "SSGEN_EXIT_CODE=%ERRORLEVEL%"', ':ssgen_popd', 'popd',
        ':ssgen_finish', 'pause', 'exit /b %SSGEN_EXIT_CODE%',
    ])
    output_dir = (args.output_dir or destination / 'LoRA').resolve()
    target = (args.comfyui_dir / 'models' / 'loras' / args.family / args.name).resolve()
    metadata = {'slider': {'family': args.family, 'name': args.name, 'sd_scripts_dir': str(root),
                          'python': str(python), 'temp_dir': str(temp_dir), 'output_dir': str(output_dir),
                          'comfyui_dir': str(args.comfyui_dir.absolute()),
                          'copy_retries': args.copy_retries, 'copy_wait': args.copy_wait}}
    outputs[destination / 'slider.toml'] = dump_toml(metadata).encode('utf-8')
    outputs[destination / 'train.bat'] = train
    outputs[destination / 'copylora.bat'] = copy_batch(output_dir, target, args.name + '.safetensors', args.copy_retries, args.copy_wait)
    # 元テンプレート・入力画像は上書きしない。全検証が終わってからまとめて書く。
    ask_generated_timestep_settings(args, outputs)
    return write_outputs(outputs)


def slider_checkpoints(temp_dir, side):
    choices = []
    for path in temp_dir.glob(f'{side}*.safetensors'):
        if not path.is_file():
            continue
        match = re.fullmatch(re.escape(side) + r'-(\d+)\.safetensors', path.name)
        if match:
            choices.append((int(match[1]), path))
        elif path.name == f'{side}.safetensors':
            choices.append((100, path))
    return sorted(choices, key=lambda item: (item[0], item[1].name))


def read_lora_rank(path):
    """safetensorsのJSONヘッダーだけを読み、LoRAモジュールの最大rankを取得する。"""
    try:
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('safetensorsヘッダーがありません')
            size = int.from_bytes(prefix, 'little')
            if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
                raise ValueError('safetensorsヘッダーのサイズが不正です')
            header = json.loads(stream.read(size).decode('utf-8'))
        if not isinstance(header, dict):
            raise ValueError('safetensorsヘッダーはオブジェクトである必要があります')
        ranks = []
        for key, tensor in header.items():
            if not key.endswith('.lora_down.weight'):
                continue
            shape = tensor.get('shape') if isinstance(tensor, dict) else None
            up = header.get(key.removesuffix('.lora_down.weight') + '.lora_up.weight')
            up_shape = up.get('shape') if isinstance(up, dict) else None
            for dimensions in (shape, up_shape):
                if not isinstance(dimensions, list) or len(dimensions) not in (2, 4) or any(
                    type(value) is not int or value < 1 for value in dimensions
                ):
                    raise ValueError(f'LoRAのテンソル形状が不正です: {key}')
            if shape[0] != up_shape[1]:
                raise ValueError(f'LoRAのup/downのrankが一致しません: {key}')
            ranks.append(shape[0])
        if not ranks:
            raise ValueError('LoRAのrankを取得できません（lora_down/up.weightがありません）')
        return max(ranks)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f'{path}: {error}') from error


def generate_slider_merge(args):
    destination = args.merge_slider.resolve()
    metadata = read_toml(destination / 'slider.toml')['slider']
    selected = []
    interactive = args.epoch_a is None or args.epoch_b is None
    if interactive:
        enable_terminal()
    for side, epoch in [('a', args.epoch_a), ('b', args.epoch_b)]:
        choices = slider_checkpoints(Path(metadata['temp_dir']), side)
        if not choices:
            raise ValueError(f'{side.upper()}のチェックポイントがありません。先に学習してください')
        if epoch is not None:
            matches = [path for number, path in choices if number == epoch]
            if len(matches) != 1:
                raise ValueError(f'{side.upper()}のepoch {epoch}が見つからないか複数あります。ファイル選択で指定してください')
            selected.append(matches[0])
        else:
            selected.append(select_option(f'{side.upper()}のマージ対象',
                [(f'epoch {number}: {path.name}', path) for number, path in choices]))
    ranks = [read_lora_rank(path) for path in selected]
    new_rank = max(ranks)
    print(f'マージrank: A={ranks[0]}, B={ranks[1]} → {new_rank}')
    output_dir = Path(metadata['output_dir'])
    save_to = output_dir / (metadata['name'] + '.safetensors')
    root = Path(metadata['sd_scripts_dir'])
    command = ' '.join(map(bat_quote, [metadata['python'], str(root / 'networks' / 'svd_merge_lora.py'),
        '--save_to', str(save_to), '--models', str(selected[0]), str(selected[1]), '--ratios', '-1', '1',
        '--new_rank', str(new_rank), '--device', 'cuda', '--save_precision', 'bf16']))
    checks = [f'if not exist {bat_quote(str(path))} goto ssgen_finish' for path in selected]
    checks += [f'if not exist {bat_quote(str(output_dir))} mkdir {bat_quote(str(output_dir))}',
               f'if not exist {bat_quote(str(output_dir))} goto ssgen_finish']
    target = Path(metadata['comfyui_dir']) / 'models' / 'loras' / metadata['family'] / metadata['name']
    return write_outputs({destination / 'merge.bat': command_batch(root, command, checks),
        destination / 'copylora.bat': copy_batch(output_dir, target, save_to.name, metadata['copy_retries'], metadata['copy_wait'])})


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('1以上を指定してください')
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('0以上を指定してください')
    return number


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_dir', nargs='?', type=Path, help='LoRAフォルダ。単独指定で学習設定ウィザード、追加オプション指定でデータセット編集・追記')
    parser.add_argument('--family', choices=['sdxl', 'anima', 'moss'])
    parser.add_argument('--kind', default='character', choices=['character', 'style', 'concept', 'slider'])
    parser.add_argument('--interactive', action='store_true', help='引数なしで起動した場合と同じ対話設定を開始')
    parser.add_argument('--merge-slider', type=Path, metavar='LORA_DIR', help='学習済みA/Bからmerge.batを生成（マージ自体は実行しない）')
    parser.add_argument('--epoch-a', type=positive, help='Aのマージ対象エポック。省略時はファイルをカーソル選択')
    parser.add_argument('--epoch-b', type=positive, help='Bのマージ対象エポック。省略時はファイルをカーソル選択')
    parser.add_argument('--name', help='LoRAの出力名。省略して --add-dataset で既存設定に追記')
    parser.add_argument('--model', help='省略時はテンプレートのモデルを使用')
    parser.add_argument('--templates-dir', type=Path, default=Path(__file__).parent / 'templates')
    parser.add_argument('--template', type=Path)
    parser.add_argument('--destination', type=Path, help='生成先、またはデータセット追記対象のフォルダ')
    parser.add_argument('--sd-scripts-dir', type=Path, help='省略時はテンプレートのtrain.batから取得')
    parser.add_argument('--python', type=Path)
    parser.add_argument('--train-script')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--logging-dir', type=Path)
    parser.add_argument('--dataset-config', type=Path, nargs='?', const=Path('dataset.toml'),
                        help='--name指定時は生成元TOML。それ以外はMOSSのdataset/config編集ウィザード（省略時dataset.toml）')
    parser.add_argument('--add-dataset', type=Path, action='append', default=[], metavar='IMAGE_DIR')
    parser.add_argument('--add-dataset-with-prefix', action='append', default=[], metavar='PREFIX',
                        help='対象フォルダ以下を再帰探索し、名前がPREFIXから始まるディレクトリを一括追記（重複は保持）')
    parser.add_argument('--repeats', type=positive, default=10)
    parser.add_argument('--dataset-index', type=nonnegative, default=0, help='追加先datasetsの番号（0始まり）')
    parser.add_argument('--keep-template-datasets', action='store_true', help='元の画像パスも引き継ぐ')
    parser.add_argument('--allow-empty-dataset', action='store_true', help='画像パスを後から追加するため空のデータセットで生成')
    parser.add_argument('--comfyui-dir', type=Path, default=DEFAULT_COMFYUI)
    parser.add_argument('--copy-retries', type=nonnegative, default=60)
    parser.add_argument('--copy-wait', type=positive, default=5, help='リトライ間隔（秒）')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--overwrite', '--force', dest='force', action='store_true',
                        help='既存設定を引き継がずテンプレートから新規作成（--forceも同義）。単独指定で対話設定を開始')
    return parser


def main():
    parser = make_parser()
    try:
        args = parser.parse_args()
        wizard_options = {'--interactive', '--overwrite', '--force'}
        project_wizard = (args.project_dir is not None and args.dataset_config is None
                          and all(token in wizard_options for token in sys.argv[1:] if token.startswith('--')))
        if project_wizard:
            args.destination = args.project_dir.absolute()
            args.name = args.destination.name
            args._preset_project_name = True
            args.project_dir = None
            args = interactive_settings(parser, args)
        elif all(token in wizard_options for token in sys.argv[1:]):
            args = interactive_settings(parser, args)
        else:
            if args.interactive:
                parser.error('--interactive と併用できるオプションは --overwrite / --force のみです')
        if args.merge_slider:
            allowed = {'--merge-slider', '--epoch-a', '--epoch-b'}
            if args.project_dir or any(token.split('=')[0] not in allowed for token in sys.argv[1:] if token.startswith('--')):
                parser.error('--merge-slider と併用できるオプションは --epoch-a / --epoch-b のみです')
            for path in generate_slider_merge(args):
                print(path)
            return
        if args.epoch_a is not None or args.epoch_b is not None:
            parser.error('--epoch-a / --epoch-b は --merge-slider と指定してください')
        if args.dataset_config is not None and not args.name:
            allowed = {'--dataset-config', '--family', '--destination'}
            if any(token.split('=')[0] not in allowed for token in sys.argv[1:] if token.startswith('--')):
                parser.error('データセット編集では --dataset-config / --family / --destination とLoRAフォルダのみ指定できます')
            if args.project_dir and args.destination:
                parser.error('LoRAフォルダと --destination は併用できません')
            edit_dataset_config(args)
            return
        if args.project_dir:
            if args.destination or args.name:
                parser.error('位置引数のLoRAフォルダは --destination / --name と併用できません')
            args.destination = args.project_dir
        if args.name:
            if args.add_dataset_with_prefix:
                parser.error('--add-dataset-with-prefix は生成済みのLoRAフォルダに対して指定してください')
            if not args.family:
                parser.error('新規生成には --family が必要です')
            paths = generate(args)
        else:
            if not (args.add_dataset or args.add_dataset_with_prefix) or not args.destination:
                parser.error('追記にはLoRAフォルダと --add-dataset または --add-dataset-with-prefix が必要です')
            # 追記モードで指定した生成オプションを黙って無視しない。
            allowed = {'--destination', '--add-dataset', '--add-dataset-with-prefix', '--repeats', '--dataset-index'}
            if any(token.split('=')[0] not in allowed for token in sys.argv[1:] if token.startswith('--')):
                parser.error('追記モードではLoRAフォルダと --destination / --add-dataset / --add-dataset-with-prefix / --repeats / --dataset-index のみ指定できます')
            path = args.destination.resolve() / 'dataset.toml'
            dataset = read_toml(path)
            candidates = list(args.add_dataset)
            if args.add_dataset_with_prefix:
                candidates.extend(find_datasets_with_prefix(path.parent, args.add_dataset_with_prefix))
            added, skipped = add_datasets(dataset, candidates, args.repeats, args.dataset_index,
                                          skip_existing=bool(args.add_dataset_with_prefix))
            content = dump_toml(dataset)
            tomllib.loads(content)
            if added:
                path.write_text(content, encoding='utf-8')
            if args.add_dataset_with_prefix:
                print(f'追加: {added}件 / 登録済みのため維持: {skipped}件')
            paths = [path]
        for path in paths:
            print(path)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f'エラー: {error}\n')
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, '\n設定を中止しました。\n')


if __name__ == '__main__':
    main()

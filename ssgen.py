"""sd-scripts 用の学習 TOML と Windows bat を生成する。"""

import argparse
from collections.abc import MutableMapping
from pathlib import Path, PureWindowsPath
import re
import sys
import os

import tomllib
import json
from datetime import date, datetime, time


DEFAULT_COMFYUI = Path('C:/Software/ComfyUI')
FAMILIES = [('SDXL', 'sdxl'), ('Anima', 'anima'), ('MOSS', 'moss')]
PURPOSES = [('キャラLoRA', 'character'), ('画風LoRA', 'style'),
            ('コンセプトLoRA', 'concept'), ('スライダーLoRA（準備中）', 'slider')]


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
    while True:
        name = input('\n2. LoRA名: ').strip()
        try:
            validate_name(name)
            destination = Path.cwd() / name
            if destination.exists():
                raise ValueError('同名のファイルまたはフォルダが存在します。別のLoRA名を入力してください')
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
    while True:
        kind = select_option('5. 用途', PURPOSES)
        if kind != 'slider':
            args.kind = kind
            return
        print('スライダーLoRA（A/Bの過学習後マージ）は準備中です。他の用途を選択してください。')


def ask_datasets(args):
    print('\n6. 画像フォルダを1つずつ入力してください（空欄で終了、後から --add-dataset でも追加可能）。')
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
WIZARD_STEPS = [ask_family, ask_name, ask_comfyui, ask_model, ask_purpose, ask_datasets]


def interactive_settings(parser):
    enable_terminal()
    args = parser.parse_args([])
    print('ssgen - 学習設定ウィザード（Ctrl+C / 選択画面のEscで中止）')
    print(f'スクリプト: {Path(__file__).absolute()}')
    for step in WIZARD_STEPS:
        step(args)
    print(f'\n出力先: {args.destination}')
    if not args.add_dataset:
        print('データセットは未登録です。学習前に --add-dataset で追加してください。')
    return args

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
        return '{' + ', '.join(f'{toml_string(key)} = {toml_value(item)}'
                               for key, item in value.items()) + '}'
    raise TypeError(f'TOMLに出力できない値です: {type(value).__name__}')


def dump_toml(document):
    """辞書をTOMLへ出力。通常テーブル・配列テーブルを再帰的に扱う。"""
    lines = []

    def array_of_tables(value):
        return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)

    def write_table(table, path=(), array=False):
        if path:
            if lines:
                lines.append('')
            name = '.'.join(toml_string(key) for key in path)
            lines.append(f'[[{name}]]' if array else f'[{name}]')
        for key, value in table.items():
            if not isinstance(value, dict) and not array_of_tables(value):
                lines.append(f'{toml_string(key)} = {toml_value(value)}')
        for key, value in table.items():
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


def generate(args):
    validate_name(args.name)
    if args.kind == 'slider':
        raise ValueError('スライダーLoRA（A/Bの過学習後マージ）は準備中です')
    template_dir = args.templates_dir / args.family
    # 用途別設定があれば優先。なければファミリー共通設定をそのまま使用する。
    config_template = args.template or template_dir / args.kind / 'config.toml'
    if not args.template and not config_template.exists():
        config_template = template_dir / 'config.toml'
    dataset_template = args.dataset_config or template_dir / 'dataset.toml'
    document = read_toml(config_template)
    dataset = read_toml(dataset_template)
    destination = (args.destination or Path('generated') / args.family / args.name).resolve()
    files = [destination / name for name in ('config.toml', 'dataset.toml', 'train.bat', 'copylora.bat')]
    if any(path.exists() for path in files) and not args.force:
        raise ValueError('出力ファイルが存在します。追記には --add-dataset と --destination のみ、再生成には --force を指定してください')
    sources = {config_template.resolve(), dataset_template.resolve(), (template_dir / 'train.bat').resolve(), (template_dir / 'copylora.bat').resolve()}
    if any(path in sources for path in files):
        raise ValueError('出力先をテンプレートと別の場所にしてください')
    if not args.keep_template_datasets:
        for entry in dataset.get('datasets', []):
            if 'subsets' in entry:
                del entry['subsets']
    if args.add_dataset:
        add_datasets(dataset, args.add_dataset, args.repeats, args.dataset_index)
    if not args.allow_empty_dataset and not any(entry.get('subsets') for entry in dataset.get('datasets', [])):
        raise ValueError('--add-dataset で画像フォルダを指定してください')
    overrides = {}
    for assignment in args.set:
        key, separator, raw = assignment.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_]+', key):
            raise ValueError('--set は key=TOML値 の形式で指定してください')
        overrides[key] = tomllib.loads(f'value = {raw}')['value']
    if args.model:
        overrides['pretrained_model_name_or_path'] = args.model
    output_dir = (args.output_dir or destination / 'LoRA').resolve()
    overrides.update(output_name=args.name, output_dir=str(output_dir),
                     logging_dir=str((args.logging_dir or destination / 'log').resolve()),
                     dataset_config=str(files[1]))
    for key, value in overrides.items():
        update_key(document, key, value)
    default_root, default_python, default_script = training_defaults(template_dir)
    root = (args.sd_scripts_dir or default_root).resolve()
    python = args.python or (root / 'venv' / 'Scripts' / 'python.exe' if args.sd_scripts_dir else default_python)
    command = ' '.join(map(bat_quote, [str(python.resolve()), '-m', 'accelerate.commands.launch',
                                      '--num_cpu_threads_per_process', '1', args.train_script or default_script,
                                      '--config_file', str(files[0])]))
    train = batch_text([f'pushd {bat_quote(str(root))}', 'if errorlevel 1 exit /b 1', command,
                        'set "SSGEN_EXIT_CODE=%ERRORLEVEL%"', 'popd', 'exit /b %SSGEN_EXIT_CODE%'])
    target = (args.comfyui_dir / 'models' / 'loras' / args.family / args.name).resolve()
    copy = batch_text([
        f'if not exist {bat_quote(str(output_dir / "*.safetensors"))} exit /b 1',
        f'robocopy {bat_quote(str(output_dir))} {bat_quote(str(target))} "*.safetensors" /COPY:DAT /R:{args.copy_retries} /W:{args.copy_wait}',
        'if errorlevel 8 exit /b %ERRORLEVEL%', 'exit /b 0',
    ])
    contents = [dump_toml(document).encode('utf-8'), dump_toml(dataset).encode('utf-8'), train, copy]
    for content in contents[:2]:
        tomllib.loads(content.decode('utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    for path, content in zip(files, contents):
        path.write_bytes(content)
    return files


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
    parser.add_argument('project_dir', nargs='?', type=Path, help='データセット追記対象のLoRAフォルダ')
    parser.add_argument('--family', choices=['sdxl', 'anima', 'moss'])
    parser.add_argument('--kind', default='character', choices=['character', 'style', 'concept', 'slider'])
    parser.add_argument('--interactive', action='store_true', help='引数なしで起動した場合と同じ対話設定を開始')
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
    parser.add_argument('--dataset-config', type=Path, help='生成元のデータセットTOML')
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
    parser.add_argument('--force', action='store_true')
    return parser


def main():
    parser = make_parser()
    try:
        if len(sys.argv) == 1 or sys.argv[1:] == ['--interactive']:
            args = interactive_settings(parser)
        else:
            args = parser.parse_args()
            if args.interactive:
                parser.error('--interactive は他のオプションと併用できません')
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

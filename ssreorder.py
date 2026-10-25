"""画像をPNGへ変換し、更新日時の古い順に0000.pngから連番へ整理する。"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid


CONVERT_EXTENSIONS = {'.jpeg', '.jpg', '.webp', '.wepb'}


def pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError('画像変換にはPillowが必要です。python -m pip install Pillow を実行してください。') from error
    return Image, ImageOps


def convert_to_png(source):
    """PNGを検証してから元画像を削除する。既存PNGは上書きしない。"""
    Image, ImageOps = pillow()
    original_stat = source.stat()
    with tempfile.NamedTemporaryFile(prefix='.ssreorder-', suffix='.png', dir=source.parent, delete=False) as temporary:
        temp = Path(temporary.name)
    try:
        with Image.open(source) as image:
            info = image.info.copy()
            frames, durations = [], []
            for index in range(getattr(image, 'n_frames', 1)):
                image.seek(index)
                frame = ImageOps.exif_transpose(image)
                mode = 'RGBA' if 'A' in frame.getbands() or 'transparency' in frame.info else 'RGB'
                frames.append(frame.convert(mode))
                durations.append(image.info.get('duration', 0))
            options = {}
            if info.get('icc_profile'):
                options['icc_profile'] = info['icc_profile']
            exif = frames[0].getexif()
            if exif:
                options['exif'] = exif.tobytes()
            if len(frames) > 1:
                options.update(save_all=True, append_images=frames[1:], duration=durations, loop=info.get('loop', 0))
            frames[0].save(temp, format='PNG', **options)
        with Image.open(temp) as saved:
            if saved.format != 'PNG' or getattr(saved, 'n_frames', 1) != len(frames):
                raise ValueError('PNGの形式またはフレーム数を確認できません')
            for index in range(len(frames)):
                saved.seek(index)
                saved.load()
                if saved.size != frames[index].size:
                    raise ValueError('PNGの画像サイズが変わっています')
        os.utime(temp, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        index = 0
        caption = source.with_suffix('.txt')
        while True:
            destination = source.with_name(source.stem + (f'_{index}' if index else '') + '.png')
            if destination.stem != source.stem and caption.is_file() and destination.with_suffix('.txt').exists():
                index += 1
                continue
            try:
                target = destination.open('xb')
            except FileExistsError:
                index += 1
                continue
            try:
                with target, temp.open('rb') as encoded:
                    shutil.copyfileobj(encoded, target)
                os.utime(destination, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            break
        # 同名PNGとの衝突で名前を変えた場合、共有キャプションも新しい画像に複製する。
        if destination.stem != source.stem and caption.is_file():
            with destination.with_suffix('.txt').open('xb') as target, caption.open('rb') as original:
                shutil.copyfileobj(original, target)
        source.unlink()
        return destination
    finally:
        temp.unlink(missing_ok=True)


def move_no_replace(source, destination):
    if destination.exists():
        raise FileExistsError(f'移動先が既に存在します: {destination}')
    source.rename(destination)


def reorder_pngs(directory):
    """2段階リネームで既存の連番との衝突を避け、失敗時は元の名前へ戻す。"""
    images = sorted((path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == '.png'),
                    key=lambda path: (path.stat().st_mtime_ns, path.name.casefold(), path.name))
    moves = []
    for index, source in enumerate(images):
        destination = directory / f'{index:04d}.png'
        moves.append((source, destination))
        caption = source.with_suffix('.txt')
        if caption.is_file():
            moves.append((caption, destination.with_suffix('.txt')))
    sources = {os.path.normcase(str(source.absolute())) for source, _ in moves}
    for _, destination in moves:
        caption = destination.with_suffix('.txt')
        if destination.suffix == '.png' and caption.exists() and os.path.normcase(str(caption.absolute())) not in sources:
            raise FileExistsError(f'画像と無関係なキャプションが移動先に存在します: {caption}')
    for _, destination in moves:
        if destination.exists() and os.path.normcase(str(destination.absolute())) not in sources:
            raise FileExistsError(f'既存ファイルを上書きできません: {destination}')
    locations = {source: source for source, _ in moves}
    try:
        for source, _ in moves:
            temporary = directory / f'.ssreorder-{uuid.uuid4().hex}'
            move_no_replace(source, temporary)
            locations[source] = temporary
        for source, destination in moves:
            move_no_replace(locations[source], destination)
            locations[source] = destination
    except BaseException:
        # 元の名前同士が循環しても復旧できるよう、復旧も2段階で行う。
        for source, current in list(locations.items()):
            temporary = directory / f'.ssreorder-{uuid.uuid4().hex}'
            move_no_replace(current, temporary)
            locations[source] = temporary
        for source, current in locations.items():
            move_no_replace(current, source)
        raise
    for source, destination in moves:
        if source.suffix.lower() == '.png':
            print(f'{source.name} -> {destination.name}')
    return len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='整理する画像フォルダ（直下のみ）')
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        parser.exit(1, f'エラー: フォルダが見つかりません: {directory}\n')
    try:
        sources = sorted((path for path in directory.iterdir()
                          if path.is_file() and path.suffix.lower() in CONVERT_EXTENSIONS), key=lambda path: path.name.casefold())
        if sources:
            pillow()
        errors = 0
        for source in sources:
            try:
                destination = convert_to_png(source)
                print(f'PNG変換: {source.name} -> {destination.name}')
            except Exception as error:
                errors += 1
                print(f'エラー: {source.name}: {error}', file=sys.stderr)
        if errors:
            parser.exit(1, f'{errors}件の変換に失敗したため連番への変更を中止しました。失敗した元画像を確認してください。成功済みの変換は保持します。\n')
        count = reorder_pngs(directory)
        print(f'完了: PNG変換 {len(sources)}件 / 連番整理 {count}件')
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'エラー: {error}\n')


if __name__ == '__main__':
    main()

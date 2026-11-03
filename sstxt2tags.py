"""同じフォルダに同名画像がある .txt だけを .tags にリネームします。"""

import argparse
import os
from pathlib import Path
import sys


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".jxl"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path,
                        help="対象フォルダ（複数指定可、省略時はカレントフォルダ）")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True,
                        help="サブフォルダも探索（既定で有効、直下のみは --no-recursive）")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="変更予定のみ表示")
    args = parser.parse_args(argv)
    roots = [p.expanduser().resolve() for p in (args.directories or [Path.cwd()])]
    for root in roots:
        if not root.is_dir():
            print(f"エラー: フォルダが見つかりません: {root}", file=sys.stderr)
            return 2
    renamed = skipped = failed = 0
    try:
        sources = {}
        for root in roots:
            paths = root.rglob("*") if args.recursive else root.iterdir()
            for path in paths:
                if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".txt":
                    # 親子フォルダや同じフォルダを指定しても1回だけ処理する。
                    sources.setdefault(os.path.normcase(str(path)), (path, root))
        image_stems = {}
        for source, root in sorted(sources.values(), key=lambda pair: str(pair[0]).casefold()):
            if source.stem.lower().endswith("_knowledge"):
                print(f"スキップ（補助知識ファイル）: {source.relative_to(root)}")
                skipped += 1
                continue
            if source.parent not in image_stems:
                image_stems[source.parent] = {
                    p.stem.casefold() for p in source.parent.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                }
            if source.stem.casefold() not in image_stems[source.parent]:
                print(f"スキップ（同名画像なし）: {source.relative_to(root)}")
                skipped += 1
                continue
            destination = source.with_suffix(".tags")
            if os.path.lexists(destination):
                print(f"スキップ（移動先が存在）: {source.relative_to(root)}")
                skipped += 1
                continue
            try:
                if not args.dry_run:
                    if os.name == "nt":
                        # Windowsのrenameは移動先が存在すると失敗する。
                        source.rename(destination)
                    else:
                        # POSIXのrenameによる既存ファイル上書きを避ける。
                        os.link(source, destination)
                        source.unlink()
                print(f"{source.relative_to(root)} -> {destination.relative_to(root)}")
                renamed += 1
            except OSError as error:
                print(f"エラー: {source}: {error}", file=sys.stderr)
                failed += 1
    except OSError as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1
    label = "変更予定" if args.dry_run else "変更"
    print(f"{label} {renamed}件 / スキップ {skipped}件 / 失敗 {failed}件")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

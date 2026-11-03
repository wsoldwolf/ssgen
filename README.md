# ssgen

sd-scriptsの学習設定生成と、画像・キャプション整理、キャプション生成、学習ログの可視化を行うツール集です。Windows用batとPythonスクリプトを用意しています。

## ツール別ドキュメント

| ツール | 用途 | ドキュメント |
| --- | --- | --- |
| ssgen | 学習用TOML・batの生成、MOSS設定編集、スライダーLoRA | [ssgenの使い方](docs/ssgen.md) |
| ssntagger | Gemma 4による画像キャプション生成 | [ssntaggerの使い方](docs/ssntagger.md) |
| sstxt2tags | txtをtagsへ一括リネーム | [sstxt2tagsの使い方](docs/sstxt2tags.md) |
| ssreorder | 画像のPNG変換と連番整理 | [ssreorderの使い方](docs/ssreorder.md) |
| ssplot | 学習ログからグラフPNGを作成 | [ssplotの使い方](docs/ssplot.md) |

## セットアップ

`ssgen` 本体はPython 3.11以上の標準ライブラリだけで動作します。`setup.bat` は共通の `venv` を作成し、ssplot・ssreorder用の依存パッケージもインストールします。

`ssntagger` は `setup_ssntagger.bat` で専用の `venv-ssntagger` を準備します。CUDAとC++ビルド環境が必要です。詳しい前提条件と起動方法は各ツールのドキュメントを参照してください。

## Tips

[Tipsの索引](tips/README.md)を参照してください。sd-scriptsの追加パッケージについては[Windows / cmd.exe向け補足事項](tips/sd-scripts.md)に記載しています。

## 検証

```powershell
python -m unittest discover -s tests -v
```

外部パッケージに依存しないことは `python -S -m unittest discover -s tests -v` でも確認できます。

## ライセンス

本プロジェクトは GNU General Public License version 3（GPL-3.0-only）で公開しています。ライセンス全文は [LICENSE](LICENSE) を参照してください。

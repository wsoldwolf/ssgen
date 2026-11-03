# sstxt2tags

[プロジェクトREADMEに戻る](../README.md)

コマンド例はプロジェクト直下から実行する想定です。

指定したすべてのフォルダと、その全サブフォルダにある `.txt` のうち、同じフォルダに同名画像が存在するものだけを `.tags` にリネームします。例えば `0001.png` があれば `0001.txt` を変更します。ファイル名・拡張子の大文字小文字は区別しません。画像拡張子は `.png` / `.jpg` / `.jpeg` / `.webp` / `.bmp` / `.gif` / `.tif` / `.tiff` / `.avif` / `.jxl` に対応します。

`_knowledge.txt` と画像別の `*_knowledge.txt` は、同名画像があっても変更しません。ファイル内容は変更せず、同名の `.tags` が存在する場合は元のtxtを残してスキップします。指定フォルダが重複・包含していても同じファイルは1回だけ処理します。旧版で変更された `_knowledge.tags` は `_knowledge.txt` へ戻してください。

```powershell
.\sstxt2tags.bat "E:\datasets\images"
# 複数フォルダをまとめて処理（サブフォルダも含む）
.\sstxt2tags.bat "E:\datasets\images1" "E:\datasets\images2"
# 実行前に変更予定を確認
.\sstxt2tags.bat "E:\datasets\images" --dry_run
# 直下のファイルだけ処理
.\sstxt2tags.bat "E:\datasets\images" --no-recursive
```

フォルダ省略時はカレントフォルダを使用します。`python sstxt2tags.py "画像フォルダ"` でも実行でき、外部パッケージは不要です。変更後はssntaggerの `--reference_extension .tags` で既存タグを参照できます。


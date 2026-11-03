# ssplot

[プロジェクトREADMEに戻る](../README.md)

コマンド例はプロジェクト直下から実行する想定です。

`ssplot` はsd-scriptsのTensorBoardイベントログを読み取り、lossや学習率のグラフをPNGに保存します。ログファイルを1つ指定してください。

## 初回セットアップ

Python 3.11以上とWindowsの `py` コマンドが必要です。プロジェクト直下で次を実行します。`setup.bat` は `py -m venv venv` で環境を作成し、その環境に本プロジェクト・TensorBoard・Matplotlibをインストールします。既存の `venv` がある場合は再利用します。初回インストールにはインターネット接続が必要です。

```powershell
.\setup.bat
```

手動でセットアップする場合は次を実行します。

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -e ".[plot]"
```

## 実行方法

PowerShell・cmd.exeのどちらでも、プロジェクト直下から次のように実行できます。仮想環境の有効化は不要です。空白を含むパスにも対応するため、ログのパスを引用符で囲んでください。

```powershell
.\ssplot.bat "E:\DataSetMoss\log\events.out.tfevents.1792572975.PCX03.51700.0"
# インストールされたコマンドを直接実行する場合
.\venv\Scripts\ssplot.exe "path/to/logfile"
# Pythonスクリプトとして実行する場合
.\venv\Scripts\python.exe ssplot.py "path/to/logfile"
```

`venv\Scripts` がPATHに含まれる環境では、`ssplot "path/to/logfile"` でも実行できます。

入力ファイル名に `.png` を付けた画像を同じディレクトリに保存し、保存先の絶対パスを表示します（`path/to/logfile` → `path/to/logfile.png`）。上の例では `E:\DataSetMoss\log\events.out.tfevents.1792572975.PCX03.51700.0.png` を生成します。既存の同名PNGは更新します。ログは変更しません。

## 指標とオプション

`loss/average`・`loss/current`・`lr/unet`・`loss/epoch_average`を優先し、ログに記録されたその他のスカラー指標も自動でプロットします。`loss/current`には既定で200サンプルの後方移動平均を重ねます。

```powershell
# 移動平均の窓幅を500サンプルに変更
.\ssplot.bat "path/to/logfile" --window 500
# ヘルプを表示
.\ssplot.bat --help
```

`--window` は1以上の整数です。学習途中のログも、その時点で読み取れるイベントまで出力します。更新するには同じコマンドを再実行してください。環境や依存パッケージが見つからない場合は `setup.bat` を再実行してください。


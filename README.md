# ssgen

sd-scriptsの学習用TOMLとWindows用batを生成します。Python 3.11以上の標準ライブラリだけで動作します。

## 起動

外部パッケージのインストールは不要です。

```powershell
python ssgen.py
python ssgen.py LoRA名/ --add-dataset-with-prefix "charname_"
```

以下の説明にある `ssgen` は `python ssgen.py` に置き換えて実行できます。どのフォルダからも起動する場合は `python E:\ComfyUI\projects\ssgen\ssgen.py` のようにスクリプトのフルパスを指定します。

`ssgen` コマンドとして登録したい場合だけ、任意で次を実行してください。

```powershell
python -m pip install -e .
ssgen --help
```

テンプレートはこのソースフォルダとともに使用します。

## 対話形式で設定

出力先にしたいカレントディレクトリで、引数を付けずに起動します。

```powershell
ssgen
# または
python E:\ComfyUI\projects\ssgen\ssgen.py
```

1. ファミリ（SDXL / Anima / MOSS）を選択
2. LoRA名を入力
3. ComfyUIパスを入力（既定 `C:\Software\ComfyUI\`）
4. 探索された学習ベースモデルを選択
5. 用途（キャラ / 画風 / コンセプト / スライダー）を選択
6. 画像フォルダを必要な数だけ入力し、リピート数を指定（既定10）

選択画面では上下キーで移動し、Enterで決定します。長い一覧はスクロールします。Ctrl+C、または選択画面でEscを押すと中止します。Windows Terminal、PowerShellなどの対話端末から実行してください。`ssgen --interactive` でも開始できます。

モデルは、指定したComfyUIパスを基準に次のフォルダ以下を再帰探索します（`<family>` は `sdxl` / `anima` / `moss`）。

- `<family>/`
- `models/<family>/`
- `models/checkpoints/<family>/`
- `models/diffusion_models/<family>/`

対応する拡張子は `.safetensors`・`.ckpt`・`.pt`・`.pth`・`.bin` です。選択したファイルの絶対パスをconfig.tomlへ設定します。VAE・テキストエンコーダーのパスはテンプレートの値を保持します。

設定後、**カレントディレクトリの `LoRA名/`** に4ファイルを生成します。同名フォルダが存在すると別名の入力を求めます。学習や複製は自動実行しません。

画像フォルダの入力は空欄で終了できます。未登録の場合は、学習前に次のように追加してください。

```powershell
ssgen --destination ".\LoRA名" --add-dataset "E:\datasets\images"
```

スライダーLoRAのA/B過学習・マージは準備中のため、選択すると他の用途の選択へ戻ります。キャラ・画風・コンセプトの学習条件は用途別テンプレートがあればそれを使い、なければ共通設定を保持します。

対話処理を含む実装は `ssgen.py` の1ファイルです。項目を追加するときは設定関数を定義して `WIZARD_STEPS` に追加します。ファミリと用途の選択肢は `FAMILIES` / `PURPOSES` で管理します。既存のCLIによる生成・データセット追記も利用できます。

## テンプレート

`templates/sdxl/`、`templates/anima/`、`templates/moss/` の各フォルダを使用します。元ファイルは変更しません。

| ファイル | 用途 |
| --- | --- |
| config.toml | グローバル学習設定。モデル、LoRA名、保存先などを更新 |
| dataset.toml | general・datasetsの設定を保持して画像フォルダを設定 |
| train.bat | activate.batのパスと学習スクリプト名を読み取り、batを再生成 |
| copylora.bat | 旧バッチの参考資料。新しい複製処理で再生成 |

`--kind character`（既定）・`style`・`concept` を指定できます。`templates/anima/style/config.toml` のような用途別設定があれば優先し、なければファミリー共通のconfig.tomlを使います。用途名だけで学習条件は変更しません。`slider` は準備中のため生成できません。

配置済みのmossテンプレートはAnimaと同じ設定・学習スクリプトです。必要に応じてテンプレートを修正するか `--train-script` などで指定してください。

## 新規生成

```powershell
ssgen --family anima --name my_character --add-dataset "E:\datasets\my_character"
```

`generated/anima/my_character/` に `config.toml`、`dataset.toml`、`train.bat`、`copylora.bat` を生成します。

モデルはテンプレートの値を使用します。`output_name` は指定したLoRA名、`output_dir` は生成先の `LoRA/`、`logging_dir` は生成先の `log/`、`dataset_config` は生成したdataset.tomlの絶対パスに書き換えます。

テンプレートの既存画像パスは新規生成時に除き、`--add-dataset` のパスを設定します。リピート数の既定値は10です。元の画像パスも保持したい場合だけ `--keep-template-datasets` を指定してください。画像フォルダは未作成でも登録できます。

```powershell
ssgen --family sdxl --kind style --name my_style --model "E:\models\base.safetensors" --destination "E:\training\my_style" --output-dir "E:\LoRA\my_style" --add-dataset "E:\datasets\style_a" --add-dataset "E:\datasets\style_b" --repeats 20 --sd-scripts-dir "E:\sd-scripts" --python "E:\sd-scripts\venv\Scripts\python.exe" --train-script sdxl_train_network.py
```

`--repeats` は同じコマンドで追加したすべてのパスに適用します。複数の `[[datasets]]` がある場合、`--dataset-index 1` のように追加先を0始まりで選べます。既存ファイルへの再生成には `--force` が必要です。

## 生成済みデータセットへの追記

```powershell
ssgen --destination "generated/anima/my_character" --add-dataset "E:\datasets\additional"
ssgen --destination "generated/anima/my_character" --add-dataset "E:\datasets\important" --repeats 30
```

`--name` を省略すると追記モードになります。dataset.tomlだけを更新し、既存のサブセットと各リピート数は保持します。同じパスの重複登録はエラーになります。

### プレフィックスで一括追加

```powershell
ssgen LoRA名/ --add-dataset-with-prefix "charname_"
```

`LoRA名/` 以下を再帰探索し、ディレクトリ名が `charname_` から始まるフォルダを、その直下の `dataset.toml` に一括追加します。例えば `charname_normal/`、`images/charname_smile/` が対象です。ファイルや、名前の途中にだけプレフィックスを含むフォルダは対象外です。Windowsに合わせ、大文字・小文字を区別しません。

新規項目のリピート数は既定10です。`--repeats 20` などで変更できます。既存のパスは別の `[[datasets]]` に登録されている場合もスキップし、古い項目のリピート数・追加設定を保持します。再実行しても重複登録しません。一致なし・全件登録済みの場合はファイルを書き換えません。

`--dataset-index` で追加先を変更できます。`--add-dataset-with-prefix` は複数指定でき、従来の `--destination LoRA名/` でも対象フォルダを指定できます。

## 学習と複製

`train.bat` を実行するとテンプレートから取得したsd-scriptsフォルダで、仮想環境のPythonからAccelerateを起動します。`--sd-scripts-dir` 指定時はそのフォルダ内の `venv/Scripts/python.exe` を使用します。`--python` で別のPythonも選べます。学習環境にはsd-scriptsの依存パッケージとAccelerateが必要です。

学習後に `copylora.bat` を実行すると、出力先の `*.safetensors` を `ComfyUI/models/loras/[sdxl,anima,moss]/LoRA名/` へ複製します。学習バッチからの自動実行はしません。

ComfyUIの場所の既定値は `C:\Software\ComfyUI\` です。対話設定で入力するか、CLIでは `--comfyui-dir` で変更できます。

Robocopyで複製し、ロックなどによる失敗時は既定で5秒間隔・最大60回再試行します。`--copy-retries` と `--copy-wait` で変更できます。複製先の既存ファイルを削除せず、同名ファイルは更新します。終了コード0〜7は成功、8以上は失敗として返します。対象ファイルがない場合も失敗します。仕様参照: [Microsoft Robocopy](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/robocopy)。

## その他の設定

- `--template PATH`: 任意のconfig.tomlを指定
- `--dataset-config PATH`: 生成元のdataset.tomlを指定
- `--logging-dir PATH`: ログ保存先を指定
- `--set 'network_dim=32' --set 'learning_rate=0.0001'`: TOML形式で学習設定を上書き

生成先・LoRA名・モデルなどの専用オプションは `--set` より優先します。TOMLは標準の `tomllib` で読み込み、`ssgen.py` 内の出力処理で再生成します。設定値は保持しますが、コメント・空行・元の引用符などの書式は引き継ぎません。元のテンプレートは変更しません。batはUTF-8・CRLFです。テンプレート内の追加モデルなどの相対パスは学習時のsd-scriptsフォルダ基準になるため、絶対パスを推奨します。

## 検証

```powershell
python -m unittest discover -s tests -v
```

外部パッケージに依存しないことは `python -S -m unittest discover -s tests -v` でも確認できます。

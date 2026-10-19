# ssgen

sd-scriptsの学習用TOMLとWindows用batを生成します。Python 3.11以上の標準ライブラリだけで動作します。

sd-scripts の追加パッケージについては [Windows / cmd.exe 向け補足事項](tips/sd-scripts.md) を参照してください。causal-conv1d のビルドと、flash-linear-attention / triton-windows のインストール手順を記載しています。

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
6. sd-scriptsパスを入力（ファミリ別の既定値を表示）
7. 画像フォルダを必要な数だけ入力し、リピート数を指定（既定10）

sd-scriptsパスは空欄でEnterを押すと次の値を使います。指定した場所を `train.bat` の `pushd` と `venv/Scripts/python.exe` の起点にします。フォルダが未作成でも設定できます。

| ファミリ | 既定のsd-scriptsパス |
| --- | --- |
| SDXL | `E:\DataSetSDXL\sd-scripts` |
| Anima | `E:\DataSetAnima\sd-scripts` |
| MOSS | `E:\DataSetMoss\sd-scripts` |

選択画面では上下キーで移動し、Enterで決定します。長い一覧はスクロールします。Ctrl+C、または選択画面でEscを押すと中止します。Windows Terminal、PowerShellなどの対話端末から実行してください。`ssgen --interactive` でも開始できます。

モデルは、指定したComfyUIパスを基準に次のフォルダ以下を再帰探索します（`<family>` は `sdxl` / `anima` / `moss`）。

- `<family>/`
- `models/<family>/`
- `models/checkpoints/<family>/`
- `models/diffusion_models/<family>/`

対応する拡張子は `.safetensors`・`.ckpt`・`.pt`・`.pth`・`.bin` です。選択したファイルの絶対パスをconfig.tomlへ設定します。VAE・テキストエンコーダーはファミリ別テンプレートを使用し、相対パスは入力したComfyUIパスを基準に絶対パス化します。外部の絶対パスを指定した場合はその値を保持します。

設定後、**カレントディレクトリの `LoRA名/`** に設定とbatを生成します。通常の用途では4ファイル、スライダーでは後述のA/B個別ファイルを生成します。同名フォルダが存在すると通常は別名の入力を求めます。既存フォルダへ再生成する場合は次のように起動してください。

```powershell
python ssgen.py --overwrite
# または
ssgen --interactive --overwrite
```

`--overwrite` 付きでは既存のLoRA名を入力して、そのまま対話設定を進められます。`config.toml`・`dataset.toml`・`train.bat`・`copylora.bat` は今回の設定で再生成します。既存dataset.tomlへの追記ではありません。画像フォルダ・学習済みLoRA・その他のファイルは保持します。同名の通常ファイルには上書きできません。CLI生成でも `--overwrite` を使用でき、従来の `--force` は同じ意味の別名として残しています。学習や複製は自動実行しません。

画像フォルダの入力は空欄で終了できます。未登録の場合は、学習前に次のように追加してください。

```powershell
ssgen --destination ".\LoRA名" --add-dataset "E:\datasets\images"
```

スライダーLoRAでは画像フォルダの入力を省略し、`LoRA名/a.png` と `LoRA名/b.png` を個別に学習する設定を生成します。キャラ・画風・コンセプトの学習条件は用途別テンプレートがあればそれを使い、なければ共通設定を保持します。

対話処理を含む実装は `ssgen.py` の1ファイルです。項目を追加するときは設定関数を定義して `WIZARD_STEPS` に追加します。ファミリと用途の選択肢は `FAMILIES` / `PURPOSES` で管理します。既存のCLIによる生成・データセット追記も利用できます。

## テンプレート

`templates/sdxl/`、`templates/anima/`、`templates/moss/` の各フォルダを使用します。元ファイルは変更しません。

| ファイル | 用途 |
| --- | --- |
| config.toml | グローバル学習設定。モデル、LoRA名、保存先などを更新 |
| dataset.toml | general・datasetsの設定を保持して画像フォルダを設定 |
| train.bat | activate.batのパスと学習スクリプト名を読み取り、batを再生成 |
| copylora.bat | 旧バッチの参考資料。新しい複製処理で再生成 |

`--kind character`（既定）・`style`・`concept`・`slider` を指定できます。`templates/anima/style/config.toml` のような用途別設定があれば優先し、なければファミリー共通のconfig.tomlを使います。キャラ・画風・コンセプトは用途名だけで学習条件は変更しません。スライダーは100エポックなど専用の設定を適用します。

学習スクリプトと補助モデルの引数は `FAMILY_SETTINGS` でファミリ別に管理します。学習スクリプトの明示変更には `--train-script` を使用してください。

### MOSS-Image設定

[最新sd-scriptsのMOSSガイド](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md)と[学習コード](https://github.com/kohya-ss/sd-scripts/blob/main/moss_image_train_network.py)を確認し、次の構成を使用します（2026-10-19確認）。

| 項目 | 設定 |
| --- | --- |
| 学習スクリプト | `moss_image_train_network.py` |
| LoRAモジュール | `networks.lora_generic` |
| テキストエンコーダー引数 | `text_encoder`（Animaの`qwen3`は使用しません） |
| テキストエンコーダー | `models/text_encoders/Moss/qwen3.5_9b_int8_convrot.safetensors` |
| VAE | `models/vae/Moss/flux2-vae.safetensors` |

この2つの相対パスは選択したComfyUIパスから解決します。Qwen3.5-9BのConvRot INT8版を使用し、非対応のW4A8 GPTQ版は使用しません。公式分割モデルなど別のファイルを使う場合は、テンプレートのパスか `--set 'text_encoder="..."'` / `--set 'vae="..."'` で変更できます。

MOSSテンプレートではbf16・DiTのみの学習を設定し、公式の事前学習と同じタイムステップ分布を指定しています。学習率・rank・エポックなどはテンプレートで調整してください。Anima用のVAEキャッシュ設定はMOSSの出力から取り除きます。生成済みのconfig.tomlとtrain.batは再生成が必要です。

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

`--repeats` は同じコマンドで追加したすべてのパスに適用します。複数の `[[datasets]]` がある場合、`--dataset-index 1` のように追加先を0始まりで選べます。既存ファイルへの再生成には `--overwrite`（または `--force`）が必要です。

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

## スライダーLoRA

### A/Bの過学習

先に `LoRA名/a.png` と `LoRA名/b.png` を配置してください。同じフォルダに `a.txt` / `b.txt` があれば、それぞれのキャプションとして使用します。ない場合はLoRA名を共通キャプションとして使用します。

対話では `ssgen --overwrite` で起動し、既存のLoRAフォルダ名と用途「スライダーLoRA」を選択します。CLIでも生成できます。

```powershell
ssgen --family anima --kind slider --name boldline --destination ".\boldline" --model "C:\Software\ComfyUI\models\diffusion_models\Anima\anima_basev10.safetensors" --sd-scripts-dir "E:\DataSetAnima\sd-scripts"
```

画像は混ざらないよう `slider_data/a/`・`slider_data/b/` にコピーします。元画像を変更した場合は `--overwrite` で再生成してください。A/Bとも次の条件を固定します。

```toml
max_train_epochs = 100
save_every_n_epochs = 10
gradient_accumulation_steps = 1
```

各1画像・10リピート・バッチサイズ1・単一プロセスのため、各1000ステップになります。通常用途の10エポックとは別設定です。チェックポイントを残すため保存世代数の制限を除き、`max_train_steps` など競合する設定は除去します。

生成ファイルは次の構成です。

```text
boldline/
  a.png / b.png                 # 元画像
  config_a.toml / config_b.toml
  dataset_a.toml / dataset_b.toml
  train_a.bat / train_b.bat      # 個別に学習する場合
  train.bat                     # A成功後にBを学習。マージは実行しない
  copylora.bat                   # マージ後の1ファイルだけを複製
  slider.toml                    # 次のマージ設定生成に使うパス・ファミリ等
  slider_data/a/a.png
  slider_data/b/b.png
  LoRA.Temp/                    # 学習実行後にA/Bの途中経過と最終LoRAを保存
    a-000010.safetensors ... a-000090.safetensors
    b-000010.safetensors ... b-000090.safetensors
    a.safetensors / b.safetensors
```

`train.bat` を実行し、学習後はチェックポイントを使って過学習の状態を目視確認してください。ここで処理を区切り、自動的にはマージしません。最終100エポックはsd-scriptsの仕様で `a.safetensors` / `b.safetensors` に保存されます。

### 確認したチェックポイントからマージ

```powershell
ssgen --merge-slider boldline --epoch-a 90 --epoch-b 60
```

指定を省略した側は、safetensorsの一覧から上下キーとEnterで選択します。

```powershell
ssgen --merge-slider boldline
ssgen --merge-slider boldline --epoch-a 90
```

`--epoch-a 100` / `--epoch-b 100` は最終ファイルに対応します。保存されていないエポックを指定するとエラーになり、別のファイルを自動選択しません。

このコマンドは `merge.bat` を生成し、`copylora.bat` を更新して終了します。再指定すると、この2ファイルだけを更新します。画像、学習設定、A/Bの重みは変更しません。実際のマージは `merge.bat` を実行したときに、保存したsd-scripts環境のPythonで付属の [networks/svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) を呼び出します。

```text
--ratios -1 1 --new_rank <A/Bから検出した最大rank> --device cuda --save_precision bf16
```

Aを-1、Bを+1でマージし、既定では `boldline/LoRA/boldline.safetensors` に保存します。`--new_rank` は選択したA/Bそれぞれのsafetensorsをスキャンし、大きい方のrankを自動採用します。例えばAが16、Bが64なら64です。モジュールごとにrankが異なる場合も、その最大値を使用します。メタデータのdimやalphaではなく実際の `lora_down.weight` / `lora_up.weight` の形状から判定します。[safetensorsの形式](https://github.com/safetensors/safetensors#format)に沿ってヘッダーだけを読み、重み本体のロードや外部パッケージは不要です。判定できないファイルではエラーを返し、既存のbatは更新しません。

保存先は初回生成時の `--output-dir` も反映します。生成後に `slider.toml` と一式を別の場所へ移動すると、保存された絶対パスの更新が必要です。

マージ後に `copylora.bat` を実行すると、通常のLoRAと同じ `ComfyUI/models/loras/<family>/boldline/` へマージ済みの1ファイルだけをRobocopyで複製します。ロック時のリトライと各batの終了前の `pause` は通常用途と共通です。

## 学習と複製

`train.bat` を実行すると、対話で入力したsd-scriptsフォルダで仮想環境のPythonからAccelerateを起動します。CLIでは `--sd-scripts-dir` で指定でき、省略時は従来どおりテンプレートのtrain.batから取得します。対話またはCLIでパスを指定した場合、そのフォルダ内の `venv/Scripts/python.exe` を使用します。CLIの `--python` で別のPythonも選べます。学習環境にはsd-scriptsの依存パッケージとAccelerateが必要です。

生成する `train.bat` と `copylora.bat` は、成功・失敗どちらでも終了前に `pause` でキー入力を待ちます。処理の終了コードは保持します。既存のbatへ反映するには `--overwrite` で再生成してください。

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

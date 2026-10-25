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

Windowsでは `ssgen.bat` でも起動できます。初回に `setup.bat` で `venv` を準備してください。仮想環境の有効化は不要で、引数はそのまま `ssgen.py` に渡します。生成先の基準は実行時のカレントディレクトリです。

```powershell
.\setup.bat
.\ssgen.bat
.\ssgen.bat --help
.\ssgen.bat LoRA名/ --add-dataset-with-prefix "charname_"
```

`ssgen` コマンドとして登録したい場合だけ、任意で次を実行してください。

```powershell
python -m pip install -e .
ssgen --help
```

テンプレートはこのソースフォルダとともに使用します。

## 学習ログのプロット（ssplot）

`ssplot` はsd-scriptsのTensorBoardイベントログを読み取り、lossや学習率のグラフをPNGに保存します。ログファイルを1つ指定してください。

### 初回セットアップ

Python 3.11以上とWindowsの `py` コマンドが必要です。プロジェクト直下で次を実行します。`setup.bat` は `py -m venv venv` で環境を作成し、その環境に本プロジェクト・TensorBoard・Matplotlibをインストールします。既存の `venv` がある場合は再利用します。初回インストールにはインターネット接続が必要です。

```powershell
.\setup.bat
```

手動でセットアップする場合は次を実行します。

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -e ".[plot]"
```

### 実行方法

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

### 指標とオプション

`loss/average`・`loss/current`・`lr/unet`・`loss/epoch_average`を優先し、ログに記録されたその他のスカラー指標も自動でプロットします。`loss/current`には既定で200サンプルの後方移動平均を重ねます。

```powershell
# 移動平均の窓幅を500サンプルに変更
.\ssplot.bat "path/to/logfile" --window 500
# ヘルプを表示
.\ssplot.bat --help
```

`--window` は1以上の整数です。学習途中のログも、その時点で読み取れるイベントまで出力します。更新するには同じコマンドを再実行してください。環境や依存パッケージが見つからない場合は `setup.bat` を再実行してください。

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
8. MOSSの場合、最後にtimestep samplingモードとサブセットoffsetの個別・一括設定

sd-scriptsパスは空欄でEnterを押すと次の値を使います。指定した場所を `train.bat` の `pushd` と `venv/Scripts/python.exe` の起点にします。フォルダが未作成でも設定できます。

| ファミリ | 既定のsd-scriptsパス |
| --- | --- |
| SDXL | `E:\DataSetSDXL\sd-scripts` |
| Anima | `E:\DataSetAnima\sd-scripts` |
| MOSS | `E:\DataSetMoss\sd-scripts` |

ファミリがMOSSの場合は、続けて `blocks_to_swap = 20` の有効・無効を上下キーとEnterで選択します（初期選択は無効）。有効時は `config.toml` に `blocks_to_swap = 20` を設定し、併用できない `unsloth_offload_checkpointing` と `cpu_offload_checkpointing` は `false` にします。無効時は `blocks_to_swap = 0` とし、既存のoffload設定を保持します。スライダーではA/B両方に適用します。SDXL・Animaではこの質問は表示しません。

選択画面では上下キーで移動し、Enterで決定します。長い一覧はスクロールします。Ctrl+C、または選択画面でEscを押すと中止します。Windows Terminal、PowerShellなどの対話端末から実行してください。`ssgen --interactive` でも開始できます。

モデルは、指定したComfyUIパスを基準に次のフォルダ以下を再帰探索します（`<family>` は `sdxl` / `anima` / `moss`）。

- `<family>/`
- `models/<family>/`
- `models/checkpoints/<family>/`
- `models/diffusion_models/<family>/`

対応する拡張子は `.safetensors`・`.ckpt`・`.pt`・`.pth`・`.bin` です。選択したファイルの絶対パスをconfig.tomlへ設定します。VAE・テキストエンコーダーはファミリ別テンプレートを使用し、相対パスは入力したComfyUIパスを基準に絶対パス化します。外部の絶対パスを指定した場合はその値を保持します。

設定後、**カレントディレクトリの `LoRA名/`** に設定とbatを生成します。通常の用途では4ファイル、スライダーでは後述のA/B個別ファイルを生成します。既存の同名フォルダも指定でき、既定では既存のconfig.tomlとdataset.tomlを読み込んで更新します。テンプレートから初期化したい場合だけ次のように起動してください。

```powershell
python ssgen.py --overwrite
# または
ssgen --interactive --overwrite
```

既定では既存の画像パス・リピート数・custom_attributesを保持し、新たに指定した画像フォルダを追記します。登録済みパスは既存の項目を維持します。`--overwrite` 付きでは既存設定を引き継がず、テンプレートから作成し直します。`config.toml`・`dataset.toml`・`train.bat`・`copylora.bat` は今回の設定で再生成します。既存dataset.tomlへの追記ではありません。画像フォルダ・学習済みLoRA・その他のファイルは保持します。同名の通常ファイルには上書きできません。CLI生成でも `--overwrite` を使用でき、従来の `--force` は同じ意味の別名として残しています。学習や複製は自動実行しません。

画像フォルダの入力は空欄で終了できます。未登録の場合は、学習前に次のように追加してください。

```powershell
ssgen --destination ".\LoRA名" --add-dataset "E:\datasets\images"
```

スライダーLoRAでは画像フォルダの入力を省略し、`LoRA名/a.png` と `LoRA名/b.png` を個別に学習する設定を生成します。キャラ・画風・コンセプトの学習条件は用途別テンプレートがあればそれを使い、なければ共通設定を保持します。

対話処理を含む実装は `ssgen.py` の1ファイルです。項目を追加するときは設定関数を定義して `WIZARD_STEPS` に追加します。ファミリと用途の選択肢は `FAMILIES` / `PURPOSES` で管理します。既存のCLIによる生成・データセット追記も利用できます。

## 画像をPNGに変換して連番へ整理

```powershell
python ssreorder.py "path/to/image/dir/"
# または
.\ssreorder.bat "path/to/image/dir/"
```

指定フォルダ直下の `.jpeg`・`.jpg`・`.webp` をPNGへ変換し、保存後に読み戻して確認できた元画像を削除します（拡張子の大文字・小文字を区別せず、`.wepb` 表記も対象）。同名PNGがある場合は一時的に連番付きの名前を使い、既存の画像を上書きしません。

その後、既存PNGを含めて**更新日時の古い順**に `0000.png`、`0001.png`…へリネームします。変換前の更新日時を引き継ぎ、同日時の場合はファイル名順です。同名の `.txt` キャプションも対応する番号へ変更します。サブフォルダとその他のファイルは変更しません。

変換に失敗した場合はエラーを表示して連番整理を中止します。成功済みの変換は保持します。リネーム途中の失敗時は元の名前へ戻します。画像と無関係な既存の連番キャプションを上書きする場合も、リネーム前に中止します。画像のEXIF回転と透過を反映し、アニメーションWebPはフレームを保持したアニメーションPNGとして保存します。

画像変換にはPillowが必要です。`setup.bat` を再実行するとプロジェクト環境へ追加されます。直接実行するPythonへは、次のいずれかでインストールしてください（ssgen本体には不要です）。

```powershell
python -m pip install Pillow
# または、ssreorderコマンドも登録する場合
python -m pip install -e ".[image]"
```

`ssreorder.bat` はプロジェクトの `venv` があればそれを使い、なければ `py -3` で起動します。終了前に `pause` し、終了コードを保持します。

## テンプレート

`templates/sdxl/`、`templates/anima/`、`templates/moss/` の各フォルダを使用します。元ファイルは変更しません。

配布テンプレートに具体的なキャラ名・LoRA名・画像フォルダは含めません。`output_name`・`output_dir`・`logging_dir`・`dataset_config` は生成時に追加します。dataset.tomlは解像度・バッチサイズなどの共通設定だけを持ち、画像フォルダは対話または `--add-dataset` などで登録します。テンプレートのbatは生成元の資料として扱い、実行にはssgenが生成したbatを使ってください。

`enable_bucket = true`・`caption_extension = '.txt'`・`keep_tokens = 1` はconfig.tomlの通常キーとして設定します。配布dataset.tomlには `[general]` を置きません。sd-scriptsはデータセット側で未指定の項目を学習設定から補います。カスタムdataset.tomlで指定した場合はデータセット側が優先されます。

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

`--repeats` は同じコマンドで追加したすべてのパスに適用します。複数の `[[datasets]]` がある場合、`--dataset-index 1` のように追加先を0始まりで選べます。既存ファイルは通常、読み込んで更新します。設定をクリアして新規作成する場合は `--overwrite`（または `--force`）を指定します。

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

## MOSSのtimestep samplingとサブセットoffsetの編集

`ssgen "LoRA名/"` は、ファミリ・ComfyUIパス・ベースモデル・用途・sd-scriptsパスなどを含む**学習設定ウィザード全体**を開始します。LoRA名は引数から取得し、config.tomlとdataset.tomlを生成または更新します。MOSSでは最後にtimestep sampling設定を表示します。既存設定は保持し、`--overwrite` / `--force` を併用した場合だけテンプレートから新規作成します。

データセットとtimestep samplingだけを編集する場合は、`--dataset-config` を指定してください。

生成済みの設定は次のウィザードで更新できます。

```powershell
ssgen --dataset-config
ssgen "LoRA名/"
ssgen --dataset-config "LoRA名/"
ssgen --dataset-config "LoRA名/dataset.toml"
ssgen LoRA名/ --dataset-config
```

引数なしの `--dataset-config` はカレントのdataset.tomlを使用し、見つからなければパスを入力します。対応するconfig.tomlからファミリを判定し、不明な場合は選択します（`--family moss` で明示も可能）。この編集機能はMOSS用です。スライダーの `dataset_a.toml` / `dataset_b.toml` は対応する `config_a.toml` / `config_b.toml` を使用します。

1. `config.toml` の `timestep_sampling` を `sigmoid`・`shift`・`flux_shift` から上下キーで選択します。
   MOSSで`shift`を選ぶと、続けて `discrete_flow_shift` を入力します。初期値は `1.0`、既存設定があればその値を表示し、空欄で維持します。0より大きい有限の数値を指定してください。正式な引数名は `--discrete_flow_shift` です。`sigmoid`・`flux_shift`ではこの入力を省略します。
2. offsetの変更方法を「個別に設定」・「一括変更」から上下キーで選びます。個別では従来どおり対象サブセットを選んで数値入力します。一括では入力した1つの値を、すべての `[[datasets]]` の全サブセットに適用します。空欄では現在値を保持します。
3. 一括変更はoffset入力後、そのままconfig.tomlとdataset.tomlを保存します。個別設定は必要なサブセットを編集後、「保存して終了」で保存します。個別設定の「変更を破棄して終了」や保存前のEsc／Ctrl+Cでは保存しません。

offsetは `custom_attributes.timestep_sampling.offset` に設定します。既存のリピート数・画像パス・他のcustom_attributesは保持します。生成時の `--name ... --dataset-config PATH` は従来どおり生成元データセットの指定です。

値の意味とモードの違いは [MOSSのtimestep sampling offset](tips/moss-timestep-sampling-offset.md) を参照してください。

## スライダーLoRA

### A/Bの過学習

画像がなくても設定を生成できます。生成後、学習を始めるまでに `LoRA名/a.png` と `LoRA名/b.png` を配置してください。同じフォルダに `a.txt` / `b.txt` があれば、それぞれのキャプションとして使用します。ない場合はLoRA名を共通キャプションとして使用します。

対話では `ssgen` で起動し、LoRAフォルダ名と用途「スライダーLoRA」を選択します。既存のA/Bデータセット設定も保持します。初期化には `--overwrite` を指定します。CLIでも生成できます。

```powershell
ssgen --family anima --kind slider --name boldline --destination ".\boldline" --model "C:\Software\ComfyUI\models\diffusion_models\Anima\anima_basev10.safetensors" --sd-scripts-dir "E:\DataSetAnima\sd-scripts"
```

`train_a.bat` / `train_b.bat` の実行時に、画像とキャプションを `slider_data/a/`・`slider_data/b/` にコピーします。画像を後から配置・変更しても設定の再生成は不要です。元のキャプションを削除した場合は学習用コピーも削除し、LoRA名を使用します。`train.bat` はA/B両方の元画像を確認してから学習を開始し、個別batは対象側の画像だけを確認します。画像がない場合や学習用フォルダに余分な画像がある場合は、学習せずに終了します。画像差し替え時のキャッシュ削除は[Tips](tips/latent-cache-refresh.md)を参照してください。A/Bとも次の条件を固定します。

```toml
max_train_epochs = 100
save_every_n_epochs = 10
gradient_accumulation_steps = 1
```

各1画像・10リピート・バッチサイズ1・単一プロセスのため、各1000ステップになります。通常用途の10エポックとは別設定です。チェックポイントを残すため保存世代数の制限を除き、`max_train_steps` など競合する設定は除去します。

生成ファイルは次の構成です。

```text
boldline/
  a.png / b.png                 # 設定生成後に配置できる元画像
  config_a.toml / config_b.toml
  dataset_a.toml / dataset_b.toml
  train_a.bat / train_b.bat      # 個別に学習する場合
  train.bat                     # A成功後にBを学習。マージは実行しない
  copylora.bat                   # マージ後の1ファイルだけを複製
  slider.toml                    # 次のマージ設定生成に使うパス・ファミリ等
  slider_data/a/a.png            # train_a.bat実行時に複製
  slider_data/b/b.png            # train_b.bat実行時に複製
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

生成する `train.bat` と `copylora.bat` は、成功・失敗どちらでも終了前に `pause` でキー入力を待ちます。処理の終了コードは保持します。既存のbatへ反映するには設定生成を再実行してください。

学習後に `copylora.bat` を実行すると、出力先の `*.safetensors` を `ComfyUI/models/loras/[sdxl,anima,moss]/LoRA名/` へ複製します。学習バッチからの自動実行はしません。

ComfyUIの場所の既定値は `C:\Software\ComfyUI\` です。対話設定で入力するか、CLIでは `--comfyui-dir` で変更できます。

Robocopyで複製し、ロックなどによる失敗時は既定で5秒間隔・最大60回再試行します。`--copy-retries` と `--copy-wait` で変更できます。複製先の既存ファイルを削除せず、同名ファイルは更新します。終了コード0〜7は成功、8以上は失敗として返します。対象ファイルがない場合も失敗します。仕様参照: [Microsoft Robocopy](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/robocopy)。

## その他の設定

- `--template PATH`: 任意のconfig.tomlを指定
- `--dataset-config [PATH]`: `--name`指定時は生成元のdataset.toml。それ以外はMOSSの既存dataset/configを編集するウィザード
- `--logging-dir PATH`: ログ保存先を指定
- `--set 'network_dim=32' --set 'learning_rate=0.0001'`: TOML形式で学習設定を上書き

生成先・LoRA名・モデルなどの専用オプションは `--set` より優先します。TOMLは標準の `tomllib` で読み込み、`ssgen.py` 内の出力処理で再生成します。設定値は保持しますが、コメント・空行・元の引用符などの書式は引き継ぎません。元のテンプレートは変更しません。batはUTF-8・CRLFです。テンプレート内の追加モデルなどの相対パスは学習時のsd-scriptsフォルダ基準になるため、絶対パスを推奨します。

## Tips

[Tipsの索引](tips/README.md)を参照してください。

## 検証

```powershell
python -m unittest discover -s tests -v
```

外部パッケージに依存しないことは `python -S -m unittest discover -s tests -v` でも確認できます。

## ライセンス

本プロジェクトは GNU General Public License version 3（GPL-3.0-only）で公開しています。ライセンス全文は [LICENSE](LICENSE) を参照してください。

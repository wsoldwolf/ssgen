# ssntagger

[プロジェクトREADMEに戻る](../README.md)

コマンド例はプロジェクト直下から実行する想定です。

`ssntagger.bat` はコンソールとPythonの出力をUTF-8に設定します。保存したログもUTF-8として開いてください。組み込みプロンプトの数値範囲は、文字コードに依存しない `0-1000` 表記を使用します。

[kohya氏のcaption_images_moss.py](https://gist.github.com/kohya-ss/6bd9bd21247361e24c25c4c038efaa96)（revision `5867ebae0e5fc71c50f860df0f860359bdcc9d2e`）を基に、llama-cpp-pythonで直接推論する `ssntagger` を追加しています。APIサーバやComfyUIの起動は不要です。元コードのMOSSプロンプト、補助知識、long / medium / shortキャプション、sidecar再利用に対応します。

## セットアップ

Python 3.11以上、Visual Studio C++ビルドツール、CUDA Toolkitを用意し、x64 Native Tools Command Promptで実行します。標準パスにあるVisual Studio 2022 Build Toolsは自動で初期化します。CUDA 13.0ではVisual Studio 2022を使用してください。専用の `venv-ssntagger` に、Gemma 4対応の `llama-cpp-python==0.3.36` をソースからビルドしてインストールします。既存のssgen / sd-scripts / ComfyUIのvenvは使用しません。

```bat
setup_ssntagger.bat
```

Python実行ファイルは環境変数 `SSNTAGGER_PYTHON` で指定できます。既定は `python` です。CPU専用ビルドは事前に `set "CMAKE_ARGS=-DGGML_CUDA=OFF"` を設定し、実行時に `--n_gpu_layers 0` を指定します。GPUビルドの既定は `-DGGML_CUDA=ON` です。既存環境を再ビルドする場合は専用venvでllama-cpp-pythonをアンインストールしてからセットアップを実行してください。

## 実行

指定フォルダ以下の全サブフォルダを既定で探索し、すべての対応画像を処理します。出力txt / JSONはそれぞれの画像と同じフォルダに保存します。直下の画像だけ処理する場合は `--no-recursive` を指定してください。従来の `--recursive` も引き続き指定できます。既存txtのスキップやsidecar再利用はサブフォルダ内でも同じです。

```powershell
.\ssntagger.bat "E:\datasets\images"
.\ssntagger.bat "E:\datasets\images" --knowledge "The character is Foo from Bar."
.\ssntagger.bat "E:\datasets\images" --no-recursive
.\ssntagger.bat "E:\datasets\images" --model "path\Gemma4.gguf" --mmproj "path\mmproj.gguf"
.\ssntagger.bat --help
```

この環境では31B Q4_K_Sで次の設定による画像推論とtxt / JSON出力を確認しています。空きVRAMが少ない場合の起動例です。

```powershell
.\ssntagger.bat "E:\datasets\images" --n_gpu_layers 16 --n_ctx 8192 --max_tokens 2048
```

モデル未指定時は `C:\ComfyUI\models\LLM`、`C:\Software\ComfyUI\models\LLM`、`E:\ComfyUI\models\LLM` の順に探索します。最初に見つかったルートで、動作実績のある31B Q4_K_S版を優先します。この環境では `gemma-4-31b-it-heretic-ara.Q4_K_S.gguf` と同フォルダの `gemma-4-31b-it-heretic-ara.mmproj-f16.gguf` を選択します。31B Q4_K_Sがない場合は最小サイズのGemma 4 GGUFを選択します。`--model_dir` で探索先を変更できます。mmproj未指定時はモデルと同じフォルダに1個だけ存在するmmprojを使用し、複数ある場合は明示指定を求めます。モデルとmmprojは対応する組を使用してください。

画像の隣に `<stem>_caption.json` と `<stem>.txt` を保存します。txtは `---===--- # long` などで区切ったsd-scripts用複数キャプションです。学習時には `enable_multiline_caption = true` を設定してください。既存txtは既定でスキップし、`--overwrite` で再推論して更新します。

```powershell
# sidecarから短いキャプションだけ再出力（モデル不要）
.\ssntagger.bat "E:\datasets\images" --from_sidecar --overwrite --candidates short
# 推論せずプロンプトを確認
.\ssntagger.bat "E:\datasets\images" --dry_run
```

補助知識は `--knowledge` / `--knowledge_file`、階層別 `_knowledge.txt`、画像別 `<stem>_knowledge.txt` から読み込みます。既存タグの参照には `--reference_extension .tags` を指定します。

ローカル推論は `--max_workers 1` のみ対応します。既定はGPU全レイヤー（`--n_gpu_layers -1`）、GPUでのFlash Attention有効、コンテキスト8192、出力上限4096トークンです。以前の既定値32768 / 16384から変更し、KVキャッシュのVRAM使用量を抑えています。長い補助知識や出力が必要な場合は `--n_ctx` / `--max_tokens` を増やしてください（`n_ctx > max_tokens` が必要です）。Gemma 4の対応はllama-cpp-pythonの `Gemma4ChatHandler` を使用します。

## 高速化設定

RTX 5090でVRAMに余裕がある場合は全レイヤーをGPUへ載せてください。16レイヤー指定はCPUでも計算するため低速になります。次のコマンドではFlash Attentionも既定で有効です。

```powershell
.\ssntagger.bat "E:\datasets\images" --n_gpu_layers -1
```

2026-11-03の31B Q4_K_S / RTX 5090による小さなテスト画像2枚の比較では、16レイヤーGPU・Flash Attention無効・バッチ512に対し、全レイヤーGPU・Flash Attention有効・バッチ1024で、1枚目は77.62秒から12.86秒、2枚目は68.09秒から3.40秒になりました。1枚目にはモデル読み込みが含まれます。コンテキスト8192、出力上限1024、temperature 0、seed 42で比較し、出力は187 / 201トークンでした。実画像での速度は画像サイズや生成内容によって変わります。全GPU実行時のGPUメモリ使用量は他のプロセスを含めて約29GBでした。

`--n_batch`（既定1024）/ `--n_ubatch` はプロンプト処理のバッチサイズです。`n_ubatch` は省略時に `n_batch` と同じ値を使います。Gemma 4の画像処理では非因果注意の評価バッチが物理バッチを超えるとネイティブのassertで停止するため、ローカル推論では両方を同じ値にしてください。明示的に小さい `n_ubatch` を指定した場合は推論前にエラーにします。メモリを節約する場合は `--n_batch 512` のように指定すると物理バッチも512になります。Flash Attentionは `--no-flash_attn` で無効にできます。CPU実行（`--n_gpu_layers 0`）では自動で無効になります。`--seed` で比較時の乱数シードを指定できます。

VRAM不足時は他のGPUアプリを終了するか、`--n_gpu_layers 16 --n_ctx 8192 --max_tokens 2048` などで使用量を抑えてください。画像ごとの経過時間と出力トークン数をログに表示し、所要時間はJSONの `elapsed_seconds` にも保存します（画像のエンコード前処理を除き、初回のみモデル読み込みを含みます）。出力上限を下げるだけでは、元々短く生成されるキャプションは速くなりません。

元コードのAPIモードは `--backend api --base_url http://localhost:8000/v1 --model MODEL` で利用できます。`--extra_body` の既定は `{}` に変更しています。vLLM / SGLang用のthinking設定などは必要に応じてAPIモードで指定してください。


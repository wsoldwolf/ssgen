# sd-scripts の補足事項（Windows / cmd.exe）

以下は **コマンドプロンプト（cmd.exe）** 用の手順です。PowerShell の `$env:...` とは書式が異なります。
例では sd-scripts を `E:\DataSetMoss\sd-scripts`、仮想環境をその中の `venv` としています。実際の配置に合わせて変更してください。

## causal-conv1d のビルドとインストール

CUDA Toolkit と Visual Studio の C++ ビルドツールが必要です。今回の環境では CUDA Toolkit 13.0 と Visual Studio 2022 Build Tools を使用しました。
Visual Studio Installer では「C++ によるデスクトップ開発」（MSVC x64/x86 ビルドツールと Windows SDK）を用意してください。

cmd.exe を開き、次を順番に実行します。`vcvars64.bat` で C++ コンパイラに必要な環境変数も設定します。

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

cd /d E:\DataSetMoss\sd-scripts
call venv\Scripts\activate.bat

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"

set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"
set "CFLAGS=/Zc:preprocessor"
set "CXXFLAGS=/Zc:preprocessor"
set "DISTUTILS_USE_SDK=1"
set "CAUSAL_CONV1D_FORCE_BUILD=TRUE"

where python
where cl
"%CUDA_HOME%\bin\nvcc.exe" --version

python -m pip install ninja
python -m pip install causal-conv1d --no-build-isolation -v
```

`where python` の先頭が sd-scripts の `venv\Scripts\python.exe` であること、`where cl` がコンパイラのパスを返すことを確認してください。開発環境の初期化や確認でエラーが出た場合は、先にその設定を修正してください。

`CUDA_HOME` には **`bin\nvcc.exe` が実在する CUDA Toolkit のルート**を指定します。`<YOUR_VENV>` のような説明用の文字列をそのまま設定しないでください。venv 内に CUDA 関連パッケージがあっても、そこにコンパイラがあるとは限りません。

インストール後、同じ cmd.exe で読み込みを確認します。

```bat
python -c "import torch; import causal_conv1d; import causal_conv1d_cuda; print('import OK')"
```

これは読み込み確認です。実際の GPU 演算や学習の成功まで確認するものではありません。今回のログではコンパイラの環境設定前までを確認しており、ビルド完了は未確認です。

### 今回のエラーの読み方

- `get_cuda_bare_metal_version` 内の `WinError 2`：`CUDA_HOME` 配下の `nvcc` が見つかっていません。
- `Error checking compiler version for cl`、続いて `cl.exe` 起動時の `WinError 2`：C++ コンパイラが見つかっていません。`vcvars64.bat` を実行した同じ cmd.exe でビルドしてください。
- `Precompiled wheel not found` / HTTP 404：環境に対応するビルド済み wheel が取得できず、ソースビルドへ進んでいます。上記では `CAUSAL_CONV1D_FORCE_BUILD=TRUE` で直接ビルドします。
- `ninja` が見つからないという警告：ビルド前に `python -m pip install ninja` を実行します。
- CUDA Toolkit 13.0 と PyTorch の CUDA 13.2 の差：今回のログでは警告であり、停止原因は `cl.exe` が見つからないことでした。ほかの組み合わせも含め、ビルドや実行の互換性を保証するものではありません。

## flash-linear-attention と triton-windows

sd-scripts の venv を有効にした cmd.exe で、pip からインストールします。

```bat
cd /d E:\DataSetMoss\sd-scripts
call venv\Scripts\activate.bat

python -m pip install triton-windows
python -m pip install flash-linear-attention
```

インポート名は配布パッケージ名と異なり、それぞれ `triton` と `fla` です。

```bat
python -m pip show triton-windows flash-linear-attention
python -c "import torch; import triton; import fla; print('import OK')"
python -m pip check
```

sd-scripts 側でバージョン指定がある場合は、その指定に合わせてください。インポート成功後も、使用するモデルの学習を実行して動作を確認してください。

## 参考

- [causal-conv1d 公式リポジトリ](https://github.com/Dao-AILab/causal-conv1d)
- [Visual Studio 開発者用コマンド環境](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022)

# 画像を差し替えた場合のlatentキャッシュ

`cache_latents_to_disk = true` のとき、同じファイル名・画像サイズのまま画像を差し替えても、latentキャッシュは自動再生成されません。sd-scriptsのキャッシュ判定は元画像の更新日時・内容を比較しないためです（[判定処理](https://github.com/kohya-ss/sd-scripts/blob/main/library/strategy_base.py)）。

画像を更新したら、その画像に対応する次のキャッシュを削除してください。次の学習開始時に再生成されます。

- MOSS: `画像名_幅x高さ_moss.npz`
- Anima: `画像名_幅x高さ_anima.npz`

`ssgen --overwrite` は設定・batの再生成であり、既存のキャッシュは削除しません。

スライダーでは `train_a.bat` / `train_b.bat` の実行時に、元の `a.png` / `b.png` を学習用フォルダへコピーします。画像を変更した場合は、実行前に `slider_data/a/`・`slider_data/b/` 内の対応するlatentキャッシュを削除してください。設定の再生成は不要です。

[Tipsの索引に戻る](README.md)

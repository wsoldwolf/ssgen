# 画像を差し替えた場合のlatentキャッシュ

`cache_latents_to_disk = true` のとき、同じファイル名・画像サイズのまま画像を差し替えても、latentキャッシュは自動再生成されません。sd-scriptsのキャッシュ判定は元画像の更新日時・内容を比較しないためです（[判定処理](https://github.com/kohya-ss/sd-scripts/blob/main/library/strategy_base.py)）。

画像を更新したら、その画像に対応する次のキャッシュを削除してください。次の学習開始時に再生成されます。

- MOSS: `画像名_幅x高さ_moss.npz`
- Anima: `画像名_幅x高さ_anima.npz`

`ssgen --overwrite` は設定・batの再生成であり、既存のキャッシュは削除しません。

スライダーでは元の `a.png` / `b.png` を変更後に `--overwrite` で学習用コピーを更新し、`slider_data/a/`・`slider_data/b/` 内の対応するlatentキャッシュも削除してください。

[Tipsの索引に戻る](README.md)

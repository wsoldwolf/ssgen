# MOSSのtimestep sampling offset

画像フォルダ（サブセット）ごとに、学習するノイズレベルの分布を調整します。

- 負の値: 低ノイズ側のタイムステップに偏り、細部・質感の学習を重視します。
- 正の値: 高ノイズ側のタイムステップに偏り、全体の構造・構図の学習を重視します。
- `0`: オフセットなし。公式の推奨範囲は `-0.5` ～ `0.5` です。

例えば、顔のアップや質感を重視したフォルダには `-0.5`、全身や構図を重視したフォルダには `0.5` を設定する使い方があります。値の影響は非線形で、絶対値を大きくすると一部のノイズレベルが学習されにくくなります。

offsetが有効な`timestep_sampling`は `sigmoid`・`shift`・`flux_shift` です。配布MOSSテンプレートの `sigma` と `uniform` では反映されません。ウィザードではこの3つの有効なモードから選び、config.tomlも合わせて更新します。`shift` は `discrete_flow_shift`、`flux_shift` は解像度によっても分布が変わります。

```powershell
ssgen --dataset-config "LoRA名/dataset.toml"
```

モードを上下キーで選択し、offsetを「個別に設定」するか「一括変更」するか選びます。個別では対象サブセットを選んでoffsetを入力し、「保存して終了」で更新します。一括では1つの値を全データセットの全サブセットへ適用し、入力後にそのまま保存します。空欄では値を維持します。既存の他の設定は保持します。

設定例（config.toml）:

```toml
timestep_sampling = "sigmoid"
```

設定例（dataset.toml）:

```toml
[[datasets.subsets]]
image_dir = "closeup_shots"
[datasets.subsets.custom_attributes]
timestep_sampling = { offset = -0.5 }
```

ssgenも上記のインラインテーブル形式で出力します。詳細は [sd-scripts公式ガイド](https://github.com/kohya-ss/sd-scripts/blob/main/docs/timestep_sampling_offset.md) と [ノイズ分布の実装](https://github.com/kohya-ss/sd-scripts/blob/main/library/flux_train_utils.py) を参照してください。

[Tipsの索引に戻る](README.md)

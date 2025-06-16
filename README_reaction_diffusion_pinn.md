# 反応拡散方程式PINNs実装

## 概要

このプロジェクトは、Physics-Informed Neural Networks (PINNs) を用いてGray-Scott反応拡散方程式を学習するニューラルネットワークモデルを実装しています。

## 実装内容

### Gray-Scott方程式
```
∂U/∂t = du * ∇²U - UV² + feed(1-U)
∂V/∂t = dv * ∇²V + UV² - (feed+kill)V
```

### 入出力仕様
- **入力**: [t, x, y, du, dv, feed, kill] (7次元)
  - t: 時間
  - x, y: 空間座標
  - du, dv: 拡散係数
  - feed: 供給率
  - kill: 除去率

- **出力**: [U, V] (2次元)
  - U(t,x,y): 時刻t、座標(x,y)における化学物質Uの濃度
  - V(t,x,y): 時刻t、座標(x,y)における化学物質Vの濃度

## ファイル構成

- `reaction_diffusion_pinn.py`: メインの実装ファイル
- `simple_test.py`: TuringPatternクラス（データ生成に使用）
- `pinns_sample.py`: 参考実装（1次元振動方程式のPINNs）

## 主要クラス・関数

### ReactionDiffusionPINN
- Gray-Scott方程式を学習するPINNsモデル
- 7入力→2出力のニューラルネットワーク
- physics_loss()でPDE残差を計算

### generate_training_data()
- TuringPatternクラスを使用した訓練データ生成
- 複数のパラメータセットでシミュレーション実行
- 時空間サンプリングによるデータ収集

### train_pinn()
- 初期条件損失 + 物理法則損失による学習
- Adam最適化手法を使用
- 自動微分による偏微分計算

### visualize_prediction()
- 学習済みモデルの予測結果可視化
- 異なる時間ステップでのU, V成分を表示

## 使用方法

```bash
# 実行
uv run reaction_diffusion_pinn.py
```

## 実装の特徴

1. **TuringPatternとの連携**: simple_test.pyのTuringPatternクラスを活用してリアルな訓練データを生成

2. **物理法則の組み込み**: 自動微分を使用してGray-Scott方程式の残差を損失関数に組み込み

3. **安全な自動微分**: safe_grad()関数により、微分計算の例外処理を実装

4. **GPU対応**: CUDA利用可能時は自動的にGPU計算を使用

5. **可視化機能**: 学習結果をmatplotlibで可視化

## 学習結果

- 初期損失: ~0.59
- 最終損失: ~0.018
- エポック数: 3000
- 訓練データ: 18,000サンプル
- 初期条件データ: 1,200サンプル

## パラメータ設定

デフォルトのパラメータセット:
- du: 0.14 (Uの拡散係数)
- dv: 0.06 (Vの拡散係数)  
- feed: 0.035 (供給率)
- kill: 0.058 (除去率)

## 技術的詳細

### 損失関数構成
1. **データ損失**: 初期条件でのMSE損失
2. **物理損失**: PDE残差のMSE損失
3. **全体損失**: データ損失 + α × 物理損失

### ネットワーク構造
- 入力層: 7次元
- 隠れ層: 4層 × 128ニューロン (Tanh活性化)
- 出力層: 2次元
- 初期化: Xavier正規分布

### 自動微分による偏微分
- ∂U/∂t, ∂V/∂t: 時間微分
- ∂²U/∂x², ∂²U/∂y²: 空間二階微分（ラプラシアン）
- torch.autograd.gradを使用した計算グラフ構築

## 今後の拡張可能性

1. より複雑な境界条件の実装
2. 3次元空間への拡張
3. 他の反応拡散方程式への適用
4. アンサンブル学習の導入
5. 不確実性定量化の実装

## 依存関係

- PyTorch
- NumPy
- matplotlib
- tqdm
- simple_test.py (TuringPatternクラス)

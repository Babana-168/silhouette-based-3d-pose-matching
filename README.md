# Silhouette-Based 3D Pose Matching

シルエット画像から3Dモデルの最適な姿勢（回転・スケール・位置）を推定するツールです。GPU加速によりリアルタイムでIoU（Intersection over Union）を最大化する最適解を探索します。

## 主な特徴

- **GPU高速化**: PyTorch + CUDAによる高速な姿勢探索
- **高精度マッチング**: 粗探索→微探索の2段階アプローチ
- **柔軟な最適化**: 回転・スケール・平行移動を統合的に最適化
- **視覚的検証**: マッチング結果をリアルタイムで可視化

## インストール

### 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨、CPUでも動作可能）

### 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な使用方法

```bash
python match_3d_to_image.py
```

デフォルトでは以下のファイルを使用します：
- ターゲット画像: `Image0.png`
- 3Dモデル: `models_rabit_dae/rabit.dae`
- 出力画像: `matching_result_gpu.png`

### 処理フロー

1. **ターゲット画像の読み込み**: Otsu法による自動二値化
2. **3Dモデルの読み込み**: DAE/OBJ形式をサポート
3. **4段階最適化**:
   - Phase 1: 粗い回転探索（10度刻み）
   - Phase 2: 微細な回転探索（±10度を1度刻み）
   - Phase 3: スケール・平行移動の粗探索
   - Phase 4: スケール・平行移動の微調整
4. **結果の可視化**: オーバーレイ・差分・IoUスコア表示

## 回転順序の重要性

### なぜZYX順序を使用するのか

3D回転は**非可換**（順序を入れ替えると結果が変わる）です。

```
X回転 → Y回転 → Z回転  ≠  Z回転 → Y回転 → X回転
```

#### 簡単な実験例

本を手に持って以下の操作を試してみてください：

1. **順序A**: X軸で90°回転（前に倒す）→ Y軸で90°回転（左に回す）
2. **順序B**: Y軸で90°回転（左に回す）→ X軸で90°回転（前に倒す）

結果は全く異なる向きになります。

### Three.jsとPythonの回転順序統一

本プロジェクトでは、ビューアー（Three.js）と計算（Python）の両方でZYX順序を使用します。

#### Three.js側

```javascript
// Three.jsのデフォルトはXYZだが、ZYXに変更
mesh.rotation.order = 'ZYX';
mesh.rotation.x = theta * Math.PI / 180;  // X軸回転
mesh.rotation.y = phi * Math.PI / 180;    // Y軸回転
mesh.rotation.z = roll * Math.PI / 180;   // Z軸回転

// 内部適用順序: Z回転 → Y回転 → X回転
```

#### Python側

```python
# ZYX順序での回転行列合成
R = Rx @ Ry @ Rz  # Z回転 → Y回転 → X回転

# 各軸の回転行列
Rx = [[1,      0,       0     ],
      [0, cos(rx), -sin(rx)],
      [0, sin(rx),  cos(rx)]]

Ry = [[cos(ry),  0, sin(ry)],
      [0,        1,      0  ],
      [-sin(ry), 0, cos(ry)]]

Rz = [[cos(rz), -sin(rz), 0],
      [sin(rz),  cos(rz), 0],
      [0,       0,       1]]
```

### 過去の問題と解決

**問題**: ビューアーでZYX順序で調整した角度（θ=104°, φ=-90°, roll=51°）を、XYZ順序のPythonコードで計算すると、IoU=47%と低かった。

**原因**: ビューアーとPythonで回転順序が異なっていた。

**解決**: Python側をZYX順序に統一することで、同じ角度パラメータで同じ姿勢を再現できるようになった。

### 回転順序の慣習

分野によって慣習が異なります：

- **航空・ロボティクス**: ZYX（ヨー→ピッチ→ロール）が一般的
- **ゲームエンジン**: エンジンによって異なる
- **Three.js**: デフォルトはXYZだが変更可能

**重要**: どの順序を選ぶかより、**すべてのコードで同じ順序を使うこと**が重要です。

## プロジェクト構成

```
silhouette-based-3d-pose-matching/
├── match_3d_to_image.py        # メインスクリプト（GPU高速化版）
├── viewer_app.py               # インタラクティブビューアー（Three.js）
├── generate_features_json.py   # 特徴量データベース生成
├── models_rabit_dae/           # 3Dモデルディレクトリ
│   └── rabit.dae
├── Image0.png                  # サンプルターゲット画像
├── requirements.txt            # Python依存関係
├── README.md                   # このファイル
└── LICENSE                     # ライセンス情報
```

## 技術詳細

### アルゴリズム

1. **シルエットレンダリング**:
   - 3Dモデルの頂点を回転行列で変換
   - 正投影による2D投影
   - デプスソートによる隠面処理

2. **IoU計算**:
   ```
   IoU = (Target ∩ Rendered) / (Target ∪ Rendered)
   ```

3. **探索空間削減**:
   - 粗探索: 10度刻み（全体を高速スキャン）
   - 微探索: 1度刻み（局所的に精密化）
   - 段階的絞り込みで計算量を大幅削減

### パフォーマンス

- **GPU使用時**: 約10,000試行/秒
- **総探索回数**: 約30,000〜50,000試行
- **処理時間**: 3〜5秒（NVIDIA RTX 3000シリーズ）

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 参考資料

- [3D回転の数学](https://en.wikipedia.org/wiki/Rotation_matrix)
- [Euler angles](https://en.wikipedia.org/wiki/Euler_angles)
- [Three.js Rotation](https://threejs.org/docs/#api/en/math/Euler)
- [PyTorch CUDA](https://pytorch.org/docs/stable/cuda.html)

## お問い合わせ

問題が発生した場合は、GitHubのIssuesでご報告ください。

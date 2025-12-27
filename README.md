# ※一応作成途中です。コード共有の時だけpublicにしてます 

# Silhouette-Based 3D Pose Matching

シルエット画像から3Dモデルの最適な姿勢（回転・スケール・位置）を推定するツールです。GPU加速によりリアルタイムでIoU（Intersection over Union）を最大化する最適解を探索します。

## 主な特徴

- **超高速マッチング**: 事前計算された40万パターンから数秒で最適解を検索（SQLite使用）
- **高精度位置合わせ**: 回転・位置シフト・スケールを統合的に最適化
- **テクスチャ付き合成**: pyrenderによるリアルな3Dレンダリング
- **複数アプローチ**: 用途に応じた5つの最適化手法を提供

## インストール

### 必要な環境

- Python 3.8以上
- OpenGL環境（pyrender使用時）

### 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

## 使い方

本プロジェクトには、目的に応じた5つのスクリプトがあります：

### 1. test2.py - 高速角度マッチング（メイン）

事前計算されたSQLiteデータベースから最適な回転角度を検索します。

```bash
python test2.py
```

**出力**: `rotation_results/` フォルダに上位候補の画像とIoUスコアを保存

### 2. test3.py - テクスチャ付き合成（基本版）

test2で見つけた最適角度を使用して、元画像に3Dモデルを合成します。

```bash
python test3.py
```

**出力**: `final_result.png`

### 3. test4.py - テクスチャ付き合成（安定版）

自動スケール・位置合わせ機能を備えた安定版です。

```bash
python test4.py
```

### 4. test5.py - 位置シフト付き最適化

test2の上位候補に対して、位置シフト（dx, dy）をグリッド探索してIoU最大化します。

```bash
python test5.py
```

**特徴**: 角度だけでなく位置も同時に最適化

### 5. test6.py - 位置情報保持マッチング

入力画像のマスク位置を保持したまま、3Dシルエットをシフトして比較します。

```bash
python test6.py
```

**特徴**: 入力画像内でのオブジェクト位置情報を活用

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
├── test2.py                    # 高速角度マッチング（SQLite使用）
├── test3.py                    # テクスチャ付き合成（基本版）
├── test4.py                    # テクスチャ付き合成（安定版）
├── test5.py                    # 位置シフト付き最適化
├── test6.py                    # 位置情報保持マッチング
├── viewer_app.py               # インタラクティブビューアー（Three.js）
├── models_rabit_obj/           # 3Dモデルディレクトリ
│   ├── rabit.obj
│   └── rabit.mtl
├── Image0.png                  # サンプルターゲット画像
├── requirements.txt            # Python依存関係
├── README.md                   # このファイル
└── LICENSE                     # ライセンス情報
```

## 技術詳細

### データベース事前計算方式（test2〜test6）

1. **特徴量データベース**:
   - 事前に40万パターンの回転角度でシルエットを計算
   - SQLiteに32x32の2値画像として保存
   - 検索時は数秒で全パターンとのIoU比較が可能

2. **IoU計算**:
   ```
   IoU = (Target ∩ Rendered) / (Target ∪ Rendered)
   ```

3. **最適化手法**:
   - **test2**: 角度のみを最適化（最速）
   - **test5**: 角度 + 位置シフト（グリッド探索）
   - **test6**: 入力画像の位置情報を活用

4. **pyrenderによる合成**:
   - OpenGLオフスクリーンレンダリング
   - テクスチャとマテリアルを保持
   - 最適角度でリアルな合成画像を生成

### パフォーマンス

- **データベース検索**: 約2〜5秒（40万パターン）
- **位置シフト最適化**: 約10〜30秒（グリッド探索）
- **テクスチャ合成**: 約1〜2秒

## ライセンス

All rights reserved.

This repository is provided for viewing purposes only.
Redistribution, modification, and commercial use are not permitted
without explicit permission from the author.

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 参考資料

- [3D回転の数学](https://en.wikipedia.org/wiki/Rotation_matrix)
- [Euler angles](https://en.wikipedia.org/wiki/Euler_angles)
- [Three.js Rotation](https://threejs.org/docs/#api/en/math/Euler)
- [PyTorch CUDA](https://pytorch.org/docs/stable/cuda.html)

## お問い合わせ

問題が発生した場合は、GitHubのIssuesでご報告ください。

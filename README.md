# Silhouette-Based 3D Pose Matching

写真に写ったオブジェクトの3D姿勢（回転角度・カメラ位置）を、シルエットのIoU（Intersection over Union）最大化により推定するシステムです。

深度画像から初期角度を推定し、多段階グリッドサーチで最適解を探索します。SQLiteによる40万パターン事前検索は不要で、深度画像1枚から直接推定できます。

## 処理パイプライン

```
入力画像 + 深度画像
       │
       ▼
[1] 深度画像解析 ─── B-G深度勾配から theta/phi/roll を推定
       │
       ▼
[2] 粗い探索 ─────── ±30° / 5°ステップ（低解像度レンダリング）
       │
       ▼
[3] 中間探索 ─────── ±8° / 1°ステップ
       │
       ▼
[4] カメラ位置最適化 ─ cam_x, cam_y, cam_z の座標降下法
       │
       ▼
[5] 精密探索 ─────── ±1° / 0.2°ステップ
       │
       ▼
[6] テクスチャ付きオーバーレイ出力
```

## 実装

### C++ 版（推奨）

C++ + OpenCV + OpenMP による高速実装です。全パイプラインを1ファイルに統合しています。

**必要環境:**
- Visual Studio 2022
- OpenCV 4.x（`C:/opencv` にインストール済み）
- CMake

**ビルドと実行:**

```bash
cd cpp_pipeline
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\Release\pose_match.exe
```

**性能:**
- 総実行時間: 約35秒
- 最終IoU: 97.0%
- テクスチャ描画: 0.15秒

### Python 版

Python + OpenCV による実装です。

```bash
pip install -r requirements.txt
```

| スクリプト | 説明 |
|-----------|------|
| `estimate_angle_from_depth.py` | 深度画像から初期角度を推定 |
| `pose_match_unified.py` | SQLiteベースの統合マッチング |
| `generate_features_sqlite.py` | 特徴量データベース生成（40万パターン） |

## 深度画像からの角度推定

構造化光による深度画像（B=手前、G=奥）から3つの回転角度を推定します。

| パラメータ | 推定手法 | 推定誤差 |
|-----------|---------|---------|
| theta（水平回転） | `asin(-grad_x / 0.35)` + 左右面積比補正 | 25.5° |
| phi（垂直傾き） | `-90 + grad_y × 50` | 1.3° |
| roll（回転） | 深度等高線の角度変化トレンド × 6 | 4.5° |

3つとも ±30° の探索範囲内に収まるため、SQLiteの事前検索を完全に代替できます。

## 座標系と回転順序

- 回転順序: **ZYX**（Z→Y→X の順に適用）
- 初期回転: X軸に -90°（OBJモデルの座標系変換）
- カメラ: 透視投影（FOV 45°）

```
R = Rx(INITIAL_RX + phi) × Ry(-theta) × Rz(roll)
```

## プロジェクト構成

```
├── cpp_pipeline/
│   ├── CMakeLists.txt          # ビルド設定
│   └── main.cpp                # C++ 全パイプライン
├── estimate_angle_from_depth.py  # 深度→角度推定
├── pose_match_unified.py         # 統合マッチング (Python)
├── generate_features_sqlite.py   # 特徴量DB生成
├── models_rabit_obj/
│   ├── rabit.obj               # 高ポリモデル (427K faces)
│   ├── rabit_low.obj           # 低ポリモデル (60K faces)
│   ├── rabit.mtl               # マテリアル定義
│   └── rabit01.jpg             # テクスチャ
├── Image0.png                  # 入力画像
├── Image0_depth.png            # 深度画像
└── requirements.txt
```

## 出力例

最適化の結果:
- **theta**: 93.59° / **phi**: -80.66° / **roll**: 57.65°
- **IoU**: 97.03%

出力される画像:
- `rendered_textured.png` - テクスチャ付き3Dレンダリング
- `overlay_50.png` - 50%ブレンドオーバーレイ
- `overlay_contour.png` - 輪郭オーバーレイ

## ライセンス

MIT License

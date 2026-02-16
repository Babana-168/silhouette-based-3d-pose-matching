# Silhouette-Based 3D Pose Matching

写真に写ったオブジェクトの3D姿勢（回転角度・カメラ位置）を、シルエットのIoU（Intersection over Union）最大化により推定するシステムです。

**深度画像から初期角度を推定し、多段階グリッドサーチで最適解を探索します。** SQLiteによる40万パターン事前検索は不要で、深度画像1枚から直接推定できます。

## 結果

入力画像から3Dモデルの姿勢を推定し、テクスチャ付きで重ね合わせた結果です。

![Pipeline Result](docs/pipeline_result.png)

**最終IoU: 97.0%** | 実行時間: 35秒（C++版）

## 処理の流れ

### 1. 入力

撮影画像と構造化光による深度画像を入力として使用します。

| 撮影画像 | 深度画像 |
|:---:|:---:|
| ![Input](docs/input.png) | ![Depth](docs/depth.png) |

### 2. 深度画像から角度推定

深度画像のB-G値を深度として解析し、勾配相関と等高線から3軸の回転角度を推定します。

![Depth Analysis](docs/depth_analysis.png)

| パラメータ | 推定手法 | 推定誤差 |
|-----------|---------|---------|
| theta（水平回転） | `asin(-grad_x / 0.35)` + 左右面積比補正 | 25.5° |
| phi（垂直傾き） | `-90 + grad_y * 50` | 1.3° |
| roll（回転） | 深度等高線の角度変化トレンド × 6 | 4.5° |

### 3. 多段階最適化

推定角度を初期値として、粗い探索→細かい探索→カメラ位置調整を段階的に実行します。

```
Phase 1: ±30° / 5°ステップ (低解像度) ───→ 3.3秒
Phase 2: ±8°  / 1°ステップ            ───→ 8.3秒
Phase 3: カメラ位置の座標降下法        ───→ 6.5秒
Phase 4: ±1°  / 0.2°ステップ (精密)   ───→ 14秒
Phase 5: clip_y (足元クリッピング)     ───→ 0.2秒
```

### 4. テクスチャ付きオーバーレイ

最適パラメータで高ポリモデル（427K面）をテクスチャ付きでレンダリングし、元画像に重ね合わせます。

| テクスチャ付きレンダリング | 輪郭オーバーレイ | 50%ブレンド |
|:---:|:---:|:---:|
| ![Rendered](docs/rendered.png) | ![Contour](docs/overlay_contour.png) | ![Overlay](docs/overlay_50.png) |

## 3Dモデル

使用している3Dモデル（低ポリ版）を直接確認できます。

[rabit_low.stl をブラウザで3D表示](models_rabit_obj/rabit_low.stl)

## 実装

### C++ 版（推奨）

C++ + OpenCV + OpenMP による高速実装です。

**必要環境:**
- Visual Studio 2022
- OpenCV 4.x
- CMake

**ビルドと実行:**

```bash
cd cpp_pipeline
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\Release\pose_match.exe
```

### Python 版

```bash
pip install -r requirements.txt
```

| スクリプト | 説明 |
|-----------|------|
| `estimate_angle_from_depth.py` | 深度画像から初期角度を推定 |
| `pose_match_unified.py` | SQLiteベースの統合マッチング |
| `generate_features_sqlite.py` | 特徴量データベース生成（40万パターン） |

## 性能比較

| | Python版 | C++版 |
|---|---------|-------|
| IoU | 97.05% | 97.03% |
| 実行時間 | 数分〜十数分 | **35秒** |
| 事前DB | 113MB 必要 | **不要** |
| 前処理 | DB生成に数時間 | **不要** |
| テクスチャ描画 | 数秒 | **0.15秒** |

## 座標系

- 回転順序: **ZYX**（Z→Y→X の順に適用）
- 初期回転: X軸に -90°（OBJモデルの座標系変換）
- カメラ: 透視投影（FOV 45°）

## プロジェクト構成

```
├── cpp_pipeline/
│   ├── CMakeLists.txt            # ビルド設定
│   └── main.cpp                  # C++ 全パイプライン
├── estimate_angle_from_depth.py  # 深度→角度推定 (Python)
├── pose_match_unified.py         # 統合マッチング (Python)
├── generate_features_sqlite.py   # 特徴量DB生成 (Python)
├── models_rabit_obj/
│   ├── rabit.obj                 # 高ポリモデル (427K faces)
│   ├── rabit_low.obj             # 低ポリモデル (60K faces)
│   ├── rabit_low.stl             # STL版 (GitHub 3D表示用)
│   └── rabit01.jpg               # テクスチャ
├── docs/                         # README用画像
├── Image0.png                    # 入力画像
├── Image0_depth.png              # 深度画像
└── requirements.txt
```

## ライセンス

MIT License

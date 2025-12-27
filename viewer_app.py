"""
3Dモデルビューア - Flask + Three.js
OBJファイルを表示し、カメラ角度・スケールをリアルタイム調整
"""

from flask import Flask, render_template_string, send_from_directory, jsonify
from pathlib import Path
import os

app = Flask(__name__)

BASE_PATH = Path(__file__).resolve().parent
MODEL_DIR = BASE_PATH / "models_rabit_obj"

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>3D Model Viewer - Overlay Mode</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }
        #container {
            display: flex;
            height: 100vh;
        }
        #viewer {
            flex: 1;
            position: relative;
        }
        #background-image {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            opacity: 0.8;
            z-index: 0;
        }
        #three-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }
        #controls {
            width: 350px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-left: 2px solid #0f3460;
            z-index: 10;
        }
        h2 {
            color: #e94560;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #0f3460;
        }
        .control-group {
            margin-bottom: 15px;
        }
        .control-group label {
            display: block;
            margin-bottom: 5px;
            color: #94a3b8;
            font-size: 12px;
        }
        .control-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .control-row span {
            width: 60px;
            font-weight: bold;
        }
        input[type="range"] {
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 3px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #e94560;
            border-radius: 50%;
            cursor: pointer;
        }
        input[type="number"] {
            width: 70px;
            padding: 5px;
            background: #0f3460;
            border: 1px solid #1a1a2e;
            color: #eee;
            border-radius: 4px;
            text-align: center;
        }
        #params-display {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 15px;
        }
        #params-display .value {
            color: #4ade80;
        }
        button {
            width: 100%;
            padding: 12px;
            margin-top: 8px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: background 0.2s;
        }
        button:hover {
            background: #ff6b6b;
        }
        button.secondary {
            background: #0f3460;
        }
        button.secondary:hover {
            background: #1a4a7a;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 5;
        }
        .checkbox-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }
        .checkbox-row input {
            width: 18px;
            height: 18px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="viewer">
            <img id="background-image" src="/image/Image0.png" alt="Background">
            <div id="info">
                スライダーで調整 | 背景画像に3Dモデルを重ねて確認
            </div>
        </div>
        <div id="controls">
            <h2>Overlay Viewer</h2>

            <div class="checkbox-row">
                <input type="checkbox" id="showBackground" checked>
                <label for="showBackground">背景画像を表示</label>
            </div>
            <div class="checkbox-row">
                <input type="checkbox" id="wireframe">
                <label for="wireframe">ワイヤーフレーム表示</label>
            </div>

            <div class="control-group">
                <label>モデル透明度</label>
                <div class="control-row">
                    <span>opacity</span>
                    <input type="range" id="opacity" min="0" max="100" value="70" step="5">
                    <input type="number" id="opacity-val" value="70">
                </div>
            </div>

            <h2>Camera Parameters</h2>

            <div class="control-group">
                <label>Theta (水平回転) -180° ~ 180°</label>
                <div class="control-row">
                    <span>θ</span>
                    <input type="range" id="theta" min="-180" max="180" value="0" step="1">
                    <input type="number" id="theta-val" value="0">
                </div>
            </div>

            <div class="control-group">
                <label>Phi (垂直回転) -90° ~ 90°</label>
                <div class="control-row">
                    <span>φ</span>
                    <input type="range" id="phi" min="-90" max="90" value="0" step="1">
                    <input type="number" id="phi-val" value="0">
                </div>
            </div>

            <div class="control-group">
                <label>Roll (傾き) -180° ~ 180°</label>
                <div class="control-row">
                    <span>roll</span>
                    <input type="range" id="roll" min="-180" max="180" value="0" step="1">
                    <input type="number" id="roll-val" value="0">
                </div>
            </div>

            <div class="control-group">
                <label>Scale (スケール)</label>
                <div class="control-row">
                    <span>scale</span>
                    <input type="range" id="scale" min="10" max="200" value="100" step="1">
                    <input type="number" id="scale-val" value="100">
                </div>
            </div>

            <div class="control-group">
                <label>Position X</label>
                <div class="control-row">
                    <span>X</span>
                    <input type="range" id="posX" min="-200" max="200" value="0" step="1">
                    <input type="number" id="posX-val" value="0">
                </div>
            </div>

            <div class="control-group">
                <label>Position Y</label>
                <div class="control-row">
                    <span>Y</span>
                    <input type="range" id="posY" min="-200" max="200" value="0" step="1">
                    <input type="number" id="posY-val" value="0">
                </div>
            </div>

            <div id="params-display">
                <div>theta: <span class="value" id="disp-theta">0</span>°</div>
                <div>phi: <span class="value" id="disp-phi">0</span>°</div>
                <div>roll: <span class="value" id="disp-roll">0</span>°</div>
                <div>scale: <span class="value" id="disp-scale">100</span></div>
                <div>posX: <span class="value" id="disp-posX">0</span></div>
                <div>posY: <span class="value" id="disp-posY">0</span></div>
            </div>

            <button onclick="copyParams()">📋 パラメータをコピー</button>
            <button class="secondary" onclick="resetParams()">🔄 リセット</button>
            <button class="secondary" onclick="applyPreset()">📥 プリセット適用 (θ=68, φ=-56, roll=-57)</button>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>

    <script>
        let scene, camera, renderer, model, modelMaterial;
        const INITIAL_RX = -90; // モデルを立たせる

        // 初期化
        function init() {
            const container = document.getElementById('viewer');

            // シーン（透明背景）
            scene = new THREE.Scene();

            // 正射影カメラ（平行投影）- Pythonコードと同じ
            const aspect = container.clientWidth / container.clientHeight;
            const frustumSize = 3;
            camera = new THREE.OrthographicCamera(
                frustumSize * aspect / -2,
                frustumSize * aspect / 2,
                frustumSize / 2,
                frustumSize / -2,
                0.1,
                1000
            );
            camera.position.set(0, 0, 5);

            // レンダラー（透明背景対応）
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setClearColor(0x000000, 0);
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.domElement.id = 'three-canvas';
            container.appendChild(renderer.domElement);

            // ライト
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);

            const backLight = new THREE.DirectionalLight(0xffffff, 0.4);
            backLight.position.set(-5, -5, -5);
            scene.add(backLight);

            // OBJモデル読み込み
            const loader = new THREE.OBJLoader();
            loader.load('/model/rabit.obj', function(obj) {
                model = obj;

                // マテリアル設定（半透明対応）
                modelMaterial = new THREE.MeshPhongMaterial({
                    color: 0x00ff00,  // 緑色（重ね合わせ確認用）
                    flatShading: false,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.7
                });

                model.traverse(function(child) {
                    if (child instanceof THREE.Mesh) {
                        child.material = modelMaterial;
                    }
                });

                // 中心に配置
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                model.position.sub(center);

                // 初期回転（モデルを立たせる）
                model.rotation.x = THREE.MathUtils.degToRad(INITIAL_RX);

                scene.add(model);

                updateModelTransform();
            });

            // リサイズ対応
            window.addEventListener('resize', onWindowResize);

            // スライダーイベント
            setupSliderEvents();
            setupCheckboxEvents();

            animate();
        }

        function setupCheckboxEvents() {
            document.getElementById('showBackground').addEventListener('change', (e) => {
                document.getElementById('background-image').style.display = e.target.checked ? 'block' : 'none';
            });

            document.getElementById('wireframe').addEventListener('change', (e) => {
                if (modelMaterial) {
                    modelMaterial.wireframe = e.target.checked;
                }
            });

            document.getElementById('opacity').addEventListener('input', (e) => {
                document.getElementById('opacity-val').value = e.target.value;
                if (modelMaterial) {
                    modelMaterial.opacity = e.target.value / 100;
                }
            });

            document.getElementById('opacity-val').addEventListener('change', (e) => {
                document.getElementById('opacity').value = e.target.value;
                if (modelMaterial) {
                    modelMaterial.opacity = e.target.value / 100;
                }
            });
        }

        function setupSliderEvents() {
            const params = ['theta', 'phi', 'roll', 'scale', 'posX', 'posY'];

            params.forEach(param => {
                const slider = document.getElementById(param);
                const input = document.getElementById(param + '-val');

                slider.addEventListener('input', () => {
                    input.value = slider.value;
                    updateModelTransform();
                    updateDisplay();
                });

                input.addEventListener('change', () => {
                    slider.value = input.value;
                    updateModelTransform();
                    updateDisplay();
                });
            });
        }

        function updateModelTransform() {
            if (!model) return;

            const theta = parseFloat(document.getElementById('theta').value);
            const phi = parseFloat(document.getElementById('phi').value);
            const roll = parseFloat(document.getElementById('roll').value);
            const scale = parseFloat(document.getElementById('scale').value) / 100;
            const posX = parseFloat(document.getElementById('posX').value) / 100;
            const posY = parseFloat(document.getElementById('posY').value) / 100;

            // 回転適用（spherical_to_rotation相当）
            const rx = INITIAL_RX + phi;
            const ry = -theta;
            const rz = roll;

            // ZYX順序で回転を適用（Pythonコードと一致させる）
            model.rotation.order = 'ZYX';
            model.rotation.x = THREE.MathUtils.degToRad(rx);
            model.rotation.y = THREE.MathUtils.degToRad(ry);
            model.rotation.z = THREE.MathUtils.degToRad(rz);

            model.scale.set(scale, scale, scale);
            model.position.x = posX;
            model.position.y = posY;
        }

        function updateDisplay() {
            document.getElementById('disp-theta').textContent = document.getElementById('theta').value;
            document.getElementById('disp-phi').textContent = document.getElementById('phi').value;
            document.getElementById('disp-roll').textContent = document.getElementById('roll').value;
            document.getElementById('disp-scale').textContent = document.getElementById('scale').value;
            document.getElementById('disp-posX').textContent = document.getElementById('posX').value;
            document.getElementById('disp-posY').textContent = document.getElementById('posY').value;
        }

        function copyParams() {
            const params = {
                theta: parseInt(document.getElementById('theta').value),
                phi: parseInt(document.getElementById('phi').value),
                roll: parseInt(document.getElementById('roll').value),
                scale: parseInt(document.getElementById('scale').value),
                posX: parseInt(document.getElementById('posX').value),
                posY: parseInt(document.getElementById('posY').value)
            };

            const text = JSON.stringify(params, null, 2);
            navigator.clipboard.writeText(text).then(() => {
                alert('パラメータをクリップボードにコピーしました！\\n\\n' + text);
            });
        }

        function resetParams() {
            document.getElementById('theta').value = 0;
            document.getElementById('phi').value = 0;
            document.getElementById('roll').value = 0;
            document.getElementById('scale').value = 100;
            document.getElementById('posX').value = 0;
            document.getElementById('posY').value = 0;

            ['theta', 'phi', 'roll', 'scale', 'posX', 'posY'].forEach(p => {
                document.getElementById(p + '-val').value = document.getElementById(p).value;
            });

            updateModelTransform();
            updateDisplay();
        }

        function applyPreset() {
            // 前回の最良結果を適用
            document.getElementById('theta').value = 68;
            document.getElementById('phi').value = -56;
            document.getElementById('roll').value = -57;
            document.getElementById('scale').value = 100;
            document.getElementById('posX').value = 0;
            document.getElementById('posY').value = 0;

            ['theta', 'phi', 'roll', 'scale', 'posX', 'posY'].forEach(p => {
                document.getElementById(p + '-val').value = document.getElementById(p).value;
            });

            updateModelTransform();
            updateDisplay();
        }

        function onWindowResize() {
            const container = document.getElementById('viewer');
            const aspect = container.clientWidth / container.clientHeight;
            const frustumSize = 3;

            camera.left = frustumSize * aspect / -2;
            camera.right = frustumSize * aspect / 2;
            camera.top = frustumSize / 2;
            camera.bottom = frustumSize / -2;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }

        init();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/model/<path:filename>')
def serve_model(filename):
    return send_from_directory(MODEL_DIR, filename)

@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory(BASE_PATH, filename)

@app.route('/api/params', methods=['GET'])
def get_params():
    return jsonify({
        "theta": 68,
        "phi": -56,
        "roll": -57,
        "scale": 100
    })

if __name__ == '__main__':
    print("=" * 50)
    print("  3D Model Viewer")
    print("=" * 50)
    print(f"\nModel directory: {MODEL_DIR}")
    print("\nブラウザで開いてください:")
    print("  http://localhost:5000")
    print("\n終了: Ctrl+C")
    print("=" * 50)
    app.run(debug=True, port=5000)

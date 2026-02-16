// pose_match pipeline - C++ version
// Full pipeline: depth angle estimation -> grid search -> optimization -> textured overlay
//
// Build: cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release
// Run:   build\Release\pose_match.exe

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

// =============================================================================
// Configuration
// =============================================================================

const std::string BASE_DIR = "C:/nagano/3Dnagano";
const std::string INPUT_IMAGE = BASE_DIR + "/Image0.png";
const std::string DEPTH_IMAGE = BASE_DIR + "/Image0_depth.png";
const std::string MODEL_PATH = BASE_DIR + "/models_rabit_obj/rabit_low.obj";
const std::string MODEL_HI_PATH = BASE_DIR + "/models_rabit_obj/rabit.obj";
const std::string TEXTURE_PATH = BASE_DIR + "/models_rabit_obj/rabit01.jpg";
const std::string OUT_DIR = BASE_DIR + "/rotation_results_cpp";

const float CAMERA_FOV = 45.0f;
const float INITIAL_RX = -90.0f;
const int RENDER_SIZE = 512;
const int NORM_SIZE = 256;
const float BBOX_MARGIN_RATIO = 0.12f;

// =============================================================================
// Timer
// =============================================================================

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
    std::string label_;
public:
    Timer(const std::string& label) : label_(label) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    ~Timer() {
        printf("  [%s] %.1f ms\n", label_.c_str(), elapsed_ms());
    }
};

// =============================================================================
// 3D Math
// =============================================================================

struct Vec3 { float x, y, z; };
struct Vec2 { float u, v; };
struct Face { int v[3]; int vt[3]; }; // vertex indices + UV indices

inline float deg2rad(float d) { return d * 3.14159265358979323846f / 180.0f; }

struct Mat3 {
    float m[3][3];
    Vec3 operator*(const Vec3& v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r = {};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    r.m[i][j] += m[i][k] * o.m[k][j];
        return r;
    }
};

Mat3 rot_x(float deg) {
    float a = deg2rad(deg), c = cosf(a), s = sinf(a);
    return {{{1,0,0},{0,c,-s},{0,s,c}}};
}
Mat3 rot_y(float deg) {
    float a = deg2rad(deg), c = cosf(a), s = sinf(a);
    return {{{c,0,s},{0,1,0},{-s,0,c}}};
}
Mat3 rot_z(float deg) {
    float a = deg2rad(deg), c = cosf(a), s = sinf(a);
    return {{{c,-s,0},{s,c,0},{0,0,1}}};
}

// =============================================================================
// OBJ Loader
// =============================================================================

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Vec2> uvs;
    std::vector<Face> faces;
};

Mesh load_obj(const std::string& path) {
    Mesh mesh;
    std::ifstream file(path);
    if (!file.is_open()) {
        printf("ERROR: Cannot open %s\n", path.c_str());
        return mesh;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            Vec3 v;
            iss >> v.x >> v.y >> v.z;
            mesh.vertices.push_back(v);
        }
        else if (prefix == "vt") {
            Vec2 vt;
            iss >> vt.u >> vt.v;
            mesh.uvs.push_back(vt);
        }
        else if (prefix == "f") {
            std::vector<int> vi, vti;
            std::string tok;
            while (iss >> tok) {
                int v_idx = 0, vt_idx = 0;
                size_t p1 = tok.find('/');
                if (p1 == std::string::npos) {
                    v_idx = std::stoi(tok) - 1;
                } else {
                    v_idx = std::stoi(tok.substr(0, p1)) - 1;
                    size_t p2 = tok.find('/', p1 + 1);
                    std::string vt_str = tok.substr(p1 + 1, (p2 != std::string::npos ? p2 - p1 - 1 : std::string::npos));
                    if (!vt_str.empty()) vt_idx = std::stoi(vt_str) - 1;
                }
                vi.push_back(v_idx);
                vti.push_back(vt_idx);
            }
            // Triangulate
            for (size_t i = 1; i + 1 < vi.size(); i++) {
                Face f;
                f.v[0] = vi[0]; f.v[1] = vi[i]; f.v[2] = vi[i+1];
                f.vt[0] = vti[0]; f.vt[1] = vti[i]; f.vt[2] = vti[i+1];
                mesh.faces.push_back(f);
            }
        }
    }
    return mesh;
}

void normalize_vertices(Mesh& mesh) {
    Vec3 center = {0, 0, 0};
    for (auto& v : mesh.vertices) {
        center.x += v.x; center.y += v.y; center.z += v.z;
    }
    float n = (float)mesh.vertices.size();
    center.x /= n; center.y /= n; center.z /= n;

    float max_abs = 0;
    for (auto& v : mesh.vertices) {
        v.x -= center.x; v.y -= center.y; v.z -= center.z;
        max_abs = std::max(max_abs, std::max({fabsf(v.x), fabsf(v.y), fabsf(v.z)}));
    }
    float scale = max_abs * 2.0f;
    for (auto& v : mesh.vertices) {
        v.x /= scale; v.y /= scale; v.z /= scale;
    }
}

// =============================================================================
// Image I/O (Unicode paths)
// =============================================================================

cv::Mat imread_unicode(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return cv::Mat();
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uchar> buf(sz);
    fread(buf.data(), 1, sz, fp);
    fclose(fp);
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}

// =============================================================================
// Depth-based Angle Estimation
// =============================================================================

struct AngleEstimate {
    float theta, phi, roll;
};

AngleEstimate estimate_angles_from_depth(const cv::Mat& depth_bgr, const cv::Mat& mask) {
    int H = depth_bgr.rows, W = depth_bgr.cols;

    // Compute B-G depth
    std::vector<float> depth_map(H * W, 0);
    for (int y = 0; y < H; y++) {
        const uchar* row = depth_bgr.ptr<uchar>(y);
        for (int x = 0; x < W; x++) {
            depth_map[y * W + x] = (float)row[x * 3] - (float)row[x * 3 + 1]; // B - G
        }
    }

    // Find valid pixels
    std::vector<int> xs, ys;
    std::vector<float> dvals;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (mask.at<uchar>(y, x) > 0) {
                xs.push_back(x);
                ys.push_back(y);
                dvals.push_back(depth_map[y * W + x]);
            }
        }
    }

    if (xs.empty()) return {0, 0, 0};

    int x_min = *std::min_element(xs.begin(), xs.end());
    int x_max = *std::max_element(xs.begin(), xs.end());
    int y_min = *std::min_element(ys.begin(), ys.end());
    int y_max = *std::max_element(ys.begin(), ys.end());
    float cx = (x_min + x_max) / 2.0f;
    float cy = (y_min + y_max) / 2.0f;
    float w = (float)(x_max - x_min);
    float h = (float)(y_max - y_min);
    if (w < 1 || h < 1) return {0, 0, 0};

    // Normalize
    size_t N = xs.size();
    float d_mean = 0;
    for (auto d : dvals) d_mean += d;
    d_mean /= N;
    float d_std = 0;
    for (auto d : dvals) d_std += (d - d_mean) * (d - d_mean);
    d_std = sqrtf(d_std / N) + 1e-10f;

    // Gradient correlation
    double sum_xd = 0, sum_x2 = 0, sum_yd = 0, sum_y2 = 0, sum_xn = 0, sum_yn = 0, sum_dn = 0;
    for (size_t i = 0; i < N; i++) {
        float xn = (xs[i] - cx) / w;
        float yn = (ys[i] - cy) / h;
        float dn = (dvals[i] - d_mean) / d_std;
        sum_xd += xn * dn;
        sum_x2 += xn * xn;
        sum_yd += yn * dn;
        sum_y2 += yn * yn;
        sum_xn += xn;
        sum_yn += yn;
        sum_dn += dn;
    }
    // Correlation
    double mean_x = sum_xn / N, mean_d = sum_dn / N, mean_y = sum_yn / N;
    double cov_xd = sum_xd / N - mean_x * mean_d;
    double var_x = sum_x2 / N - mean_x * mean_x;
    double cov_yd = sum_yd / N - mean_y * mean_d;
    double var_y = sum_y2 / N - mean_y * mean_y;
    float grad_x = (var_x > 1e-10) ? (float)(cov_xd / sqrt(var_x * (1.0))) : 0; // simplified corr
    float grad_y = (var_y > 1e-10) ? (float)(cov_yd / sqrt(var_y * (1.0))) : 0;

    // Full correlation
    double var_d = 1.0; // d_norm has std=1
    grad_x = (var_x > 1e-10) ? (float)(cov_xd / sqrt(var_x * var_d)) : 0;
    grad_y = (var_y > 1e-10) ? (float)(cov_yd / sqrt(var_y * var_d)) : 0;

    printf("  grad_x=%.3f, grad_y=%.3f\n", grad_x, grad_y);

    // Area ratio (left vs right)
    int left_area = 0, right_area = 0;
    for (size_t i = 0; i < N; i++) {
        if (xs[i] < (int)cx) left_area++;
        else right_area++;
    }
    float area_ratio = (float)(right_area - left_area) / (float)(left_area + right_area);

    // Theta estimation
    float gx_clamped = std::clamp(-grad_x / 0.35f, -1.0f, 1.0f);
    float theta_base = asinf(gx_clamped) * 180.0f / 3.14159265f;
    float theta_est;
    if (area_ratio > 0)
        theta_est = std::max(theta_base, 0.0f) + area_ratio * 30.0f;
    else
        theta_est = std::min(theta_base, 0.0f) + area_ratio * 30.0f;

    // Phi estimation
    float phi_est = -90.0f + grad_y * 50.0f;

    // Roll estimation: contour angle trend
    float d_min_v = *std::min_element(dvals.begin(), dvals.end());
    float d_max_v = *std::max_element(dvals.begin(), dvals.end());

    std::vector<float> contour_angles;
    float levels[] = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    for (float lp : levels) {
        float level = d_min_v + (d_max_v - d_min_v) * lp;
        std::vector<float> c_xs, c_ys;
        for (size_t i = 0; i < N; i++) {
            if (fabsf(dvals[i] - level) < 8.0f) {
                c_xs.push_back((float)xs[i]);
                c_ys.push_back((float)ys[i]);
            }
        }
        if (c_xs.size() > 20) {
            float mx = 0, my = 0;
            for (size_t i = 0; i < c_xs.size(); i++) { mx += c_xs[i]; my += c_ys[i]; }
            mx /= c_xs.size(); my /= c_ys.size();

            double cxx = 0, cyy = 0, cxy = 0;
            for (size_t i = 0; i < c_xs.size(); i++) {
                float dx = c_xs[i] - mx, dy = c_ys[i] - my;
                cxx += dx * dx; cyy += dy * dy; cxy += dx * dy;
            }
            // PCA: eigenvalues of 2x2 covariance
            double a = cxx, b = cxy, d = cyy;
            double trace = a + d, det = a * d - b * b;
            double disc = trace * trace / 4.0 - det;
            if (disc >= 0) {
                double sqrtd = sqrt(disc);
                double ev1 = trace / 2.0 + sqrtd; // larger eigenvalue
                // Eigenvector for ev1: (b, ev1-a) or (ev1-d, b)
                double vx = b, vy = ev1 - a;
                double len = sqrt(vx * vx + vy * vy);
                if (len > 1e-10) {
                    float ang = atan2f((float)vy, (float)vx) * 180.0f / 3.14159265f;
                    contour_angles.push_back(ang);
                }
            }
        }
    }

    float roll_est = 0;
    if (contour_angles.size() >= 2) {
        float trend = contour_angles.back() - contour_angles.front();
        roll_est = trend * 6.0f;
        printf("  contour_trend=%.1f, roll_est=%.1f\n", trend, roll_est);
    }

    printf("  Estimated: theta=%.1f, phi=%.1f, roll=%.1f\n", theta_est, phi_est, roll_est);
    return {theta_est, phi_est, roll_est};
}

// =============================================================================
// Software Silhouette Renderer
// =============================================================================

struct BBox { int x0, y0, x1, y1; };

BBox bbox_from_mask(const cv::Mat& mask) {
    BBox bb = {mask.cols, mask.rows, 0, 0};
    bool found = false;
    for (int y = 0; y < mask.rows; y++) {
        const uchar* row = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; x++) {
            if (row[x] > 0) {
                bb.x0 = std::min(bb.x0, x);
                bb.y0 = std::min(bb.y0, y);
                bb.x1 = std::max(bb.x1, x);
                bb.y1 = std::max(bb.y1, y);
                found = true;
            }
        }
    }
    if (!found) return {-1,-1,-1,-1};
    return bb;
}

BBox expand_bbox(BBox bb, int W, int H, float margin) {
    int bw = bb.x1 - bb.x0, bh = bb.y1 - bb.y0;
    int m = (int)(std::max(bw, bh) * margin);
    return {
        std::max(0, bb.x0 - m), std::max(0, bb.y0 - m),
        std::min(W - 1, bb.x1 + m), std::min(H - 1, bb.y1 + m)
    };
}

// Render silhouette with world-Y clipping
cv::Mat render_silhouette(const Mesh& mesh, float theta, float phi, float roll,
                          float cam_x, float cam_y, float cam_z,
                          int size, float clip_y = -999.0f) {
    Mat3 R = rot_x(INITIAL_RX + phi) * rot_y(-theta) * rot_z(roll);

    int nv = (int)mesh.vertices.size();
    std::vector<Vec3> tv(nv); // transformed
    std::vector<float> px(nv), py(nv);

    float f = size / (2.0f * tanf(deg2rad(CAMERA_FOV / 2.0f)));

    for (int i = 0; i < nv; i++) {
        tv[i] = R * mesh.vertices[i];
        float vx = tv[i].x - cam_x;
        float vy = tv[i].y - cam_y;
        float vz = std::max(0.1f, cam_z - tv[i].z);
        px[i] = (vx / vz) * f + size * 0.5f;
        py[i] = size * 0.5f - (vy / vz) * f;
    }

    cv::Mat sil = cv::Mat::zeros(size, size, CV_8UC1);
    cv::Point pts_arr[3];

    for (const auto& face : mesh.faces) {
        int a = face.v[0], b = face.v[1], c = face.v[2];
        if (a < 0 || a >= nv || b < 0 || b >= nv || c < 0 || c >= nv) continue;

        if (clip_y > -990.0f) {
            if (tv[a].y < clip_y || tv[b].y < clip_y || tv[c].y < clip_y)
                continue;
        }

        pts_arr[0] = {(int)px[a], (int)py[a]};
        pts_arr[1] = {(int)px[b], (int)py[b]};
        pts_arr[2] = {(int)px[c], (int)py[c]};
        cv::fillConvexPoly(sil, pts_arr, 3, cv::Scalar(255));
    }

    return sil;
}

// =============================================================================
// IoU
// =============================================================================

float calc_iou(const cv::Mat& a, const cv::Mat& b) {
    int inter = 0, uni = 0;
    for (int y = 0; y < a.rows; y++) {
        const uchar* ra = a.ptr<uchar>(y);
        const uchar* rb = b.ptr<uchar>(y);
        for (int x = 0; x < a.cols; x++) {
            bool va = ra[x] > 0, vb = rb[x] > 0;
            if (va && vb) inter++;
            if (va || vb) uni++;
        }
    }
    return uni > 0 ? (float)inter / uni : 0.0f;
}

cv::Mat crop_and_resize(const cv::Mat& img, BBox bb, int size) {
    cv::Rect roi(bb.x0, bb.y0, bb.x1 - bb.x0 + 1, bb.y1 - bb.y0 + 1);
    cv::Mat crop = img(roi);
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(size, size), 0, 0, cv::INTER_NEAREST);
    return resized;
}

float compute_iou(const Mesh& mesh, const cv::Mat& target_norm,
                  float theta, float phi, float roll,
                  float cam_x, float cam_y, float cam_z, float clip_y,
                  int render_size = RENDER_SIZE) {
    cv::Mat sil = render_silhouette(mesh, theta, phi, roll, cam_x, cam_y, cam_z, render_size, clip_y);
    BBox bb = bbox_from_mask(sil);
    if (bb.x0 < 0) return 0.0f;
    BBox bbe = expand_bbox(bb, render_size, render_size, BBOX_MARGIN_RATIO);
    cv::Mat sil_norm = crop_and_resize(sil, bbe, NORM_SIZE);
    return calc_iou(target_norm, sil_norm);
}

// =============================================================================
// Fill holes
// =============================================================================

cv::Mat fill_holes(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat filled = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(filled, contours, -1, cv::Scalar(1), cv::FILLED);
    return filled;
}

// =============================================================================
// Multi-stage Optimization
// =============================================================================

struct Params {
    float theta, phi, roll;
    float cam_x, cam_y, cam_z;
    float clip_y;
    float iou;
};

Params optimize(const Mesh& mesh, const cv::Mat& target_norm,
                float init_theta, float init_phi, float init_roll) {
    Params best;
    best.theta = init_theta;
    best.phi = init_phi;
    best.roll = init_roll;
    best.cam_x = -0.45f;
    best.cam_y = -0.45f;
    best.cam_z = 2.72f;
    best.clip_y = -0.39f;

    best.iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                            best.cam_x, best.cam_y, best.cam_z, best.clip_y);
    printf("  [Init] IoU: %.4f%%\n", best.iou * 100);

    // Phase 1: Coarse rotation (+-30 deg, step 5) at LOW resolution for speed
    {
        Timer t("Phase1 Coarse");
        float range = 30.0f, step = 5.0f;
        float b_th = best.theta, b_ph = best.phi, b_ro = best.roll;
        float b_iou = 0.0f; // re-evaluate at low res

        const int LO_RES = 256;

        int n_th = (int)(2*range/step) + 1;
        int n_ph = (int)(2*range/step) + 1;
        int n_ro = (int)(2*range/step) + 1;
        int total = n_th * n_ph * n_ro;
        #pragma omp parallel for schedule(dynamic, 64)
        for (int idx = 0; idx < total; idx++) {
            int it = idx / (n_ph * n_ro);
            int ip = (idx / n_ro) % n_ph;
            int ir = idx % n_ro;
            float th = init_theta - range + it * step;
            float ph = init_phi - range + ip * step;
            float ro = init_roll - range + ir * step;
            float iou = compute_iou(mesh, target_norm, th, ph, ro,
                                   best.cam_x, best.cam_y, best.cam_z, best.clip_y, LO_RES);
            #pragma omp critical
            {
                if (iou > b_iou) {
                    b_iou = iou;
                    b_th = th; b_ph = ph; b_ro = ro;
                }
            }
        }
        // Re-evaluate top result at full resolution
        best.iou = compute_iou(mesh, target_norm, b_th, b_ph, b_ro,
                               best.cam_x, best.cam_y, best.cam_z, best.clip_y);
        best.theta = b_th; best.phi = b_ph; best.roll = b_ro; best.iou = b_iou;
        printf("  [Phase1] theta=%.1f phi=%.1f roll=%.1f IoU=%.4f%%\n",
               best.theta, best.phi, best.roll, best.iou * 100);
    }

    // Phase 2: Medium rotation (+-8 deg, step 1)
    {
        Timer t("Phase2 Medium rotation");
        float range = 8.0f, step = 1.0f;
        float b_th = best.theta, b_ph = best.phi, b_ro = best.roll;
        float b_iou = best.iou;
        float c_th = best.theta, c_ph = best.phi, c_ro = best.roll;

        int n_th2 = (int)(2*range/step) + 1;
        int n_ph2 = (int)(2*range/step) + 1;
        int n_ro2 = (int)(2*range/step) + 1;
        int total2 = n_th2 * n_ph2 * n_ro2;
        #pragma omp parallel for schedule(dynamic, 64)
        for (int idx = 0; idx < total2; idx++) {
            int it = idx / (n_ph2 * n_ro2);
            int ip = (idx / n_ro2) % n_ph2;
            int ir = idx % n_ro2;
            float th = c_th - range + it * step;
            float ph = c_ph - range + ip * step;
            float ro = c_ro - range + ir * step;
            float iou = compute_iou(mesh, target_norm, th, ph, ro,
                                   best.cam_x, best.cam_y, best.cam_z, best.clip_y);
            #pragma omp critical
            {
                if (iou > b_iou) {
                    b_iou = iou;
                    b_th = th; b_ph = ph; b_ro = ro;
                }
            }
        }
        best.theta = b_th; best.phi = b_ph; best.roll = b_ro; best.iou = b_iou;
        printf("  [Phase2] theta=%.1f phi=%.1f roll=%.1f IoU=%.4f%%\n",
               best.theta, best.phi, best.roll, best.iou * 100);
    }

    // Phase 3: Camera position (coordinate descent, using raw IoU without normalization)
    {
        Timer t("Phase3 Camera");
        // For camera search, render at RENDER_SIZE and compare against target
        // placed at same scale - we need to compare silhouettes at the same
        // resolution including position/scale effects
        for (int iter = 0; iter < 2; iter++) {
            // cam_x
            float cx0 = best.cam_x;
            for (float dx = -0.2f; dx <= 0.201f; dx += 0.005f) {
                float cx = cx0 + dx;
                float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                       cx, best.cam_y, best.cam_z, best.clip_y);
                if (iou > best.iou) { best.iou = iou; best.cam_x = cx; }
            }
            // cam_y
            float cy0 = best.cam_y;
            for (float dy = -0.2f; dy <= 0.201f; dy += 0.005f) {
                float cy = cy0 + dy;
                float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                       best.cam_x, cy, best.cam_z, best.clip_y);
                if (iou > best.iou) { best.iou = iou; best.cam_y = cy; }
            }
            // cam_z
            float cz0 = best.cam_z;
            for (float dz = -0.5f; dz <= 0.501f; dz += 0.01f) {
                float cz = cz0 + dz;
                float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                       best.cam_x, best.cam_y, cz, best.clip_y);
                if (iou > best.iou) { best.iou = iou; best.cam_z = cz; }
            }
            // Fine clip_y interleaved
            float cly0 = best.clip_y;
            for (float cl = cly0 - 0.1f; cl <= cly0 + 0.1f; cl += 0.005f) {
                float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                       best.cam_x, best.cam_y, best.cam_z, cl);
                if (iou > best.iou) { best.iou = iou; best.clip_y = cl; }
            }
        }
        printf("  [Phase3] cam=(%.3f,%.3f,%.3f) clip_y=%.3f IoU=%.4f%%\n",
               best.cam_x, best.cam_y, best.cam_z, best.clip_y, best.iou * 100);
    }

    // Phase 4: Very fine rotation (+-1 deg, step 0.2)
    {
        Timer t("Phase4 Very fine");
        float range = 1.0f, step = 0.2f;
        float b_th = best.theta, b_ph = best.phi, b_ro = best.roll;
        float b_iou = best.iou;
        float c_th = best.theta, c_ph = best.phi, c_ro = best.roll;

        for (float dt = -range; dt <= range + 0.01f; dt += step) {
            for (float dp = -range; dp <= range + 0.01f; dp += step) {
                for (float dr = -range; dr <= range + 0.01f; dr += step) {
                    float iou = compute_iou(mesh, target_norm,
                                           c_th + dt, c_ph + dp, c_ro + dr,
                                           best.cam_x, best.cam_y, best.cam_z, best.clip_y);
                    if (iou > b_iou) {
                        b_iou = iou;
                        b_th = c_th + dt; b_ph = c_ph + dp; b_ro = c_ro + dr;
                    }
                }
            }
        }
        best.theta = b_th; best.phi = b_ph; best.roll = b_ro; best.iou = b_iou;
        printf("  [Phase4] theta=%.2f phi=%.2f roll=%.2f IoU=%.4f%%\n",
               best.theta, best.phi, best.roll, best.iou * 100);
    }

    // Phase 5: Clip Y
    {
        Timer t("Phase5 ClipY");
        float b_clip = best.clip_y;
        float b_iou = best.iou;
        for (float cy = -0.5f; cy <= 0.01f; cy += 0.02f) {
            float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                   best.cam_x, best.cam_y, best.cam_z, cy);
            if (iou > b_iou) { b_iou = iou; b_clip = cy; }
        }
        best.clip_y = b_clip; best.iou = b_iou;
        printf("  [Phase5] clip_y=%.3f IoU=%.4f%%\n", best.clip_y, best.iou * 100);
    }

    return best;
}

// =============================================================================
// Textured Rendering (Painter's Algorithm)
// =============================================================================

void render_textured_overlay(const Mesh& mesh, const cv::Mat& texture,
                             const cv::Mat& original, const cv::Mat& orig_mask,
                             const Params& params, const std::string& out_dir) {
    Timer t("Textured overlay");
    int H = original.rows, W = original.cols;

    Mat3 R = rot_x(INITIAL_RX + params.phi) * rot_y(-params.theta) * rot_z(params.roll);

    int nv = (int)mesh.vertices.size();
    std::vector<Vec3> tv(nv);
    std::vector<float> px(nv), py(nv);

    float f_proj = RENDER_SIZE / (2.0f * tanf(deg2rad(CAMERA_FOV / 2.0f)));

    // Transform vertices
    for (int i = 0; i < nv; i++) {
        tv[i] = R * mesh.vertices[i];
    }

    // Find valid faces (clip_y filter) and compute bbox
    struct FaceDepth { int idx; float depth; };
    std::vector<FaceDepth> valid_faces;
    float min_px = 1e9, max_px = -1e9, min_py = 1e9, max_py = -1e9;

    for (int fi = 0; fi < (int)mesh.faces.size(); fi++) {
        const Face& face = mesh.faces[fi];
        int a = face.v[0], b = face.v[1], c = face.v[2];
        if (a < 0 || a >= nv || b < 0 || b >= nv || c < 0 || c >= nv) continue;

        if (tv[a].y < params.clip_y || tv[b].y < params.clip_y || tv[c].y < params.clip_y)
            continue;

        float vz_a = std::max(0.1f, params.cam_z - tv[a].z);
        float vz_b = std::max(0.1f, params.cam_z - tv[b].z);
        float vz_c = std::max(0.1f, params.cam_z - tv[c].z);

        float px_a = (tv[a].x - params.cam_x) / vz_a * f_proj + RENDER_SIZE * 0.5f;
        float py_a = RENDER_SIZE * 0.5f - (tv[a].y - params.cam_y) / vz_a * f_proj;
        float px_b = (tv[b].x - params.cam_x) / vz_b * f_proj + RENDER_SIZE * 0.5f;
        float py_b = RENDER_SIZE * 0.5f - (tv[b].y - params.cam_y) / vz_b * f_proj;
        float px_c = (tv[c].x - params.cam_x) / vz_c * f_proj + RENDER_SIZE * 0.5f;
        float py_c = RENDER_SIZE * 0.5f - (tv[c].y - params.cam_y) / vz_c * f_proj;

        min_px = std::min({min_px, px_a, px_b, px_c});
        max_px = std::max({max_px, px_a, px_b, px_c});
        min_py = std::min({min_py, py_a, py_b, py_c});
        max_py = std::max({max_py, py_a, py_b, py_c});

        float avg_z = (tv[a].z + tv[b].z + tv[c].z) / 3.0f;
        valid_faces.push_back({fi, avg_z});
    }

    if (valid_faces.empty()) {
        printf("  No valid faces for textured render\n");
        return;
    }

    // Sort by depth (far to near = painter's algorithm)
    std::sort(valid_faces.begin(), valid_faces.end(),
              [](const FaceDepth& a, const FaceDepth& b) { return a.depth < b.depth; });

    // Target bbox from original mask
    BBox target_bb = bbox_from_mask(orig_mask);
    if (target_bb.x0 < 0) return;

    // Compute scale and offset
    float render_w = max_px - min_px;
    float render_h = max_py - min_py;
    float target_w = (float)(target_bb.x1 - target_bb.x0);
    float target_h = (float)(target_bb.y1 - target_bb.y0);

    float scale = std::min(target_w / render_w, target_h / render_h);
    float off_x = target_bb.x0 + target_w * 0.5f - (min_px + max_px) * 0.5f * scale + min_px * scale - min_px * scale;
    // Simpler: map render center to target center
    float rcx = (min_px + max_px) / 2.0f;
    float rcy = (min_py + max_py) / 2.0f;
    float tcx = (target_bb.x0 + target_bb.x1) / 2.0f;
    float tcy = (target_bb.y0 + target_bb.y1) / 2.0f;

    // Project and scale all vertices to image space
    std::vector<float> img_px(nv), img_py(nv);
    for (int i = 0; i < nv; i++) {
        float vz = std::max(0.1f, params.cam_z - tv[i].z);
        float rx = (tv[i].x - params.cam_x) / vz * f_proj + RENDER_SIZE * 0.5f;
        float ry = RENDER_SIZE * 0.5f - (tv[i].y - params.cam_y) / vz * f_proj;
        img_px[i] = (rx - rcx) * scale + tcx;
        img_py[i] = (ry - rcy) * scale + tcy;
    }

    // Render textured image
    cv::Mat rendered = cv::Mat::zeros(H, W, CV_8UC4); // BGRA (A=0 means transparent)
    int tex_h = texture.rows, tex_w = texture.cols;

    for (const auto& fd : valid_faces) {
        const Face& face = mesh.faces[fd.idx];
        int a = face.v[0], b = face.v[1], c = face.v[2];

        float x0f = img_px[a], y0f = img_py[a];
        float x1f = img_px[b], y1f = img_py[b];
        float x2f = img_px[c], y2f = img_py[c];

        // Bounding box of triangle
        int tri_x0 = std::max(0, (int)std::min({x0f, x1f, x2f}));
        int tri_y0 = std::max(0, (int)std::min({y0f, y1f, y2f}));
        int tri_x1 = std::min(W - 1, (int)std::max({x0f, x1f, x2f}) + 1);
        int tri_y1 = std::min(H - 1, (int)std::max({y0f, y1f, y2f}) + 1);

        // UV coordinates
        float u0 = 0, v0 = 0, u1 = 0, v1 = 0, u2 = 0, v2 = 0;
        if (!mesh.uvs.empty()) {
            int vt0 = face.vt[0], vt1 = face.vt[1], vt2 = face.vt[2];
            if (vt0 >= 0 && vt0 < (int)mesh.uvs.size()) { u0 = mesh.uvs[vt0].u; v0 = mesh.uvs[vt0].v; }
            if (vt1 >= 0 && vt1 < (int)mesh.uvs.size()) { u1 = mesh.uvs[vt1].u; v1 = mesh.uvs[vt1].v; }
            if (vt2 >= 0 && vt2 < (int)mesh.uvs.size()) { u2 = mesh.uvs[vt2].u; v2 = mesh.uvs[vt2].v; }
        }

        // Barycentric rasterization
        float denom = (y1f - y2f) * (x0f - x2f) + (x2f - x1f) * (y0f - y2f);
        if (fabsf(denom) < 0.001f) continue;
        float inv_denom = 1.0f / denom;

        for (int py = tri_y0; py <= tri_y1; py++) {
            for (int px = tri_x0; px <= tri_x1; px++) {
                float w0 = ((y1f - y2f) * (px - x2f) + (x2f - x1f) * (py - y2f)) * inv_denom;
                float w1 = ((y2f - y0f) * (px - x2f) + (x0f - x2f) * (py - y2f)) * inv_denom;
                float w2 = 1.0f - w0 - w1;

                if (w0 >= -0.001f && w1 >= -0.001f && w2 >= -0.001f) {
                    float u = w0 * u0 + w1 * u1 + w2 * u2;
                    float v = w0 * v0 + w1 * v1 + w2 * v2;

                    // Sample texture
                    int tx = ((int)(u * tex_w)) % tex_w;
                    int ty = ((int)((1.0f - v) * tex_h)) % tex_h;
                    if (tx < 0) tx += tex_w;
                    if (ty < 0) ty += tex_h;

                    const uchar* trow = texture.ptr<uchar>(ty);
                    uchar* orow = rendered.ptr<uchar>(py);
                    orow[px * 4 + 0] = trow[tx * 3 + 0]; // B
                    orow[px * 4 + 1] = trow[tx * 3 + 1]; // G
                    orow[px * 4 + 2] = trow[tx * 3 + 2]; // R
                    orow[px * 4 + 3] = 255; // A
                }
            }
        }
    }

    // Clip with original mask
    for (int y = 0; y < H; y++) {
        const uchar* m = orig_mask.ptr<uchar>(y);
        uchar* r = rendered.ptr<uchar>(y);
        for (int x = 0; x < W; x++) {
            if (m[x] == 0) {
                r[x * 4 + 3] = 0; // transparent
            }
        }
    }

    // Save rendered
    cv::Mat rendered_bgr;
    cv::cvtColor(rendered, rendered_bgr, cv::COLOR_BGRA2BGR);
    // Where alpha=0, set to black
    for (int y = 0; y < H; y++) {
        const uchar* a = rendered.ptr<uchar>(y);
        uchar* b = rendered_bgr.ptr<uchar>(y);
        for (int x = 0; x < W; x++) {
            if (a[x * 4 + 3] == 0) {
                b[x * 3] = b[x * 3 + 1] = b[x * 3 + 2] = 0;
            }
        }
    }
    cv::imwrite(out_dir + "/rendered_textured.png", rendered_bgr);

    // Overlays
    cv::Mat overlay50 = original.clone();
    cv::Mat overlay30 = original.clone();
    for (int y = 0; y < H; y++) {
        const uchar* rr = rendered.ptr<uchar>(y);
        uchar* o50 = overlay50.ptr<uchar>(y);
        uchar* o30 = overlay30.ptr<uchar>(y);
        for (int x = 0; x < W; x++) {
            if (rr[x * 4 + 3] > 0) {
                for (int c = 0; c < 3; c++) {
                    o50[x * 3 + c] = (uchar)(o50[x * 3 + c] * 0.5f + rr[x * 4 + c] * 0.5f);
                    o30[x * 3 + c] = (uchar)(o30[x * 3 + c] * 0.7f + rr[x * 4 + c] * 0.3f);
                }
            }
        }
    }
    cv::imwrite(out_dir + "/overlay_50.png", overlay50);
    cv::imwrite(out_dir + "/overlay_30.png", overlay30);

    // Contour overlay
    cv::Mat render_mask;
    cv::cvtColor(rendered_bgr, render_mask, cv::COLOR_BGR2GRAY);
    cv::threshold(render_mask, render_mask, 1, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(render_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat contour_overlay = original.clone();
    cv::drawContours(contour_overlay, contours, -1, cv::Scalar(0, 255, 0), 2);
    cv::imwrite(out_dir + "/overlay_contour.png", contour_overlay);

    printf("  Textured overlay saved to %s\n", out_dir.c_str());
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("============================================================\n");
    printf("Pose Match Pipeline (C++ + OpenMP)\n");
    printf("============================================================\n");

    Timer total_timer("Total");

    fs::create_directories(OUT_DIR);

    // 1. Load images
    printf("\n[1/6] Loading images...\n");
    cv::Mat original = imread_unicode(INPUT_IMAGE);
    cv::Mat depth_bgr = imread_unicode(DEPTH_IMAGE);
    if (original.empty() || depth_bgr.empty()) {
        printf("ERROR: Cannot load images\n");
        return 1;
    }
    int H = original.rows, W = original.cols;
    printf("  Image: %dx%d\n", W, H);

    // 2. Create mask
    printf("\n[2/6] Creating mask...\n");
    cv::Mat gray, mask_raw, mask;
    cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask_raw, 10, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask_raw, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // Largest connected component
    cv::Mat labels, stats, centroids;
    int num = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    if (num > 1) {
        int largest = 1;
        int max_area = 0;
        for (int i = 1; i < num; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > max_area) { max_area = area; largest = i; }
        }
        mask = (labels == largest);
        mask.convertTo(mask, CV_8UC1, 255);
    }

    cv::Mat mask_filled = fill_holes(mask);
    printf("  Mask created\n");

    // Target normalization
    BBox bb_target = bbox_from_mask(mask_filled);
    BBox bb_target_e = expand_bbox(bb_target, W, H, BBOX_MARGIN_RATIO);
    cv::Mat target_norm = crop_and_resize(mask_filled, bb_target_e, NORM_SIZE);

    // 3. Depth-based angle estimation
    printf("\n[3/6] Estimating angles from depth...\n");
    cv::Mat depth_gray_img;
    cv::cvtColor(depth_bgr, depth_gray_img, cv::COLOR_BGR2GRAY);
    cv::Mat depth_mask;
    cv::threshold(depth_gray_img, depth_mask, 5, 255, cv::THRESH_BINARY);
    cv::Mat dk = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(depth_mask, depth_mask, cv::MORPH_CLOSE, dk);
    cv::morphologyEx(depth_mask, depth_mask, cv::MORPH_OPEN, dk);
    // Largest CC for depth mask
    cv::Mat dlabels, dstats, dcent;
    int dnum = cv::connectedComponentsWithStats(depth_mask, dlabels, dstats, dcent);
    if (dnum > 1) {
        int dl = 1; int dma = 0;
        for (int i = 1; i < dnum; i++) {
            int a = dstats.at<int>(i, cv::CC_STAT_AREA);
            if (a > dma) { dma = a; dl = i; }
        }
        depth_mask = (dlabels == dl);
        depth_mask.convertTo(depth_mask, CV_8UC1, 255);
    }

    AngleEstimate est = estimate_angles_from_depth(depth_bgr, depth_mask);
    printf("  Estimated: theta=%.1f, phi=%.1f, roll=%.1f\n", est.theta, est.phi, est.roll);

    // 4. Load 3D model
    printf("\n[4/6] Loading 3D model...\n");
    Mesh mesh = load_obj(MODEL_PATH);
    normalize_vertices(mesh);
    printf("  Low-poly: %zu vertices, %zu faces\n", mesh.vertices.size(), mesh.faces.size());

    // 5. Optimization
    printf("\n[5/6] Optimization...\n");
    Timer opt_timer("Optimization");
    Params best = optimize(mesh, target_norm, est.theta, est.phi, est.roll);
    printf("\n  === BEST RESULT ===\n");
    printf("  theta=%.4f, phi=%.4f, roll=%.4f\n", best.theta, best.phi, best.roll);
    printf("  cam=(%.4f, %.4f, %.4f), clip_y=%.4f\n", best.cam_x, best.cam_y, best.cam_z, best.clip_y);
    printf("  IoU=%.4f%%\n", best.iou * 100);

    // 6. Textured overlay (hi-poly model)
    printf("\n[6/6] Textured overlay...\n");
    Mesh mesh_hi = load_obj(MODEL_HI_PATH);
    normalize_vertices(mesh_hi);
    printf("  Hi-poly: %zu vertices, %zu faces\n", mesh_hi.vertices.size(), mesh_hi.faces.size());

    cv::Mat texture = imread_unicode(TEXTURE_PATH);
    if (texture.empty()) {
        printf("  WARNING: Cannot load texture, skipping overlay\n");
    } else {
        render_textured_overlay(mesh_hi, texture, original, mask_filled, best, OUT_DIR);
    }

    // Save result
    std::ofstream result(OUT_DIR + "/result.txt");
    result << "=== Pose Match C++ Result ===\n\n";
    result << "[Best Parameters]\n";
    result << "  theta: " << best.theta << "\n";
    result << "  phi: " << best.phi << "\n";
    result << "  roll: " << best.roll << "\n";
    result << "  cam_x: " << best.cam_x << "\n";
    result << "  cam_y: " << best.cam_y << "\n";
    result << "  cam_z: " << best.cam_z << "\n";
    result << "  clip_y: " << best.clip_y << "\n\n";
    result << "[Result]\n";
    result << "  IoU: " << (best.iou * 100) << "%\n";
    result.close();

    printf("\n============================================================\n");
    printf("Output: %s\n", OUT_DIR.c_str());
    printf("============================================================\n");

    return 0;
}

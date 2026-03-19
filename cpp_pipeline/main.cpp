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
std::string INPUT_IMAGE = BASE_DIR + "/Image0.png";
const std::string DEPTH_IMAGE = BASE_DIR + "/Image0_depth.png";
const std::string MODEL_PATH = BASE_DIR + "/models_rabit_obj/rabit_low.obj";
const std::string MODEL_HI_PATH = BASE_DIR + "/models_rabit_obj/rabit.obj";
const std::string TEXTURE_PATH = BASE_DIR + "/models_rabit_obj/rabit01.jpg";
const std::string DEFAULT_OUT_DIR = BASE_DIR + "/rotation_results_cpp";

const float CAMERA_FOV = 45.0f;
const float INITIAL_RX = -90.0f;
const int RENDER_SIZE = 512;
const int DEFAULT_NORM_SIZE = 256;
const float DEFAULT_BBOX_MARGIN_RATIO = 0.12f;

struct TuneConfig {
    float edge_tiebreaker_scale = 0.01f;
    float sym_contour_weight = 0.5f;
    float contour_match_threshold_px = 3.0f;
    float phase6_iou_decrease_tolerance = 0.0001f;
    float phase6_r2_rot_step = 0.01f;
    float phase6_r2_cam_step = 0.001f;
    int phase6_enable_round4 = 0;
    float phase6_r3_rot_step = 0.005f;
    float phase6_r3_rot_range = 0.02f;
    float phase6_r3_cam_step = 0.0005f;

    float boundary_weight_factor = 0.2f;
    float boundary_weight_distance_px = 10.0f;

    float phase7_match_threshold = 0.3f;
    float phase7_confidence_scale = 30.0f;
    float phase7_base_lr = 0.3f;
    float phase7_rotation_scale_factor = 6.0f;

    float phase4_range_deg = 1.0f;
    float phase4_step_deg = 0.2f;

    int norm_size = DEFAULT_NORM_SIZE;
    float bbox_margin_ratio = DEFAULT_BBOX_MARGIN_RATIO;
    float mask_threshold = 10.0f;
    float camera_fov = 45.0f;
    float clip_slope_x = 0.0f;  // tilted clip plane: clip if y < clip_y + slope*x
};

TuneConfig g_tune;

struct RuntimeConfig {
    std::string output_dir = DEFAULT_OUT_DIR;
    bool enable_phase7 = true;
    bool save_phase7_images = true;
    bool save_final_overlay = true;
    std::vector<cv::Rect> ignore_rects;  // occlusion rects in original image coords
    int ignore_above_y = -1;  // ignore all pixels above this y (image coords), -1=disabled
    int ignore_below_y = -1;  // ignore all pixels below this y (image coords), -1=disabled
    int ignore_left_x = -1;   // ignore all pixels left of this x, -1=disabled
    int ignore_right_x = -1;  // ignore all pixels right of this x, -1=disabled
};

RuntimeConfig g_runtime;

// Global ignore mask in normalized target space (same size as target_norm)
// Pixels > 0 are excluded from IoU calculation
cv::Mat g_ignore_norm;

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

    float f = size / (2.0f * tanf(deg2rad(g_tune.camera_fov / 2.0f)));

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
            float sl = g_tune.clip_slope_x;
            if (tv[a].y < clip_y + sl*tv[a].x || tv[b].y < clip_y + sl*tv[b].x || tv[c].y < clip_y + sl*tv[c].x)
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
    bool has_ignore = !g_ignore_norm.empty();
    for (int y = 0; y < a.rows; y++) {
        const uchar* ra = a.ptr<uchar>(y);
        const uchar* rb = b.ptr<uchar>(y);
        const uchar* ri = has_ignore ? g_ignore_norm.ptr<uchar>(y) : nullptr;
        for (int x = 0; x < a.cols; x++) {
            if (ri && ri[x] > 0) continue;  // skip ignored pixels
            bool va = ra[x] > 0, vb = rb[x] > 0;
            if (va && vb) inter++;
            if (va || vb) uni++;
        }
    }
    return uni > 0 ? (float)inter / uni : 0.0f;
}

// Pose quality analysis: measures error concentration via multiple metrics
// 1. Gini coefficient: angular distribution of errors (higher=more concentrated)
// 2. Mean boundary distance: avg distance between target & render contours (lower=better)
// 3. Boundary error ratio: fraction of contour with significant displacement (lower=better)
// Good pose: errors concentrated at known defects, low mean distance, few error hotspots
// Bad pose: errors spread everywhere, higher mean distance, many error hotspots
struct PoseQuality {
    float iou;
    float gini;               // Gini coefficient (0-1, higher=more concentrated=better)
    float mean_boundary_dist;  // Mean contour-to-contour distance (lower=better)
    float p90_boundary_dist;   // 90th percentile boundary distance (lower=better)
    float error_ratio;         // Fraction of contour with displacement >1px (lower=better)
    int fn_count, fp_count;
    int active_sectors;
    int total_sectors;
    float combined_score;      // IoU weighted by concentration metrics
};

PoseQuality compute_pose_quality(const cv::Mat& target, const cv::Mat& rendered) {
    PoseQuality q = {};
    int ns = target.rows;
    q.total_sectors = 12;
    const int N_SEC = 12;
    bool has_ignore = !g_ignore_norm.empty();

    // Collect FN/FP positions
    float ucx = 0, ucy = 0;
    int ucnt = 0, inter = 0, uni = 0;
    std::vector<cv::Point> fn_pts, fp_pts;

    for (int y = 0; y < ns; y++) {
        const uchar* t = target.ptr<uchar>(y);
        const uchar* r = rendered.ptr<uchar>(y);
        const uchar* ig = has_ignore ? g_ignore_norm.ptr<uchar>(y) : nullptr;
        for (int x = 0; x < ns; x++) {
            if (ig && ig[x] > 0) continue;
            bool vt = t[x] > 0, vr = r[x] > 0;
            if (vt || vr) { ucx += x; ucy += y; ucnt++; uni++; }
            if (vt && vr) inter++;
            if (vt && !vr) { fn_pts.push_back({x, y}); q.fn_count++; }
            if (!vt && vr) { fp_pts.push_back({x, y}); q.fp_count++; }
        }
    }

    q.iou = uni > 0 ? (float)inter / uni : 0.0f;
    if (ucnt == 0 || (fn_pts.empty() && fp_pts.empty())) {
        q.gini = 1.0f; q.mean_boundary_dist = 0; q.p90_boundary_dist = 0;
        q.error_ratio = 0; q.combined_score = q.iou;
        return q;
    }
    ucx /= ucnt; ucy /= ucnt;

    // === Metric 1: Gini coefficient on angular sectors ===
    int sector_err[N_SEC] = {};
    for (auto& p : fn_pts) {
        int s = (int)((atan2f(p.y - ucy, p.x - ucx) + 3.14159f) / (2 * 3.14159f) * N_SEC) % N_SEC;
        sector_err[s]++;
    }
    for (auto& p : fp_pts) {
        int s = (int)((atan2f(p.y - ucy, p.x - ucx) + 3.14159f) / (2 * 3.14159f) * N_SEC) % N_SEC;
        sector_err[s]++;
    }

    std::vector<int> sorted_err(sector_err, sector_err + N_SEC);
    std::sort(sorted_err.begin(), sorted_err.end());
    float gini_num = 0, gini_den = 0;
    for (int i = 0; i < N_SEC; i++) {
        gini_num += (2.0f * (i + 1) - N_SEC - 1) * sorted_err[i];
        gini_den += sorted_err[i];
    }
    q.gini = gini_den > 0 ? gini_num / (N_SEC * gini_den) : 0;

    int total_err = q.fn_count + q.fp_count;
    float threshold = total_err * 0.05f;
    q.active_sectors = 0;
    for (int i = 0; i < N_SEC; i++) {
        if (sector_err[i] > threshold) q.active_sectors++;
    }

    // === Metric 2&3: Boundary distance distribution ===
    // Extract contours
    std::vector<std::vector<cv::Point>> tc, rc;
    {
        cv::Mat t_c = target.clone(), r_c = rendered.clone();
        if (has_ignore) {
            for (int y = 0; y < ns; y++) {
                const uchar* ig = g_ignore_norm.ptr<uchar>(y);
                uchar* tt = t_c.ptr<uchar>(y);
                uchar* rr = r_c.ptr<uchar>(y);
                for (int x = 0; x < ns; x++) {
                    if (ig[x] > 0) { tt[x] = 0; rr[x] = 0; }
                }
            }
        }
        cv::findContours(t_c, tc, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::findContours(r_c, rc, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    }

    q.mean_boundary_dist = 0;
    q.p90_boundary_dist = 0;
    q.error_ratio = 0;

    if (!tc.empty() && !rc.empty()) {
        auto& tgt_c = *std::max_element(tc.begin(), tc.end(),
            [](const auto& a, const auto& b) { return a.size() < b.size(); });
        auto& rnd_c = *std::max_element(rc.begin(), rc.end(),
            [](const auto& a, const auto& b) { return a.size() < b.size(); });

        // For each target contour point, find distance to nearest render contour point
        std::vector<float> distances;
        distances.reserve(tgt_c.size());

        for (const auto& tp : tgt_c) {
            float min_dsq = 1e9f;
            for (const auto& rp : rnd_c) {
                float dsq = (float)(tp.x - rp.x) * (tp.x - rp.x) +
                            (float)(tp.y - rp.y) * (tp.y - rp.y);
                if (dsq < min_dsq) min_dsq = dsq;
            }
            distances.push_back(std::sqrt(min_dsq));
        }

        // Also reverse direction: render→target
        for (const auto& rp : rnd_c) {
            float min_dsq = 1e9f;
            for (const auto& tp : tgt_c) {
                float dsq = (float)(rp.x - tp.x) * (rp.x - tp.x) +
                            (float)(rp.y - tp.y) * (rp.y - tp.y);
                if (dsq < min_dsq) min_dsq = dsq;
            }
            distances.push_back(std::sqrt(min_dsq));
        }

        if (!distances.empty()) {
            std::sort(distances.begin(), distances.end());

            float sum = 0;
            for (float d : distances) sum += d;
            q.mean_boundary_dist = sum / distances.size();
            q.p90_boundary_dist = distances[(int)(distances.size() * 0.9)];

            // Error ratio: fraction of points with distance > 1.5px
            int over_thresh = 0;
            for (float d : distances) {
                if (d > 1.5f) over_thresh++;
            }
            q.error_ratio = (float)over_thresh / distances.size();
        }
    }

    // Combined score: IoU boosted by concentration, penalized by boundary distance
    // Higher is better
    float concentration_bonus = 0.01f * q.gini;
    float distance_penalty = 0.005f * q.mean_boundary_dist;
    q.combined_score = q.iou * (1.0f + concentration_bonus - distance_penalty);

    return q;
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
    BBox bbe = expand_bbox(bb, render_size, render_size, g_tune.bbox_margin_ratio);
    int norm_size = std::max(64, g_tune.norm_size);
    cv::Mat sil_norm = crop_and_resize(sil, bbe, norm_size);
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
// Forward declarations
// =============================================================================

struct Params;
cv::Mat render_textured_fast(const Mesh& mesh, const cv::Mat& texture,
                             const Params& params, int size,
                             const cv::Mat& orig_img, const cv::Mat& orig_mask);
cv::Mat render_silhouette_fullres(const Mesh& mesh, const Params& params,
                                   const cv::Mat& orig_mask);
float compute_edge_score(const cv::Mat& original, const cv::Mat& orig_mask,
                          const cv::Mat& render_mask);
cv::Mat preprocess_for_matching(const cv::Mat& img);

struct MatchResult {
    int num_inliers;
    float mean_dx, mean_dy;       // average offset of inliers
    float top_dy, bottom_dy;      // offset in top/bottom halves
    float left_dx, right_dx;      // offset in left/right halves
    std::vector<cv::DMatch> inlier_matches;
    std::vector<cv::KeyPoint> kp1, kp2;
};
MatchResult compute_feature_matches(const cv::Mat& orig, const cv::Mat& rendered,
                                     const cv::Mat& orig_mask, const cv::Mat& rend_mask);

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
                float init_theta, float init_phi, float init_roll,
                float init_cam_x = -0.45f, float init_cam_y = -0.45f,
                float init_cam_z = 2.72f, float init_clip_y = -0.39f,
                const cv::Mat& texture = cv::Mat(),
                const cv::Mat& original = cv::Mat(),
                const cv::Mat& orig_mask = cv::Mat(),
                const Mesh* mesh_hi = nullptr) {
    Params best;
    best.theta = init_theta;
    best.phi = init_phi;
    best.roll = init_roll;
    best.cam_x = init_cam_x;
    best.cam_y = init_cam_y;
    best.cam_z = init_cam_z;
    best.clip_y = init_clip_y;

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
        float range = std::max(0.05f, g_tune.phase4_range_deg);
        float step = std::max(0.01f, g_tune.phase4_step_deg);
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

        // Clip slope sweep (tilted clip plane)
        float b_slope = g_tune.clip_slope_x;
        for (float sl = -0.3f; sl <= 0.301f; sl += 0.02f) {
            g_tune.clip_slope_x = sl;
            float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                   best.cam_x, best.cam_y, best.cam_z, best.clip_y);
            if (iou > best.iou) { best.iou = iou; b_slope = sl; }
        }
        // Fine sweep around best
        float coarse_slope = b_slope;
        for (float sl = coarse_slope - 0.02f; sl <= coarse_slope + 0.021f; sl += 0.002f) {
            g_tune.clip_slope_x = sl;
            float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                   best.cam_x, best.cam_y, best.cam_z, best.clip_y);
            if (iou > best.iou) { best.iou = iou; b_slope = sl; }
        }
        g_tune.clip_slope_x = b_slope;

        printf("  [Phase5] clip_y=%.3f clip_slope_x=%.4f IoU=%.4f%%\n", best.clip_y, g_tune.clip_slope_x, best.iou * 100);
    }

    // Phase 5b: FOV + cam_z joint refinement
    // FOV affects perspective distortion, cam_z affects scale. They interact.
    // Sweep after rotation/position are fine-tuned.
    {
        Timer t("Phase5b FOV");
        float best_fov = g_tune.camera_fov;
        float best_iou_fov = best.iou;
        float best_cz_fov = best.cam_z;
        // Coarse: ±5° in 1° steps, re-optimize cam_z for each FOV
        float fov_center = g_tune.camera_fov;
        for (float fov = fov_center - 5.0f; fov <= fov_center + 5.01f; fov += 1.0f) {
            if (fov < 20.0f || fov > 90.0f) continue;
            g_tune.camera_fov = fov;
            // Re-optimize cam_z for this FOV
            float bcz = best.cam_z;
            float biou = 0.0f;
            for (float dz = -0.3f; dz <= 0.301f; dz += 0.01f) {
                float cz = best.cam_z + dz;
                float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                       best.cam_x, best.cam_y, cz, best.clip_y);
                if (iou > biou) { biou = iou; bcz = cz; }
            }
            if (biou > best_iou_fov) {
                best_iou_fov = biou; best_fov = fov; best_cz_fov = bcz;
            }
        }
        // Fine: ±1° in 0.25° steps around best coarse
        float coarse_fov = best_fov;
        for (float fov = coarse_fov - 1.0f; fov <= coarse_fov + 1.01f; fov += 0.25f) {
            if (fov < 20.0f || fov > 90.0f) continue;
            g_tune.camera_fov = fov;
            float bcz = best_cz_fov;
            float biou = 0.0f;
            for (float dz = -0.05f; dz <= 0.051f; dz += 0.005f) {
                float cz = best_cz_fov + dz;
                float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                       best.cam_x, best.cam_y, cz, best.clip_y);
                if (iou > biou) { biou = iou; bcz = cz; }
            }
            if (biou > best_iou_fov) {
                best_iou_fov = biou; best_fov = fov; best_cz_fov = bcz;
            }
        }
        g_tune.camera_fov = best_fov;
        best.cam_z = best_cz_fov;
        best.iou = best_iou_fov;
        printf("  [Phase5b] FOV=%.2f cam_z=%.4f IoU=%.4f%%\n", g_tune.camera_fov, best.cam_z, best.iou * 100);
    }

    // Phase 6: Edge-guided refinement
    // Uses Chamfer distance between rendered silhouette edges and original image edges
    // This captures internal contour alignment that IoU alone misses
    if (!original.empty() && !orig_mask.empty()) {
        Timer t("Phase6 Edge");
        float base_iou = best.iou;

        // Step 1: Compute baseline edge score
        cv::Mat base_sil = render_silhouette_fullres(mesh, best, orig_mask);
        float base_edge = compute_edge_score(original, orig_mask, base_sil);
        printf("  [Phase6] Base edge score: %.2f (chamfer dist = %.2f px)\n", base_edge, -base_edge);

        // Step 2: Coordinate descent with combined IoU + edge score
        // Edge score is cheap (no texture render), so we can evaluate more candidates
        struct Candidate { Params p; float iou; float edge; float combined; };

        Params best6 = best;
        // IoU primary + edge as tiebreaker
        float best_combined = base_iou + base_edge * g_tune.edge_tiebreaker_scale;
        best6.iou = base_iou;

        // 3 rounds (default): coarse -> medium -> fine
        // Optional 4th round can be enabled from tune config for final micro refinement.
        int total_rounds = (g_tune.phase6_enable_round4 > 0) ? 4 : 3;
        for (int round = 0; round < total_rounds; round++) {
            float rot_step, rot_range, cam_step;
            if (round == 0) { rot_step = 0.05f; rot_range = 0.3f; cam_step = 0.005f; }
            else if (round == 1) { rot_step = 0.02f; rot_range = 0.1f; cam_step = 0.002f; }
            else if (round == 2) {
                rot_step = std::max(0.002f, g_tune.phase6_r2_rot_step);
                rot_range = 0.05f;
                cam_step = std::max(0.0001f, g_tune.phase6_r2_cam_step);
            } else {
                rot_step = std::max(0.001f, g_tune.phase6_r3_rot_step);
                rot_range = std::max(rot_step, g_tune.phase6_r3_rot_range);
                cam_step = std::max(0.0001f, g_tune.phase6_r3_cam_step);
            }

            // Helper lambda: evaluate candidate
            // IoU must not decrease; edge score used to break ties
            auto eval = [&](Params trial) -> float {
                float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                       trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                // Require IoU >= baseline (no decrease allowed)
                float eval_tol = std::max(0.0f, g_tune.phase6_iou_decrease_tolerance * 0.5f);
                if (iou < best6.iou - eval_tol) return -1e6f;
                cv::Mat sil = render_silhouette_fullres(mesh, trial, orig_mask);
                float edge = compute_edge_score(original, orig_mask, sil);
                // IoU improvement is primary, edge is secondary tiebreaker
                return iou + edge * g_tune.edge_tiebreaker_scale;
            };

            // Theta
            float saved_th = best6.theta;
            for (float d = -rot_range; d <= rot_range + 0.001f; d += rot_step) {
                Params trial = best6; trial.theta = saved_th + d;
                float c = eval(trial);
                if (c > best_combined) {
                    best_combined = c;
                    best6.theta = trial.theta;
                    best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                           best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                }
            }
            // Phi
            float saved_ph = best6.phi;
            for (float d = -rot_range; d <= rot_range + 0.001f; d += rot_step) {
                Params trial = best6; trial.phi = saved_ph + d;
                float c = eval(trial);
                if (c > best_combined) {
                    best_combined = c;
                    best6.phi = trial.phi;
                    best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                           best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                }
            }
            // Roll
            float saved_ro = best6.roll;
            for (float d = -rot_range; d <= rot_range + 0.001f; d += rot_step) {
                Params trial = best6; trial.roll = saved_ro + d;
                float c = eval(trial);
                if (c > best_combined) {
                    best_combined = c;
                    best6.roll = trial.roll;
                    best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                           best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                }
            }
            // Camera X
            float saved_cx = best6.cam_x;
            for (float d = -0.02f; d <= 0.021f; d += cam_step) {
                Params trial = best6; trial.cam_x = saved_cx + d;
                float c = eval(trial);
                if (c > best_combined) {
                    best_combined = c;
                    best6.cam_x = trial.cam_x;
                    best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                           best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                }
            }
            // Camera Y
            float saved_cy = best6.cam_y;
            for (float d = -0.02f; d <= 0.021f; d += cam_step) {
                Params trial = best6; trial.cam_y = saved_cy + d;
                float c = eval(trial);
                if (c > best_combined) {
                    best_combined = c;
                    best6.cam_y = trial.cam_y;
                    best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                           best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                }
            }
            // Camera Z
            float saved_cz = best6.cam_z;
            for (float d = -0.02f; d <= 0.021f; d += cam_step) {
                Params trial = best6; trial.cam_z = saved_cz + d;
                float c = eval(trial);
                if (c > best_combined) {
                    best_combined = c;
                    best6.cam_z = trial.cam_z;
                    best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                           best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                }
            }
            // Clip Y
            float saved_clip = best6.clip_y;
            for (float d = -0.05f; d <= 0.051f; d += 0.01f) {
                Params trial = best6; trial.clip_y = saved_clip + d;
                float c = eval(trial);
                if (c > best_combined) {
                    best_combined = c;
                    best6.clip_y = trial.clip_y;
                    best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                           best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                }
            }
            // Clip slope X (tilted clip plane)
            {
                float slope_step = (round <= 1) ? 0.01f : 0.005f;
                float slope_range = (round <= 1) ? 0.05f : 0.02f;
                float saved_slope = g_tune.clip_slope_x;
                for (float d = -slope_range; d <= slope_range + 0.001f; d += slope_step) {
                    g_tune.clip_slope_x = saved_slope + d;
                    Params trial = best6;
                    float c = eval(trial);
                    if (c > best_combined) {
                        best_combined = c;
                        saved_slope = g_tune.clip_slope_x;
                        best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                               best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                    }
                }
                g_tune.clip_slope_x = saved_slope;
            }
            // FOV (only in coarse rounds to avoid over-tuning)
            if (round <= 1) {
                float fov_step = (round == 0) ? 0.5f : 0.25f;
                float fov_range = (round == 0) ? 2.0f : 0.5f;
                float saved_fov = g_tune.camera_fov;
                for (float d = -fov_range; d <= fov_range + 0.001f; d += fov_step) {
                    float trial_fov = saved_fov + d;
                    if (trial_fov < 20.0f || trial_fov > 90.0f) continue;
                    g_tune.camera_fov = trial_fov;
                    Params trial = best6;
                    float c = eval(trial);
                    if (c > best_combined) {
                        best_combined = c;
                        saved_fov = trial_fov;
                        best6.iou = compute_iou(mesh, target_norm, best6.theta, best6.phi, best6.roll,
                                               best6.cam_x, best6.cam_y, best6.cam_z, best6.clip_y);
                    }
                }
                g_tune.camera_fov = saved_fov;
            }
        }

        // Final edge score
        cv::Mat final_sil = render_silhouette_fullres(mesh, best6, orig_mask);
        float final_edge = compute_edge_score(original, orig_mask, final_sil);

        printf("  [Phase6] theta=%.2f phi=%.2f roll=%.2f cam=(%.3f,%.3f) clip_y=%.3f\n",
               best6.theta, best6.phi, best6.roll, best6.cam_x, best6.cam_y, best6.clip_y);
        printf("  [Phase6] IoU=%.4f%% edge=%.2f (was IoU=%.4f%% edge=%.2f)\n",
               best6.iou * 100, final_edge, base_iou * 100, base_edge);

        if (best6.iou >= best.iou - g_tune.phase6_iou_decrease_tolerance) {
            best = best6;
            printf("  [Phase6] Accepted!\n");
        } else {
            printf("  [Phase6] Rejected (IoU decreased too much)\n");
        }
    } else {
        printf("  [Phase6] Skipped (missing original/mask)\n");
    }

    // Phase 6b: Post-Phase6 fine re-sweep of FOV + clip_slope + cam_z
    // Phase 6 may have shifted rotation/cam — re-optimize FOV and slope for the new pose
    {
        Timer t("Phase6b Resweep");
        float best_iou_6b = best.iou;
        // Fine FOV re-sweep (±1° in 0.1° steps)
        float best_fov_6b = g_tune.camera_fov;
        float fov_center_6b = g_tune.camera_fov;
        for (float fov = fov_center_6b - 1.0f; fov <= fov_center_6b + 1.01f; fov += 0.1f) {
            if (fov < 20.0f || fov > 90.0f) continue;
            g_tune.camera_fov = fov;
            float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                   best.cam_x, best.cam_y, best.cam_z, best.clip_y);
            if (iou > best_iou_6b) { best_iou_6b = iou; best_fov_6b = fov; }
        }
        g_tune.camera_fov = best_fov_6b;
        best.iou = best_iou_6b;

        // Fine cam_z re-sweep
        float best_cz_6b = best.cam_z;
        for (float dz = -0.02f; dz <= 0.021f; dz += 0.001f) {
            float cz = best.cam_z + dz;
            float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                   best.cam_x, best.cam_y, cz, best.clip_y);
            if (iou > best.iou) { best.iou = iou; best_cz_6b = cz; }
        }
        best.cam_z = best_cz_6b;

        // Fine clip_slope_x re-sweep
        float best_slope_6b = g_tune.clip_slope_x;
        float slope_center_6b = g_tune.clip_slope_x;
        for (float sl = slope_center_6b - 0.03f; sl <= slope_center_6b + 0.031f; sl += 0.002f) {
            g_tune.clip_slope_x = sl;
            float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                   best.cam_x, best.cam_y, best.cam_z, best.clip_y);
            if (iou > best.iou) { best.iou = iou; best_slope_6b = sl; }
        }
        g_tune.clip_slope_x = best_slope_6b;

        // Fine clip_y re-sweep
        float best_clip_6b = best.clip_y;
        for (float cl = best.clip_y - 0.02f; cl <= best.clip_y + 0.021f; cl += 0.001f) {
            float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                   best.cam_x, best.cam_y, best.cam_z, cl);
            if (iou > best.iou) { best.iou = iou; best_clip_6b = cl; }
        }
        best.clip_y = best_clip_6b;

        printf("  [Phase6b] FOV=%.2f cam_z=%.4f slope_x=%.4f clip_y=%.4f IoU=%.4f%%\n",
               g_tune.camera_fov, best.cam_z, g_tune.clip_slope_x, best.clip_y, best.iou * 100);
    }

    // Phase 7: Feature-guided refinement (SIFT + RANSAC on Sobel gradients)
    // Renders textured model, matches internal features, corrects camera/rotation
    if (g_runtime.enable_phase7 && !texture.empty() && !original.empty() && !orig_mask.empty() && mesh_hi != nullptr) {
        Timer t("Phase7 Feature");
        float base_iou = best.iou;

        // Compute pixel-to-camera conversion factor
        float f_proj = RENDER_SIZE / (2.0f * tanf(deg2rad(g_tune.camera_fov / 2.0f)));
        // Estimate render-to-image scale from bbox mapping
        Mat3 R7 = rot_x(INITIAL_RX + best.phi) * rot_y(-best.theta) * rot_z(best.roll);
        int nv7 = (int)mesh.vertices.size();
        float rmin_px = 1e9, rmax_px = -1e9, rmin_py = 1e9, rmax_py = -1e9;
        for (int i = 0; i < nv7; i++) {
            Vec3 tv7 = R7 * mesh.vertices[i];
            if (tv7.y < best.clip_y + g_tune.clip_slope_x * tv7.x) continue;
            float vz = std::max(0.1f, best.cam_z - tv7.z);
            float px7 = (tv7.x - best.cam_x) / vz * f_proj + RENDER_SIZE * 0.5f;
            float py7 = RENDER_SIZE * 0.5f - (tv7.y - best.cam_y) / vz * f_proj;
            rmin_px = std::min(rmin_px, px7); rmax_px = std::max(rmax_px, px7);
            rmin_py = std::min(rmin_py, py7); rmax_py = std::max(rmax_py, py7);
        }
        BBox tbb7 = bbox_from_mask(orig_mask);
        float rw7 = rmax_px - rmin_px, rh7 = rmax_py - rmin_py;
        float tw7 = (float)(tbb7.x1 - tbb7.x0), th7 = (float)(tbb7.y1 - tbb7.y0);
        float img_scale = std::min(tw7 / std::max(1.0f, rw7), th7 / std::max(1.0f, rh7));
        float avg_z = best.cam_z;
        // px_to_cam: convert image-space pixel offset to camera coordinate offset
        float px_to_cam = avg_z / (f_proj * img_scale);
        // px_per_deg: approximate pixels per degree of rotation
        float px_per_deg = f_proj * 0.01745f / avg_z * img_scale;

        printf("  [Phase7] scale=%.2f px_to_cam=%.4f px_per_deg=%.1f\n",
               img_scale, px_to_cam, px_per_deg);

        // Step A: Phase correlation for robust translation estimation
        {
            cv::Mat rendered = render_textured_fast(*mesh_hi, texture, best,
                                                     RENDER_SIZE, original, orig_mask);
            // Preprocess both to Sobel gradient
            cv::Mat orig_grad = preprocess_for_matching(original);
            cv::Mat rend_grad = preprocess_for_matching(rendered);

            // Crop to overlap ROI
            cv::Mat omask255;
            { double mx; cv::minMaxLoc(orig_mask, nullptr, &mx);
              if (mx <= 1.0) orig_mask.convertTo(omask255, CV_8UC1, 255);
              else omask255 = orig_mask; }
            cv::Mat rg; cv::cvtColor(rendered, rg, cv::COLOR_BGR2GRAY);
            cv::Mat rmask; cv::threshold(rg, rmask, 1, 255, cv::THRESH_BINARY);
            cv::Mat overlap; cv::bitwise_and(omask255, rmask, overlap);
            BBox pbb = bbox_from_mask(overlap);

            if (pbb.x0 >= 0) {
                int pm = 10;
                int px0 = std::max(0, pbb.x0 - pm), py0 = std::max(0, pbb.y0 - pm);
                int px1 = std::min(original.cols - 1, pbb.x1 + pm);
                int py1 = std::min(original.rows - 1, pbb.y1 + pm);
                cv::Rect proi(px0, py0, px1 - px0 + 1, py1 - py0 + 1);

                cv::Mat c_orig, c_rend;
                orig_grad(proi).convertTo(c_orig, CV_64F);
                rend_grad(proi).convertTo(c_rend, CV_64F);

                cv::Mat hann;
                cv::createHanningWindow(hann, c_orig.size(), CV_64F);
                double response;
                cv::Point2d shift = cv::phaseCorrelate(c_orig, c_rend, hann, &response);

                printf("  [Phase7] PhaseCorr: shift=(%.1f, %.1f) response=%.3f\n",
                       shift.x, shift.y, response);

                // Apply if response is significant and shift is small.
                // Try multiple step sizes and both signs, then keep only the best IoU candidate.
                if (response > 0.05 && fabsf((float)shift.x) < 30 && fabsf((float)shift.y) < 30) {
                    const float lrs[] = {0.70f, 0.35f, 0.15f};
                    const float signs[] = {1.0f, -1.0f};
                    float old_iou = best.iou;
                    float best_candidate_iou = old_iou;
                    Params best_candidate = best;
                    bool found_better = false;

                    for (float sgn : signs) {
                        for (float lr : lrs) {
                            Params trial = best;
                            float sx = (float)shift.x * px_to_cam * lr * sgn;
                            float sy = (float)shift.y * px_to_cam * lr * sgn;
                            trial.cam_x -= sx;
                            trial.cam_y += sy;

                            float cand_iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                                         trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                            printf("  [Phase7] PhaseCorr cand sign=%.0f lr=%.2f IoU=%.4f%%\n",
                                   sgn, lr, cand_iou * 100);
                            if (cand_iou > best_candidate_iou + 1e-7f) {
                                best_candidate_iou = cand_iou;
                                best_candidate = trial;
                                found_better = true;
                            }
                        }
                    }

                    if (found_better) {
                        best_candidate.iou = best_candidate_iou;
                        best = best_candidate;
                        printf("  [Phase7] PhaseCorr accepted: %.4f%% -> %.4f%%\n",
                               old_iou * 100, best_candidate_iou * 100);
                    } else {
                        printf("  [Phase7] PhaseCorr no improving candidate (IoU stays %.4f%%)\n",
                               old_iou * 100);
                    }
                }
            }
        }

        // Step B: SIFT+RANSAC iterative refinement
        for (int iter = 0; iter < 3; iter++) {
            cv::Mat rendered = render_textured_fast(*mesh_hi, texture, best,
                                                     RENDER_SIZE, original, orig_mask);

            MatchResult match = compute_feature_matches(original, rendered,
                                                         orig_mask, cv::Mat());

            printf("  [Phase7 iter%d] inliers=%d offset=(%.1f, %.1f)px\n",
                   iter, match.num_inliers, match.mean_dx, match.mean_dy);
            printf("  [Phase7 iter%d] top_dy=%.1f bot_dy=%.1f left_dx=%.1f right_dx=%.1f\n",
                   iter, match.top_dy, match.bottom_dy, match.left_dx, match.right_dx);

            if (match.num_inliers < 5) {
                printf("  [Phase7] Too few SIFT inliers, stopping\n");
                break;
            }

            float offset_mag = sqrtf(match.mean_dx * match.mean_dx +
                                     match.mean_dy * match.mean_dy);
            if (offset_mag < 2.0f) {
                printf("  [Phase7] Offsets small (%.1f px), converged\n", offset_mag);
                break;
            }

            // Scale learning rate by confidence (more inliers = more confident)
            float conf_scale = std::max(1.0f, g_tune.phase7_confidence_scale);
            float confidence = std::min(1.0f, match.num_inliers / conf_scale);
            float lr_base = g_tune.phase7_base_lr * confidence;

            // Rotation from spatial distribution
            float dy_diff = match.top_dy - match.bottom_dy;
            float dx_diff = match.left_dx - match.right_dx;
            float rot_scale = std::max(1.0f, g_tune.phase7_rotation_scale_factor);

            // Backtracking search: try multiple scales and choose best IoU.
            const float step_scales[] = {1.0f, 0.5f, 0.25f, -0.25f, -0.5f};
            float old_iou = best.iou;
            float best_candidate_iou = old_iou;
            Params best_candidate = best;
            float accepted_scale = 0.0f;

            for (float s : step_scales) {
                float lr = lr_base * s;
                Params trial = best;
                trial.cam_x -= match.mean_dx * px_to_cam * lr;
                trial.cam_y += match.mean_dy * px_to_cam * lr;

                if (fabsf(dy_diff) > 3.0f && px_per_deg > 0.1f) {
                    trial.phi += dy_diff / (px_per_deg * rot_scale) * lr;
                }
                if (fabsf(dx_diff) > 3.0f && px_per_deg > 0.1f) {
                    trial.theta -= dx_diff / (px_per_deg * rot_scale) * lr;
                }

                if (fabsf(match.bottom_dy) > 5.0f) {
                    trial.clip_y -= match.bottom_dy * px_to_cam * lr * 0.3f;
                    trial.clip_y = std::max(-0.6f, std::min(-0.1f, trial.clip_y));
                }

                float cand_iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                             trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);

                printf("  [Phase7 iter%d] cand scale=%+.2f IoU=%.4f%% (cam=%.3f,%.3f clip_y=%.3f)\n",
                       iter, s, cand_iou * 100, trial.cam_x, trial.cam_y, trial.clip_y);

                if (cand_iou > best_candidate_iou + 1e-7f) {
                    best_candidate_iou = cand_iou;
                    best_candidate = trial;
                    accepted_scale = s;
                }
            }

            if (best_candidate_iou > old_iou + 1e-7f) {
                best_candidate.iou = best_candidate_iou;
                best = best_candidate;
                printf("  [Phase7 iter%d] Accepted scale=%+.2f IoU: %.4f%% -> %.4f%%\n",
                       iter, accepted_scale, old_iou * 100, best_candidate_iou * 100);
            } else {
                printf("  [Phase7 iter%d] No improving candidate, stop\n", iter);
                break;
            }
        }

        // Final clip_y fine-tune (foot correction)
        {
            float b_clip = best.clip_y;
            float b_iou = best.iou;
            for (float cy = b_clip - 0.05f; cy <= b_clip + 0.05f; cy += 0.005f) {
                float iou = compute_iou(mesh, target_norm, best.theta, best.phi, best.roll,
                                       best.cam_x, best.cam_y, best.cam_z, cy);
                if (iou > b_iou) { b_iou = iou; b_clip = cy; }
            }
            if (b_iou > best.iou) {
                best.clip_y = b_clip;
                best.iou = b_iou;
                printf("  [Phase7] clip_y fine-tuned: %.3f IoU=%.4f%%\n", best.clip_y, best.iou * 100);
            }
        }

        // Save feature match visualizations
        if (g_runtime.save_phase7_images) {
            cv::Mat rendered_final = render_textured_fast(*mesh_hi, texture, best,
                                                          RENDER_SIZE, original, orig_mask);
            MatchResult final_match = compute_feature_matches(original, rendered_final,
                                                               orig_mask, cv::Mat());

            if (!final_match.inlier_matches.empty()) {
                // 1. Side-by-side drawMatches (traditional feature match visualization)
                cv::Mat match_vis;
                cv::drawMatches(original, final_match.kp1, rendered_final, final_match.kp2,
                                final_match.inlier_matches, match_vis,
                                cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
                                std::vector<char>(),
                                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imwrite(g_runtime.output_dir + "/feature_matches.png", match_vis);

                // 2. Offset vector overlay on original image
                // Shows displacement arrows at each match point, color-coded by offset magnitude
                cv::Mat offset_vis = original.clone();
                // If original is grayscale, convert for color arrows
                if (offset_vis.channels() == 1)
                    cv::cvtColor(offset_vis, offset_vis, cv::COLOR_GRAY2BGR);

                float max_offset = 0;
                for (auto& m : final_match.inlier_matches) {
                    float dx = final_match.kp1[m.queryIdx].pt.x - final_match.kp2[m.trainIdx].pt.x;
                    float dy = final_match.kp1[m.queryIdx].pt.y - final_match.kp2[m.trainIdx].pt.y;
                    max_offset = std::max(max_offset, sqrtf(dx*dx + dy*dy));
                }
                if (max_offset < 1.0f) max_offset = 1.0f;

                for (auto& m : final_match.inlier_matches) {
                    cv::Point2f p1 = final_match.kp1[m.queryIdx].pt;
                    cv::Point2f p2 = final_match.kp2[m.trainIdx].pt;
                    float dx = p1.x - p2.x;
                    float dy = p1.y - p2.y;
                    float dist = sqrtf(dx*dx + dy*dy);

                    // Color: green (small offset) yellow red (large offset)
                    float ratio = std::min(1.0f, dist / max_offset);
                    int r = (int)(ratio * 255);
                    int g = (int)((1.0f - ratio) * 255);
                    cv::Scalar color(0, g, r); // BGR

                    // Draw arrow from original point toward rendered point (x5 scale for visibility)
                    cv::Point2f arrow_end(p1.x - dx * 5.0f, p1.y - dy * 5.0f);
                    cv::arrowedLine(offset_vis, cv::Point(p1), cv::Point(arrow_end), color, 2, cv::LINE_AA, 0, 0.3);
                    cv::circle(offset_vis, cv::Point(p1), 3, color, -1);
                }

                // Add legend text
                char legend[128];
                snprintf(legend, sizeof(legend), "Matches: %d  Avg offset: %.1fpx  Max: %.1fpx",
                         final_match.num_inliers,
                         sqrtf(final_match.mean_dx*final_match.mean_dx + final_match.mean_dy*final_match.mean_dy),
                         max_offset);
                cv::putText(offset_vis, legend, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                            0.7, cv::Scalar(255, 255, 255), 2);

                cv::imwrite(g_runtime.output_dir + "/offset_vectors.png", offset_vis);

                float final_offset = sqrtf(final_match.mean_dx * final_match.mean_dx +
                                          final_match.mean_dy * final_match.mean_dy);
                printf("  [Phase7] Saved feature_matches.png: %d matches, offset=%.1fpx (max=%.1fpx)\n",
                       final_match.num_inliers, final_offset, max_offset);
                printf("  [Phase7] Saved offset_vectors.png (color-coded displacement arrows)\n");
            }
        } else {
            printf("  [Phase7] Visualization export disabled (--no-phase7-images)\n");
        }

        printf("  [Phase7] Final: IoU=%.4f%% (was %.4f%%)\n", best.iou * 100, base_iou * 100);
    } else {
        if (!g_runtime.enable_phase7) {
            printf("  [Phase7] Skipped (--no-phase7)\n");
        } else {
            printf("  [Phase7] Skipped (missing texture/original/mask/hi-poly)\n");
        }
    }

    // =====================================================================
    // Phase 7b: Hand Region Interior SIFT Feature Matching Refinement
    // Uses SIFT features on Sobel-gradient images *inside* the object
    // (not just contour edges), focused on the hand/paw area.
    // =====================================================================
    if (g_runtime.enable_phase7 && !texture.empty() && !original.empty() && !orig_mask.empty() && mesh_hi != nullptr) {
        Timer t7b("Phase7b Hand");
        float base_iou_7b = best.iou;

        // Render current best textured model
        cv::Mat rendered_7b = render_textured_fast(*mesh_hi, texture, best,
                                                    RENDER_SIZE, original, orig_mask);

        // Prepare masks
        cv::Mat omask_7b;
        { double mx; cv::minMaxLoc(orig_mask, nullptr, &mx);
          if (mx <= 1.0) orig_mask.convertTo(omask_7b, CV_8UC1, 255);
          else omask_7b = orig_mask.clone();
          if (omask_7b.type() != CV_8UC1) omask_7b.convertTo(omask_7b, CV_8UC1); }

        cv::Mat rg_7b;
        if (rendered_7b.channels() == 3) cv::cvtColor(rendered_7b, rg_7b, cv::COLOR_BGR2GRAY);
        else rg_7b = rendered_7b;
        cv::Mat rmask_7b;
        cv::threshold(rg_7b, rmask_7b, 1, 255, cv::THRESH_BINARY);

        // Interior masks: erode to avoid boundary effects
        cv::Mat kern7b = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
        cv::Mat o_interior, r_interior;
        cv::erode(omask_7b, o_interior, kern7b);
        cv::erode(rmask_7b, r_interior, kern7b);
        cv::Mat interior_mask;
        cv::bitwise_and(o_interior, r_interior, interior_mask);

        // Hand region: central portion of the object where paws are
        BBox bb7b = bbox_from_mask(orig_mask);
        int ow7b = bb7b.x1 - bb7b.x0, oh7b = bb7b.y1 - bb7b.y0;
        int hx0 = bb7b.x0 + (int)(ow7b * 0.10f);
        int hy0 = bb7b.y0 + (int)(oh7b * 0.20f);
        int hw = (int)(ow7b * 0.80f);
        int hh = (int)(oh7b * 0.60f);
        cv::Rect hand_roi(hx0, hy0, hw, hh);
        hand_roi &= cv::Rect(0, 0, original.cols, original.rows);

        // Create hand-focus mask (interior AND hand region)
        cv::Mat hand_mask = cv::Mat::zeros(original.size(), CV_8UC1);
        interior_mask(hand_roi).copyTo(hand_mask(hand_roi));

        int hand_nonzero = cv::countNonZero(hand_mask);
        printf("  [Phase7b] Hand region: (%d,%d)-(%d,%d), interior pixels=%d\n",
               hx0, hy0, hx0 + hw, hy0 + hh, hand_nonzero);

        if (hand_nonzero > 100) {
            // Preprocess to Sobel gradient (same as Phase7)
            cv::Mat orig_grad_7b = preprocess_for_matching(original);
            cv::Mat rend_grad_7b = preprocess_for_matching(rendered_7b);

            // === Method 1: SIFT on gradient images ===
            auto sift7b = cv::SIFT::create(500, 3, 0.03, 10, 1.6);
            std::vector<cv::KeyPoint> kp1_7b, kp2_7b;
            cv::Mat desc1_7b, desc2_7b;
            sift7b->detectAndCompute(orig_grad_7b, hand_mask, kp1_7b, desc1_7b);
            sift7b->detectAndCompute(rend_grad_7b, hand_mask, kp2_7b, desc2_7b);

            printf("  [Phase7b] SIFT features: orig=%d rend=%d\n",
                   (int)kp1_7b.size(), (int)kp2_7b.size());

            float sift_dx = 0, sift_dy = 0;
            int sift_good = 0;
            std::vector<cv::Point2f> sift_pts1, sift_pts2;

            if (!desc1_7b.empty() && !desc2_7b.empty() && desc1_7b.rows >= 3 && desc2_7b.rows >= 3) {
                auto matcher7b = cv::BFMatcher::create(cv::NORM_L2);
                std::vector<std::vector<cv::DMatch>> knn7b;
                matcher7b->knnMatch(desc1_7b, desc2_7b, knn7b, 2);

                for (auto& m : knn7b) {
                    if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) {
                        cv::Point2f p1 = kp1_7b[m[0].queryIdx].pt;
                        cv::Point2f p2 = kp2_7b[m[0].trainIdx].pt;
                        float dx = p1.x - p2.x, dy = p1.y - p2.y;
                        // Reject outliers (> 20px displacement)
                        if (sqrtf(dx*dx + dy*dy) < 20.0f) {
                            sift_dx += dx; sift_dy += dy;
                            sift_good++;
                            sift_pts1.push_back(p1);
                            sift_pts2.push_back(p2);
                        }
                    }
                }
            }

            // === Method 2: Dense grid template matching (interior) ===
            int patch_h7b = 12;   // 25x25 patches
            int search_r7b = 15;  // ±15 pixel search
            int grid_step7b = 18; // grid spacing
            float grid_dx = 0, grid_dy = 0;
            int grid_matches = 0;

            for (int gy = hand_roi.y; gy < hand_roi.y + hand_roi.height; gy += grid_step7b) {
                for (int gx = hand_roi.x; gx < hand_roi.x + hand_roi.width; gx += grid_step7b) {
                    if (hand_mask.at<uchar>(gy, gx) == 0) continue;

                    int x0g = gx - patch_h7b - search_r7b;
                    int y0g = gy - patch_h7b - search_r7b;
                    int x1g = gx + patch_h7b + search_r7b;
                    int y1g = gy + patch_h7b + search_r7b;
                    if (x0g < 0 || y0g < 0 || x1g >= original.cols || y1g >= original.rows) continue;

                    cv::Rect sr7b(gx - search_r7b, gy - search_r7b, 2*search_r7b+1, 2*search_r7b+1);
                    if (cv::countNonZero(rmask_7b(sr7b)) < search_r7b) continue;

                    cv::Mat patch7b = orig_grad_7b(cv::Rect(gx - patch_h7b, gy - patch_h7b,
                                                             2*patch_h7b+1, 2*patch_h7b+1));
                    cv::Mat search7b = rend_grad_7b(cv::Rect(x0g, y0g,
                                                              2*(patch_h7b+search_r7b)+1,
                                                              2*(patch_h7b+search_r7b)+1));

                    cv::Mat tmres;
                    cv::matchTemplate(search7b, patch7b, tmres, cv::TM_CCOEFF_NORMED);
                    double mx_val; cv::Point mx_loc;
                    cv::minMaxLoc(tmres, nullptr, &mx_val, nullptr, &mx_loc);

                    if (mx_val > 0.25) {
                        float match_x = (float)(x0g + mx_loc.x + patch_h7b);
                        float match_y = (float)(y0g + mx_loc.y + patch_h7b);
                        float dx = gx - match_x, dy = gy - match_y;
                        if (sqrtf(dx*dx + dy*dy) < 15.0f) {
                            grid_dx += dx; grid_dy += dy;
                            grid_matches++;
                        }
                    }
                }
            }

            // Combine SIFT and grid results
            float total_dx = 0, total_dy = 0;
            int total_matches = sift_good + grid_matches;
            if (total_matches > 0) {
                total_dx = (sift_dx + grid_dx) / total_matches;
                total_dy = (sift_dy + grid_dy) / total_matches;
            }

            printf("  [Phase7b] SIFT good=%d (dx=%.2f dy=%.2f), Grid=%d (dx=%.2f dy=%.2f)\n",
                   sift_good, sift_good > 0 ? sift_dx/sift_good : 0.0f,
                   sift_good > 0 ? sift_dy/sift_good : 0.0f,
                   grid_matches, grid_matches > 0 ? grid_dx/grid_matches : 0.0f,
                   grid_matches > 0 ? grid_dy/grid_matches : 0.0f);
            printf("  [Phase7b] Combined: %d matches, displacement=(%.2f, %.2f) px\n",
                   total_matches, total_dx, total_dy);

            if (total_matches >= 3 && sqrtf(total_dx*total_dx + total_dy*total_dy) > 0.3f) {
                // Pixel-to-camera factor (same as Phase7)
                float f_proj_7b = RENDER_SIZE / (2.0f * tanf(deg2rad(g_tune.camera_fov / 2.0f)));
                Mat3 R7b = rot_x(INITIAL_RX + best.phi) * rot_y(-best.theta) * rot_z(best.roll);
                float rmin_x7 = 1e9, rmax_x7 = -1e9, rmin_y7 = 1e9, rmax_y7 = -1e9;
                for (int vi = 0; vi < (int)mesh.vertices.size(); vi++) {
                    Vec3 tv7b = R7b * mesh.vertices[vi];
                    if (tv7b.y < best.clip_y + g_tune.clip_slope_x * tv7b.x) continue;
                    float vz = std::max(0.1f, best.cam_z - tv7b.z);
                    float px = (tv7b.x - best.cam_x) / vz * f_proj_7b + RENDER_SIZE * 0.5f;
                    float py = RENDER_SIZE * 0.5f - (tv7b.y - best.cam_y) / vz * f_proj_7b;
                    rmin_x7 = std::min(rmin_x7, px); rmax_x7 = std::max(rmax_x7, px);
                    rmin_y7 = std::min(rmin_y7, py); rmax_y7 = std::max(rmax_y7, py);
                }
                float tw_7b = (float)(bb7b.x1 - bb7b.x0);
                float rw_7b = rmax_x7 - rmin_x7;
                float scale_7b = tw_7b / std::max(1.0f, rw_7b);
                float px_to_cam_7b = best.cam_z / (f_proj_7b * scale_7b);

                // Step 1: Try camera position correction from hand features
                float best_iou_cand = best.iou;
                Params best_cand = best;

                const float cam_lrs[] = {0.8f, 0.5f, 0.3f, 0.15f, 0.07f};
                for (float lr : cam_lrs) {
                    Params trial = best;
                    trial.cam_x -= total_dx * px_to_cam_7b * lr;
                    trial.cam_y += total_dy * px_to_cam_7b * lr;
                    float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                            trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                    if (iou > best_iou_cand + 1e-7f) {
                        best_iou_cand = iou;
                        best_cand = trial;
                        printf("  [Phase7b] cam lr=%.2f IoU=%.4f%%\n", lr, iou * 100);
                    }
                }

                // Step 2: Micro-rotation grid search (guided by hand displacement direction)
                // Try very fine theta/phi/roll adjustments
                for (float dtheta = -0.15f; dtheta <= 0.151f; dtheta += 0.03f) {
                    for (float dphi = -0.15f; dphi <= 0.151f; dphi += 0.03f) {
                        for (float droll = -0.10f; droll <= 0.101f; droll += 0.05f) {
                            if (fabsf(dtheta) < 0.01f && fabsf(dphi) < 0.01f && fabsf(droll) < 0.01f) continue;
                            Params trial = best;
                            trial.theta += dtheta;
                            trial.phi += dphi;
                            trial.roll += droll;
                            float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                                    trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                            if (iou > best_iou_cand + 1e-7f) {
                                best_iou_cand = iou;
                                best_cand = trial;
                            }
                        }
                    }
                }

                // Step 3: If position was updated, also try combined cam + rotation
                if (best_iou_cand > best.iou + 1e-7f) {
                    Params base_7b = best_cand;
                    for (float dtheta = -0.06f; dtheta <= 0.061f; dtheta += 0.02f) {
                        for (float dphi = -0.06f; dphi <= 0.061f; dphi += 0.02f) {
                            Params trial = base_7b;
                            trial.theta += dtheta;
                            trial.phi += dphi;
                            float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                                    trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                            if (iou > best_iou_cand + 1e-7f) {
                                best_iou_cand = iou;
                                best_cand = trial;
                            }
                        }
                    }
                }

                if (best_iou_cand > best.iou + 1e-7f) {
                    best_cand.iou = best_iou_cand;
                    best = best_cand;
                    printf("  [Phase7b] Accepted: IoU %.4f%% -> %.4f%%\n",
                           base_iou_7b * 100, best_iou_cand * 100);
                } else {
                    printf("  [Phase7b] No IoU improvement (stays %.4f%%)\n", best.iou * 100);
                }
            } else {
                printf("  [Phase7b] Too few matches or displacement too small, skipping\n");
            }

            // Save hand feature match visualization
            if (g_runtime.save_phase7_images && (sift_good >= 3 || grid_matches >= 3)) {
                cv::Mat hand_vis = original.clone();
                if (hand_vis.channels() == 1)
                    cv::cvtColor(hand_vis, hand_vis, cv::COLOR_GRAY2BGR);

                // Draw hand region rectangle
                cv::rectangle(hand_vis, hand_roi, cv::Scalar(255, 255, 0), 2);

                // Draw SIFT matches (cyan circles + arrows)
                for (int si = 0; si < (int)sift_pts1.size(); si++) {
                    cv::Point2f p1 = sift_pts1[si], p2 = sift_pts2[si];
                    float dx = p1.x - p2.x, dy = p1.y - p2.y;
                    cv::Point2f arrow_end(p1.x - dx * 5.0f, p1.y - dy * 5.0f);
                    cv::arrowedLine(hand_vis, cv::Point(p1), cv::Point(arrow_end),
                                    cv::Scalar(255, 255, 0), 2, cv::LINE_AA, 0, 0.3);
                    cv::circle(hand_vis, cv::Point(p1), 4, cv::Scalar(255, 255, 0), -1);
                }

                // Legend
                char leg7b[128];
                snprintf(leg7b, sizeof(leg7b), "Hand: SIFT=%d Grid=%d dx=%.1f dy=%.1f",
                         sift_good, grid_matches, total_dx, total_dy);
                cv::putText(hand_vis, leg7b, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                            0.6, cv::Scalar(0, 255, 255), 2);
                cv::imwrite(g_runtime.output_dir + "/hand_features.png", hand_vis);
                printf("  [Phase7b] Saved hand_features.png\n");
            }
        } else {
            printf("  [Phase7b] Insufficient interior pixels, skipping\n");
        }
    }

    // =========================================================================
    // Phase 8: Spatial Error Correction
    // Analyzes boundary displacement vectors between target and render contours
    // to derive and apply micro-corrections for pose parameters.
    // Key idea: the spatial pattern of FN/FP at boundary encodes rotation,
    //           translation, and scale errors that can be decoded analytically.
    // =========================================================================
    {
        Timer t("Phase8 SpatialCorrect");
        printf("\n=== Phase 8: Spatial Error Correction ===\n");

        const int P8_MAX_ITERS = 5;
        int ns = std::max(64, g_tune.norm_size);

        for (int p8_iter = 0; p8_iter < P8_MAX_ITERS; p8_iter++) {
            // Render current best
            cv::Mat sil_p8 = render_silhouette(mesh, best.theta, best.phi, best.roll,
                                                best.cam_x, best.cam_y, best.cam_z, RENDER_SIZE, best.clip_y);
            BBox bb_p8 = bbox_from_mask(sil_p8);
            if (bb_p8.x0 < 0) break;

            BBox bbe_p8 = expand_bbox(bb_p8, RENDER_SIZE, RENDER_SIZE, g_tune.bbox_margin_ratio);
            cv::Mat sil_norm_p8 = crop_and_resize(sil_p8, bbe_p8, ns);

            // Mask out ignored regions for contour extraction
            cv::Mat tgt_for_contour = target_norm.clone();
            cv::Mat rnd_for_contour = sil_norm_p8.clone();
            if (!g_ignore_norm.empty()) {
                for (int y = 0; y < ns; y++) {
                    const uchar* ig = g_ignore_norm.ptr<uchar>(y);
                    uchar* t = tgt_for_contour.ptr<uchar>(y);
                    uchar* r = rnd_for_contour.ptr<uchar>(y);
                    for (int x = 0; x < ns; x++) {
                        if (ig[x] > 0) { t[x] = 0; r[x] = 0; }
                    }
                }
            }

            // Extract contours
            std::vector<std::vector<cv::Point>> tc_p8, rc_p8;
            cv::findContours(tgt_for_contour, tc_p8, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            cv::findContours(rnd_for_contour, rc_p8, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            if (tc_p8.empty() || rc_p8.empty()) break;

            // Get largest contour from each
            auto& tgt_contour = *std::max_element(tc_p8.begin(), tc_p8.end(),
                [](const auto& a, const auto& b) { return a.size() < b.size(); });
            auto& rnd_contour = *std::max_element(rc_p8.begin(), rc_p8.end(),
                [](const auto& a, const auto& b) { return a.size() < b.size(); });

            float cx = ns / 2.0f, cy = ns / 2.0f;

            // For each target contour point, find nearest render contour point
            struct DispVec { float px, py, dx, dy; };
            std::vector<DispVec> disps;
            disps.reserve(tgt_contour.size());
            float max_dist = ns * 0.1f;  // max 10% of image size

            for (const auto& tp : tgt_contour) {
                float min_dsq = 1e9f;
                cv::Point nearest(-1, -1);
                for (const auto& rp : rnd_contour) {
                    float dsq = (float)(tp.x - rp.x) * (tp.x - rp.x) +
                                (float)(tp.y - rp.y) * (tp.y - rp.y);
                    if (dsq < min_dsq) { min_dsq = dsq; nearest = rp; }
                }
                if (min_dsq < max_dist * max_dist) {
                    disps.push_back({(float)tp.x, (float)tp.y,
                                     (float)(nearest.x - tp.x), (float)(nearest.y - tp.y)});
                }
            }

            if (disps.size() < 20) {
                printf("  [Iter %d] Too few boundary points (%zu), stopping\n", p8_iter + 1, disps.size());
                break;
            }

            // === Analyze displacement field ===

            // 1. Translation (mean displacement)
            float mean_dx = 0, mean_dy = 0;
            for (auto& d : disps) { mean_dx += d.dx; mean_dy += d.dy; }
            mean_dx /= disps.size();
            mean_dy /= disps.size();

            // 2. Roll component: fit rotation model
            //    For roll ω around center: dx = -ω*(y-cy), dy = ω*(x-cx)
            //    Least-squares: ω = Σ[-ddx*(y-cy) + ddy*(x-cx)] / Σ[(x-cx)² + (y-cy)²]
            float rot_num = 0, rot_den = 0;
            for (auto& d : disps) {
                float rx = d.px - cx, ry = d.py - cy;
                float ddx = d.dx - mean_dx, ddy = d.dy - mean_dy;
                rot_num += (-ddx * ry + ddy * rx);
                rot_den += (rx * rx + ry * ry);
            }
            float omega = rot_den > 0 ? rot_num / rot_den : 0;

            // 3. Scale component (radial displacement)
            float radial_sum = 0, radial_cnt = 0;
            for (auto& d : disps) {
                float rx = d.px - cx, ry = d.py - cy;
                float r = std::sqrt(rx * rx + ry * ry);
                if (r > 3) {
                    float ddx = d.dx - mean_dx, ddy = d.dy - mean_dy;
                    radial_sum += (ddx * rx + ddy * ry) / r;
                    radial_cnt += 1;
                }
            }
            float mean_radial = radial_cnt > 0 ? radial_sum / radial_cnt : 0;

            // 4. Quadrant analysis (top vs bottom, left vs right)
            float top_dx = 0, top_dy = 0, bot_dx = 0, bot_dy = 0;
            int top_n = 0, bot_n = 0;
            float left_dx = 0, left_dy = 0, right_dx = 0, right_dy = 0;
            int left_n = 0, right_n = 0;
            for (auto& d : disps) {
                if (d.py < cy) { top_dx += d.dx; top_dy += d.dy; top_n++; }
                else            { bot_dx += d.dx; bot_dy += d.dy; bot_n++; }
                if (d.px < cx) { left_dx += d.dx; left_dy += d.dy; left_n++; }
                else            { right_dx += d.dx; right_dy += d.dy; right_n++; }
            }
            if (top_n > 0) { top_dx /= top_n; top_dy /= top_n; }
            if (bot_n > 0) { bot_dx /= bot_n; bot_dy /= bot_n; }
            if (left_n > 0) { left_dx /= left_n; left_dy /= left_n; }
            if (right_n > 0) { right_dx /= right_n; right_dy /= right_n; }

            if (p8_iter == 0) {
                printf("  Boundary points: %zu\n", disps.size());
                printf("  Mean displacement: dx=%.3f, dy=%.3f px\n", mean_dx, mean_dy);
                printf("  Roll component: omega=%.5f (%.3f deg)\n", omega, omega * 180.0f / 3.14159f);
                printf("  Radial (scale): %.3f px\n", mean_radial);
                printf("  Top:    dx=%+.2f dy=%+.2f   Bottom: dx=%+.2f dy=%+.2f\n",
                       top_dx, top_dy, bot_dx, bot_dy);
                printf("  Left:   dx=%+.2f dy=%+.2f   Right:  dx=%+.2f dy=%+.2f\n",
                       left_dx, left_dy, right_dx, right_dy);
            }

            // === Convert pixel displacements to parameter corrections ===
            float fov_rad = deg2rad(g_tune.camera_fov / 2.0f);
            float focal = RENDER_SIZE / (2.0f * tanf(fov_rad));
            float bbox_w = (float)(bbe_p8.x1 - bbe_p8.x0 + 1);
            float norm_to_render = bbox_w / ns;

            // cam_x, cam_y: pixel shift → camera translation
            float dcam_x = -mean_dx * norm_to_render * best.cam_z / focal;
            float dcam_y =  mean_dy * norm_to_render * best.cam_z / focal;

            // Roll: direct from omega (radians → degrees)
            float droll = omega * 180.0f / 3.14159f;

            // Phi/theta from top-vs-bottom and left-vs-right asymmetry
            float tb_diff_dx = top_dx - bot_dx;
            float lr_diff_dy = left_dy - right_dy;
            float dphi   = -tb_diff_dx * 0.03f;
            float dtheta =  lr_diff_dy * 0.03f;

            if (p8_iter == 0) {
                printf("\n  Estimated corrections:\n");
                printf("    cam_x: %+.5f  cam_y: %+.5f\n", dcam_x, dcam_y);
                printf("    roll: %+.4f deg\n", droll);
                printf("    theta: %+.4f deg  phi: %+.4f deg\n", dtheta, dphi);
            }

            // Skip if displacements are negligible
            float total_disp = std::sqrt(mean_dx * mean_dx + mean_dy * mean_dy);
            if (total_disp < 0.05f && std::abs(omega) < 0.0002f) {
                printf("  [Iter %d] Displacement < 0.05px, converged.\n", p8_iter + 1);
                break;
            }

            // === Try corrections with multiple learning rates ===
            Params p8_best = best;
            float p8_best_iou = best.iou;
            bool p8_improved = false;

            float lr_list[] = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.3f, 1.6f, 2.0f};
            for (float lr : lr_list) {
                // Translation only
                {
                    Params trial = best;
                    trial.cam_x += dcam_x * lr;
                    trial.cam_y += dcam_y * lr;
                    float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                             trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                    if (iou > p8_best_iou) {
                        p8_best = trial; p8_best.iou = iou; p8_best_iou = iou; p8_improved = true;
                    }
                }
                // Translation + roll
                {
                    Params trial = best;
                    trial.cam_x += dcam_x * lr;
                    trial.cam_y += dcam_y * lr;
                    trial.roll  += droll * lr;
                    float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                             trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                    if (iou > p8_best_iou) {
                        p8_best = trial; p8_best.iou = iou; p8_best_iou = iou; p8_improved = true;
                    }
                }
                // Translation + roll + theta/phi
                {
                    Params trial = best;
                    trial.cam_x += dcam_x * lr;
                    trial.cam_y += dcam_y * lr;
                    trial.roll  += droll * lr;
                    trial.theta += dtheta * lr;
                    trial.phi   += dphi * lr;
                    float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                             trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                    if (iou > p8_best_iou) {
                        p8_best = trial; p8_best.iou = iou; p8_best_iou = iou; p8_improved = true;
                    }
                }
                // Roll only
                {
                    Params trial = best;
                    trial.roll += droll * lr;
                    float iou = compute_iou(mesh, target_norm, trial.theta, trial.phi, trial.roll,
                                             trial.cam_x, trial.cam_y, trial.cam_z, trial.clip_y);
                    if (iou > p8_best_iou) {
                        p8_best = trial; p8_best.iou = iou; p8_best_iou = iou; p8_improved = true;
                    }
                }
            }

            if (p8_improved) {
                printf("  [Iter %d] IMPROVED: %.4f%% -> %.4f%% (+%.4f%%)\n",
                       p8_iter + 1, best.iou * 100, p8_best_iou * 100, (p8_best_iou - best.iou) * 100);
                best = p8_best;
            } else {
                printf("  [Iter %d] No improvement (disp=%.2fpx, omega=%.4f), stopping\n",
                       p8_iter + 1, total_disp, omega);
                break;
            }
        }

        // Save displacement visualization
        {
            cv::Mat sil_p8 = render_silhouette(mesh, best.theta, best.phi, best.roll,
                                                best.cam_x, best.cam_y, best.cam_z, RENDER_SIZE, best.clip_y);
            BBox bb_p8 = bbox_from_mask(sil_p8);
            if (bb_p8.x0 >= 0) {
                BBox bbe_p8 = expand_bbox(bb_p8, RENDER_SIZE, RENDER_SIZE, g_tune.bbox_margin_ratio);
                cv::Mat sil_norm_p8 = crop_and_resize(sil_p8, bbe_p8, ns);

                cv::Mat vis(ns, ns, CV_8UC3, cv::Scalar(0, 0, 0));
                // Draw target contour (green) and render contour (blue)
                {
                    cv::Mat t_c = target_norm.clone(), r_c = sil_norm_p8.clone();
                    std::vector<std::vector<cv::Point>> tc2, rc2;
                    cv::findContours(t_c, tc2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    cv::findContours(r_c, rc2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    cv::drawContours(vis, tc2, -1, cv::Scalar(0, 200, 0), 1);
                    cv::drawContours(vis, rc2, -1, cv::Scalar(255, 100, 0), 1);
                }
                // Upscale
                cv::Mat vis_large;
                cv::resize(vis, vis_large, cv::Size(ns * 3, ns * 3), 0, 0, cv::INTER_NEAREST);
                cv::putText(vis_large, "Green=Target  Blue=Render", cv::Point(10, 25),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                if (g_runtime.save_phase7_images) {
                    cv::imwrite(g_runtime.output_dir + "/phase8_contours.png", vis_large);
                    printf("  Saved phase8_contours.png\n");
                }
            }
        }

        // Compute and display pose quality (error concentration)
        {
            cv::Mat sil_q = render_silhouette(mesh, best.theta, best.phi, best.roll,
                                              best.cam_x, best.cam_y, best.cam_z, RENDER_SIZE, best.clip_y);
            BBox bb_q = bbox_from_mask(sil_q);
            if (bb_q.x0 >= 0) {
                BBox bbe_q = expand_bbox(bb_q, RENDER_SIZE, RENDER_SIZE, g_tune.bbox_margin_ratio);
                cv::Mat sil_norm_q = crop_and_resize(sil_q, bbe_q, ns);
                PoseQuality pq = compute_pose_quality(target_norm, sil_norm_q);
                printf("\n  === Pose Quality ===\n");
                printf("  IoU: %.4f%%\n", pq.iou * 100);
                printf("  Error Gini: %.4f (higher=concentrated=better)\n", pq.gini);
                printf("  Mean boundary dist: %.3f px (lower=better)\n", pq.mean_boundary_dist);
                printf("  P90 boundary dist: %.3f px\n", pq.p90_boundary_dist);
                printf("  Error ratio (>1.5px): %.1f%% (lower=better)\n", pq.error_ratio * 100);
                printf("  Active sectors: %d/%d\n", pq.active_sectors, pq.total_sectors);
                printf("  FN=%d FP=%d\n", pq.fn_count, pq.fp_count);
                printf("  Combined score: %.6f\n", pq.combined_score);
            }
        }

        printf("  Phase 8 final IoU: %.4f%%\n", best.iou * 100);
    }

    return best;
}

// =============================================================================
// Fast Textured Rendering (for feature matching, low-poly, small size)
// =============================================================================

cv::Mat render_textured_fast(const Mesh& mesh, const cv::Mat& texture,
                             const Params& params, int size,
                             const cv::Mat& orig_img, const cv::Mat& orig_mask) {
    Mat3 R = rot_x(INITIAL_RX + params.phi) * rot_y(-params.theta) * rot_z(params.roll);

    int nv = (int)mesh.vertices.size();
    std::vector<Vec3> tv(nv);
    float f_proj = size / (2.0f * tanf(deg2rad(g_tune.camera_fov / 2.0f)));

    for (int i = 0; i < nv; i++)
        tv[i] = R * mesh.vertices[i];

    // Find valid faces and bbox
    struct FD { int idx; float z; };
    std::vector<FD> vfaces;
    float min_px = 1e9, max_px = -1e9, min_py = 1e9, max_py = -1e9;

    for (int fi = 0; fi < (int)mesh.faces.size(); fi++) {
        const Face& face = mesh.faces[fi];
        int a = face.v[0], b = face.v[1], c = face.v[2];
        if (a < 0 || a >= nv || b < 0 || b >= nv || c < 0 || c >= nv) continue;
        {
            float sl = g_tune.clip_slope_x;
            if (tv[a].y < params.clip_y + sl*tv[a].x || tv[b].y < params.clip_y + sl*tv[b].x || tv[c].y < params.clip_y + sl*tv[c].x) continue;
        }

        for (int vi : {a, b, c}) {
            float vz = std::max(0.1f, params.cam_z - tv[vi].z);
            float px = (tv[vi].x - params.cam_x) / vz * f_proj + size * 0.5f;
            float py = size * 0.5f - (tv[vi].y - params.cam_y) / vz * f_proj;
            min_px = std::min(min_px, px); max_px = std::max(max_px, px);
            min_py = std::min(min_py, py); max_py = std::max(max_py, py);
        }
        float avg_z = (tv[a].z + tv[b].z + tv[c].z) / 3.0f;
        vfaces.push_back({fi, avg_z});
    }

    if (vfaces.empty()) return cv::Mat::zeros(orig_img.rows, orig_img.cols, CV_8UC3);

    std::sort(vfaces.begin(), vfaces.end(), [](const FD& a, const FD& b) { return a.z < b.z; });

    // Project to image: map render bbox to target bbox from orig_mask
    cv::Mat mask_gray;
    if (orig_mask.channels() == 3) {
        cv::cvtColor(orig_mask, mask_gray, cv::COLOR_BGR2GRAY);
    } else {
        mask_gray = orig_mask;
    }
    BBox tbb = bbox_from_mask(mask_gray);
    if (tbb.x0 < 0) return cv::Mat::zeros(orig_img.rows, orig_img.cols, CV_8UC3);
    float rcx = (min_px + max_px) / 2.0f, rcy = (min_py + max_py) / 2.0f;
    float rw = max_px - min_px, rh = max_py - min_py;
    float tw = (float)(tbb.x1 - tbb.x0), th = (float)(tbb.y1 - tbb.y0);
    float tcx = (tbb.x0 + tbb.x1) / 2.0f, tcy = (tbb.y0 + tbb.y1) / 2.0f;
    float scale = std::min(tw / rw, th / rh);

    int H = orig_img.rows, W = orig_img.cols;
    cv::Mat result = cv::Mat::zeros(H, W, CV_8UC3);
    int tex_h = texture.rows, tex_w = texture.cols;

    for (const auto& fd : vfaces) {
        const Face& face = mesh.faces[fd.idx];
        int a = face.v[0], b = face.v[1], c = face.v[2];

        float xs[3], ys[3];
        for (int k = 0; k < 3; k++) {
            int vi = face.v[k];
            float vz = std::max(0.1f, params.cam_z - tv[vi].z);
            float rx = (tv[vi].x - params.cam_x) / vz * f_proj + size * 0.5f;
            float ry = size * 0.5f - (tv[vi].y - params.cam_y) / vz * f_proj;
            xs[k] = (rx - rcx) * scale + tcx;
            ys[k] = (ry - rcy) * scale + tcy;
        }

        float u0 = 0, v0 = 0, u1 = 0, v1 = 0, u2 = 0, v2 = 0;
        if (!mesh.uvs.empty()) {
            auto get_uv = [&](int vt_idx, float& u, float& v) {
                if (vt_idx >= 0 && vt_idx < (int)mesh.uvs.size()) {
                    u = mesh.uvs[vt_idx].u; v = mesh.uvs[vt_idx].v;
                }
            };
            get_uv(face.vt[0], u0, v0);
            get_uv(face.vt[1], u1, v1);
            get_uv(face.vt[2], u2, v2);
        }

        int tri_x0 = std::max(0, (int)std::min({xs[0], xs[1], xs[2]}));
        int tri_y0 = std::max(0, (int)std::min({ys[0], ys[1], ys[2]}));
        int tri_x1 = std::min(W - 1, (int)std::max({xs[0], xs[1], xs[2]}) + 1);
        int tri_y1 = std::min(H - 1, (int)std::max({ys[0], ys[1], ys[2]}) + 1);

        float denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2]);
        if (fabsf(denom) < 0.001f) continue;
        float inv_d = 1.0f / denom;

        for (int py = tri_y0; py <= tri_y1; py++) {
            for (int px = tri_x0; px <= tri_x1; px++) {
                float w0 = ((ys[1]-ys[2])*(px-xs[2]) + (xs[2]-xs[1])*(py-ys[2])) * inv_d;
                float w1 = ((ys[2]-ys[0])*(px-xs[2]) + (xs[0]-xs[2])*(py-ys[2])) * inv_d;
                float w2 = 1.0f - w0 - w1;
                if (w0 >= -0.001f && w1 >= -0.001f && w2 >= -0.001f) {
                    float u = w0*u0 + w1*u1 + w2*u2;
                    float v = w0*v0 + w1*v1 + w2*v2;
                    int tx = ((int)(u * tex_w)) % tex_w;
                    int ty = ((int)((1.0f - v) * tex_h)) % tex_h;
                    if (tx < 0) tx += tex_w;
                    if (ty < 0) ty += tex_h;
                    result.at<cv::Vec3b>(py, px) = texture.at<cv::Vec3b>(ty, tx);
                }
            }
        }
    }
    return result;
}

// =============================================================================
// Silhouette at original image resolution (for edge matching)
// =============================================================================

cv::Mat render_silhouette_fullres(const Mesh& mesh, const Params& params,
                                   const cv::Mat& orig_mask) {
    Mat3 R = rot_x(INITIAL_RX + params.phi) * rot_y(-params.theta) * rot_z(params.roll);
    int nv = (int)mesh.vertices.size();
    std::vector<Vec3> tv(nv);
    float f_proj = RENDER_SIZE / (2.0f * tanf(deg2rad(g_tune.camera_fov / 2.0f)));

    for (int i = 0; i < nv; i++)
        tv[i] = R * mesh.vertices[i];

    // Compute render-space bbox
    float min_px = 1e9, max_px = -1e9, min_py = 1e9, max_py = -1e9;
    for (int i = 0; i < nv; i++) {
        float vz = std::max(0.1f, params.cam_z - tv[i].z);
        float px = (tv[i].x - params.cam_x) / vz * f_proj + RENDER_SIZE * 0.5f;
        float py = RENDER_SIZE * 0.5f - (tv[i].y - params.cam_y) / vz * f_proj;
        min_px = std::min(min_px, px); max_px = std::max(max_px, px);
        min_py = std::min(min_py, py); max_py = std::max(max_py, py);
    }

    // Map to original image coordinates via target bbox
    cv::Mat mask_gray;
    if (orig_mask.channels() == 3) cv::cvtColor(orig_mask, mask_gray, cv::COLOR_BGR2GRAY);
    else mask_gray = orig_mask;
    BBox tbb = bbox_from_mask(mask_gray);
    if (tbb.x0 < 0) return cv::Mat::zeros(orig_mask.rows, orig_mask.cols, CV_8UC1);

    float rcx = (min_px + max_px) / 2.0f, rcy = (min_py + max_py) / 2.0f;
    float rw = max_px - min_px, rh = max_py - min_py;
    float tw = (float)(tbb.x1 - tbb.x0), th = (float)(tbb.y1 - tbb.y0);
    float tcx = (tbb.x0 + tbb.x1) / 2.0f, tcy = (tbb.y0 + tbb.y1) / 2.0f;
    float scale = std::min(tw / rw, th / rh);

    int H = orig_mask.rows, W = orig_mask.cols;
    cv::Mat result = cv::Mat::zeros(H, W, CV_8UC1);

    for (const auto& face : mesh.faces) {
        int a = face.v[0], b = face.v[1], c = face.v[2];
        if (a < 0 || a >= nv || b < 0 || b >= nv || c < 0 || c >= nv) continue;
        {
            float sl = g_tune.clip_slope_x;
            if (tv[a].y < params.clip_y + sl*tv[a].x || tv[b].y < params.clip_y + sl*tv[b].x || tv[c].y < params.clip_y + sl*tv[c].x) continue;
        }

        cv::Point pts[3];
        for (int k = 0; k < 3; k++) {
            int vi = face.v[k];
            float vz = std::max(0.1f, params.cam_z - tv[vi].z);
            float rx = (tv[vi].x - params.cam_x) / vz * f_proj + RENDER_SIZE * 0.5f;
            float ry = RENDER_SIZE * 0.5f - (tv[vi].y - params.cam_y) / vz * f_proj;
            pts[k].x = (int)((rx - rcx) * scale + tcx);
            pts[k].y = (int)((ry - rcy) * scale + tcy);
        }
        cv::fillConvexPoly(result, pts, 3, cv::Scalar(255));
    }
    return result;
}

// =============================================================================
// Edge-based Contour Score (Chamfer distance)
// =============================================================================

// Compute edge score: contour match percentage + boundary-weighted IoU
// Combines: (1) What % of contour pixels match within 3px
//           (2) Boundary-weighted pixel agreement
// Returns score in range [0, 1] where higher = better
float compute_edge_score(const cv::Mat& original, const cv::Mat& orig_mask,
                         const cv::Mat& render_mask) {
    BBox bb1 = bbox_from_mask(orig_mask);
    BBox bb2 = bbox_from_mask(render_mask);
    if (bb1.x0 < 0 || bb2.x0 < 0) return 0.0f;

    int x0 = std::min(bb1.x0, bb2.x0) - 10;
    int y0 = std::min(bb1.y0, bb2.y0) - 10;
    int x1 = std::max(bb1.x1, bb2.x1) + 10;
    int y1 = std::max(bb1.y1, bb2.y1) + 10;
    x0 = std::max(0, x0); y0 = std::max(0, y0);
    x1 = std::min(render_mask.cols - 1, x1); y1 = std::min(render_mask.rows - 1, y1);
    cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);

    // Prepare masks in ROI
    cv::Mat omask_roi = orig_mask(roi);
    cv::Mat omask255;
    double mx; cv::minMaxLoc(omask_roi, nullptr, &mx);
    if (mx <= 1.0) omask_roi.convertTo(omask255, CV_8UC1, 255);
    else omask255 = omask_roi;
    cv::Mat rmask_roi = render_mask(roi);

    // Extract contours using morphological gradient
    cv::Mat kern = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat orig_contour, rend_contour;
    cv::morphologyEx(omask255, orig_contour, cv::MORPH_GRADIENT, kern);
    cv::morphologyEx(rmask_roi, rend_contour, cv::MORPH_GRADIENT, kern);

    int oc = cv::countNonZero(orig_contour);
    int rc = cv::countNonZero(rend_contour);
    if (oc < 10 || rc < 10) return 0.0f;

    // (1) Contour match %: render contour pixels within 3px of original contour
    cv::Mat orig_inv; cv::bitwise_not(orig_contour, orig_inv);
    cv::Mat dist_to_orig; cv::distanceTransform(orig_inv, dist_to_orig, cv::DIST_L2, 3);

    int close_count = 0;
    for (int y = 0; y < rend_contour.rows; y++) {
        const uchar* er = rend_contour.ptr<uchar>(y);
        const float* dr = dist_to_orig.ptr<float>(y);
        for (int x = 0; x < rend_contour.cols; x++) {
            if (er[x] > 0 && dr[x] <= g_tune.contour_match_threshold_px) close_count++;
        }
    }
    float contour_match = (float)close_count / rc;

    // (2) Same in reverse: original contour within 3px of render contour
    cv::Mat rend_inv; cv::bitwise_not(rend_contour, rend_inv);
    cv::Mat dist_to_rend; cv::distanceTransform(rend_inv, dist_to_rend, cv::DIST_L2, 3);

    int close_count2 = 0;
    for (int y = 0; y < orig_contour.rows; y++) {
        const uchar* er = orig_contour.ptr<uchar>(y);
        const float* dr = dist_to_rend.ptr<float>(y);
        for (int x = 0; x < orig_contour.cols; x++) {
            if (er[x] > 0 && dr[x] <= g_tune.contour_match_threshold_px) close_count2++;
        }
    }
    float contour_match_rev = (float)close_count2 / oc;

    // (3) Boundary-weighted IoU: weight each pixel by closeness to contour
    // Pixels on the boundary get weight 1.0, interior pixels get lower weight
    cv::Mat omask_dist; cv::distanceTransform(omask255, omask_dist, cv::DIST_L2, 3);
    cv::Mat rmask_dist; cv::distanceTransform(rmask_roi, rmask_dist, cv::DIST_L2, 3);

    float weighted_inter = 0, weighted_union = 0;
    float bw_factor = std::max(0.0f, g_tune.boundary_weight_factor);
    float bw_dist = std::max(0.1f, g_tune.boundary_weight_distance_px);
    for (int y = 0; y < omask_roi.rows; y++) {
        const uchar* om = omask255.ptr<uchar>(y);
        const uchar* rm = rmask_roi.ptr<uchar>(y);
        const float* od = omask_dist.ptr<float>(y);
        const float* rd = rmask_dist.ptr<float>(y);
        for (int x = 0; x < omask_roi.cols; x++) {
            bool o = om[x] > 0, r = rm[x] > 0;
            if (!o && !r) continue;
            // Weight: higher near boundary (distance < 10px)
            float min_dist = 1e6f;
            if (o) min_dist = std::min(min_dist, od[x]);
            if (r) min_dist = std::min(min_dist, rd[x]);
            float d = std::min(min_dist, bw_dist);
            float w = 1.0f / (1.0f + d * bw_factor);
            if (o && r) weighted_inter += w;
            if (o || r) weighted_union += w;
        }
    }
    float weighted_iou = (weighted_union > 0) ? weighted_inter / weighted_union : 0.0f;

    // Combined: symmetric contour match + weighted IoU
    float sym_contour = (contour_match + contour_match_rev) / 2.0f;
    float contour_w = std::max(0.0f, std::min(1.0f, g_tune.sym_contour_weight));
    return contour_w * sym_contour + (1.0f - contour_w) * weighted_iou;
}

// =============================================================================
// Cross-modal Feature Matching (Sobel + SIFT + RANSAC)
// =============================================================================

// Preprocess image for cross-modal matching:
// CLAHE 竊・bilateral filter 竊・Sobel gradient magnitude
cv::Mat preprocess_for_matching(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img.clone();
    // CLAHE for better contrast (especially on depth images)
    auto clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
    clahe->apply(gray, gray);
    // Bilateral filter: preserves edges, smooths texture noise
    cv::Mat filtered;
    cv::bilateralFilter(gray, filtered, 7, 50, 50);
    // Sobel gradient (larger kernel for structural edges)
    cv::Mat gx, gy;
    cv::Sobel(filtered, gx, CV_32F, 1, 0, 5);
    cv::Sobel(filtered, gy, CV_32F, 0, 1, 5);
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);
    cv::Mat result;
    cv::normalize(mag, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // CLAHE again on gradient for better feature detection
    clahe->apply(result, result);
    return result;
}

// Compute feature matches using contour-point template matching
// Samples points along the original contour, matches gradient patches locally
// in the rendered image. This avoids the cross-modal descriptor problem.
MatchResult compute_feature_matches(const cv::Mat& orig, const cv::Mat& rendered,
                                     const cv::Mat& orig_mask, const cv::Mat& rend_mask) {
    MatchResult result = {};

    // Prepare masks (CV_8U, 0 or 255)
    cv::Mat omask255;
    if (!orig_mask.empty()) {
        double mx; cv::minMaxLoc(orig_mask, nullptr, &mx);
        if (mx <= 1.0) orig_mask.convertTo(omask255, CV_8UC1, 255);
        else omask255 = orig_mask.clone();
        if (omask255.type() != CV_8UC1) omask255.convertTo(omask255, CV_8UC1);
    } else return result;

    cv::Mat rmask255;
    if (!rend_mask.empty()) {
        rmask255 = rend_mask.clone();
    } else {
        cv::Mat rg;
        if (rendered.channels() == 3) cv::cvtColor(rendered, rg, cv::COLOR_BGR2GRAY);
        else rg = rendered;
        cv::threshold(rg, rmask255, 1, 255, cv::THRESH_BINARY);
    }

    // Preprocess both to Sobel gradient (edges are common to both modalities)
    cv::Mat orig_grad = preprocess_for_matching(orig);
    cv::Mat rend_grad = preprocess_for_matching(rendered);

    // Extract contour points from original mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(omask255.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return result;

    // Find largest contour
    int best_idx = 0;
    for (int i = 1; i < (int)contours.size(); i++)
        if (contours[i].size() > contours[best_idx].size()) best_idx = i;

    const auto& contour = contours[best_idx];
    int n_contour = (int)contour.size();
    if (n_contour < 20) return result;

    // Sample ~60 points evenly along the contour
    int n_samples = std::min(60, n_contour / 3);
    int sample_step = n_contour / n_samples;

    int patch_half = 15;  // 31x31 patch
    int search_radius = 25; // search within +-25 pixels
    int H = orig.rows, W = orig.cols;

    float sum_dx = 0, sum_dy = 0;
    int n_top = 0, n_bot = 0, n_left = 0, n_right = 0;
    float sum_top_dy = 0, sum_bot_dy = 0, sum_left_dx = 0, sum_right_dx = 0;
    float cy = H / 2.0f, cx = W / 2.0f;
    int match_idx = 0;

    for (int i = 0; i < n_contour; i += sample_step) {
        cv::Point pt = contour[i];

        // Check bounds for patch + search region
        int x0 = pt.x - patch_half - search_radius;
        int y0 = pt.y - patch_half - search_radius;
        int x1 = pt.x + patch_half + search_radius;
        int y1 = pt.y + patch_half + search_radius;
        if (x0 < 0 || y0 < 0 || x1 >= W || y1 >= H) continue;

        // Check that rendered image has content in search area
        cv::Rect search_rect(pt.x - search_radius, pt.y - search_radius,
                             2 * search_radius + 1, 2 * search_radius + 1);
        if (cv::countNonZero(rmask255(search_rect)) < search_radius) continue;

        // Extract patch from original gradient
        cv::Mat patch = orig_grad(cv::Rect(pt.x - patch_half, pt.y - patch_half,
                                           2 * patch_half + 1, 2 * patch_half + 1));

        // Search region in rendered gradient (larger than patch by search_radius on each side)
        cv::Mat search_img = rend_grad(cv::Rect(x0, y0,
                                                2 * (patch_half + search_radius) + 1,
                                                2 * (patch_half + search_radius) + 1));

        // Template matching (normalized cross-correlation)
        cv::Mat match_result;
        cv::matchTemplate(search_img, patch, match_result, cv::TM_CCOEFF_NORMED);

        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(match_result, nullptr, &max_val, nullptr, &max_loc);

        if (max_val > g_tune.phase7_match_threshold) {
            // Match location in original image coordinates
            float match_x = (float)(x0 + max_loc.x + patch_half);
            float match_y = (float)(y0 + max_loc.y + patch_half);

            float dx = pt.x - match_x;
            float dy = pt.y - match_y;

            // Create KeyPoint pair for visualization
            result.kp1.push_back(cv::KeyPoint((float)pt.x, (float)pt.y, (float)(2 * patch_half)));
            result.kp2.push_back(cv::KeyPoint(match_x, match_y, (float)(2 * patch_half)));
            result.inlier_matches.push_back(cv::DMatch(match_idx, match_idx, (float)(1.0 - max_val)));
            match_idx++;

            sum_dx += dx; sum_dy += dy;
            if (pt.y < cy) { sum_top_dy += dy; n_top++; }
            else { sum_bot_dy += dy; n_bot++; }
            if (pt.x < cx) { sum_left_dx += dx; n_left++; }
            else { sum_right_dx += dx; n_right++; }
        }
    }

    result.num_inliers = match_idx;
    if (match_idx > 0) {
        result.mean_dx = sum_dx / match_idx;
        result.mean_dy = sum_dy / match_idx;
        result.top_dy = n_top > 0 ? sum_top_dy / n_top : 0;
        result.bottom_dy = n_bot > 0 ? sum_bot_dy / n_bot : 0;
        result.left_dx = n_left > 0 ? sum_left_dx / n_left : 0;
        result.right_dx = n_right > 0 ? sum_right_dx / n_right : 0;
    }

    printf("    [FeatureMatch] contour=%d sampled=%d matched=%d offset=(%.1f,%.1f)px\n",
           n_contour, n_samples, match_idx, result.mean_dx, result.mean_dy);

    return result;
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

    float f_proj = RENDER_SIZE / (2.0f * tanf(deg2rad(g_tune.camera_fov / 2.0f)));

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

        {
            float sl = g_tune.clip_slope_x;
            if (tv[a].y < params.clip_y + sl*tv[a].x || tv[b].y < params.clip_y + sl*tv[b].x || tv[c].y < params.clip_y + sl*tv[c].x)
                continue;
        }

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
        // Back-face culling: skip faces pointing away from camera
        if (denom > 0) continue;
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
// AI init JSON loader
// =============================================================================

bool load_ai_init_json(const std::string& path, Params& out) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    cv::FileNode root = fs["init_pose"];
    if (root.empty()) return false;

    auto read_num = [&](const char* key, float& dst) -> bool {
        cv::FileNode n = root[key];
        if (n.empty()) return false;
        dst = static_cast<float>((double)n);
        return true;
    };

    if (!read_num("theta", out.theta)) return false;
    if (!read_num("phi", out.phi)) return false;
    if (!read_num("roll", out.roll)) return false;
    if (!read_num("cam_x", out.cam_x)) return false;
    if (!read_num("cam_y", out.cam_y)) return false;
    if (!read_num("cam_z", out.cam_z)) return false;
    if (!read_num("clip_y", out.clip_y)) return false;

    out.iou = 0.0f;
    return true;
}

bool load_tune_config_json(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    auto read_float = [&](const char* key, float& dst) {
        cv::FileNode n = fs[key];
        if (!n.empty()) dst = static_cast<float>((double)n);
    };
    auto read_int = [&](const char* key, int& dst) {
        cv::FileNode n = fs[key];
        if (!n.empty()) dst = static_cast<int>((double)n);
    };

    read_float("edge_tiebreaker_scale", g_tune.edge_tiebreaker_scale);
    read_float("sym_contour_weight", g_tune.sym_contour_weight);
    read_float("contour_match_threshold", g_tune.contour_match_threshold_px);
    read_float("phase6_iou_tolerance", g_tune.phase6_iou_decrease_tolerance);
    read_float("phase6_r2_rot_step", g_tune.phase6_r2_rot_step);
    read_float("phase6_r2_cam_step", g_tune.phase6_r2_cam_step);
    read_int("phase6_enable_round4", g_tune.phase6_enable_round4);
    read_float("phase6_r3_rot_step", g_tune.phase6_r3_rot_step);
    read_float("phase6_r3_rot_range", g_tune.phase6_r3_rot_range);
    read_float("phase6_r3_cam_step", g_tune.phase6_r3_cam_step);

    read_float("boundary_weight_factor", g_tune.boundary_weight_factor);
    read_float("boundary_weight_distance", g_tune.boundary_weight_distance_px);

    read_float("phase7_match_threshold", g_tune.phase7_match_threshold);
    read_float("phase7_confidence_scale", g_tune.phase7_confidence_scale);
    read_float("phase7_base_lr", g_tune.phase7_base_lr);
    read_float("phase7_rotation_scale_factor", g_tune.phase7_rotation_scale_factor);

    read_float("phase4_range", g_tune.phase4_range_deg);
    read_float("phase4_step", g_tune.phase4_step_deg);

    read_int("norm_size", g_tune.norm_size);
    read_float("bbox_margin_ratio", g_tune.bbox_margin_ratio);
    read_float("mask_threshold", g_tune.mask_threshold);
    read_float("camera_fov", g_tune.camera_fov);
    read_float("clip_slope_x", g_tune.clip_slope_x);

    g_tune.sym_contour_weight = std::max(0.0f, std::min(1.0f, g_tune.sym_contour_weight));
    g_tune.phase6_iou_decrease_tolerance = std::max(0.0f, g_tune.phase6_iou_decrease_tolerance);
    g_tune.phase6_r2_rot_step = std::max(0.002f, g_tune.phase6_r2_rot_step);
    g_tune.phase6_r2_cam_step = std::max(0.0001f, g_tune.phase6_r2_cam_step);
    g_tune.phase6_enable_round4 = g_tune.phase6_enable_round4 > 0 ? 1 : 0;
    g_tune.phase6_r3_rot_step = std::max(0.001f, g_tune.phase6_r3_rot_step);
    g_tune.phase6_r3_rot_range = std::max(g_tune.phase6_r3_rot_step, g_tune.phase6_r3_rot_range);
    g_tune.phase6_r3_cam_step = std::max(0.0001f, g_tune.phase6_r3_cam_step);
    g_tune.boundary_weight_factor = std::max(0.0f, g_tune.boundary_weight_factor);
    g_tune.boundary_weight_distance_px = std::max(0.1f, g_tune.boundary_weight_distance_px);
    g_tune.phase7_match_threshold = std::max(0.0f, std::min(1.0f, g_tune.phase7_match_threshold));
    g_tune.phase7_confidence_scale = std::max(1.0f, g_tune.phase7_confidence_scale);
    g_tune.phase4_step_deg = std::max(0.01f, g_tune.phase4_step_deg);
    g_tune.phase4_range_deg = std::max(g_tune.phase4_step_deg, g_tune.phase4_range_deg);
    g_tune.norm_size = std::max(64, g_tune.norm_size);
    g_tune.bbox_margin_ratio = std::max(0.0f, g_tune.bbox_margin_ratio);
    g_tune.mask_threshold = std::max(0.0f, g_tune.mask_threshold);
    g_tune.camera_fov = std::max(20.0f, std::min(90.0f, g_tune.camera_fov));
    g_tune.clip_slope_x = std::max(-1.0f, std::min(1.0f, g_tune.clip_slope_x));

    return true;
}

void print_tune_config_summary() {
    printf("  [Tune] edge_tiebreaker_scale=%.5f sym_contour_weight=%.3f contour_match_threshold=%.2f\n",
           g_tune.edge_tiebreaker_scale, g_tune.sym_contour_weight, g_tune.contour_match_threshold_px);
    printf("  [Tune] phase6_iou_tolerance=%.6f phase6_r2_rot_step=%.5f phase6_r2_cam_step=%.5f\n",
           g_tune.phase6_iou_decrease_tolerance, g_tune.phase6_r2_rot_step, g_tune.phase6_r2_cam_step);
    printf("  [Tune] phase6_round4=%d phase6_r3_rot_step=%.5f phase6_r3_rot_range=%.5f phase6_r3_cam_step=%.5f\n",
           g_tune.phase6_enable_round4, g_tune.phase6_r3_rot_step,
           g_tune.phase6_r3_rot_range, g_tune.phase6_r3_cam_step);
    printf("  [Tune] boundary_weight_factor=%.3f boundary_weight_distance=%.2f\n",
           g_tune.boundary_weight_factor, g_tune.boundary_weight_distance_px);
    printf("  [Tune] phase7_match_threshold=%.3f confidence_scale=%.2f base_lr=%.3f rot_scale=%.2f\n",
           g_tune.phase7_match_threshold, g_tune.phase7_confidence_scale,
           g_tune.phase7_base_lr, g_tune.phase7_rotation_scale_factor);
    printf("  [Tune] phase4_range=%.3f phase4_step=%.3f norm_size=%d bbox_margin=%.3f mask_threshold=%.1f fov=%.2f\n",
           g_tune.phase4_range_deg, g_tune.phase4_step_deg, g_tune.norm_size,
           g_tune.bbox_margin_ratio, g_tune.mask_threshold, g_tune.camera_fov);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("Pose Match Pipeline (C++ + OpenMP)\n");
    printf("============================================================\n");

    Timer total_timer("Total");

    std::string ai_init_path;
    std::string tune_config_path;
    int requested_threads = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printf("Usage: pose_match.exe [--ai-init <path_to_cpp_init.json>] [--tune-config <path_to_tune.json>]\n");
            printf("                      [--out-dir <path>] [--threads <n>] [--fast]\n");
            printf("                      [--input-image <path>] [--ignore-rect x0,y0,x1,y1]\n");
            printf("                      [--no-phase7] [--no-phase7-images] [--no-final-overlay]\n");
            return 0;
        }
        if (arg == "--ai-init" && i + 1 < argc) {
            ai_init_path = argv[++i];
            continue;
        }
        if (arg == "--tune-config" && i + 1 < argc) {
            tune_config_path = argv[++i];
            continue;
        }
        if (arg == "--out-dir" && i + 1 < argc) {
            g_runtime.output_dir = argv[++i];
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            try {
                requested_threads = std::stoi(argv[++i]);
            } catch (...) {
                printf("WARNING: Invalid --threads value. Ignored.\n");
                requested_threads = 0;
            }
            continue;
        }
        if (arg == "--input-image" && i + 1 < argc) {
            INPUT_IMAGE = argv[++i];
            continue;
        }
        if (arg == "--ignore-rect" && i + 1 < argc) {
            // Parse "x0,y0,x1,y1"
            std::string val = argv[++i];
            int x0, y0, x1, y1;
            if (sscanf(val.c_str(), "%d,%d,%d,%d", &x0, &y0, &x1, &y1) == 4) {
                g_runtime.ignore_rects.push_back(cv::Rect(x0, y0, x1 - x0, y1 - y0));
                printf("  Ignore rect: (%d,%d)-(%d,%d)\n", x0, y0, x1, y1);
            } else {
                printf("WARNING: Invalid --ignore-rect format '%s'. Expected x0,y0,x1,y1\n", val.c_str());
            }
            continue;
        }
        if (arg == "--ignore-above-y" && i + 1 < argc) {
            g_runtime.ignore_above_y = std::stoi(argv[++i]);
            printf("  Ignore above y=%d\n", g_runtime.ignore_above_y);
            continue;
        }
        if (arg == "--ignore-below-y" && i + 1 < argc) {
            g_runtime.ignore_below_y = std::stoi(argv[++i]);
            printf("  Ignore below y=%d\n", g_runtime.ignore_below_y);
            continue;
        }
        if (arg == "--ignore-left-x" && i + 1 < argc) {
            g_runtime.ignore_left_x = std::stoi(argv[++i]);
            printf("  Ignore left of x=%d\n", g_runtime.ignore_left_x);
            continue;
        }
        if (arg == "--ignore-right-x" && i + 1 < argc) {
            g_runtime.ignore_right_x = std::stoi(argv[++i]);
            printf("  Ignore right of x=%d\n", g_runtime.ignore_right_x);
            continue;
        }
        if (arg == "--fast") {
            g_runtime.save_phase7_images = false;
            g_runtime.save_final_overlay = false;
            continue;
        }
        if (arg == "--no-phase7") {
            g_runtime.enable_phase7 = false;
            continue;
        }
        if (arg == "--no-phase7-images") {
            g_runtime.save_phase7_images = false;
            continue;
        }
        if (arg == "--no-final-overlay") {
            g_runtime.save_final_overlay = false;
            continue;
        }
        printf("WARNING: Unknown argument ignored: %s\n", arg.c_str());
    }

    if (requested_threads > 0) {
        omp_set_num_threads(requested_threads);
        printf("  OpenMP threads forced: %d\n", requested_threads);
    }
    printf("  OpenMP max threads: %d\n", omp_get_max_threads());

    fs::create_directories(g_runtime.output_dir);
    printf("  Input image: %s\n", INPUT_IMAGE.c_str());
    printf("  Output dir: %s\n", g_runtime.output_dir.c_str());
    if (!g_runtime.enable_phase7) {
        printf("  Runtime: Phase7 disabled\n");
    } else if (!g_runtime.save_phase7_images) {
        printf("  Runtime: Phase7 image export disabled\n");
    }
    if (!g_runtime.save_final_overlay) {
        printf("  Runtime: Final textured overlay disabled\n");
    }

    if (!tune_config_path.empty()) {
        if (load_tune_config_json(tune_config_path)) {
            printf("  Loaded tune config: %s\n", tune_config_path.c_str());
            print_tune_config_summary();
        } else {
            printf("  WARNING: Failed to parse tune config: %s\n", tune_config_path.c_str());
            printf("           Continuing with default tuning parameters.\n");
        }
    }

    Params ai_init_params = {};
    bool has_ai_init = false;
    if (!ai_init_path.empty()) {
        has_ai_init = load_ai_init_json(ai_init_path, ai_init_params);
        if (has_ai_init) {
            printf("  Loaded AI init JSON: %s\n", ai_init_path.c_str());
            printf("  AI init theta=%.3f phi=%.3f roll=%.3f cam=(%.3f,%.3f,%.3f) clip_y=%.3f\n",
                ai_init_params.theta, ai_init_params.phi, ai_init_params.roll,
                ai_init_params.cam_x, ai_init_params.cam_y, ai_init_params.cam_z,
                ai_init_params.clip_y);
        } else {
            printf("  WARNING: Failed to parse AI init JSON: %s\n", ai_init_path.c_str());
            printf("           Falling back to depth-based initialization.\n");
        }
    }

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
    cv::threshold(gray, mask_raw, g_tune.mask_threshold, 255, cv::THRESH_BINARY);
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
    BBox bb_target_e = expand_bbox(bb_target, W, H, g_tune.bbox_margin_ratio);
    int norm_size = std::max(64, g_tune.norm_size);
    cv::Mat target_norm = crop_and_resize(mask_filled, bb_target_e, norm_size);

    // Create ignore mask in normalized target space
    {
        bool has_rects = !g_runtime.ignore_rects.empty();
        bool has_axis = (g_runtime.ignore_above_y >= 0 || g_runtime.ignore_below_y >= 0 ||
                         g_runtime.ignore_left_x >= 0 || g_runtime.ignore_right_x >= 0);
        if (has_rects || has_axis) {
            cv::Mat ignore_img = cv::Mat::zeros(H, W, CV_8UC1);
            // Rects
            for (const auto& r : g_runtime.ignore_rects) {
                cv::Rect safe_r = r & cv::Rect(0, 0, W, H);
                ignore_img(safe_r) = 255;
            }
            // Axis cuts
            if (g_runtime.ignore_above_y >= 0) {
                int ya = std::min(g_runtime.ignore_above_y, H);
                if (ya > 0) ignore_img(cv::Rect(0, 0, W, ya)) = 255;
            }
            if (g_runtime.ignore_below_y >= 0) {
                int yb = std::max(0, g_runtime.ignore_below_y);
                if (yb < H) ignore_img(cv::Rect(0, yb, W, H - yb)) = 255;
            }
            if (g_runtime.ignore_left_x >= 0) {
                int xl = std::min(g_runtime.ignore_left_x, W);
                if (xl > 0) ignore_img(cv::Rect(0, 0, xl, H)) = 255;
            }
            if (g_runtime.ignore_right_x >= 0) {
                int xr = std::max(0, g_runtime.ignore_right_x);
                if (xr < W) ignore_img(cv::Rect(xr, 0, W - xr, H)) = 255;
            }
            g_ignore_norm = crop_and_resize(ignore_img, bb_target_e, norm_size);
            cv::threshold(g_ignore_norm, g_ignore_norm, 128, 255, cv::THRESH_BINARY);
            int ignored_px = cv::countNonZero(g_ignore_norm);
            printf("  Ignore mask: %d pixels ignored in norm space\n", ignored_px);
        }
    }

    // 3. Initial angle estimation (AI init or depth fallback)
    printf("\n[3/6] Initial angle estimation...\n");

    AngleEstimate est;
    if (has_ai_init) {
        est = {ai_init_params.theta, ai_init_params.phi, ai_init_params.roll};
        printf("  Using AI init angles: theta=%.1f, phi=%.1f, roll=%.1f\n", est.theta, est.phi, est.roll);
    } else {
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

        est = estimate_angles_from_depth(depth_bgr, depth_mask);
        printf("  Depth-estimated: theta=%.1f, phi=%.1f, roll=%.1f\n", est.theta, est.phi, est.roll);
    }

    // 4. Load 3D model
    printf("\n[4/6] Loading 3D model...\n");
    Mesh mesh = load_obj(MODEL_PATH);
    normalize_vertices(mesh);
    printf("  Low-poly: %zu vertices, %zu faces\n", mesh.vertices.size(), mesh.faces.size());

    const bool need_textured_assets = g_runtime.enable_phase7 || g_runtime.save_final_overlay;
    cv::Mat texture;
    Mesh mesh_hi;
    bool has_mesh_hi = false;
    if (need_textured_assets) {
        texture = imread_unicode(TEXTURE_PATH);
        if (!texture.empty()) {
            printf("  Texture: %dx%d\n", texture.cols, texture.rows);
        } else {
            printf("  WARNING: Cannot load texture: %s\n", TEXTURE_PATH.c_str());
        }

        mesh_hi = load_obj(MODEL_HI_PATH);
        if (!mesh_hi.vertices.empty()) {
            normalize_vertices(mesh_hi);
            has_mesh_hi = true;
            printf("  Hi-poly: %zu vertices, %zu faces, %zu uvs\n",
                   mesh_hi.vertices.size(), mesh_hi.faces.size(), mesh_hi.uvs.size());
        } else {
            printf("  WARNING: Failed to load hi-poly mesh: %s\n", MODEL_HI_PATH.c_str());
        }
    } else {
        printf("  Textured assets skipped (Phase7 and final overlay are disabled)\n");
    }

    // 5. Optimization (with feature-point refinement in Phase 6)
    printf("\n[5/6] Optimization...\n");
    Timer opt_timer("Optimization");
    float init_cam_x = has_ai_init ? ai_init_params.cam_x : -0.45f;
    float init_cam_y = has_ai_init ? ai_init_params.cam_y : -0.45f;
    float init_cam_z = has_ai_init ? ai_init_params.cam_z : 2.72f;
    float init_clip_y = has_ai_init ? ai_init_params.clip_y : -0.39f;

    const Mesh* mesh_hi_ptr = has_mesh_hi ? &mesh_hi : nullptr;
    Params best = optimize(mesh, target_norm, est.theta, est.phi, est.roll,
                           init_cam_x, init_cam_y, init_cam_z, init_clip_y,
                           texture, original, mask_filled, mesh_hi_ptr);
    printf("\n  === BEST RESULT ===\n");
    printf("  theta=%.4f, phi=%.4f, roll=%.4f\n", best.theta, best.phi, best.roll);
    printf("  cam=(%.4f, %.4f, %.4f), clip_y=%.4f\n", best.cam_x, best.cam_y, best.cam_z, best.clip_y);
    printf("  IoU=%.4f%%\n", best.iou * 100);
    printf("  FOV=%.2f\n", g_tune.camera_fov);

    // Save diff mask visualization (Green=match, Red=target only, Blue=render only)
    {
        cv::Mat sil_best = render_silhouette(mesh, best.theta, best.phi, best.roll,
                                              best.cam_x, best.cam_y, best.cam_z, RENDER_SIZE, best.clip_y);
        BBox bb_sil = bbox_from_mask(sil_best);
        if (bb_sil.x0 >= 0) {
            BBox bb_sil_e = expand_bbox(bb_sil, RENDER_SIZE, RENDER_SIZE, g_tune.bbox_margin_ratio);
            int ns = std::max(64, g_tune.norm_size);
            cv::Mat sil_norm = crop_and_resize(sil_best, bb_sil_e, ns);

            // Diff visualization at norm_size
            cv::Mat diff_img(ns, ns, CV_8UC3, cv::Scalar(0,0,0));
            bool has_ign = !g_ignore_norm.empty();
            int fn_count = 0, fp_count = 0, tp_count = 0, ign_count = 0;
            for (int y = 0; y < ns; y++) {
                const uchar* t = target_norm.ptr<uchar>(y);
                const uchar* r = sil_norm.ptr<uchar>(y);
                const uchar* ig = has_ign ? g_ignore_norm.ptr<uchar>(y) : nullptr;
                cv::Vec3b* d = diff_img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < ns; x++) {
                    if (ig && ig[x] > 0) {
                        // Ignored region: show in gray
                        bool tgt = t[x] > 0, rnd = r[x] > 0;
                        if (tgt || rnd) { d[x] = cv::Vec3b(128, 128, 128); }
                        ign_count++;
                        continue;
                    }
                    bool tgt = t[x] > 0, rnd = r[x] > 0;
                    if (tgt && rnd) { d[x] = cv::Vec3b(0, 180, 0); tp_count++; }      // Green: match
                    else if (tgt && !rnd) { d[x] = cv::Vec3b(0, 0, 255); fn_count++; } // Red: target only (FN)
                    else if (!tgt && rnd) { d[x] = cv::Vec3b(255, 80, 0); fp_count++; } // Blue: render only (FP)
                }
            }
            cv::imwrite(g_runtime.output_dir + "/diff_mask.png", diff_img);

            // Upscale diff for better visibility
            cv::Mat diff_big;
            cv::resize(diff_img, diff_big, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
            // Add legend text
            std::string legend = "Green=Match  Red=TargetOnly  Blue=RenderOnly";
            if (has_ign) legend += "  Gray=Ignored";
            cv::putText(diff_big, legend, cv::Point(10, 500),
                       cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(255,255,255), 1);
            char buf[128];
            snprintf(buf, sizeof(buf), "FN=%d FP=%d TP=%d Union=%d IoU=%.4f%%",
                    fn_count, fp_count, tp_count, fn_count+fp_count+tp_count, best.iou*100);
            cv::putText(diff_big, buf, cv::Point(10, 480),
                       cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1);
            if (has_ign) {
                char buf2[64];
                snprintf(buf2, sizeof(buf2), "Ignored=%d pixels", ign_count);
                cv::putText(diff_big, buf2, cv::Point(10, 460),
                           cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200,200,200), 1);
            }
            cv::imwrite(g_runtime.output_dir + "/diff_mask_large.png", diff_big);
            printf("  Diff mask: FN=%d FP=%d TP=%d (saved to diff_mask_large.png)\n", fn_count, fp_count, tp_count);
            if (has_ign) printf("  Ignored pixels: %d\n", ign_count);
        }
    }

    // 6. Textured overlay (hi-poly model already loaded above)
    printf("\n[6/6] Textured overlay...\n");

    if (!g_runtime.save_final_overlay) {
        printf("  Skipped (--no-final-overlay)\n");
    } else if (texture.empty() || !has_mesh_hi) {
        printf("  WARNING: Missing textured assets, skipping overlay\n");
    } else {
        // Disable clip_y for textured rendering (show full model)
        Params render_params = best;
        render_params.clip_y = -10.0f;  // effectively no clipping
        render_textured_overlay(mesh_hi, texture, original, mask_filled, render_params, g_runtime.output_dir);
    }

    // Save result
    std::ofstream result(g_runtime.output_dir + "/result.txt");
    result << "=== Pose Match C++ Result ===\n\n";
    result << "[Best Parameters]\n";
    result << "  theta: " << best.theta << "\n";
    result << "  phi: " << best.phi << "\n";
    result << "  roll: " << best.roll << "\n";
    result << "  cam_x: " << best.cam_x << "\n";
    result << "  cam_y: " << best.cam_y << "\n";
    result << "  cam_z: " << best.cam_z << "\n";
    result << "  clip_y: " << best.clip_y << "\n";
    result << "  camera_fov: " << g_tune.camera_fov << "\n\n";
    result << "[Result]\n";
    result << "  IoU: " << (best.iou * 100) << "%\n";

    // Compute and save pose quality (error concentration)
    {
        cv::Mat sil_q = render_silhouette(mesh, best.theta, best.phi, best.roll,
                                          best.cam_x, best.cam_y, best.cam_z, RENDER_SIZE, best.clip_y);
        BBox bb_q = bbox_from_mask(sil_q);
        if (bb_q.x0 >= 0) {
            BBox bbe_q = expand_bbox(bb_q, RENDER_SIZE, RENDER_SIZE, g_tune.bbox_margin_ratio);
            int ns = std::max(64, g_tune.norm_size);
            cv::Mat sil_norm_q = crop_and_resize(sil_q, bbe_q, ns);
            PoseQuality pq = compute_pose_quality(target_norm, sil_norm_q);
            result << "\n[Quality]\n";
            result << "  Gini: " << pq.gini << "\n";
            result << "  MeanBoundaryDist: " << pq.mean_boundary_dist << "\n";
            result << "  P90BoundaryDist: " << pq.p90_boundary_dist << "\n";
            result << "  ErrorRatio: " << pq.error_ratio << "\n";
            result << "  ActiveSectors: " << pq.active_sectors << "/" << pq.total_sectors << "\n";
            result << "  FN: " << pq.fn_count << "\n";
            result << "  FP: " << pq.fp_count << "\n";
            result << "  CombinedScore: " << pq.combined_score << "\n";
        }
    }
    result.close();

    printf("\n============================================================\n");
    printf("Output: %s\n", g_runtime.output_dir.c_str());
    printf("============================================================\n");

    return 0;
}







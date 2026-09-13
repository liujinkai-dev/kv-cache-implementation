/*
 * KV Cache 基础概念演示 (C++ 版)
 * ================================
 * 对应 python/kv_cache_basics.py，用 C++ 展示同样的概念。
 *
 * 编译: g++ -O2 -o kv_cache_demo kv_cache_demo.cpp
 * 运行: ./kv_cache_demo
 *
 * 核心概念:
 *   Transformer 自回归生成时，每个 token 需要和之前所有 token 做 attention。
 *   Attention = softmax(Q @ K^T / sqrt(d)) @ V
 *   - 无 Cache: 每步重新计算所有 token 的 K, V → O(n^2)
 *   - 有 Cache: 只算新 token 的 K, V，历史的从缓存取 → O(n)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace std;

const int D_MODEL = 64;
const int D_HEAD = 64;
const int SEQ_LEN = 128;

// 简单矩阵: 用 vector<vector<float>> 表示
using Matrix = vector<vector<float>>;
using Vector = vector<float>;

// 生成随机矩阵
Matrix random_matrix(int rows, int cols, float scale = 0.1f) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> d(0, scale);
    Matrix m(rows, Vector(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[i][j] = d(gen);
    return m;
}

// 矩阵 × 向量: (rows, cols) @ (cols,) → (rows,)
Vector mat_vec_mul(const Matrix& W, const Vector& x) {
    int rows = W.size(), cols = W[0].size();
    Vector out(rows, 0.0f);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[i] += W[i][j] * x[j];
    return out;
}

// 向量 × 矩阵: (cols,) @ (rows, cols)^T → 等价于 W^T @ x → (rows,)
// 这里 W 是 (d_model, d_head), x 是 (d_model,), 输出 (d_head,)
Vector transposed_mat_vec(const Matrix& W, const Vector& x) {
    int d_model = W.size(), d_head = W[0].size();
    Vector out(d_head, 0.0f);
    for (int j = 0; j < d_head; j++)
        for (int i = 0; i < d_model; i++)
            out[j] += W[i][j] * x[i];
    return out;
}

// softmax
void softmax_inplace(Vector& x) {
    float max_val = *max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (auto& v : x) { v = expf(v - max_val); sum += v; }
    for (auto& v : x) v /= sum;
}

// attention: q (d_head,), k (T, d_head), v (T, d_head) → (d_head,)
Vector attention(const Vector& q, const vector<Vector>& k, const vector<Vector>& v) {
    int T = k.size();
    int dh = q.size();
    Vector scores(T, 0.0f);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < dh; j++)
            scores[t] += q[j] * k[t][j];
    float scale = 1.0f / sqrtf((float)dh);
    for (auto& s : scores) s *= scale;
    softmax_inplace(scores);
    Vector out(dh, 0.0f);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < dh; j++)
            out[j] += scores[t] * v[t][j];
    return out;
}

// ── 方式一：无 KV Cache ──
vector<Vector> generate_without_cache(const vector<Vector>& tokens) {
    Matrix W_Q = random_matrix(D_MODEL, D_HEAD);
    Matrix W_K = random_matrix(D_MODEL, D_HEAD);
    Matrix W_V = random_matrix(D_MODEL, D_HEAD);
    vector<Vector> outputs;
    for (int t = 0; t < (int)tokens.size(); t++) {
        // 每步重新计算所有 token 的 K, V
        vector<Vector> k_all, v_all;
        for (int i = 0; i <= t; i++) {
            k_all.push_back(transposed_mat_vec(W_K, tokens[i]));
            v_all.push_back(transposed_mat_vec(W_V, tokens[i]));
        }
        Vector q = transposed_mat_vec(W_Q, tokens[t]);
        outputs.push_back(attention(q, k_all, v_all));
    }
    return outputs;
}

// ── 方式二：有 KV Cache ──
vector<Vector> generate_with_cache(const vector<Vector>& tokens) {
    Matrix W_Q = random_matrix(D_MODEL, D_HEAD);
    Matrix W_K = random_matrix(D_MODEL, D_HEAD);
    Matrix W_V = random_matrix(D_MODEL, D_HEAD);
    vector<Vector> k_cache, v_cache;  // KV Cache
    vector<Vector> outputs;
    for (int t = 0; t < (int)tokens.size(); t++) {
        // 只算当前 token 的 K, V
        Vector k_new = transposed_mat_vec(W_K, tokens[t]);
        Vector v_new = transposed_mat_vec(W_V, tokens[t]);
        k_cache.push_back(k_new);  // 存入 cache
        v_cache.push_back(v_new);  // 存入 cache
        Vector q = transposed_mat_vec(W_Q, tokens[t]);
        // 用 cache 中的所有 K, V
        outputs.push_back(attention(q, k_cache, v_cache));
    }
    return outputs;
}

int main() {
    cout << "============================================================" << endl;
    cout << "KV Cache 概念演示 (C++ 版)" << endl;
    cout << "============================================================" << endl;

    // 生成随机 token 序列
    mt19937 gen(42);
    normal_distribution<float> dist(0, 1);
    vector<Vector> tokens(SEQ_LEN, Vector(D_MODEL));
    for (auto& t : tokens)
        for (auto& v : t) v = dist(gen);

    cout << "\n配置: d_model=" << D_MODEL << ", seq_len=" << SEQ_LEN << endl;

    // 性能对比
    cout << "\n[性能对比]" << endl;

    auto t1 = chrono::high_resolution_clock::now();
    auto out_no = generate_without_cache(tokens);
    auto t2 = chrono::high_resolution_clock::now();
    auto out_yes = generate_with_cache(tokens);
    auto t3 = chrono::high_resolution_clock::now();

    double ms_no = chrono::duration<double, milli>(t2 - t1).count();
    double ms_yes = chrono::duration<double, milli>(t3 - t2).count();

    cout << fixed << setprecision(1);
    cout << "    无 Cache: " << ms_no << " ms" << endl;
    cout << "    有 Cache: " << ms_yes << " ms" << endl;
    cout << "    加速比:   " << (ms_no / ms_yes) << "x" << endl;

    // 逐步计算量
    cout << "\n[每步 K/V 计算量对比 (前 10 步)]" << endl;
    cout << "    步数 |   无Cache   |   有Cache" << endl;
    cout << "    ----+-------------+----------" << endl;
    for (int t = 1; t <= 10; t++) {
        int no_ops = t * D_MODEL * D_HEAD;
        int yes_ops = 1 * D_MODEL * D_HEAD;
        cout << "    " << setw(3) << t << " | " << setw(11) << no_ops << " | " << setw(9) << yes_ops << endl;
    }

    cout << "\n结论: 序列越长，KV Cache 优势越大。" << endl;
    cout << "      无 Cache: O(n^2)，有 Cache: O(n)。" << endl;
    return 0;
}

"""
KV Cache 基础概念演示
=====================
本文件只关注 KV Cache 的核心概念，不涉及训练、反向传播等内容。
用最简单的方式展示：有 cache vs 无 cache 的区别。

依赖: numpy (pip install numpy)

运行: python kv_cache_basics.py

核心概念:
--------
Transformer 生成文本时，每次生成一个 token（自回归生成）。
每个 token 需要和之前所有 token 做 attention 计算。
Attention 公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

问题: 每生成一个新 token，如果重新计算所有历史 token 的 K 和 V，太慢了。
解决: 把已经算过的 K 和 V 存起来（缓存），下次只算新 token 的 K 和 V。

  无 Cache: 第 t 步要计算 t 个 token 的 K, V → 总计算量 O(n^2)
  有 Cache: 第 t 步只算 1 个新 token 的 K, V → 总计算量 O(n)
"""
import numpy as np
import time
import warnings

np.random.seed(42)
# 抑制 Apple Silicon 上 float32 BLAS 的误报警告（计算结果正确）
np.seterr(all='ignore')

# ── 模型参数（随机初始化，仅用于演示前向传播） ──
d_model = 256         # 嵌入维度（增大以让计算差异更明显）
d_head = 256          # 注意力头维度（单头）
seq_len = 256         # 要生成的序列长度

# 随机初始化权重矩阵（缩小范围避免数值溢出）
W_Q = (np.random.randn(d_model, d_head) * 0.02).astype(np.float32)
W_K = (np.random.randn(d_model, d_head) * 0.02).astype(np.float32)
W_V = (np.random.randn(d_model, d_head) * 0.02).astype(np.float32)

# 生成随机 token 序列（缩小范围）
tokens = [(np.random.randn(d_model) * 0.1).astype(np.float32) for _ in range(seq_len)]


def softmax(x):
    """数值稳定的 softmax"""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(q, k, v):
    """
    单个 token 的 attention 计算。
    q: (d_head,)        当前 token 的 Query
    k: (T, d_head)      所有 token 的 Key（包括历史的）
    v: (T, d_head)      所有 token 的 Value（包括历史的）
    返回: (d_head,)     attention 输出
    """
    # Q @ K^T / sqrt(d) → (T,)
    scores = (q @ k.T) / np.sqrt(d_head)
    weights = softmax(scores)
    # weights @ V → (d_head,)
    return weights @ v


# ════════════════════════════════════════════════════════
# 方式一：无 KV Cache —— 每步都重新计算所有 token 的 K 和 V
# ════════════════════════════════════════════════════════
def generate_without_cache(tokens):
    """
    每生成一个新 token，都要把之前所有 token 重新过一遍 W_K 和 W_V。
    第 t 步：计算 t 个 token 的 K, V → 越来越慢。
    """
    all_tokens = np.stack(tokens)  # (seq_len, d_model) 一次性转成矩阵
    outputs = []
    for t in range(len(tokens)):
        # 每次都要重新计算 ALL tokens 的 K 和 V（浪费！）
        k = all_tokens[:t+1] @ W_K  # (t+1, d_head)  ← 重复计算！
        v = all_tokens[:t+1] @ W_V  # (t+1, d_head)  ← 重复计算！
        q = tokens[t] @ W_Q          # (d_head,)
        out = attention(q, k, v)
        outputs.append(out)
    return outputs


# ════════════════════════════════════════════════════════
# 方式二：有 KV Cache —— 只算新 token 的 K 和 V，历史的复用
# ════════════════════════════════════════════════════════
def generate_with_cache(tokens):
    """
    把已经算过的 K 和 V 存起来（缓存），每步只算 1 个新 token 的 K 和 V。
    第 t 步：只计算 1 个 token 的 K, V → 速度恒定。
    """
    # 预分配 cache 空间（实际框架中也是预分配的）
    k_cache = np.zeros((seq_len, d_head), dtype=np.float32)
    v_cache = np.zeros((seq_len, d_head), dtype=np.float32)
    outputs = []
    for t in range(len(tokens)):
        # 只计算当前 token 的 K 和 V（不重复计算历史的）
        k_cache[t] = tokens[t] @ W_K  # (d_head,)  ← 只算 1 个！
        v_cache[t] = tokens[t] @ W_V  # (d_head,)  ← 只算 1 个！
        q = tokens[t] @ W_Q            # (d_head,)
        # 用 cache 中的所有 K, V 做 attention
        out = attention(q, k_cache[:t+1], v_cache[:t+1])
        outputs.append(out)
    return outputs


# ── 验证两种方式结果完全一致 ──
print("=" * 60)
print("KV Cache 概念演示")
print("=" * 60)

print(f"\n配置: d_model={d_model}, seq_len={seq_len}")
print()

# 验证结果一致性
print("[1] 验证两种方式结果一致...")
out_no_cache = generate_without_cache(tokens)
out_with_cache = generate_with_cache(tokens)
diff = max(np.max(np.abs(a - b)) for a, b in zip(out_no_cache, out_with_cache))
print(f"    最大差异: {diff:.2e} (应接近 0，说明两种方式等价)")
print()

# 性能对比
print("[2] 性能对比...")

start = time.time()
for _ in range(3):  # 跑 3 轮取平均
    _ = generate_without_cache(tokens)
t_no = (time.time() - start) / 3

start = time.time()
for _ in range(3):
    _ = generate_with_cache(tokens)
t_yes = (time.time() - start) / 3

print(f"    无 Cache: {t_no*1000:.1f} ms")
print(f"    有 Cache: {t_yes*1000:.1f} ms")
print(f"    加速比:   {t_no/t_yes:.2f}x")
print()

# 展示逐步增长的计算量
print("[3] 每步计算量对比（前 10 步）:")
print(f"    {'步数':>4} | {'无Cache K/V计算量':>20} | {'有Cache K/V计算量':>20}")
print(f"    {'-'*4}-+-{'-'*20}-+-{'-'*20}")
for t in range(1, 11):
    no_cache_ops = t * d_model * d_head  # 每步算 t 个 token
    with_cache_ops = 1 * d_model * d_head  # 每步只算 1 个 token
    print(f"    {t:4d} | {no_cache_ops:20d} | {with_cache_ops:20d}")

print()
print("结论: 序列越长，KV Cache 的优势越明显。")
print("      无 Cache 计算量随序列平方增长 O(n^2)，")
print("      有 Cache 计算量随序列线性增长 O(n)。")

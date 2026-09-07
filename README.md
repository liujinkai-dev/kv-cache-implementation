# KV Cache 学习项目

LLM 推理中 KV Cache 的学习与实现，从零开始，逐步深入。

## 什么是 KV Cache？

Transformer 自回归生成（逐个 token 生成文本）时，每个新 token 需要和之前所有 token 做 Attention。Attention 需要 Q (Query)、K (Key)、V (Value) 三个矩阵。

- **无 Cache**：每生成一个新 token，重新计算所有历史 token 的 K 和 V → 计算量 O(n²)
- **有 Cache**：把历史 token 的 K 和 V 存起来，只算新 token 的 K 和 V → 计算量 O(n)

这是 LLM 推理加速最核心的技术之一。

## 项目结构

```
KV Cache/
├── python/
│   ├── kv_cache_basics.py   # 入门 demo：用 NumPy 演示 KV Cache 核心概念
│   └── microgpt.py           # karpathy 的纯 Python GPT 实现（含 KV Cache）
├── cpp/
│   └── kv_cache_demo.cpp     # C++ 版 KV Cache 概念演示
├── .gitignore
└── README.md
```

## 快速开始

### 1. Python 入门 demo（推荐先看这个）

```bash
cd python
python kv_cache_basics.py
```

只关注 KV Cache 本身，不涉及训练。对比有无 Cache 的结果一致性和性能差异。

### 2. C++ 版 demo

```bash
cd cpp
g++ -O2 -o kv_cache_demo kv_cache_demo.cpp
./kv_cache_demo
```

### 3. karpathy microgpt.py（完整 GPT）

```bash
cd python
python microgpt.py
```

纯 Python 零依赖实现，包含训练 + 推理。KV Cache 体现在 `gpt()` 函数中的 `keys` 和 `values` 列表。运行较慢（纯 Python 标量级自动微分），耐心等待。

## 学习路径

1. **先跑 `kv_cache_basics.py`**：理解 KV Cache 的核心概念（为什么需要、怎么做、加速多少）
2. **跑 `kv_cache_demo.cpp`**：用 C++ 验证同样的概念
3. **读 `microgpt.py`**：在完整 GPT 中看 KV Cache 如何融入实际模型
4. 后期逐步优化：多头注意力、batch 推理、内存管理、PagedAttention 等

## 来源

- `microgpt.py` 来自 [@karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## License

MIT

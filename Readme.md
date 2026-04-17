# FlashAttention-2 — Triton Implementation

A from-scratch implementation of **FlashAttention-2** in OpenAI Triton, with full PyTorch autograd support, causal masking, and benchmarks against PyTorch SDPA.


---

## What this is

A high-performance GPU kernel implementation of memory-efficient attention designed for modern large language models.
Standard attention computes the full N×N attention matrix, which is O(N²) in memory. This becomes the bottleneck at long sequence lengths — not compute, but memory bandwidth.

FlashAttention solves this by never materializing the full matrix. Instead it:
- Tiles Q, K, V into blocks that fit in GPU SRAM
- Computes attention block-by-block using **online softmax**
- Fuses QKᵀ → softmax → V multiplication into a single Triton kernel

The result: **O(N) memory**, fewer memory roundtrips, and significant wall-clock speedup.

---

## Implementation

Core components are modularized for clarity and reuse:

```
flash-attention-v2-triton/
├── flash_attention/
│   ├── triton_impl.py     # Core Triton kernel (forward + backward)
│   ├── autograd.py        # PyTorch autograd wrapper
│   └── reference.py       # Naive O(N²) reference for correctness checks
├── benchmarks/
│   └── benchmark.py       # Latency + memory benchmarks
├── examples/
│   └── model.py           # Drop-in usage example
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install torch triton
```

```python
import torch
from flash_attention.autograd import FlashAttentionFunction

B, N, D = 2, 1024, 64  # batch, seq_len, head_dim

q = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)

# causal=True for autoregressive (decoder) attention
out = FlashAttentionFunction.apply(q, k, v, True)
```

---

## Run Benchmark

```bash
pip install -r requirements.txt
python -m benchmarks.benchmark
```

---

## 📊 Benchmark Results (Tesla T4, bfloat16)

### Forward + Backward Latency (ms)

#### Head Dim = 16

| Seq Len | PyTorch | FlashAttention | Speedup |
|--------|--------|----------------|--------|
| 128    | 0.7701 | 0.1499         | ~5.1×  |
| 256    | 0.7510 | 0.1635         | ~4.6×  |
| 512    | 0.8236 | 0.2245         | ~3.7×  |
| 1024   | 1.0151 | 0.6485         | ~1.6×  |

---

#### Head Dim = 32

| Seq Len | PyTorch | FlashAttention | Speedup |
|--------|--------|----------------|--------|
| 128    | 0.7502 | 0.1672         | ~4.5×  |
| 256    | 0.7301 | 0.1926         | ~3.8×  |
| 512    | 0.7818 | 0.2808         | ~2.8×  |
| 1024   | 1.0504 | 0.7396         | ~1.4×  |

### Memory (GB) — PyTorch vs FlashAttention

| Seq Len | PyTorch SDPA | This impl | Reduction |
|---------|-------------|-----------|-----------|
| 1024    | allocates N×N attention matrix | O(N) only | ~8× at seq=1K |
| 4096    | OOM on T4    | fits       | —         |

---

## Key implementation details

### Tiling
Rather than computing `softmax(QKᵀ)V` in one shot, we process blocks:
```
for each Q_block:
    for each K_block, V_block:
        scores = Q_block @ K_block.T
        update running (m, l, O) via online softmax
```
This keeps the working set in SRAM throughout.

### Online softmax
Maintains two running statistics per row without a second pass:
- `m` — running max (for numerical stability)
- `l` — running sum of exponentials (normalization factor)

On each new block, rescales the previous output and accumulates:
```
m_new = max(m, max(scores))
l_new = exp(m - m_new) * l + sum(exp(scores - m_new))
O = (exp(m - m_new) * l * O + exp(scores - m_new) @ V) / l_new
```

### Kernel fusion
QKᵀ, softmax, and the V multiplication are fused into a single Triton kernel. This eliminates multiple global memory roundtrips that standard attention incurs between each op.

---

## Correctness

Outputs are validated against the naive reference implementation:

```python
from flash_attention.reference import naive_attention
from flash_attention.autograd import FlashAttentionFunction

ref = naive_attention(q, k, v, causal=True)
out = FlashAttentionFunction.apply(q, k, v, True)

torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)  # bfloat16 tolerance
```

---

## Limitations

- `head_dim` must be a power of 2 (current kernel constraint)
- Single-head implementation — multi-head via batching over heads
- Performance degrades for `head_dim >= 64` on T4 due to SRAM limits
- No GQA / MQA support yet


---

## References

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (ICLR 2024)
- OpenAI, [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://triton-lang.org/)

---

## License

MIT

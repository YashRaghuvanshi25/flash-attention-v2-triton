import torch
import triton
import triton.testing
from flash_attention.autograd import FlashAttentionFunction


def benchmark_flash(B, N, D, dtype):

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA device required for benchmarking"
    torch.cuda.empty_cache()

    Q = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)

    def pytorch_forward():
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

        # causal mask (upper triangular)
        mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        P = torch.softmax(scores, dim=-1)
        out = torch.matmul(P, V)
        torch.cuda.synchronize()
        return out

    def flash_forward():
        out = FlashAttentionFunction.apply(Q, K, V, True)
        torch.cuda.synchronize()
        return out

    def pytorch_backward():
        Q.grad = K.grad = V.grad = None
        O = pytorch_forward()
        O.sum().backward()
        torch.cuda.synchronize()

    def flash_backward():
        Q.grad = K.grad = V.grad = None
        O = FlashAttentionFunction.apply(Q, K, V, True)
        O.sum().backward()
        torch.cuda.synchronize()

    def pytorch_full():
        Q.grad = K.grad = V.grad = None
        O = pytorch_forward()
        O.sum().backward()
        torch.cuda.synchronize()

    def flash_full():
        Q.grad = K.grad = V.grad = None
        O = FlashAttentionFunction.apply(Q, K, V, True)
        O.sum().backward()
        torch.cuda.synchronize()

    fwd_pt = triton.testing.do_bench(pytorch_forward, warmup=25, rep=100)
    fwd_flash = triton.testing.do_bench(flash_forward, warmup=25, rep=100)

    bwd_pt = triton.testing.do_bench(pytorch_backward, warmup=25, rep=100)
    bwd_flash = triton.testing.do_bench(flash_backward, warmup=25, rep=100)

    full_pt = triton.testing.do_bench(pytorch_full, warmup=25, rep=100)
    full_flash = triton.testing.do_bench(flash_full, warmup=25, rep=100)

    print(f"N={N}, D={D}, dtype={dtype}")
    print(f"{'Metric':<20} {'PyTorch':<15} {'FlashAttention':<15}")
    print(f"{'Forward':<20} {fwd_pt:<15.4f} {fwd_flash:<15.4f}")
    print(f"{'Backward':<20} {bwd_pt:<15.4f} {bwd_flash:<15.4f}")
    print(f"{'Forward+Backward':<20} {full_pt:<15.4f} {full_flash:<15.4f}")
    print("-" * 60)


if __name__ == "__main__":
    B = 1  # assignment requires batch size 1

    seq_lengths = [128, 256, 512, 1024]
    embed_dims = [16, 32]
    dtypes = [torch.float32, torch.bfloat16]

    for dtype in dtypes:
        print(f"\n===== DTYPE: {dtype} =====")
        for D in embed_dims:
            for N in seq_lengths:
                try:
                    benchmark_flash(B, N, D, dtype)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"N={N}, D={D}, dtype={dtype} → OOM")
                        torch.cuda.empty_cache()
                    else:
                        raise e
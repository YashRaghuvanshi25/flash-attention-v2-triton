

import torch
import triton
import triton.language as tl


# ================================
# Triton Kernel
# ================================

@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,
    B, N,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each Triton program (block) processes BLOCK_SIZE queries
    for one batch element.
    """

    # program id along sequence
    pid_n = tl.program_id(0)
    # program id along batch
    pid_b = tl.program_id(1)

    # Starting position of this query block
    q_start = pid_n * BLOCK_SIZE
    offs_q = q_start + tl.arange(0, BLOCK_SIZE)

    # Mask to prevent OOB access
    q_mask = offs_q < N

    # Compute pointers for this batch
    Q_batch = Q_ptr + pid_b * stride_qb
    K_batch = K_ptr + pid_b * stride_kb
    V_batch = V_ptr + pid_b * stride_vb
    O_batch = O_ptr + pid_b * stride_ob
    L_batch = L_ptr + pid_b * stride_lb

    # Load Q tile
    Q_block = tl.load(
        Q_batch + offs_q[:, None] * stride_qn + tl.arange(0, D)[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    # Running softmax statistics
    m = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    O_block = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)

    # scale = 1.0 / tl.sqrt(tl.float32(D))

    # Iterate over K/V tiles
    for k_start in range(0, N, BLOCK_SIZE):

        offs_k = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = offs_k < N

        K_block = tl.load(
            K_batch + offs_k[:, None] * stride_kn + tl.arange(0, D)[None, :] * stride_kd,
            mask=k_mask[:, None],
            other=0.0,
        )

        V_block = tl.load(
            V_batch + offs_k[:, None] * stride_vn + tl.arange(0, D)[None, :] * stride_vd,
            mask=k_mask[:, None],
            other=0.0,
        )

        # Compute attention scores for this tile
        scores = tl.dot(Q_block, tl.trans(K_block)) * scale

        # Causal masking (correct + stable)
        if is_causal:
            q_idx = offs_q[:, None]
            k_idx = offs_k[None, :]

            causal_mask = k_idx > q_idx
            full_mask = causal_mask | (~k_mask[None, :])

            scores = tl.where(full_mask, -float("inf"), scores)

        # Online softmax (stable)
        row_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, row_max)

        alpha = tl.exp(m - m_new)
        p_tilde = tl.exp(scores - m_new[:, None]).to(tl.float32)

        l_new = alpha * l + tl.sum(p_tilde, axis=1)

        p_tilde_cast = p_tilde.to(V_block.dtype)
        O_block = alpha[:, None] * O_block + tl.dot(p_tilde_cast, V_block)

        m = m_new
        l = l_new

    # Final normalization
    O_block = O_block / l[:, None]
    L_block = m + tl.log(l)

    # Store output
    tl.store(
        O_batch + offs_q[:, None] * stride_on + tl.arange(0, D)[None, :] * stride_od,
        O_block,
        mask=q_mask[:, None],
    )
    tl.store(
        L_batch + offs_q * stride_ln,
        L_block,
        mask=q_mask,
    )


# ================================
# Python Wrapper
# ================================

def flash_attention_triton(Q, K, V, is_causal=False, block_size=64):
    """
    Python wrapper to launch Triton kernel.
    """

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, N, D = Q.shape

    O = torch.empty_like(Q)
    L = torch.empty((B, N), device=Q.device, dtype=torch.float32)

    scale = 1.0 / (D ** 0.5)

    grid = (
        triton.cdiv(N, block_size),  # number of sequence blocks
        B,                           # batch dimension
    )

    flash_attention_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        B, N,
        scale,
        is_causal,
        D=D,
        BLOCK_SIZE=block_size,
    )

    return O, L

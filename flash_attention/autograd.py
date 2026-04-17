import torch
import math


class FlashAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal):
        if Q.is_cuda:
            from .triton_impl import flash_attention_triton
            O, L = flash_attention_triton(Q, K, V, is_causal=causal)
        else:
            from .reference import flash_attention_reference
            O = flash_attention_reference(Q, K, V)
            # compute L (log-sum-exp) manually
            scale = 1.0 / math.sqrt(Q.shape[-1])
            S = torch.matmul(Q, K.transpose(-1, -2)) * scale
            if causal:
                seq_len = S.size(-1)
                mask = torch.triu(torch.ones(seq_len, seq_len, device=S.device, dtype=torch.bool), diagonal=1)
                S = S.masked_fill(mask, float('-inf'))
            L = torch.logsumexp(S, dim=-1)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.causal = causal
        ctx.scale = 1.0 / math.sqrt(Q.shape[-1])

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dO = dO.to(torch.float32)
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)
        O = O.to(torch.float32)
        L = L.to(torch.float32)

        scale = ctx.scale
        causal = ctx.causal

        # Recompute S
        S = torch.matmul(Q, K.transpose(-1, -2)) * scale

        if causal:
            seq_len = S.size(-1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=S.device, dtype=torch.bool), diagonal=1)
            S = S.masked_fill(mask, float('-inf'))

        # Recompute P using L
        P = torch.exp(S - L.unsqueeze(-1))

        # D vector
        D = torch.sum(O * dO, dim=-1, keepdim=True)

        # dV
        dV = torch.matmul(P.transpose(-1, -2), dO)

        # dP
        dP = torch.matmul(dO, V.transpose(-1, -2))

        # dS
        dS = P * (dP - D)

        # apply scale
        dS = dS * scale

        # dQ
        dQ = torch.matmul(dS, K)

        # dK
        dK = torch.matmul(dS.transpose(-1, -2), Q)

        return dQ, dK, dV, None

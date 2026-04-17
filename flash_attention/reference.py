import torch
import torch.nn.functional as F


def flash_attention_reference(Q, K, V, block_size: int = 128):
  

    B, N, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    # Output tensor
    O = torch.zeros_like(Q)

    # Process Q in blocks (row-wise tiling)
    for q_start in range(0, N, block_size):
        q_end = min(q_start + block_size, N)

        Q_block = Q[:, q_start:q_end, :]  # (B, Bq, D)
        Bq = Q_block.shape[1]

        # Running softmax statistics
        # m: running max
        # l: running denominator (sum of exp)
        m = torch.full((B, Bq), float("-inf"), device=Q.device)
        l = torch.zeros((B, Bq), device=Q.device)

        # Running output accumulator
        O_block = torch.zeros((B, Bq, D), device=Q.device)

        # Iterate over K/V in blocks (column-wise tiling)
        for k_start in range(0, N, block_size):
            k_end = min(k_start + block_size, N)

            K_block = K[:, k_start:k_end, :]  # (B, Bk, D)
            V_block = V[:, k_start:k_end, :]  # (B, Bk, D)

            # Compute attention scores for this tile
            # (B, Bq, D) @ (B, D, Bk) -> (B, Bq, Bk)
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

            # Compute max over new block
            m_new = torch.maximum(m, scores.max(dim=-1).values)

            # Compute exp(scores - m_new)
            exp_scores = torch.exp(scores - m_new.unsqueeze(-1))

            # Update denominator
            l_new = torch.exp(m - m_new) * l + exp_scores.sum(dim=-1)

            # Update output accumulator
            # Adjust old accumulator
            O_block = (
                torch.exp(m - m_new).unsqueeze(-1) * O_block
                + torch.matmul(exp_scores, V_block)
            )

            # Update running stats
            m = m_new
            l = l_new

        # Final normalization
        O_block = O_block / l.unsqueeze(-1)

        # Write back to output
        O[:, q_start:q_end, :] = O_block

    return O


def naive_attention(Q, K, V):
    """
    Standard attention implementation that explicitly
    materializes the full attention matrix.
    Used for correctness checking.
    """

    D = Q.shape[-1]
    scale = 1.0 / (D ** 0.5)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, V)
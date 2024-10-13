import torch.nn as nn

class Attention(nn.Module):
    """
    Attention Mechanism

    Parameters:
    -----------
    dim       : int   | Input and Output dimension of per token features
    n_heads   : int   | Number of attention heads
    qkv_bias  : bool  | Add bias to query, key and value projections
    qk_scale  : float | Normalizing constant for dot product
    attn_drop : float | Dropout probability for qkv projections
    proj_drop : float | Dropout probabilty for output tensor
    """

    def __init__(
        self,
        dim,
        n_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.qk_scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Parameters:
        -----------
        x : tensor.Tensor | Shape `(batch_size, n_patches, dim)`

        Returns:
        --------
        tensor.Tensor     | Shape `(batch_size, n_patches, dim)`
        """
        batch_size, n_patches, _ = x.shape
        qkv = self.qkv(x)  # (batch_size, n_patches, dim * 3)
        qkv = qkv.reshape(
            batch_size, n_patches, 3, self.n_heads, self.head_dim
        )  # (batch_size, n_patches, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (batch_size, n_head, head_dim, n_patches)
        dp = (q @ k_t) * self.qk_scale  # (batch_size, n_head, n_patches, n_patches)
        attn = dp.softmax(dim=-1)  # (batch_size, n_head, n_patches, n_patches)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (batch_size, n_head, n_patches, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (batch_size, n_patches, n_head, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (batch_size, n_patches, dim)
        x = self.proj(weighted_avg)  # (batch_size, n_patches, dim)
        x = self.proj_drop(x)

        return x
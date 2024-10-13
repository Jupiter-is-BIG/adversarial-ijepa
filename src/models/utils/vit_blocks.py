import torch.nn as nn

from .vit_attention import Attention
from .vit_mlp import MLP

class Block(nn.Module):
    """
    Transformer Block

    Parameters:
    -----------
    dim             : int   | Embedding dimension
    n_heads         : int   | Number of attention heads
    mlp_ratio       : float | Determines the dimension size of the MLP module wrt dim
    qkv_bias        : bool  | Add bias to query, key and value projections
    qk_scale        : float | Normalizing constant for dot product
    drop, attn_drop : float | Dropout probability
    """

    def __init__(
        self,
        dim,
        n_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            dim, hidden_features=hidden_features, out_features=dim, drop=drop
        )

    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor | Shape `(batch_size, n_patches + 1, dim)`

        Returns:
        --------
        torch.Tensor     | Shape `(batch_size, n_patches + 1, dim)`
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
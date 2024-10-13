import torch
import torch.nn as nn

from src.models.utils.pos_embed import get_2d_sincos_pos_embed
from src.models.utils.vit_blocks import Block
from src.masks.utils import apply_masks, repeat_interleave_batch

class VisionTransformerGenerator(nn.Module):
    """
    Vision Transformer Generator (ViTG)

    Parameters:
    -----------
    n_patches           : int
    embed_dim           : int
    predictor_embed_dim : int
    depth               : int
    n_heads             : int
    mlp_ratio           : float
    qkv_bias            : bool
    drop, attn_drop     : float
    """
    def __init__(
            self,
            n_patches,
            embed_dim=768,
            predictor_embed_dim=364,
            depth=6,
            n_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, n_patches, predictor_embed_dim),requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1], int(n_patches**.5), cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(drop)

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
    
    def forward(self, x, masks_x, masks):
        """
        Parameters:
        -----------
        x       : torch.Tensor | Shape `(batch_size, n_patches, embed_dim)`
        masks_x : list         | context mask 
        masks   : list         | target mask

        Returns:
        --------
        torch.Tensor | Shape `(batch_size * n_mask_x * n_masks, n' + m', embed_dim)`
        """
        assert masks and masks_x
        masks_x = masks_x if isinstance(masks_x, list) else [masks_x]
        masks = masks if isinstance(masks, list) else [masks]

        batch_size = len(x) // len(masks_x)

        x = self.predictor_embed(x)
        x_pos_embed = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)
        assert x.shape[1] == masks_x[0].shape[1]

        pos_embs = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, batch_size, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)

        x = self.norm(x)

        assert x.shape[0] == batch_size * len(masks) * len(masks_x)
        assert x.shape[1] == masks[0].shape[1] + masks_x[0].shape[1]

        x = x[:, masks_x[0].shape[1]:]
        x = self.predictor_proj(x)

        return x

def vit_generator(**kwargs):
    model = VisionTransformerGenerator(
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model
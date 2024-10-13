import torch
import torch.nn as nn

from src.models.utils.pos_embed import get_2d_sincos_pos_embed
from src.models.utils.vit_patch_embed import PatchEmbed
from src.models.utils.vit_blocks import Block
from src.masks.utils import apply_masks

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)

    Parameters:
    -----------
    img_size        : list[int]
    patch_size      : int
    in_chans        : int
    embed_dim       : int
    depth           : int
    n_heads         : int
    mlp_ratio       : float
    qkv_bias        : bool
    drop, attn_drop : float
    """

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.n_patches**0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
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

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, masks=None):
        """
        Parameters:
        -----------
        x      : torch.Tensor       | Shape `(batch_size, in_chans, img_size, img_size)`
        ?masks : list[torch.Tensor] | context mask

        Returns:
        --------
        torch.Tensor     | Shape `(batch_size * n_masks, num_kept_patches, dim)`
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        _B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if masks is not None:
            x = apply_masks(x, masks)
            assert x.shape[0] == _B * len(masks)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )

    return model
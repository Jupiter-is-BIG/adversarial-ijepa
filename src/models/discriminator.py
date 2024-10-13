import torch
import torch.nn as nn

from src.models.utils.pos_embed import get_2d_sincos_pos_embed
from src.masks.utils import apply_masks

class Discriminator(nn.Module):
    """
    Discriminator

    Parameters:
    -----------
    embed_dim   : int   | Dimension of the latent space
    hidden_dim  : int   | Dimension of discriminator latent space
    drop        : float | Dropout probability
    p           : float | LeakyReLU parameter
    """
    def __init__(self, n_patches, embed_dim=768, hidden_dim=256, drop=0.0, p=0.2):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm_context = nn.LayerNorm(embed_dim)
        self.norm_input = nn.LayerNorm(embed_dim)

        self.pos_embed_ctx = nn.Parameter(torch.zeros(1, n_patches, embed_dim), requires_grad=False)
        pos_embed_ctx = get_2d_sincos_pos_embed(embed_dim, int(n_patches**0.5), cls_token=False)
        self.pos_embed_ctx.data.copy_(torch.from_numpy(pos_embed_ctx).float().unsqueeze(0))

        self.pos_embed_pred = nn.Parameter(torch.zeros(1, n_patches, embed_dim), requires_grad=False)
        pos_embed_pred = get_2d_sincos_pos_embed(embed_dim, int(n_patches**0.5), cls_token=False)
        self.pos_embed_pred.data.copy_(torch.from_numpy(pos_embed_pred).float().unsqueeze(0))

        self.context_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(p),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(p),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(p),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, ctx, mask_ctx, mask_pred):
        """
        Parameters:
        -----------
        x   : torch.Tensor         | Shape `(batch_size * n_ctx * n_targets, target_size, embed_dim)`
        ctx : torch.Tensor         | Shape `(batch_size * n_ctx * n_targets, ctx_size, embed_dim)`

        Returns:
        --------
        probability : torch.Tensor | Shape `(batch_size * n_ctx * n_targets)`
        """

        batch_size = x.shape[0] // (len(mask_ctx) * len(mask_pred))

        x_pos_embed_ctx = self.pos_embed_ctx.repeat(batch_size, 1, 1)
        x_pos_embed_ctx = apply_masks(x_pos_embed_ctx, mask_ctx)
        x_pos_embed_ctx = x_pos_embed_ctx.repeat(len(mask_pred), 1, 1)
        ctx = ctx + x_pos_embed_ctx

        x_pos_embed_pred = self.pos_embed_pred.repeat(batch_size * len(mask_ctx), 1, 1)
        x_pos_embed_pred = apply_masks(x_pos_embed_pred, mask_pred)
        x = x + x_pos_embed_pred

        ctx = self.norm_context(ctx)
        x = self.norm_input(x)

        target_pooled = x.mean(dim=1)                                            # Shape: (batch_size * n_ctx * n_targets, embed_dim)
        context_pooled = ctx.mean(dim=1)                                         # Shape: (batch_size * n_ctx * n_targets, embed_dim)

        target_processed = self.input_net(target_pooled)                         # Shape: (batch_size * n_ctx * n_targets, hidden_dim)
        context_processed = self.context_net(context_pooled)                     # Shape: (batch_size * n_ctx * n_targets, hidden_dim)

        latent_image = torch.cat((context_processed, target_processed), dim=-1)  # Shape: (batch_size * n_ctx * n_targets, hidden_dim * 2)

        output = self.fusion_net(latent_image)

        return output.squeeze(1)

def discriminator(**kwargs):
    model = Discriminator(
        embed_dim=768,
        hidden_dim=256,
        **kwargs
    )

    return model
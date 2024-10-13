import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Segreate the image into patches and compute the embeddings

    Parameters:
    -----------
        img_size   : int | dimensions of the image (assuming square)
        patch_size : int | size of each patch (assuming square)
        in_chans   : int | number of input channels
        embed_dim  : int | embedding dimmensions
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor | Shape `(batch_size, in_chans, img_size, img_size)`

        Returns:
        --------
        torch.Tensor     | Shape `(batch_size, n_patches, embed_dim)`
        """
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)

        return x
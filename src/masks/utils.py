import torch

def apply_masks(x, masks):
    """
    Parameters:
    -----------
    x     : torch.Tensor       | Shape `(batch_size, n_patches, dim)`
    masks : List[torch.Tensor] | list of tensors containing indices of patches in [N] to keep

    Returns:
    --------
    torch.Tensor               | Shape `(batch_size * n_masks, num_kept_patches, dim)`
    """
    all_x = []
    
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]

    return torch.cat(all_x, dim=0)

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

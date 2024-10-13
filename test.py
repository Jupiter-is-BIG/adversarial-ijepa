import os
import sys
import copy

import numpy as np

import torch
import torch.nn.functional as F

from src.utils import save_checkpoint, init_model, init_opt
from src.transforms import make_transforms
from src.masks.utils import apply_masks, repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.masks.multiblock import MaskCollator as MBMaskCollator

# model = vit_base()

# x = torch.rand([5,3,224,224])
# masks = [torch.tensor([1, 3, 5]), torch.tensor([0, 2, 6])]

# with torch.no_grad():
#     result = model(x, masks)

# print(result.shape)
if __name__ == '__main__':
    # --
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_size = 16
    crop_size = 224
    pred_depth = 12
    pred_emb_dim = 384
    model_name = 'vit_base'
    num_epochs = 300
    final_lr = 1.0e-06
    final_wd = 0.4
    ipe_scale = 1.0
    lr = 0.001
    start_lr = 0.0002
    warmup = 40
    wd = 0.04
    use_bfloat16 = False
    ema = [0.996, 1.0]
    folder = 'checkpoints/'
    tag = 'ajepa'
    batch_size = 128
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    checkpoint_freq = 50
    # --

    encoder, predictor, discriminator = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name
    )

    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    mask_collator = MBMaskCollator(
        input_size=224, 
        patch_size=16, 
        aspect_ratio=(1.0, 1.0), 
        enc_mask_scale=(0.35, 0.35), 
        pred_mask_scale=(0.1, 0.1),
        nenc=2,
        npred=3
    )
    transform = make_transforms(crop_size=[224])
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
                transform=transform,
                batch_size=128,
                collator=mask_collator,
                pin_mem=True,
                training=False,
                root_path="tiny-imagenet-200",
                drop_last=True)
    print(len(unsupervised_loader))
    i = 0
    for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
        # print('ffffffffffffffffff')
        # print(len(unsupervised_loader))
        # print(len(masks_enc))
        # print(len(masks_pred))
        # print(masks_enc[0].shape)
        # print(masks_enc[1].shape)
        # print(masks_pred[0].shape)
        # print(masks_pred[1].shape)
        def load_imgs():
            imgs = udata[0].to(device, non_blocking=True)
            masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
            return (imgs, masks_1, masks_2)
        
        imgs, masks_enc, masks_pred = load_imgs()
    
        def forward_context():
            z = encoder(imgs, masks_enc)
            z = predictor(z, masks_enc, masks_pred)
            return z
        
        def forward_target():
            with torch.no_grad():
                h = target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                B = len(h)

                h_context = apply_masks(h, masks_enc)
                h_context = h_context.repeat(len(masks_pred), 1, 1)

                h_target = apply_masks(h, masks_pred)
                h_target = repeat_interleave_batch(h_target, B, repeat=len(masks_enc))
                return h_context, h_target
        
        def forward_discriminator(x, ctx):
            with torch.no_grad():
                p = discriminator(x, ctx)
                return p
        
        h_target_fake = forward_context()
        # h_context, h_target_real = forward_target()
        # zero_values = forward_discriminator(h_target_fake, h_context)
        # one_values = forward_discriminator(h_target_real, h_context)
        print('------------------')
        print(h_target_fake.shape)
        # print(zero_values.shape)                
        # print(one_values.shape)                
        print('------------------')
        i+=1
        if i == 1:
            break
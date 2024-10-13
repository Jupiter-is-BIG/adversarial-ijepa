import os
import sys
import copy

import numpy as np

import torch
import torch.nn.functional as F

from src.utils import save_checkpoint, init_model, init_opt, AverageMeter
from src.transforms import make_transforms
from src.masks.utils import apply_masks, repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.masks.multiblock import MaskCollator as MBMaskCollator

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

def main():
    # --
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_size = 16
    crop_size = 224
    pred_depth = 12
    pred_emb_dim = 384
    model_name = 'vit_base'
    num_epochs = 10
    final_lr = 1.0e-06
    final_wd = 0.4
    ipe_scale = 1.0
    lr = 0.001
    start_lr = 0.0002
    warmup = 40
    wd = 0.04
    use_bfloat16 = False
    ema = [0.996, 1.0]
    folder = './checkpoints/'
    tag = 'ajepa'
    batch_size = 128
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    checkpoint_freq = 50
    alpha = 0.5
    # --

    encoder, predictor, discriminator = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    mask_collator = MBMaskCollator(
        input_size=crop_size, 
        patch_size=patch_size, 
        aspect_ratio=(0.75, 1.5), 
        enc_mask_scale=(0.85, 1.0), 
        pred_mask_scale=(0.15, 0.2)
    )
    transform = make_transforms(crop_size=[crop_size])

    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=True,
            training=True,
            root_path="tiny-imagenet-200",
            drop_last=True
        )
    ipe = len(unsupervised_loader)

    optimizer, discriminator_optimizer, scaler, scheduler, discriminator_scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        discriminator=discriminator,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        start_lr_disc=start_lr,
        ref_lr=lr,
        ref_lr_disc=lr,
        final_lr=final_lr,
        final_lr_disc=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        print('[INFO] Epoch %d' % (epoch + 1))

        unsupervised_sampler.set_epoch(epoch)

        reconstruction_loss_meter = AverageMeter()
        generator_loss_meter = AverageMeter()
        discriminator_loss_meter = AverageMeter()
        loss_meter = AverageMeter()

        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        N = len(unsupervised_loader)

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            
            imgs, masks_enc, masks_pred = load_imgs()

            maskA_meter.update(len(masks_enc[0][0]))  # n_patches in context
            maskB_meter.update(len(masks_pred[0][0])) # n_patches in target

            def train_step():
                scheduler.step()
                discriminator_scheduler.step()
                wd_scheduler.step()

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
                h_context, h_target_real = forward_target()

                fake_preds = forward_discriminator(h_target_fake, h_context)
                reconstruction_loss = F.smooth_l1_loss(h_target_fake, h_target_real)

                g_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))

                combined_loss = alpha * reconstruction_loss + (1 - alpha) * g_loss

                combined_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                discriminator.requires_grad_(True)
                for _ in range(5):
                    h_target_fake = forward_context()
                    real_preds = forward_discriminator(h_target_real, h_context)
                    fake_preds = forward_discriminator(h_target_fake, h_context)

                    d_loss_real = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
                    d_loss_fake = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
                    d_loss = (d_loss_real + d_loss_fake) / 2

                    d_loss.backward()
                    discriminator_optimizer.step()
                    discriminator_optimizer.zero_grad()

                with torch.no_grad():
                    real_preds = forward_discriminator(h_target_real, h_context)
                    fake_preds = forward_discriminator(h_target_fake, h_context)

                d_loss_real = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
                d_loss_fake = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
                d_loss = (d_loss_real + d_loss_fake) / 2

                mean_real_preds = torch.mean(real_preds).item()
                mean_fake_preds = torch.mean(fake_preds).item()

                print(f"Mean of real predictions: {mean_real_preds}")
                print(f"Mean of fake predictions: {mean_fake_preds}")

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (reconstruction_loss.item(), g_loss.item(), d_loss.item(), combined_loss.item())
            
            reconstruction_loss, g_loss, d_loss, combined_loss = train_step()
            
            reconstruction_loss_meter.update(reconstruction_loss)
            generator_loss_meter.update(g_loss)
            discriminator_loss_meter.update(d_loss)
            loss_meter.update(combined_loss)
            print(f'[INFO] Episode {itr} / {N} completed')
            print('[INFO] avg. loss: %.3f | avg. reconstruction loss: %.3f | avg. generator loss: %.3f | avg. discriminator loss: %.3f | mask_ctx: %.3f | mask_target: %.3f' % (
                loss_meter.avg,
                reconstruction_loss_meter.avg,
                generator_loss_meter.avg,
                discriminator_loss_meter.avg,
                maskA_meter.avg,
                maskB_meter.avg
            ))

        print('[INFO] avg. loss: %.3f | avg. reconstruction loss: %.3f | avg. generator loss: %.3f | avg. discriminator loss: %.3f | mask_ctx: %.3f | mask_target: %.3f' % (
            loss_meter.avg,
            reconstruction_loss_meter.avg,
            generator_loss_meter.avg,
            discriminator_loss_meter.avg,
            maskA_meter.avg,
            maskB_meter.avg
        ))

        save_checkpoint(
            epoch=epoch+1,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            discriminator=discriminator,
            optimizer=optimizer,
            scaler=scaler,
            loss_meter=loss_meter,
            batch_size=batch_size,
            lr=lr,
            checkpoint_freq=checkpoint_freq,
            latest_path=latest_path,
            save_path=save_path
        )

if __name__ == "__main__":
    main()
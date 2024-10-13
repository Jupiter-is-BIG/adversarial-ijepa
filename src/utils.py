import torch

import src.models.vision_transformer as vit
import src.models.vit_generator as vitg
import src.models.discriminator as disc

from src.schedulers import WarmupCosineSchedule, CosineWDSchedule

def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vitg.__dict__['vit_generator'](
        n_patches=encoder.patch_embed.n_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        n_heads=encoder.n_heads)
    discriminator = disc.__dict__['discriminator'](
        n_patches=encoder.patch_embed.n_patches,
    )

    encoder.to(device)
    predictor.to(device)
    discriminator.to(device)
    return encoder, predictor, discriminator

def init_opt(
    encoder,
    predictor,
    discriminator,  # Add discriminator as a parameter
    iterations_per_epoch,
    start_lr,
    start_lr_disc,
    ref_lr,
    ref_lr_disc,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    final_lr_disc=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    generator_param_groups = [
        {
            'params': [p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)]
        }, 
        {
            'params': [p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)]
        }, 
        {
            'params': [p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)],
            'WD_exclude': True,
            'weight_decay': 0
        }, 
        {
            'params': [p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)],
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    optimizer = torch.optim.AdamW(generator_param_groups)

    discriminator_param_groups = [
        {
            'params': [p for n, p in discriminator.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)]
        },
        {
            'params': [p for n, p in discriminator.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)],
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    discriminator_optimizer = torch.optim.AdamW(discriminator_param_groups)

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch)
    )

    discriminator_scheduler = WarmupCosineSchedule(
        discriminator_optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr_disc,
        ref_lr=ref_lr_disc,
        final_lr=final_lr_disc,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch)
    )
    
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch)
    )

    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

    return optimizer, discriminator_optimizer, scaler, scheduler, discriminator_scheduler, wd_scheduler


def save_checkpoint(
        epoch,
        encoder, 
        predictor, 
        target_encoder, 
        discriminator, 
        optimizer, 
        scaler, 
        loss_meter, 
        batch_size, 
        lr, 
        checkpoint_freq,
        latest_path,
        save_path
    ):
    save_dict = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'discriminator': discriminator.state_dict(),
        'opt': optimizer.state_dict(),
        'scaler': None if scaler is None else scaler.state_dict(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'batch_size': batch_size,
        'lr': lr
    }
    torch.save(save_dict, latest_path)
    if (epoch + 1) % checkpoint_freq == 0:
        torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
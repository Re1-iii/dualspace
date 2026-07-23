"""
example_usage.py — how to wire the three axes into ONE training step.

This is a documentation snippet, not a trainer: there is no data loading,
optimizer schedule, or evaluation. It shows the loss assembly for a single
step so you can drop the components into your own loop with any backbone.

Replace `model` with your segmentation network. The only contract is:
    model(x)                 -> logits (B, 1, H, W)
    model(x, return_feats=T) -> (logits, [feat1, feat2, ...])   # optional, axis-feat
"""
import torch
import torch.nn.functional as F

from dual_space import (
    UGFECTv2Augmentor,          # style axis (Fourier amplitude perturbation)
    random_scale_size, scale_consistency_loss,   # scale axis
    SWAD, update_bn,            # weight axis
    DiceBCELoss, gated_consistency_loss,
)

# ---- set up once ----
seg_loss = DiceBCELoss(bce_weight=0.5)
style_aug = UGFECTv2Augmentor()          # produces the style variant x'
swad = SWAD()                            # accumulates flat-minima weights
lam_sty, lam_sca = 1.0, 1.0
img_size = 352


def training_step(model, imgs, masks):
    """One forward/backward's worth of loss (call inside your own loop)."""
    logits = model(imgs)
    loss = seg_loss(logits, masks)                       # L_seg

    teacher = torch.sigmoid(logits).detach()             # stop-grad anchor + gate

    # --- style axis: perturb low-freq amplitude, enforce gated consistency ---
    imgs_style = style_aug(imgs, masks, model)           # x'  (fp32, no autocast)
    logits_style = model(imgs_style)
    loss = loss + lam_sty * gated_consistency_loss(teacher, logits_style)

    # --- scale axis: random downscale, enforce gated consistency ---
    s = random_scale_size(img_size, 0.5, 1.0)
    imgs_scaled = F.interpolate(imgs, size=(s, s), mode='bilinear', align_corners=False)
    logits_scaled = model(imgs_scaled)
    loss = loss + lam_sca * scale_consistency_loss(teacher, logits_scaled)

    return loss


# --- weight axis (per epoch): after each epoch, record the current weights ---
# swad.update(model, val_loss=source_val_loss)     # source-domain val only
# ... after training:
# deployed = swad.finalize_into(model)             # flat-minima averaged weights
# update_bn(train_loader, deployed, device)        # re-estimate BN on train data
#
# At inference, run `deployed(x)` alone: zero added params, zero added cost.

if __name__ == "__main__":
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(8, 1, 1))
        def forward(self, x, return_feats=False):
            return self.c(x)

    model = Tiny()
    imgs = torch.rand(2, 3, 352, 352)
    masks = (torch.rand(2, 1, 352, 352) > 0.5).float()
    loss = training_step(model, imgs, masks)
    loss.backward()
    print(f"assembled loss = {loss.item():.5f}  finite={torch.isfinite(loss).item()}")
    print("example OK — plug these components into your own training loop.")

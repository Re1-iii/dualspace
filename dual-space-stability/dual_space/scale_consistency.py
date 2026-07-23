"""
Scale axis — multi-scale consistency (training-only).

Orthogonal to the style axis: the style axis addresses appearance, this one
addresses scale. A view of the image at a random smaller resolution is fed to
the network and its prediction is forced to agree with the full-resolution
prediction (a stop-gradient teacher), making the model robust to polyp-scale
change. Downscaling is favored, which directly practises small polyps -- the
main difficulty on sets such as ETIS -- and also saves memory. Inference stays
single-scale, so the axis is plug-and-play.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def random_scale_size(base_size, scale_min=0.5, scale_max=1.0, multiple=32, min_size=96):
    """Sample a scaled side length (a multiple of 32 so every stage divides evenly)."""
    s = np.random.uniform(scale_min, scale_max)
    sz = int(round(base_size * s / multiple)) * multiple
    return max(sz, min_size)


def scale_consistency_loss(p_base_teacher, pred_scaled_logits,
                           uncertainty_weighted=True, eps=1e-6):
    """Confidence-gated consistency between the full-scale teacher and a scaled view.

    p_base_teacher:     sigmoid probs of the full-scale prediction, detached (B,1,Hb,Wb)
    pred_scaled_logits: logits of the scaled view (B,1,Hs,Ws); Hs/Ws may differ
    Returns a scalar consistency loss.
    The gate C = 1 - H(teacher)/log 2 down-weights ambiguous boundary pixels.
    """
    pred_up = F.interpolate(pred_scaled_logits, size=p_base_teacher.shape[2:],
                            mode='bilinear', align_corners=False)
    p_scaled = torch.sigmoid(pred_up)
    if uncertainty_weighted:
        pc = p_base_teacher.clamp(eps, 1.0 - eps)
        ent = -(pc * pc.log() + (1.0 - pc) * (1.0 - pc).log())
        conf = 1.0 - ent / math.log(2.0)        # more confident teacher -> stronger constraint
        return (F.mse_loss(p_scaled, p_base_teacher, reduction='none') * conf).mean()
    return F.mse_loss(p_scaled, p_base_teacher)


if __name__ == '__main__':
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(8, 1, 1))
        def forward(self, x):
            return self.c(x)

    m = Tiny()
    imgs = torch.rand(4, 3, 352, 352)

    sizes = [random_scale_size(352, 0.5, 1.0) for _ in range(12)]
    print(f"sampled sizes (multiples of 32, <= 352): {sorted(set(sizes))}")
    assert all(s % 32 == 0 and s <= 352 for s in sizes)

    pred_base = m(imgs)
    teacher = torch.sigmoid(pred_base).detach()
    ss = random_scale_size(352, 0.5, 0.9)
    x_scaled = F.interpolate(imgs, size=(ss, ss), mode='bilinear', align_corners=False)
    pred_scaled = m(x_scaled)
    print(f"teacher {tuple(teacher.shape)} | student (scaled to {ss}) {tuple(pred_scaled.shape)}")

    loss = scale_consistency_loss(teacher, pred_scaled, uncertainty_weighted=True)
    loss.backward()
    has_grad = all(p.grad is not None for p in m.parameters())
    print(f"loss={loss.item():.5f} finite={torch.isfinite(loss).item()}")
    print(f"student has grad (True)={has_grad} | teacher.requires_grad (False)={teacher.requires_grad}")
    print("scale_consistency OK" if (loss.item() >= 0 and has_grad
          and not teacher.requires_grad) else "FAIL")

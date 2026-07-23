# Dual-Space Stability — plug-and-play components

Reference implementation of the **training-only** stability components from
*Dual-Space Stability* (Multimedia Systems). The method casts domain-generalized
polyp segmentation as enforcing predictive stability along three axes. It adds
**no parameters and no inference cost** and is **backbone-agnostic**: the modules
here plug into any segmentation network's training loop.

> This repository contains the **method components only** — no training scripts,
> data loaders, evaluation code, or model/backbone definitions. Wire the
> components into your own loop (see `example_usage.py`).

## The three axes

| Axis | Space | What it does | File |
|------|-------|--------------|------|
| **Style** | input | Perturb only the low-frequency **amplitude** of the FFT, keep the phase → an acquisition-style variant `x'`. | `dual_space/fourier_style.py` |
| **Scale** | input | Random downscale to `[0.5, 1.0]·S`, enforce consistency with the full-scale prediction. | `dual_space/scale_consistency.py` |
| **Weight** | weight | Dense flat-minima weight averaging (SWAD), intended to favour a flatter solution; source-domain model selection. | `dual_space/swad.py` |

**Style-axis settings used in the paper.** The edited window `Ω_β` is centred on
the zero frequency after `fftshift`; the `ratio` argument is the **half-width**
ratio `β`, so the window spans a side of about `2βH` (with `β = 0.05` and a
352×352 input, a 35×35 window). Two interchangeable strategies are applied
inside `Ω_β` — multiplicative Gaussian noise (Strategy 1) and a batch amplitude
swap with a donor frame (Strategy 2) — and the reported results draw one of the
two per perturbed image with **equal probability** (`mode='both'`), with
`p = 0.7` and `α = 1.0`.

The two input-space axes are tied together by an **uncertainty gate** read from
the clean (stop-gradient) teacher:

```
C = 1 - H̃(ŷ),   H̃ = H / log 2,   H(p) = -p·log p - (1-p)·log(1-p)
```

Consistency is enforced only where `C` is high (the teacher is confident). The
Fourier perturbation is ineffective as plain augmentation and becomes useful only
once it is coupled with the consistency loss; the gate contributes a further
improvement by down-weighting uncertain pixels. The gate and the gated
consistency loss live in `dual_space/losses.py` (and, for the scale axis, in
`scale_consistency.py`).

## Install

```bash
pip install -r requirements.txt   # torch, numpy
```

Then import from the `dual_space` package (copy the folder into your project, or
`pip install -e .` if you add a `pyproject.toml`).

## Minimal usage

```python
import torch, torch.nn.functional as F
from dual_space import (UGFECTv2Augmentor, random_scale_size,
                        scale_consistency_loss, gated_consistency_loss, DiceBCELoss)

seg_loss  = DiceBCELoss(bce_weight=0.5)
style_aug = UGFECTv2Augmentor()

def training_step(model, imgs, masks, img_size=352, lam_sty=1.0, lam_sca=1.0):
    logits  = model(imgs)
    loss    = seg_loss(logits, masks)
    teacher = torch.sigmoid(logits).detach()            # stop-grad anchor + gate

    imgs_style   = style_aug(imgs, masks, model)        # style axis: x'
    loss += lam_sty * gated_consistency_loss(teacher, model(imgs_style))

    s = random_scale_size(img_size, 0.5, 1.0)           # scale axis
    imgs_scaled = F.interpolate(imgs, (s, s), mode='bilinear', align_corners=False)
    loss += lam_sca * scale_consistency_loss(teacher, model(imgs_scaled))
    return loss
```

Weight axis (per epoch / after training):

```python
from dual_space import SWAD, update_bn
swad = SWAD()
# each epoch:  swad.update(model, val_loss=source_val_loss)   # source val only
deployed = swad.finalize_into(model)                          # averaged weights
update_bn(train_loader, deployed, device)                     # re-estimate BN
# inference: deployed(x) alone — zero added params, zero added cost.
```

A runnable version is in `example_usage.py`. Each module also has a
`__main__` self-test (`python -m dual_space.scale_consistency`, etc.).

## Notes

- All losses are **binary** (single-channel logits + sigmoid). For multi-class,
  replace the sigmoid/binary-entropy gate with the softmax/categorical version.
- Generate the style variant **outside** mixed-precision (`autocast`): the FFT
  runs in fp32.
- `fourier_style.py` also provides simpler entry points
  (`fourier_style_perturbation`, `fourier_style_swap`, `FourierDomainAugmentor`)
  if you do not need the full `UGFECTv2Augmentor`.

## Citation

```bibtex
@article{dualspacestability,
  author  = {Shi, Junyu and Li, Xinlei and Hu, Linqiang},
  title   = {Dual-Space Stability: uncertainty-gated multi-axis consistency and
             flat-minima averaging for generalizable polyp segmentation},
  journal = {Multimedia Systems},
  year    = {2026},
  note    = {Shanghai University of International Business and Economics}
}
```

## License

MIT — see `LICENSE`.

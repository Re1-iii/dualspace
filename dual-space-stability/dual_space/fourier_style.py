"""
Style axis — Fourier low-frequency amplitude perturbation.

The amplitude spectrum of an image captures low-level acquisition-dependent
appearance (color, illumination); the phase carries structure (object
boundaries). Perturbing only the low-frequency amplitude while keeping the
phase yields an acquisition-style variant x' that is geometry-aligned with the
original: the spatial coordinate system, and hence the mask, is preserved.
Note this does not guarantee that every appearance cue is unchanged -- an
aggressive amplitude replacement may alter local contrast or highlights.

The edited window Ω_β is centred on the zero frequency after fftshift. The
`ratio` argument is the HALF-WIDTH ratio β: rows cy-βH .. cy+βH are edited, so
the window spans a side of about 2βH. With the paper's β = 0.05 and a 352x352
input this is a 35x35 window.

Two interchangeable strategies are applied inside Ω_β, and the paper's reported
results draw one of the two per perturbed image with equal probability
(`mode='both'`): multiplicative Gaussian noise (Strategy 1) and a batch
amplitude swap with a donor frame (Strategy 2).

Public API (used by the paper's method):
    fourier_style_perturbation(img, ...)      # Gaussian-noise strategy (1) on Ω_β
    fourier_style_swap(img1, img2, ...)       # batch amplitude-swap strategy (2)
    FourierDomainAugmentor                    # per-image random perturb/swap
    UGFECTv2Augmentor                         # the augmentor used in the paper

Optional (not required to reproduce the paper; provided for ablation):
    PolypAwareAugmentor        # foreground/background-adaptive perturbation
    adversarial_lowfreq_perturbation  # worst-case low-frequency perturbation

Training-only: these produce the style variant x' during training. At
inference only the clean branch runs -- zero added parameters, zero cost.
"""
import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Core perturbations (single image, C x H x W)
# ---------------------------------------------------------------------------

def fourier_style_perturbation(img_tensor, alpha=0.5, ratio=0.01):
    """Multiplicative Gaussian noise on the low-frequency amplitude, phase kept.

    ratio: half-width ratio β of the centred window Ω_β (side ≈ 2βH).
    """
    C, H, W = img_tensor.shape
    fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    amplitude = torch.abs(fft_shift)
    phase = torch.angle(fft_shift)
    cy, cx = H // 2, W // 2
    rh = max(int(H * ratio), 1)
    rw = max(int(W * ratio), 1)
    noise = torch.randn_like(amplitude[:, cy-rh:cy+rh+1, cx-rw:cx+rw+1])
    amplitude_perturbed = amplitude.clone()
    amplitude_perturbed[:, cy-rh:cy+rh+1, cx-rw:cx+rw+1] = \
        amplitude[:, cy-rh:cy+rh+1, cx-rw:cx+rw+1] * (1.0 + alpha * noise)
    fft_perturbed = amplitude_perturbed * torch.exp(1j * phase)
    fft_perturbed = torch.fft.ifftshift(fft_perturbed, dim=(-2, -1))
    img_perturbed = torch.fft.ifft2(fft_perturbed, dim=(-2, -1)).real
    return img_perturbed


def fourier_style_swap(img1_tensor, img2_tensor, alpha=0.3, ratio=0.01):
    """Mix img2's low-frequency amplitude into img1 (img1's phase kept).

    ratio: half-width ratio β of the centred window Ω_β (side ≈ 2βH).
    """
    C, H, W = img1_tensor.shape
    fft1 = torch.fft.fftshift(
        torch.fft.fft2(img1_tensor, dim=(-2, -1)), dim=(-2, -1))
    fft2 = torch.fft.fftshift(
        torch.fft.fft2(img2_tensor, dim=(-2, -1)), dim=(-2, -1))
    amp1 = torch.abs(fft1)
    amp2 = torch.abs(fft2)
    phase1 = torch.angle(fft1)
    cy, cx = H // 2, W // 2
    rh = max(int(H * ratio), 1)
    rw = max(int(W * ratio), 1)
    amp_mixed = amp1.clone()
    amp_mixed[:, cy-rh:cy+rh+1, cx-rw:cx+rw+1] = \
        (1 - alpha) * amp1[:, cy-rh:cy+rh+1, cx-rw:cx+rw+1] + \
        alpha       * amp2[:, cy-rh:cy+rh+1, cx-rw:cx+rw+1]
    fft_mixed = amp_mixed * torch.exp(1j * phase1)
    fft_mixed = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
    img_mixed = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real
    return img_mixed


class FourierDomainAugmentor:
    """Per-image random style perturbation (Gaussian-noise or amplitude-swap).

    mode='both' (the paper's setting) draws one of the two strategies per image
    with equal probability; 'perturb' / 'swap' force a single strategy.
    """

    def __init__(self, prob=0.5, alpha=0.3, ratio=0.01, mode='both'):
        self.prob = prob
        self.alpha = alpha
        self.ratio = ratio
        self.mode = mode

    def __call__(self, imgs):
        B = imgs.shape[0]
        augmented = imgs.clone()
        for i in range(B):
            if np.random.random() > self.prob:
                continue
            if self.mode == 'perturb':
                use_swap = False
            elif self.mode == 'swap':
                use_swap = True
            else:
                use_swap = np.random.random() > 0.5
            if use_swap and B > 1:
                j = np.random.randint(0, B)
                while j == i:
                    j = np.random.randint(0, B)
                augmented[i] = fourier_style_swap(
                    imgs[i], imgs[j],
                    alpha=self.alpha, ratio=self.ratio)
            else:
                augmented[i] = fourier_style_perturbation(
                    imgs[i], alpha=self.alpha, ratio=self.ratio)
        return augmented


# ---------------------------------------------------------------------------
# Optional: foreground/background-adaptive perturbation (not used in the paper)
# Applies a strong perturbation to the background and a weak one to the lesion,
# blended with a Gaussian-softened mask to avoid a seam at the boundary.
# ---------------------------------------------------------------------------

def polyp_aware_perturbation(img_tensor, mask_tensor,
                             alpha_bg=1.0, alpha_fg=0.2, ratio=0.05):
    """result = mask * weak + (1 - mask) * strong. mask may be soft in [0, 1]."""
    img_strong = fourier_style_perturbation(img_tensor, alpha=alpha_bg, ratio=ratio)
    img_weak = fourier_style_perturbation(img_tensor, alpha=alpha_fg, ratio=ratio)
    mask_3c = mask_tensor.unsqueeze(0).expand_as(img_tensor)
    return mask_3c * img_weak + (1.0 - mask_3c) * img_strong


def polyp_aware_swap(img1_tensor, img2_tensor, mask_tensor,
                     alpha_bg=1.0, alpha_fg=0.2, ratio=0.05):
    """Foreground/background-adaptive amplitude swap with a donor image."""
    img_strong = fourier_style_swap(
        img1_tensor, img2_tensor, alpha=alpha_bg, ratio=ratio)
    img_weak = fourier_style_swap(
        img1_tensor, img2_tensor, alpha=alpha_fg, ratio=ratio)
    mask_3c = mask_tensor.unsqueeze(0).expand_as(img1_tensor)
    return mask_3c * img_weak + (1.0 - mask_3c) * img_strong


class PolypAwareAugmentor:
    """Foreground/background-adaptive Fourier augmentor (optional).

    The GT mask is Gaussian-softened before blending so that the strong/weak
    perturbation boundary does not create a high-frequency seam on the lesion
    edge -- exactly where segmentation is hardest.
    """

    def __init__(self,
                 prob=0.7,
                 alpha_bg=1.0,    # background: strong perturbation
                 alpha_fg=0.2,    # lesion: weak perturbation
                 ratio=0.05,
                 mode='both',     # 'perturb' / 'swap' / 'both'
                 smooth_mask=True,
                 mask_blur_sigma=4.0,
                 mask_blur_ksize=15):
        self.prob = prob
        self.alpha_bg = alpha_bg
        self.alpha_fg = alpha_fg
        self.ratio = ratio
        self.mode = mode
        self.smooth_mask = smooth_mask
        self.mask_blur_sigma = mask_blur_sigma
        self.mask_blur_ksize = mask_blur_ksize

    @staticmethod
    def _gaussian_kernel(ksize, sigma, device, dtype):
        ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
        g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        k2d = torch.outer(g, g)
        return k2d.view(1, 1, ksize, ksize)

    def _smooth(self, mask_hw):
        """Soften a hard mask (H, W) -> soft mask (H, W) in [0, 1]."""
        if not self.smooth_mask:
            return mask_hw
        k = self._gaussian_kernel(self.mask_blur_ksize, self.mask_blur_sigma,
                                  mask_hw.device, mask_hw.dtype)
        m = mask_hw.view(1, 1, *mask_hw.shape)
        m = F.conv2d(m, k, padding=self.mask_blur_ksize // 2)
        return m.view(*mask_hw.shape).clamp_(0.0, 1.0)

    def __call__(self, imgs, masks):
        """imgs: (B, C, H, W); masks: (B, 1, H, W) GT masks."""
        B = imgs.shape[0]
        augmented = imgs.clone()

        for i in range(B):
            if np.random.random() > self.prob:
                continue

            mask_i = self._smooth(masks[i, 0])   # soft mask (H, W)

            if self.mode == 'perturb':
                use_swap = False
            elif self.mode == 'swap':
                use_swap = True
            else:
                use_swap = np.random.random() > 0.5

            if use_swap and B > 1:
                j = np.random.randint(0, B)
                while j == i:
                    j = np.random.randint(0, B)
                augmented[i] = polyp_aware_swap(
                    imgs[i], imgs[j], mask_i,
                    alpha_bg=self.alpha_bg,
                    alpha_fg=self.alpha_fg,
                    ratio=self.ratio)
            else:
                augmented[i] = polyp_aware_perturbation(
                    imgs[i], mask_i,
                    alpha_bg=self.alpha_bg,
                    alpha_fg=self.alpha_fg,
                    ratio=self.ratio)

        return augmented


# ---------------------------------------------------------------------------
# Optional: worst-case (adversarial) low-frequency perturbation (not in paper)
# ---------------------------------------------------------------------------

def adversarial_lowfreq_perturbation(imgs, model,
                                     beta=0.05, epsilon=1.0,
                                     step_size=1.0, n_steps=1):
    """Search, inside the low-frequency window Ω_β, for the perturbation that
    most changes the prediction, and use it to build the style variant x'.
    This is a hard-positive-mining analogue: instead of a random style drift,
    it moves along the model's currently most fragile low-frequency direction.

    Plug-and-play guarantees:
      - the model is treated as a black box; gradients are taken w.r.t. the
        frequency-domain perturbation only, never written to model.grad
        (torch.autograd.grad(only_inputs=True));
      - the model is temporarily set to eval() so the adversarial forward pass
        does not pollute BatchNorm running statistics;
      - a detached image is returned; inference never calls this function.

    Args:
        imgs:      (B, C, H, W) normalized training batch
        model:     segmentation network f_theta
        beta:      low-frequency window ratio (Ω_β, default 0.05)
        epsilon:   L-inf bound on the multiplicative amplitude perturbation
        step_size: per-step size (n_steps=1 with step=epsilon is FGSM)
        n_steps:   number of adversarial steps (>1 is PGD-style: stronger, slower)
    Returns:
        x_adv: (B, C, H, W), detached
    """
    was_training = model.training
    model.eval()

    imgs = imgs.detach()
    B, C, H, W = imgs.shape

    # clean prediction as a fixed target (eval, no grad)
    with torch.no_grad():
        target = torch.sigmoid(model(imgs).float())

    # frequency decomposition (fp32)
    imgs_f = imgs.float()
    fft = torch.fft.fftshift(torch.fft.fft2(imgs_f, dim=(-2, -1)), dim=(-2, -1))
    amp = torch.abs(fft)
    phase = torch.angle(fft)

    cy, cx = H // 2, W // 2
    rh = max(int(H * beta), 1)
    rw = max(int(W * beta), 1)
    # delta lives only inside the window; zero-padded outside -> phase/high-freq intact
    pad = (cx - rw, W - (cx + rw + 1), cy - rh, H - (cy + rh + 1))

    delta = torch.zeros(B, C, 2 * rh + 1, 2 * rw + 1,
                        device=imgs.device, dtype=torch.float32,
                        requires_grad=True)

    def _reconstruct(d):
        d_full = F.pad(d, pad)                    # (B,C,H,W), zero outside window
        amp_p = amp * (1.0 + d_full)              # multiplicative, window only
        fft_p = amp_p * torch.exp(1j * phase)     # phase P unchanged
        x = torch.fft.ifft2(torch.fft.ifftshift(fft_p, dim=(-2, -1)),
                            dim=(-2, -1)).real
        return x

    for _ in range(n_steps):
        x_adv = _reconstruct(delta)
        pred = torch.sigmoid(model(x_adv).float())
        # maximize the gap between the perturbed and clean predictions
        loss_adv = F.mse_loss(pred, target)
        grad = torch.autograd.grad(loss_adv, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta = (delta + step_size * grad.sign()).clamp_(-epsilon, epsilon)
        delta.requires_grad_(True)

    with torch.no_grad():
        x_adv = _reconstruct(delta)

    if was_training:
        model.train()
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Unified augmentor: selects plain / polyp-aware / adversarial by flags.
# This is the entry point used by the paper (plain path by default).
# ---------------------------------------------------------------------------

class UGFECTv2Augmentor:
    """Produce the style variant x'. The defaults reproduce the paper's setting
    (plain random Fourier with mode='both'); the flags below are for ablation:

        use_adversarial=False, use_polyp_aware=False -> plain random Fourier
        use_polyp_aware=True                         -> foreground/background-aware
        use_adversarial=True                         -> adversarial (prob adv_prob),
                                                        else falls back to the above

    Call: aug(imgs, masks, model)
        - masks is needed only for the polyp-aware path (GT at train time)
        - model is needed only for the adversarial path
    """

    def __init__(self,
                 use_polyp_aware=False,
                 use_adversarial=False,
                 adv_prob=0.5,
                 prob=0.7,
                 alpha_plain=1.0,
                 alpha_bg=1.0,
                 alpha_fg=0.2,
                 ratio=0.05,
                 mode='both',
                 smooth_mask=True,
                 mask_blur_sigma=4.0,
                 mask_blur_ksize=15,
                 adv_beta=0.05,
                 adv_epsilon=1.0,
                 adv_step=1.0,
                 adv_steps=1):
        self.use_polyp_aware = use_polyp_aware
        self.use_adversarial = use_adversarial
        self.adv_prob = adv_prob
        self.adv_beta = adv_beta
        self.adv_epsilon = adv_epsilon
        self.adv_step = adv_step
        self.adv_steps = adv_steps

        self._plain = FourierDomainAugmentor(
            prob=prob, alpha=alpha_plain, ratio=ratio, mode=mode)
        self._polyp = PolypAwareAugmentor(
            prob=prob, alpha_bg=alpha_bg, alpha_fg=alpha_fg,
            ratio=ratio, mode=mode,
            smooth_mask=smooth_mask,
            mask_blur_sigma=mask_blur_sigma,
            mask_blur_ksize=mask_blur_ksize)

    def __call__(self, imgs, masks=None, model=None):
        # adversarial path (triggered with probability adv_prob)
        if self.use_adversarial and (model is not None) \
                and (np.random.random() < self.adv_prob):
            return adversarial_lowfreq_perturbation(
                imgs, model,
                beta=self.adv_beta, epsilon=self.adv_epsilon,
                step_size=self.adv_step, n_steps=self.adv_steps)

        # otherwise: random perturbation (polyp-aware or plain), no grad
        with torch.no_grad():
            if self.use_polyp_aware:
                assert masks is not None, "polyp-aware mode requires masks"
                return self._polyp(imgs, masks)
            return self._plain(imgs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    B, C, H, W = 4, 3, 352, 352

    # ---- foreground/background-aware perturbation ----
    imgs = torch.rand(B, C, H, W)
    masks = torch.zeros(B, 1, H, W)
    masks[:, :, 100:200, 120:220] = 1.0

    aug_a = PolypAwareAugmentor(prob=1.0, alpha_bg=1.0, alpha_fg=0.2,
                                ratio=0.05, mode='both', smooth_mask=True)
    imgs_aug = aug_a(imgs, masks)
    diff = (imgs_aug - imgs).abs()
    mask_exp = masks.expand_as(diff)
    fg_diff = diff[mask_exp == 1].mean().item()
    bg_diff = diff[mask_exp == 0].mean().item()
    print(f"[fg/bg] lesion mean change:     {fg_diff:.4f}")
    print(f"[fg/bg] background mean change: {bg_diff:.4f}")
    print(f"[fg/bg] background/lesion ratio: {bg_diff/fg_diff:.2f}x  (should be > 1)")

    # ---- adversarial low-frequency perturbation (dummy model) ----
    import torch.nn as nn
    dummy = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 1, 1)
    )
    dummy.train()
    x_adv = adversarial_lowfreq_perturbation(
        imgs, dummy, beta=0.05, epsilon=1.0, step_size=1.0, n_steps=2)
    print(f"\n[adv] x_adv shape: {tuple(x_adv.shape)} | requires_grad={x_adv.requires_grad}")
    print(f"[adv] mean change after perturbation: {(x_adv - imgs).abs().mean().item():.4f}")
    print(f"[adv] model params left untouched (all None): "
          f"{all(p.grad is None for p in dummy.parameters())}")
    print(f"[adv] model restored to train(): {dummy.training}")

    # ---- unified augmentor ----
    gen = UGFECTv2Augmentor(use_polyp_aware=True, use_adversarial=True, adv_prob=0.5)
    out = gen(imgs, masks, dummy)
    print(f"\n[gen] output shape: {tuple(out.shape)}")

"""
losses.py — uncertainty-gated consistency and the segmentation loss.

This is the reference implementation of the gated consistency used by the
style axis (Eq. 3 in the paper). The confidence gate is identical to the one
in ``scale_consistency.py``: a per-pixel binary-entropy gate

    C = 1 - H(y_hat) / log 2,     H(p) = -p log p - (1-p) log(1-p),

read from the *clean* teacher prediction (stop-gradient). Consistency is
enforced only where the teacher is confident, which is what makes the Fourier
style perturbation useful instead of inert; gating it by teacher confidence
gives a further improvement.

All losses are binary (single-channel logits + sigmoid). Training-only:
none of this runs at inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def confidence_map(teacher_prob, eps=1e-6):
    """C = 1 - H_tilde(teacher). teacher_prob: sigmoid probs (B,1,H,W), detached."""
    pc = teacher_prob.clamp(eps, 1.0 - eps)
    ent = -(pc * pc.log() + (1.0 - pc) * (1.0 - pc).log())
    return 1.0 - ent / math.log(2.0)


def gated_consistency_loss(teacher_prob, student_logits, uncertainty_weighted=True,
                           eps=1e-6):
    """Confidence-gated MSE between a stop-grad teacher and a perturbed student.

    teacher_prob:     sigmoid probs of the clean prediction, detached (B,1,H,W)
    student_logits:   logits of the perturbed view (B,1,Hs,Ws); resized to match
    Mirrors the gate of scale_consistency_loss so the style and scale axes share
    one confidence map.
    """
    if student_logits.shape[2:] != teacher_prob.shape[2:]:
        student_logits = F.interpolate(student_logits, size=teacher_prob.shape[2:],
                                       mode='bilinear', align_corners=False)
    student_prob = torch.sigmoid(student_logits)
    if not uncertainty_weighted:
        return F.mse_loss(student_prob, teacher_prob)
    conf = confidence_map(teacher_prob, eps)
    return (F.mse_loss(student_prob, teacher_prob, reduction='none') * conf).mean()


class DiceBCELoss(nn.Module):
    """Segmentation loss L_seg = bce_weight * BCE + (1 - bce_weight) * softDice."""

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(1, 2, 3)) + self.smooth
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth
        dice = 1 - (num / den).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


def feature_consistency_loss(feats_a, feats_b):
    """Optional: MSE between matched shallow encoder features of two views."""
    loss, n = 0.0, 0
    for fa, fb in zip(feats_a, feats_b):
        if fa.shape[2:] != fb.shape[2:]:
            fb = F.interpolate(fb, size=fa.shape[2:], mode='bilinear',
                               align_corners=False)
        loss = loss + F.mse_loss(fa, fb)
        n += 1
    return loss / max(n, 1)


if __name__ == '__main__':
    teacher = torch.sigmoid(torch.randn(2, 1, 64, 64)).detach()
    student = torch.randn(2, 1, 48, 48, requires_grad=True)  # different scale
    lc = gated_consistency_loss(teacher, student)
    lc.backward()
    print(f"gated consistency = {lc.item():.5f}  finite={torch.isfinite(lc).item()}"
          f"  student.grad set={student.grad is not None}")
    seg = DiceBCELoss()(torch.randn(2, 1, 64, 64), (teacher > 0.5).float())
    print(f"DiceBCE = {seg.item():.5f}")
    print("losses OK")

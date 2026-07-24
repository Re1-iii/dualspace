"""
Training script for Dual-Space Stability on polyp segmentation.

Implements the three stability axes of the paper on a PVTv2-B2 + U-Net backbone:
  style  (Section 3.2/3.3) -- Fourier amplitude perturbation + uncertainty-gated consistency
  scale  (Section 3.4)     -- multi-scale consistency under the same confidence gate
  weight (Section 3.5)     -- dense flat-minima weight averaging (SWAD)

Usage
-----
  # Baseline (no domain-generalization components)
  python train.py --data_root ./data --epochs 100 --mode baseline

  # Style axis only (Fourier perturbation + uncertainty-gated consistency)
  python train.py --data_root ./data --epochs 100 --mode dg

  # Style + weight axes
  python train.py --data_root ./data --epochs 100 --mode dg --use_swad

  # Style + scale axes
  python train.py --data_root ./data --epochs 100 --mode dg --use_scale_consistency

  # Full model of the paper (style + scale + weight)
  python train.py --data_root ./data --epochs 100 --mode dg \
      --use_swad --use_scale_consistency

  # Simple weight averaging over the last N epochs (SWA control)
  python train.py --data_root ./data --epochs 100 --mode dg --use_swad --swad_mode tail

  # Resume an interrupted run ('auto' locates the checkpoint of this run)
  python train.py --data_root ./data --epochs 100 --mode dg --use_swad --resume auto

Optional components (not used for the reported results; provided for ablation):
  --use_polyp_aware        lesion-aware frequency perturbation
  --use_adversarial        adversarial worst-case low-frequency perturbation
  --use_feat_consistency   multi-level feature consistency

Pretrained encoder
------------------
  Download pvt_v2_b2.pth into ./pretrained/
  https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
import time
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PolypDataset, get_test_datasets
from models.pvtv2_unet import PVTv2UNet
# ================= Losses and feature consistency =================
from dual_space.losses import (DiceBCELoss, DiceBCEWithUncertaintyConsistency,
                               feature_consistency_loss)
from dual_space.fourier_style import DualSpaceStyleAugmentor
from dual_space.swad import SWAD, update_bn
from dual_space.scale_consistency import random_scale_size, scale_consistency_loss


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_run_tag(args):
    """Build a unique run tag from the enabled components so that different
    ablations do not overwrite each other's checkpoints."""
    if args.run_name:
        return args.run_name
    if args.mode == 'baseline':
        base = 'baseline'
    else:
        parts = []
        if args.use_polyp_aware:
            parts.append('A')
        if args.use_adversarial:
            parts.append('B')
        if args.use_feat_consistency:
            parts.append('C')
        base = 'dg' + ('_' + ''.join(parts) if parts else '')
    if getattr(args, 'use_scale_consistency', False):
        base += '_scale'
    if getattr(args, 'use_swad', False):
        base += '_swadtail' if getattr(args, 'swad_mode', 'auto') == 'tail' else '_swad'
    if args.mode == 'dg':
        if getattr(args, 'no_uncertainty_gate', False):
            base += '_nogate'
        if abs(getattr(args, 'aug_alpha', 1.0) - 1.0) > 1e-9:
            base += f"_a{args.aug_alpha}"
        if abs(getattr(args, 'aug_ratio', 0.05) - 0.05) > 1e-9:
            base += f"_b{args.aug_ratio}"
        if abs(getattr(args, 'aug_prob', 0.7) - 0.7) > 1e-9:
            base += f"_p{args.aug_prob}"
    return base


def _json_safe(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    smooth = 1e-6
    intersection = (pred_bin * target).sum()
    dice = (2.0 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)
    union = pred_bin.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return dice.item(), iou.item()


def evaluate(model, datasets, device):
    model.eval()
    results = {}
    for name, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        dices, ious = [], []
        with torch.no_grad():
            for imgs, masks, _ in loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                with autocast('cuda'):
                    preds = model(imgs)
                preds = torch.sigmoid(preds.float())
                for i in range(imgs.size(0)):
                    dice, iou = compute_metrics(preds[i], masks[i])
                    dices.append(dice)
                    ious.append(iou)
        results[name] = {'dice': np.mean(dices), 'iou': np.mean(ious), 'n': len(dices)}
    return results


@torch.no_grad()
def compute_source_val_loss(model, datasets, device, seen_names=('Kvasir', 'CVC-ClinicDB')):
    """Source-domain validation loss used for SWAD interval detection.
    Only the seen datasets are used; the unseen target domains are never touched."""
    model.eval()
    crit = DiceBCELoss(bce_weight=0.5)
    total, n = 0.0, 0
    for name in seen_names:
        if name not in datasets:
            continue
        loader = DataLoader(datasets[name], batch_size=1, shuffle=False, num_workers=0)
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device).float()
            with autocast('cuda'):
                pred = model(imgs)
                loss = crit(pred, masks)
            total += loss.item()
            n += 1
    return total / max(n, 1)


def save_ckpt(path, epoch, model, optimizer, scheduler, scaler,
              best_dice, best_results, swad, args):
    """Atomically write the full training state for --resume.
    `epoch` is the next epoch to run."""
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_dice': best_dice,
        'best_results': best_results,
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'rng_numpy': np.random.get_state(),
        'args': vars(args),
    }
    if swad is not None:
        ckpt['swad'] = swad.state_dict()
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)   # atomic replace, so an interrupted write cannot leave a partial file


def print_results(results, title=""):
    seen = ['Kvasir', 'CVC-ClinicDB']
    unseen = ['CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']
    
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")
    print(f"  {'Dataset':<22} | {'N':>5} | {'mDice':>8} | {'mIoU':>8} | {'Domain'}")
    print(f"  {'-'*64}")
    
    seen_dice, unseen_dice = [], []
    for name in seen + unseen:
        if name in results:
            r = results[name]
            domain = 'Seen' if name in seen else 'Unseen'
            print(f"  {name:<22} | {r['n']:>5} | {r['dice']:.4f}   | "
                  f"{r['iou']:.4f}   | {domain}")
            if name in seen:
                seen_dice.append(r['dice'])
            else:
                unseen_dice.append(r['dice'])
    
    print(f"  {'-'*64}")
    if seen_dice:
        print(f"  {'Seen Average':<22} |       | {np.mean(seen_dice):.4f}   |")
    if unseen_dice:
        print(f"  {'Unseen Average':<22} |       | {np.mean(unseen_dice):.4f}   |")
    if seen_dice and unseen_dice:
        gap = np.mean(seen_dice) - np.mean(unseen_dice)
        print(f"  {'Domain Gap':<22} |       | {gap:+.4f}   | "
              f"{'⚠ Large' if gap > 0.05 else '✓ Small'}")
    all_dice = seen_dice + unseen_dice
    if all_dice:
        print(f"  {'Overall Average':<22} |       | {np.mean(all_dice):.4f}   |")
    print(f"{'='*70}")


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    is_dg = (args.mode == 'dg')
    
    # datasets
    print(f"\n--- Loading datasets ---")
    train_ds = PolypDataset(
        img_dir=os.path.join(args.data_root, 'TrainDataset', 'image'),
        mask_dir=os.path.join(args.data_root, 'TrainDataset', 'masks'),
        img_size=args.img_size, augment=True
    )
    test_datasets = get_test_datasets(args.data_root, args.img_size)
    
    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    
    # pretrained encoder weights
    pretrained_path = os.path.join(args.data_root, '..', 'pretrained', 'pvt_v2_b2.pth')
    if not os.path.exists(pretrained_path):
        pretrained_path = os.path.join('.', 'pretrained', 'pvt_v2_b2.pth')
    if not os.path.exists(pretrained_path):
        pretrained_path = None
        print("  [WARNING] pvt_v2_b2.pth not found, training from scratch!")
    
    # model
    model = PVTv2UNet(
        num_classes=1,
        pretrained_path=pretrained_path,
        use_style_in=False
    ).to(device)
    
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    
    # style axis: perturbation generator (optional variants selected by flags)
    augmentor = None
    if is_dg:
        augmentor = DualSpaceStyleAugmentor(
            use_polyp_aware=args.use_polyp_aware,
            use_adversarial=args.use_adversarial,
            adv_prob=args.adv_prob,
            prob=args.aug_prob,
            alpha_plain=args.aug_alpha,
            alpha_bg=args.alpha_bg,
            alpha_fg=args.alpha_fg,
            ratio=args.aug_ratio,
            mode='both',
            smooth_mask=True,
            mask_blur_sigma=args.mask_blur_sigma,
            adv_beta=args.adv_beta,
            adv_epsilon=args.adv_epsilon,
            adv_step=args.adv_step,
            adv_steps=args.adv_steps,
        )
    
    mode_str = "DG" if is_dg else "Baseline"
    print(f"\n{'='*65}")
    print(f"  PVTv2-UNet {mode_str} | {total_p:.2f}M params")
    print(f"  Image: {args.img_size} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    if is_dg:
        print(f"  DG Components:")
        print(f"    Fourier Aug:    prob={args.aug_prob}, alpha={args.aug_alpha}")
        print(f"    Consistency:    weight={args.consistency_weight} (Uncertainty-Guided)")
        print(f"    [A] PolypAware:    {'ON' if args.use_polyp_aware else 'off'} "
              f"(alpha_bg={args.alpha_bg}, alpha_fg={args.alpha_fg})")
        print(f"    [B] Adversarial:   {'ON' if args.use_adversarial else 'off'} "
              f"(p={args.adv_prob}, eps={args.adv_epsilon}, steps={args.adv_steps})")
        print(f"    [C] FeatConsist:   {'ON' if args.use_feat_consistency else 'off'} "
              f"(mode={args.feat_mode}, w={args.feat_weight})")
        print(f"    [S] MultiScale:    {'ON' if args.use_scale_consistency else 'off'} "
              f"(range=[{args.scale_min},{args.scale_max}], w={args.scale_weight})")
    print(f"{'='*65}\n")
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if is_dg:
        crit_kwargs = dict(bce_weight=0.5, consistency_weight=args.consistency_weight)
        if args.no_uncertainty_gate:
            crit_kwargs['use_uncertainty_gate'] = False
        criterion = DiceBCEWithUncertaintyConsistency(**crit_kwargs)
    else:
        criterion = DiceBCELoss(bce_weight=0.5)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler('cuda')
    
    run_tag = build_run_tag(args)
    save_dir = os.path.join(args.data_root, f'outputs_pvt_{run_tag}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Checkpoints -> {save_dir}/  (best/last_pvt_{run_tag}.pth)\n")

    swad = None
    if args.use_swad:
        swad = SWAD(mode=args.swad_mode, n_start=args.swad_start,
                    n_end=args.swad_end, tol=args.swad_tol,
                    tail=args.swad_tail, total_epochs=args.epochs)
        print(f"  SWAD: mode={args.swad_mode} "
              f"(Ns={args.swad_start}, Ne={args.swad_end}, r={args.swad_tol}, "
              f"tail={args.swad_tail}) | val=source domain (Kvasir+ClinicDB)\n")

    best_dice = 0.0
    best_results = {}

    # --- resume: restore the full training state ---
    start_epoch = 0
    ckpt_path = os.path.join(save_dir, f'ckpt_pvt_{run_tag}.pth')
    if args.resume:
        rp = ckpt_path if args.resume == 'auto' else args.resume
        if os.path.exists(rp):
            ck = torch.load(rp, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            scheduler.load_state_dict(ck['scheduler'])
            scaler.load_state_dict(ck['scaler'])
            best_dice = ck.get('best_dice', 0.0)
            best_results = ck.get('best_results', {})
            start_epoch = ck['epoch']
            try:
                torch.set_rng_state(ck['rng_torch'])
                if ck.get('rng_cuda') is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ck['rng_cuda'])
                np.random.set_state(ck['rng_numpy'])
            except Exception as e:
                print(f"  [WARN] could not restore RNG state ({e}), continuing.")
            if swad is not None and 'swad' in ck:
                swad.load_state_dict(ck['swad'])
            print(f"  >>> Resumed from {rp} @ epoch {start_epoch} "
                  f"(best so far mDice={best_dice:.4f})\n")
        else:
            print(f"  [WARN] --resume given but {rp} not found; training from scratch.\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum, seg_sum, cons_sum, feat_sum, scale_sum, n_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()
        
        for imgs, masks, _ in train_ld:
            imgs = imgs.to(device)
            masks = masks.to(device).float()
            
            optimizer.zero_grad()
            
            if is_dg:
                # build the perturbed view x'
                # not wrapped in no_grad: the adversarial branch manages its own
                # gradients and returns a detached tensor; the plain branch uses no_grad.
                imgs_aug = augmentor(imgs, masks, model)
                
                with autocast('cuda'):
                    # two forward passes, keeping shallow features for feature consistency
                    preds_orig, feats_orig = model(imgs, return_features=True)
                    preds_aug,  feats_aug  = model(imgs_aug, return_features=True)
                    # output level: uncertainty-gated consistency + segmentation loss
                    loss, seg_loss, cons_loss = criterion(preds_orig, preds_aug, masks)
                    # optional: multi-level feature consistency
                    if args.use_feat_consistency:
                        loss_feat = feature_consistency_loss(
                            feats_orig, feats_aug, mode=args.feat_mode)
                        loss = loss + args.feat_weight * loss_feat
                        feat_val = float(loss_feat.detach())
                    else:
                        feat_val = 0.0
                
                cons_sum += cons_loss * imgs.size(0)
                feat_sum += feat_val * imgs.size(0)
            else:
                # baseline mode
                with autocast('cuda'):
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                seg_loss = loss.item()
            
            # keep the full-resolution teacher (stop-grad) for the scale axis,
            # which must happen before backward
            if is_dg and args.use_scale_consistency:
                p_base_teacher = torch.sigmoid(preds_orig).detach()

            scaler.scale(loss).backward()   # frees the base + Fourier graph

            # --- scale axis: separate forward/backward so peak memory stays at two passes ---
            if is_dg and args.use_scale_consistency:
                with autocast('cuda'):
                    ss = random_scale_size(args.img_size, args.scale_min, args.scale_max)
                    x_scaled = F.interpolate(imgs, size=(ss, ss),
                                             mode='bilinear', align_corners=False)
                    pred_scaled = model(x_scaled)
                    loss_scale = args.scale_weight * scale_consistency_loss(
                        p_base_teacher, pred_scaled)
                scaler.scale(loss_scale).backward()
                scale_sum += float(loss_scale.detach()) * imgs.size(0)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            loss_sum += loss.item() * imgs.size(0)
            seg_sum += seg_loss * imgs.size(0)
            n_samples += imgs.size(0)
        
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        
        # evaluation
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            results = evaluate(model, test_datasets, device)
            all_dice = np.mean([r['dice'] for r in results.values()])
            
            if is_dg:
                print(f"Ep [{epoch+1:3d}/{args.epochs}] "
                      f"Loss: {loss_sum/n_samples:.4f} "
                      f"(seg={seg_sum/n_samples:.4f} cons={cons_sum/n_samples:.4f} "
                      f"feat={feat_sum/n_samples:.4f} scale={scale_sum/n_samples:.4f}) | "
                      f"mDice: {all_dice:.4f} | LR: {lr:.1e} | {elapsed:.0f}s")
            else:
                print(f"Ep [{epoch+1:3d}/{args.epochs}] "
                      f"Loss: {loss_sum/n_samples:.4f} | "
                      f"mDice: {all_dice:.4f} | LR: {lr:.1e} | {elapsed:.0f}s")
            
            for name in ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB',
                         'ETIS-LaribPolypDB', 'CVC-300']:
                if name in results:
                    tag = '(S)' if name in ['Kvasir', 'CVC-ClinicDB'] else '(U)'
                    print(f"    {name:<22} {tag} Dice={results[name]['dice']:.4f} "
                          f"IoU={results[name]['iou']:.4f}")
            
            if all_dice > best_dice:
                best_dice = all_dice
                best_results = {k: dict(v) for k, v in results.items()}
                best_results['epoch'] = epoch + 1
                torch.save(model.state_dict(),
                          os.path.join(save_dir, f'best_pvt_{run_tag}.pth'))
                print(f"    >>> New Best! mDice={all_dice:.4f}")
        else:
            if (epoch + 1) % 5 == 0:
                if is_dg:
                    print(f"Ep [{epoch+1:3d}/{args.epochs}] "
                          f"Loss: {loss_sum/n_samples:.4f} "
                          f"(seg={seg_sum/n_samples:.4f} cons={cons_sum/n_samples:.4f} "
                          f"feat={feat_sum/n_samples:.4f} scale={scale_sum/n_samples:.4f}) | "
                          f"LR: {lr:.1e} | {elapsed:.0f}s")
                else:
                    print(f"Ep [{epoch+1:3d}/{args.epochs}] "
                          f"Loss: {loss_sum/n_samples:.4f} | "
                          f"LR: {lr:.1e} | {elapsed:.0f}s")
    
        # --- SWAD: fold the current weights into the running average ---
        if swad is not None:
            vloss = compute_source_val_loss(model, test_datasets, device) \
                    if args.swad_mode == 'auto' else None
            swad.update(model, vloss)

        # --- periodically save the full training state ---
        if (epoch + 1) % args.ckpt_every == 0 or (epoch + 1) == args.epochs:
            save_ckpt(ckpt_path, epoch + 1, model, optimizer, scheduler, scaler,
                      best_dice, best_results, swad, args)

    # save the final weights and the best metrics of this run
    torch.save(model.state_dict(), os.path.join(save_dir, f'last_pvt_{run_tag}.pth'))
    with open(os.path.join(save_dir, f'results_{run_tag}.json'), 'w', encoding='utf-8') as f:
        json.dump({'run_tag': run_tag, 'config': vars(args), 'best': best_results},
                  f, indent=2, ensure_ascii=False, default=_json_safe)

    print_results(best_results,
                  f"PVTv2-UNet {mode_str} Best (Epoch {best_results.get('epoch', '?')})")
    print(f"\n  Saved: best_pvt_{run_tag}.pth | last_pvt_{run_tag}.pth | results_{run_tag}.json")

    # --- SWAD: finalize, re-estimate BN statistics, evaluate and save ---
    if swad is not None:
        print(f"\n--- Finalizing SWAD --- {swad.info()}")
        swad_model = swad.finalize_into(model)
        if swad_model is None:
            print("  [WARN] SWAD produced no average (no stable low-loss band detected). "
                  "Try --swad_mode tail, or lower --swad_start.")
        else:
            update_bn(train_ld, swad_model, device, max_batches=args.swad_bn_batches)
            swad_results = evaluate(swad_model, test_datasets, device)
            print_results(swad_results, "PVTv2-UNet + SWAD")
            torch.save(swad_model.state_dict(),
                       os.path.join(save_dir, f'swad_pvt_{run_tag}.pth'))
            with open(os.path.join(save_dir, f'results_{run_tag}_swad.json'),
                      'w', encoding='utf-8') as f:
                json.dump({'run_tag': run_tag, 'swad_info': swad.info(),
                           'config': vars(args), 'swad': swad_results},
                          f, indent=2, ensure_ascii=False, default=_json_safe)
            print(f"  Saved: swad_pvt_{run_tag}.pth | results_{run_tag}_swad.json")

    return best_results


def main():
    parser = argparse.ArgumentParser(description='PVTv2-UNet Polyp Segmentation (UG-FECT v2)')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'dg'],
                        help='baseline=plain training, dg=enable the style axis')
    parser.add_argument('--run_name', type=str, default='',
                        help='custom run name; if empty it is derived from the enabled components')
    # style-axis hyper-parameters
    parser.add_argument('--aug_prob', type=float, default=0.7)
    parser.add_argument('--aug_alpha', type=float, default=1.0)
    parser.add_argument('--consistency_weight', type=float, default=1.0)
    parser.add_argument('--aug_ratio', type=float, default=0.05,
                        help='Fourier low-freq window ratio beta (default 0.05)')
    parser.add_argument('--no_uncertainty_gate', action='store_true',
                        help='ablation: disable the confidence gate in the style-consistency loss')

    # ===== optional components (ablation only) =====
    parser.add_argument('--use_polyp_aware', action='store_true',
                        help='lesion-aware frequency perturbation (stronger on background)')
    parser.add_argument('--use_adversarial', action='store_true',
                        help='adversarial worst-case low-frequency perturbation')
    parser.add_argument('--use_feat_consistency', action='store_true',
                        help='multi-level feature consistency')

    # lesion-aware perturbation
    parser.add_argument('--alpha_bg', type=float, default=1.0, help='perturbation strength on the background')
    parser.add_argument('--alpha_fg', type=float, default=0.2, help='perturbation strength inside the lesion')
    parser.add_argument('--mask_blur_sigma', type=float, default=4.0,
                        help='Gaussian sigma used to soften the mask')
    # adversarial perturbation
    parser.add_argument('--adv_prob', type=float, default=0.5,
                        help='per-batch probability of using the adversarial perturbation')
    parser.add_argument('--adv_beta', type=float, default=0.05, help='low-frequency window ratio')
    parser.add_argument('--adv_epsilon', type=float, default=1.0, help='upper bound on the adversarial magnitude')
    parser.add_argument('--adv_step', type=float, default=1.0, help='adversarial step size')
    parser.add_argument('--adv_steps', type=int, default=1, help='number of adversarial steps (1 = FGSM)')
    # feature consistency
    parser.add_argument('--feat_mode', type=str, default='stat',
                        choices=['stat', 'full'], help='feature alignment mode')
    parser.add_argument('--feat_weight', type=float, default=0.1,
                        help='feature consistency weight (stat and full have different scales)')

    # ===== weight axis: flat-minima weight averaging (SWAD) =====
    parser.add_argument('--use_swad', action='store_true', help='enable SWAD weight averaging')
    parser.add_argument('--swad_mode', type=str, default='auto', choices=['auto', 'tail'],
                        help='auto = overfit-aware SWAD, tail = plain average of the last N epochs')
    parser.add_argument('--swad_start', type=int, default=3, help='SWAD optimum patience Ns(epoch)')
    parser.add_argument('--swad_end', type=int, default=6, help='SWAD overfit patience Ne(epoch)')
    parser.add_argument('--swad_tol', type=float, default=1.3, help='SWAD tolerance ratio r')
    parser.add_argument('--swad_tail', type=int, default=20, help='number of trailing epochs averaged in tail mode')
    parser.add_argument('--swad_bn_batches', type=int, default=200, help='number of batches used to re-estimate BN statistics')

    # ===== resume =====
    parser.add_argument('--resume', type=str, default='',
                        help="resume: 'auto' locates the checkpoint of this run, or give a .pth path")
    parser.add_argument('--ckpt_every', type=int, default=1,
                        help='how often to save a full checkpoint (optimizer, scheduler, SWAD state)')

    # ===== scale axis: multi-scale consistency =====
    parser.add_argument('--use_scale_consistency', action='store_true',
                        help='enable multi-scale consistency')
    parser.add_argument('--scale_min', type=float, default=0.5, help='lower bound of the random scale factor')
    parser.add_argument('--scale_max', type=float, default=1.0, help='upper bound of the random scale factor')
    parser.add_argument('--scale_weight', type=float, default=1.0, help='scale consistency weight')
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()

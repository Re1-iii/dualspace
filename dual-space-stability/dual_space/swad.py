"""
Weight axis — Stochastic Weight Averaging Densely (SWAD).

Orthogonal to the input-space consistency: SWAD adds no loss, no augmentation,
and no architecture change. It densely averages the weights over a stable
low-loss part of the source-domain validation-loss trajectory, which is intended
to favour a flatter solution that
generalizes better across domains. Fully plug-and-play: the result is just a
better set of weights, with zero inference overhead.

Two modes:
  'auto' : overfit-aware SWAD (paper version) -- the start t_s and end t_e of
           the flat interval are detected automatically from source val loss.
  'tail' : plain SWA -- average the last `tail` epochs (robust fallback / SWA
           baseline).

Important:
  1. After averaging, BatchNorm running stats are stale; call update_bn to
     recompute them on training data.
  2. In 'auto' mode, val_loss MUST come from the source (seen) domain. Using an
     unseen target domain would be test-set leakage.
"""
import copy
from collections import deque

import torch


class SWAD:
    def __init__(self, mode='auto', n_start=3, n_end=6, tol=1.3,
                 tail=20, total_epochs=100):
        assert mode in ('auto', 'tail')
        self.mode = mode
        self.n_start = n_start        # optimum patience Ns
        self.n_end = n_end            # overfit patience Ne
        self.tol = tol                # tolerance ratio r
        self.tail = tail
        self.total_epochs = total_epochs

        self.losses = []
        self.snapshots = deque(maxlen=n_start + 1)
        self.avg = None
        self.n_avg = 0
        self.ts = None
        self.te = None
        self.l_ts = None
        self.frozen = False
        self.epoch = -1

    @staticmethod
    def _cpu_state(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _add(self, state):
        """Fold a state_dict into the running average (float tensors only;
        integer buffers take the latest value)."""
        if self.avg is None:
            self.avg = {}
            for k, v in state.items():
                self.avg[k] = v.clone().float() if v.is_floating_point() else v.clone()
            self.n_avg = 1
        else:
            self.n_avg += 1
            for k in self.avg:
                if self.avg[k].is_floating_point():
                    self.avg[k].add_((state[k].float() - self.avg[k]) / self.n_avg)
                else:
                    self.avg[k] = state[k].clone()

    def update(self, model, val_loss=None):
        self.epoch += 1
        i = self.epoch
        if self.frozen:
            return

        if self.mode == 'tail':
            # plain SWA: average the last `tail` epochs
            if i >= self.total_epochs - self.tail:
                self._add(self._cpu_state(model))
            return

        # ---- mode == 'auto': overfit-aware SWAD ----
        assert val_loss is not None, "auto mode needs a source val_loss every epoch"
        state = self._cpu_state(model)
        self.snapshots.append(state)
        self.losses.append(float(val_loss))

        if self.ts is None:
            # start: l[i-Ns] is the min of the last Ns+1 -> a stable optimum
            if i >= self.n_start:
                window = self.losses[i - self.n_start: i + 1]
                if self.losses[i - self.n_start] == min(window):
                    self.ts = i - self.n_start
                    self.l_ts = self.losses[self.ts]
                    for s in self.snapshots:        # deque holds epochs ts..i
                        self._add(s)
        elif self.te is None:
            self._add(state)                        # include the current epoch
            # end: the last Ne val losses are all > tol * l_ts -> flat region left
            if i >= self.ts + self.n_end:
                recent = self.losses[i - self.n_end + 1: i + 1]
                if min(recent) > self.tol * self.l_ts:
                    self.te = i
                    self.frozen = True

    def has_avg(self):
        return self.avg is not None

    def finalize_into(self, model):
        """Write the averaged weights into a deep copy of `model` and return it
        (BN stats still need update_bn). If 'auto' never captured a flat region,
        fall back to the last few snapshots in the deque."""
        if self.avg is None:
            if len(self.snapshots) == 0:
                return None
            for s in self.snapshots:
                self._add(s)
        swad_model = copy.deepcopy(model)
        tgt = swad_model.state_dict()
        for k in tgt:
            tgt[k].copy_(self.avg[k].to(tgt[k].device, tgt[k].dtype))
        return swad_model

    def info(self):
        return {'mode': self.mode, 'ts': self.ts, 'te': self.te,
                'n_avg': self.n_avg, 'frozen': self.frozen,
                'n_val': len(self.losses)}

    def state_dict(self):
        """Serialize SWAD state (for resuming). Once ts is set the snapshots are
        no longer needed, so they are dropped to save space."""
        return {
            'mode': self.mode, 'n_start': self.n_start, 'n_end': self.n_end,
            'tol': self.tol, 'tail': self.tail, 'total_epochs': self.total_epochs,
            'losses': list(self.losses),
            'snapshots': list(self.snapshots) if self.ts is None else [],
            'avg': self.avg, 'n_avg': self.n_avg,
            'ts': self.ts, 'te': self.te, 'l_ts': self.l_ts,
            'frozen': self.frozen, 'epoch': self.epoch,
        }

    def load_state_dict(self, sd):
        self.mode = sd['mode']
        self.n_start = sd['n_start']; self.n_end = sd['n_end']; self.tol = sd['tol']
        self.tail = sd['tail']; self.total_epochs = sd['total_epochs']
        self.losses = list(sd['losses'])
        self.snapshots = deque(sd['snapshots'], maxlen=self.n_start + 1)
        self.avg = sd['avg']; self.n_avg = sd['n_avg']
        self.ts = sd['ts']; self.te = sd['te']; self.l_ts = sd['l_ts']
        self.frozen = sd['frozen']; self.epoch = sd['epoch']


@torch.no_grad()
def update_bn(loader, model, device, max_batches=None):
    """Recompute BN running mean/var (required after SWAD/SWA averaging).
    momentum=None -> cumulative moving average, i.e. re-estimate on this data."""
    momenta = {}
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            momenta[m] = m.momentum
            m.momentum = None
    if not momenta:
        return  # no BN (e.g. pure LayerNorm): nothing to recompute

    was_training = model.training
    model.train()
    n = 0
    for batch in loader:
        imgs = batch[0].to(device)
        model(imgs)
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    for m, mom in momenta.items():
        m.momentum = mom
    if not was_training:
        model.eval()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch.nn as nn

    # ---- test 1: correctness of tail-mode averaging ----
    class Tiny(nn.Module):
        def __init__(self, val):
            super().__init__()
            self.w = nn.Parameter(torch.full((4,), float(val)))
    sw = SWAD(mode='tail', tail=3, total_epochs=5)
    for ep in range(5):                       # epoch weight = ep
        sw.update(Tiny(ep))
    sw.finalize_into(Tiny(0))                  # trigger finalize (no BN)
    got = sw.avg['w'].mean().item()
    print(f"[tail] mean of last 3 epochs (2,3,4) expect=3.0, got={got:.4f} "
          f"-> {'OK' if abs(got-3.0)<1e-6 else 'FAIL'} | n_avg={sw.n_avg}")

    # ---- test 2: auto-mode ts/te detection ----
    losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.46, 0.50, 0.55, 0.60, 0.66, 0.70]
    sw2 = SWAD(mode='auto', n_start=3, n_end=3, tol=1.1, total_epochs=len(losses))
    for ep, l in enumerate(losses):
        sw2.update(Tiny(ep), val_loss=l)
    # ts should land on the optimum epoch 5 (loss 0.44); te fires after overfitting
    print(f"[auto] ts={sw2.ts} (expect 5, the optimum), te={sw2.te}, n_avg={sw2.n_avg}, frozen={sw2.frozen}")
    print(f"[auto] {'OK' if sw2.ts == 5 and sw2.te is not None else 'CHECK'}")

    # ---- test 3: update_bn recomputes BN stats ----
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    bn = net[1]
    before = bn.running_mean.clone()
    loader = [(torch.randn(4, 3, 32, 32),) for _ in range(5)]
    update_bn(loader, net, device='cpu', max_batches=5)
    changed = not torch.allclose(before, bn.running_mean)
    print(f"[update_bn] BN running_mean updated (True)={changed} | model.train()={net.training}")

    # ---- test 4: state_dict round-trip (resume) ----
    sd = sw2.state_dict()
    sw3 = SWAD(mode='auto', n_start=3, n_end=3, tol=1.1, total_epochs=12)
    sw3.load_state_dict(sd)
    rt_ok = (sw3.ts == sw2.ts and sw3.te == sw2.te and sw3.n_avg == sw2.n_avg
             and sw3.epoch == sw2.epoch and torch.allclose(sw3.avg['w'], sw2.avg['w']))
    print(f"[state_dict] round-trip consistent (True)={rt_ok}")
    print("[swad] all passed" if (abs(got-3.0) < 1e-6 and sw2.ts == 5 and changed and rt_ok)
          else "[swad] problem")

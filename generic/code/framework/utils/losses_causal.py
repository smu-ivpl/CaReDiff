"""
DiffusionLossCoarse
===================
Adds an explicit coarse-to-fine cross-entropy term on top of the baseline
`DiffusionLoss`. The coarse head predicts the listener's 8-class facial
expression distribution per timestep; we supervise it with a masked soft
cross-entropy against the ground-truth distribution (target_emotion[..., 17:25]).

Padded / invalid frames have an all-zero target distribution and are masked out
(soft-CE against an all-zero label is naturally zero, but we additionally
renormalise valid frames to a proper distribution).
"""
import torch
import torch.nn.functional as F

from framework.utils.losses import DiffusionLoss


class DiffusionLossCoarse(DiffusionLoss):
    def __init__(self, w_coarse: float = 0.5, coarse_emo_start: int = 17,
                 coarse_classes: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.w_coarse = float(w_coarse)
        self.coarse_emo_start = int(coarse_emo_start)
        self.coarse_classes = int(coarse_classes)

    def __call__(self, output_decoder):
        losses = super().__call__(output_decoder)

        dec = output_decoder.get("output_decoder", output_decoder) \
            if isinstance(output_decoder, dict) else output_decoder

        coarse_logits = dec.get("coarse_logits") if isinstance(dec, dict) else None
        if coarse_logits is None:
            losses["loss_coarse"] = losses["loss"].new_tensor(0.0)
            return losses

        target_emotion = dec["target_emotion"]  # (bs, num_preds, T, 25)
        s0 = self.coarse_emo_start
        s1 = s0 + self.coarse_classes
        tgt = target_emotion[..., s0:s1]                       # (bs, np, T, C)

        denom = tgt.sum(dim=-1, keepdim=True)                  # (bs, np, T, 1)
        valid = (denom > 0.5).to(coarse_logits.dtype)          # real-distribution frames
        prob = tgt / denom.clamp_min(1e-6)                     # normalised soft labels
        logp = F.log_softmax(coarse_logits, dim=-1)
        ce = -(prob * logp).sum(dim=-1, keepdim=True)          # (bs, np, T, 1)
        loss_coarse = (ce * valid).sum() / valid.sum().clamp_min(1.0)

        losses["loss_coarse"] = loss_coarse
        losses["loss"] = losses["loss"] + self.w_coarse * loss_coarse
        return losses

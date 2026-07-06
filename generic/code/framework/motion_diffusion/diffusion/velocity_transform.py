"""Velocity (frame-delta) reparametrisation for the listener-emotion sequence.

The PerFRDiff decoder diffusion natively models the listener emotion sequence
``e`` of shape ``(..., T, C)`` directly in value space.  This module provides a
*bijective* map between ``e`` and a velocity representation ``d`` so the diffusion
can instead be trained/sampled in delta space (predicting frame-to-frame change),
analogous to epsilon-/v-prediction in standard diffusion:

    to_delta:    d[..., 0, :] = e[..., 0, :]                 (absolute anchor)
                 d[..., t, :] = e[..., t, :] - e[..., t-1, :] (velocity)
    from_delta:  e = cumsum(d, time)

Keeping the very first frame as an absolute anchor makes the transform invertible
(no separate "initial frame" predictor is needed) and bounds long-range drift,
while every other position is a pure velocity target.

Time axis is assumed to be ``dim=-2`` and channels ``dim=-1`` (i.e. ``(..., T, C)``),
which matches every tensor the decoder matcher feeds through here
(``(B, T, C)`` and ``(B * num_preds, T, C)``).
"""

import torch


def to_delta(x: torch.Tensor) -> torch.Tensor:
    """Map an emotion sequence ``(..., T, C)`` to its velocity representation.

    The first time step is kept as an absolute value; the rest are first-order
    differences. Autograd-safe (no in-place writes).
    """
    first = x[..., :1, :]
    diffs = x[..., 1:, :] - x[..., :-1, :]
    return torch.cat([first, diffs], dim=-2)


def from_delta(d: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_delta`: integrate the velocity over time.

    ``from_delta(to_delta(x)) == x`` (up to floating point).
    """
    return torch.cumsum(d, dim=-2)

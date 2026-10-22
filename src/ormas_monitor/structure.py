"""Channel-level tests that read weights only.

A channel is *certified dead* when its BatchNorm output is a constant <= 0 for every input.
The ReLU after it is then closed everywhere, every gradient into the channel is exactly
zero, and plain SGD or AdamW can never move it again (Raadh 2026, Theorem 2). Two weight
patterns make the BatchNorm output constant in training mode:

  * the incoming filter is all zero, so the BatchNorm input is the same everywhere
    and its normalised value is 0, leaving only the shift beta;
  * the BatchNorm scale gamma is 0, which removes the normalised value directly.

Either way the output equals beta, so beta <= 0 closes the ReLU for good.
A constant channel with beta > 0 is *flat*: alive, but it carries no information.

Assumptions the certificate needs (and the report states): a ReLU follows the BatchNorm,
BatchNorm runs in training mode, and the optimiser's momentum for the channel has decayed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

CONV_KINDS = ("Conv1d", "Conv2d", "Conv3d", "Linear")
BN_KINDS = ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm")
RELU_KINDS = (nn.ReLU, nn.ReLU6)
DECAYED_REL_TOL = 1e-5      # "decayed": filter or BatchNorm scale below this x the layer median


@dataclass
class ChannelScan:
    """Result of scanning one producer layer (and its BatchNorm, if any)."""
    certified: list[int] = field(default_factory=list)
    flat: list[int] = field(default_factory=list)
    decayed: list[int] = field(default_factory=list)   # ~0 instead of exactly 0, shift <= 0
    channels: int = 0


def zero_rows(weight: torch.Tensor) -> torch.Tensor:
    """Boolean mask over output channels whose incoming weights are all exactly zero."""
    return (weight.detach().reshape(weight.shape[0], -1).abs().amax(dim=1) == 0).cpu()


def _tiny(values: torch.Tensor, rel_tol: float) -> torch.Tensor:
    """Entries that are nonzero but negligible next to the layer's typical magnitude."""
    values = values.float().abs().cpu()
    nonzero = values[values > 0]
    if nonzero.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return (values > 0) & (values <= rel_tol * nonzero.median())


def scan_channels(weight: torch.Tensor | None, bias: torch.Tensor | None,
                  gamma: torch.Tensor | None, beta: torch.Tensor | None,
                  rel_tol: float = DECAYED_REL_TOL) -> ChannelScan:
    """Find certified-dead, flat and decayed channels from weights alone.

    `weight`/`bias` belong to the producer (Conv or Linear); `gamma`/`beta` to the BatchNorm
    that follows it, or None when there is none. *Decayed* channels have a filter or BatchNorm
    scale that is not exactly zero but below `rel_tol` x the layer's median, with shift <= 0:
    the signature of a channel that died and was then shrunk by weight decay.
    """
    if weight is None and gamma is None:
        return ChannelScan()
    n = weight.shape[0] if weight is not None else gamma.shape[0]
    zero_filter = zero_rows(weight) if weight is not None else torch.zeros(n, dtype=torch.bool)

    if beta is not None:
        # BatchNorm present: output is constant if the filter or gamma is zero; it equals beta.
        constant = zero_filter.clone()
        if gamma is not None:
            constant |= gamma.detach().reshape(-1).cpu() == 0
        shift = beta.detach().reshape(-1).cpu()
    else:
        # No BatchNorm: a zero filter outputs its bias (or 0 when there is no bias).
        constant = zero_filter
        shift = bias.detach().reshape(-1).cpu() if bias is not None else torch.zeros(n)

    constant = constant.cpu()
    certified = constant & (shift <= 0)
    flat = constant & (shift > 0)
    near = torch.zeros(n, dtype=torch.bool)
    if weight is not None:
        near |= _tiny(weight.detach().reshape(n, -1).abs().amax(dim=1), rel_tol)
    if beta is not None and gamma is not None:
        near |= _tiny(gamma.detach().reshape(-1), rel_tol)
    decayed = near & ~constant & (shift <= 0)
    return ChannelScan(certified=certified.nonzero().flatten().tolist(),
                       flat=flat.nonzero().flatten().tolist(),
                       decayed=decayed.nonzero().flatten().tolist(), channels=n)


def pair_by_order(order: list[str], kinds: dict[str, str], sizes: dict[str, int]) -> dict[str, str]:
    """Pair each BatchNorm with the Conv/Linear that ran just before it.

    `order` is module names in the order they executed in one forward pass.
    Returns {producer_name: batchnorm_name}.
    """
    pairs: dict[str, str] = {}
    prev = None
    for name in order:
        kind = kinds.get(name)
        if kind in BN_KINDS and prev is not None:
            if kinds.get(prev) in CONV_KINDS and sizes.get(prev) == sizes.get(name):
                pairs[prev] = name
        if kind in CONV_KINDS or kind in BN_KINDS:
            prev = name
    return pairs


def relu_followers(model: torch.nn.Module, example: torch.Tensor) -> set[str]:
    """Names of BatchNorm/Conv/Linear layers whose output goes straight into a ReLU module.

    Runs one forward pass. A skip connection added in place before the ReLU bumps the
    tensor's version counter, so such layers are (correctly) left out.
    """
    kinds = BN_KINDS + CONV_KINDS
    last, found, handles = [None], set(), []

    def after(name):
        def fn(module, inputs, output):
            if isinstance(output, torch.Tensor):
                last[0] = (name, output, output._version)
        return fn

    def before(module, inputs):
        if last[0] and inputs and inputs[0] is last[0][1] and inputs[0]._version == last[0][2]:
            found.add(last[0][0])

    for name, m in model.named_modules():
        if type(m).__name__ in kinds:
            handles.append(m.register_forward_hook(after(name)))
        elif isinstance(m, RELU_KINDS):
            handles.append(m.register_forward_pre_hook(before))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example)
    finally:
        model.train(was)
        for h in handles:
            h.remove()
    return found

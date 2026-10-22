"""Channel-level tests that read weights only.

A channel is *certified dead* when its BatchNorm output is a constant <= 0 for every input.
The ReLU after it is then closed everywhere, every gradient into the channel is exactly
zero, and plain SGD or AdamW can never move it again (Raadh 2026, Theorem 2). Two weight
patterns make the BatchNorm output exactly constant in training mode:

  * the incoming filter is all zero AND the producer has no bias or a bias of exactly 0, so the
    BatchNorm input is exactly 0 everywhere and its normalised value is exactly 0, leaving only
    the shift beta. (With a nonzero bias the input is a constant that the batch mean does not
    cancel exactly in floating point: the output sits up to ~1e-3 off beta, so beta = 0 does not
    close the ReLU. Such channels are reported as flat or decayed, never certified.)
  * the BatchNorm scale gamma is 0, which removes the normalised value directly.

Either way the output equals beta, so beta <= 0 closes the ReLU for good.
A constant channel with beta > 0 is *flat*: alive, but it carries no information.

Assumptions the certificate needs (and the report states): a ReLU is the only consumer of the
BatchNorm output, BatchNorm runs in training mode, and the optimiser's momentum for the channel
has decayed. The live monitor checks these; `scan_channels` reads weights only.
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


@dataclass(frozen=True)
class ChannelMasks:
    """Per-channel boolean masks on the weights' device (see `channel_masks`).

    `zero_case`: zero filter (and zero or no producer bias) with shift <= 0; `gamma_case`: zero
    BatchNorm scale with shift <= 0. `certified` is their union; the live monitor needs the two
    cases apart because each depends on different optimizer state.
    """
    certified: torch.Tensor
    flat: torch.Tensor
    decayed: torch.Tensor
    zero_case: torch.Tensor
    gamma_case: torch.Tensor


def zero_rows(weight: torch.Tensor) -> torch.Tensor:
    """Boolean mask over output channels whose incoming weights are all exactly zero."""
    return (weight.detach().reshape(weight.shape[0], -1).abs().amax(dim=1) == 0).cpu()


def _tiny(values: torch.Tensor, rel_tol: float) -> torch.Tensor:
    """Entries that are nonzero but negligible next to the layer's typical magnitude (on device).

    The median is over the nonzero entries (torch's lower median), computed without boolean
    indexing so it never synchronises with the host.
    """
    values = values.detach().float().abs()
    med = torch.nanmedian(torch.where(values > 0, values, torch.full_like(values, float("nan"))))
    return (values > 0) & (values <= rel_tol * med)      # all-zero layer: med is nan, all False


def channel_masks(weight: torch.Tensor | None, bias: torch.Tensor | None,
                  gamma: torch.Tensor | None, beta: torch.Tensor | None,
                  rel_tol: float = DECAYED_REL_TOL) -> ChannelMasks:
    """`scan_channels` as device-side masks, with no host read. Needs `weight` or `gamma`."""
    assert weight is not None or gamma is not None, "channel_masks needs a filter or a BatchNorm scale"
    ref = weight if weight is not None else gamma
    n, dev = ref.shape[0], ref.device
    no = torch.zeros(n, dtype=torch.bool, device=dev)
    if weight is not None:
        row_max = weight.detach().reshape(n, -1).abs().amax(dim=1)
        zero_filter = row_max == 0
    else:
        row_max, zero_filter = None, no
    bias_zero = bias.detach().reshape(-1) == 0 if bias is not None else ~no

    if beta is not None:
        shift = beta.detach().reshape(-1)
        exact_zero = zero_filter & bias_zero
        gamma_zero = (gamma.detach().reshape(-1) == 0) if gamma is not None else no
        # Zero filter, nonzero bias: constant only up to float rounding inside BatchNorm.
        near_constant = zero_filter & ~bias_zero & ~gamma_zero
    else:
        # No BatchNorm: a zero filter outputs exactly its bias (or 0 when there is no bias).
        shift = bias.detach().reshape(-1) if bias is not None else torch.zeros(n, device=dev)
        exact_zero, gamma_zero, near_constant = zero_filter, no, no

    closed = shift <= 0
    constant = exact_zero | gamma_zero
    zero_case, gamma_case = exact_zero & closed, gamma_zero & closed
    near = near_constant.clone()
    if row_max is not None:
        near |= _tiny(row_max, rel_tol)
    if beta is not None and gamma is not None:
        near |= _tiny(gamma.reshape(-1), rel_tol)
    return ChannelMasks(certified=zero_case | gamma_case,
                        flat=(constant | near_constant) & ~closed,
                        decayed=near & ~constant & closed,
                        zero_case=zero_case, gamma_case=gamma_case)


def scan_channels(weight: torch.Tensor | None, bias: torch.Tensor | None,
                  gamma: torch.Tensor | None, beta: torch.Tensor | None,
                  rel_tol: float = DECAYED_REL_TOL) -> ChannelScan:
    """Find certified-dead, flat and decayed channels from weights alone.

    `weight`/`bias` belong to the producer (Conv or Linear); `gamma`/`beta` to the BatchNorm
    that follows it, or None when there is none. *Decayed* channels have a filter or BatchNorm
    scale that is not exactly zero but below `rel_tol` x the layer's median, with shift <= 0:
    the signature of a channel that died and was then shrunk by weight decay. A zero filter
    with a nonzero producer bias in front of a BatchNorm is only nearly constant, so it is
    flat (shift > 0) or decayed (shift <= 0), never certified.
    """
    if weight is None and gamma is None:
        return ChannelScan()
    m = channel_masks(weight, bias, gamma, beta, rel_tol)
    packed = torch.stack([m.certified, m.flat, m.decayed]).cpu()
    return ChannelScan(certified=packed[0].nonzero().flatten().tolist(),
                       flat=packed[1].nonzero().flatten().tolist(),
                       decayed=packed[2].nonzero().flatten().tolist(), channels=packed.shape[1])


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

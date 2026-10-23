# Dead channels and the certificate

A **dead channel** is a unit that outputs nothing useful and receives no gradient, so training cannot change it. Most tools can tell you a unit *looks* inactive on the batches they saw. `ormas-monitor` can also tell you, for one common pattern, that a channel **can never recover**, and it says so only when it can prove it.

## The statement in plain words

Take one channel of a Conv (or Linear) layer that feeds a BatchNorm, which feeds a ReLU.

1. **Suppose the incoming filter is exactly zero, or the BatchNorm scale γ is exactly zero.**
   With a zero filter, the BatchNorm input is the same value at every position. Its normalised value is 0, so the BatchNorm outputs its shift β. With γ = 0, the normalised value is multiplied by zero, and the output is again β. Either way, the channel's output is the constant β for every input.
2. **Suppose also that β ≤ 0.** Then the ReLU receives a value ≤ 0 everywhere and outputs 0 everywhere. A closed ReLU passes zero gradient back. So every gradient into the channel's filter, bias, γ and β is exactly zero.
3. **Then plain SGD or AdamW can never move the channel again.** With zero gradient and no stored momentum, the update is zero. Decoupled weight decay (or SGD's decay without momentum, with `lr × weight_decay < 1`) shrinks β toward 0 but never past it, so β stays ≤ 0. The channel is in an *absorbing* state.
4. **A repair that only nudges the weights cannot get it out either.** Any repair proportional to the weights, ‖R(W) − W‖ ≤ ρ‖W‖, leaves a zero filter at zero. Only a discrete jump, such as a re-initialisation, leaves the set. This is why [repair](repair.md) re-initialises.

This is Theorem 2 of R. A. D. Raadh, *Self-Repair as a Bounded Disturbance: Input-to-State Stability of Neural Network Training Dynamics*, DeepMath 2026 (accepted poster). See [About](../about.md).

The same reasoning covers a Linear layer with no BatchNorm: a unit whose incoming weights are all zero outputs its bias, and a ReLU after it with bias ≤ 0 never fires.

## The conditions the monitor checks before it issues a certificate

The theorem has assumptions. The monitor checks them and withholds the certificate when one fails. The `manifest` event at the first check lists, per layer, whether a certificate is possible and, if not, why; a `certificate_refused` event (severity `info`) names the failed condition whenever channels that look dead from their weights are not certified.

| Assumption | How the monitor checks it | What happens if it fails |
|---|---|---|
| **The BatchNorm really reads the layer's output.** | A BatchNorm is paired with a Conv/Linear only if, in the forward pass, the BatchNorm's input *is* that layer's output tensor, unmodified (same tensor, unchanged version counter). Pairing follows the dataflow, not the order in which modules run. | The layer is not paired with that BatchNorm. |
| **A ReLU is the only consumer.** | At the first sampled step the monitor walks the autograd graph (from the loss, or from the model's outputs when no loss with a `grad_fn` is passed). It requires that the layer's output feeds only its BatchNorm, that the BatchNorm's output feeds only an `nn.ReLU` or `nn.ReLU6` module (a dtype cast in between is allowed), and that the layer and its BatchNorm each run exactly once per forward pass. A residual add, a second branch, or a module called twice fails this. If a later check sees a different call count, the layer loses eligibility for the rest of the run. | No certificate for that layer. Its zero channels are reported as *constant* (flat). |
| **The gradient is exactly zero now.** | At every check, the gradients of the channel's filter row, producer bias, γ and β must all be exactly 0. This is a guard against anything the structure check missed. | No certificate at this check (`certificate_refused`: "nonzero gradient"). |
| **The optimizer cannot move the channel.** | The optimizer must be `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam` or `Adamax` (exact type), or a type you list in [`MonitorConfig.certify_optimizers`](../reference/config.md). It must hold zero first-moment state (`momentum_buffer`, `exp_avg`) for the channel's filter, bias, γ and β. Framework wrappers (Lightning, Accelerate) are unwrapped to reach the real optimizer. | No certificate while momentum remains (in testing, Adam momentum revived a zeroed layer). An optimizer outside the list, or no optimizer at all, means no certificate is ever issued. |
| **Weight decay cannot lift the shift above 0.** | Coupled L2 decay in the Adam family (`weight_decay > 0` without decoupled decay, e.g. `Adam(weight_decay=...)`) is refused. Decoupled decay (AdamW) needs `lr × weight_decay < 1`. For SGD with momentum and weight decay, a sign rule checks that decay alone can never push a shift ≤ 0 above 0 from a zero momentum buffer (it holds for usual settings, e.g. `lr × wd < (1 − √momentum)²`). Read from the param group at every check. | No certificate while the setting fails. |
| **BatchNorm is in training mode.** | Re-checked at every check. | No certificate. In eval mode BatchNorm uses running statistics, and the zero-filter argument does not hold. |
| **The filter or γ is *exactly* zero and β ≤ 0.** | Exact floating-point comparison on the weights. A zero filter also needs the producer's bias to be absent or exactly 0 (or γ = 0): with a nonzero bias, BatchNorm's output is constant only up to float rounding. | A value that is tiny but not zero, or a zero filter with a nonzero bias, is reported as *decayed to ~0*, never certified. |

What the checks do not cover (**status: OPEN**):

- **The structure is read once,** from the autograd graph of the first sampled step. A later change of wiring that keeps every module's call count the same is not re-analysed; a tensor that escapes through a module attribute and is used elsewhere is not seen by the graph walk. Both are caught only by the per-check gradient guard, which can only see a gradient on the batches of a check.
- **A certificate already written is not withdrawn.** If a learning-rate schedule later breaks the SGD sign condition, later checks refuse new certificates, but the earlier `certificate` event stays in the record.
- **The SGD sign rule and the optimizer allow-list are the monitor's own analysis,** tested against `torch.optim` (`tests/test_core_certificate.py`), not part of the paper's theorem, which is stated for SGD and AdamW.

!!! warning "What the certificate is not"
    The certificate is a statement about one wiring pattern under the assumptions above. It does not apply to transformers, GELU/SiLU activations, LayerNorm, or a BatchNorm whose output joins a skip connection before the ReLU. On those models you get [alarms](alarms.md), loss checks and [the record](record.md), but no dead-unit proof.

## The other channel states

Not every stuck channel is provably dead. The monitor and `check` report four states, from strongest to weakest evidence. The first three are read from the weights and count as **stuck**; the fourth is an observation and is reported separately as **silent**, never as stuck:

| State | Meaning | Evidence |
|---|---|---|
| **Certified dead** | Zero filter or γ, β ≤ 0, and every condition above holds. | Proof (Theorem 2), under the conditions above. |
| **Constant (flat)** | Zero filter or γ, but β > 0, or a condition for the certificate is missing. The channel outputs one value for every input and carries no information. | Exact, from weights. Not necessarily permanent. |
| **Decayed to ~0** | Filter or γ below 10⁻⁵ × the layer's median (or a zero filter with a nonzero bias), β ≤ 0, ReLU after. What a dead channel looks like after weight decay has shrunk it for a while. | From weights. Not exact, so not certified. |
| **Silent** (monitor) / **dormant** (`check`) | The monitor saw no positive output on the sampled batches of `patience` checks in a row. `check` estimates dormancy from β ≤ −3·|γ|. | Observation or estimate. Labelled as one. |

## Prior art

Dead and dormant units are not a new observation. ReDo (Sokar et al., *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*, ICML 2023) detects and resets dormant neurons, and network slimming (Liu et al., ICCV 2017) deliberately pushes BatchNorm γ toward 0 to prune channels. `ormas-monitor` claims the certificate (a proof, with its assumptions checked at run time) and the packaging, not the discovery.

# Dead channels and the certificate

A **dead channel** is a unit that outputs nothing useful and receives no gradient, so training cannot change it. Most tools can tell you a unit *looks* inactive on the batches they saw. `ormas-monitor` can also tell you, for one common pattern, that a channel **can never recover**, and it says so only when it can prove it.

## The statement in plain words

Take one channel of a Conv (or Linear) layer that feeds a BatchNorm, which feeds a ReLU.

1. **Suppose the incoming filter is exactly zero, or the BatchNorm scale γ is exactly zero.**
   With a zero filter, the BatchNorm input is the same value at every position. Its normalised value is 0, so the BatchNorm outputs its shift β. With γ = 0, the normalised value is multiplied by zero, and the output is again β. Either way, the channel's output is the constant β for every input.
2. **Suppose also that β ≤ 0.** Then the ReLU receives a value ≤ 0 everywhere and outputs 0 everywhere. A closed ReLU passes zero gradient back. So every gradient into the channel's filter, bias, γ and β is exactly zero.
3. **Then plain SGD or AdamW can never move the channel again.** With zero gradient and no stored momentum, the update is zero. Weight decay shrinks β toward 0 but never past it, so β stays ≤ 0. The channel is in an *absorbing* state.
4. **A repair that only nudges the weights cannot get it out either.** Any repair proportional to the weights, ‖R(W) − W‖ ≤ ρ‖W‖, leaves a zero filter at zero. Only a discrete jump, such as a re-initialisation, leaves the set. This is why [repair](repair.md) re-initialises.

This is Theorem 2 of R. A. D. Raadh, *Self-Repair as a Bounded Disturbance: Input-to-State Stability of Neural Network Training Dynamics*, DeepMath 2026 (accepted poster). See [About](../about.md).

The same reasoning covers a Linear layer with no BatchNorm: a unit whose incoming weights are all zero outputs its bias, and a ReLU after it with bias ≤ 0 never fires.

## The conditions the monitor checks before it issues a certificate

The theorem has assumptions. The monitor checks them and withholds the certificate when one fails. The `manifest` event at the first check lists, per layer, whether a certificate is possible and, if not, why.

| Assumption | How the monitor checks it | What happens if it fails |
|---|---|---|
| **A ReLU directly consumes the layer's output.** | During one forward pass it records which layers' output tensors arrive, unmodified, at a `ReLU` or `ReLU6` module. A residual `out += identity` before the ReLU changes the tensor's version counter and is detected. | No certificate. The channel is reported as *constant* (flat) instead. |
| **The optimizer holds zero first-moment momentum for the channel.** | It reads the optimizer state (`exp_avg`, `momentum_buffer`) for the channel's filter, bias, γ and β. Framework wrappers (Lightning, Accelerate) are unwrapped to reach the real optimizer. | No certificate while momentum remains. In testing, Adam momentum revived a zeroed layer. Without an optimizer, no certificate is ever issued. |
| **BatchNorm is in training mode.** | Checked at every check. | No certificate. In eval mode BatchNorm uses running statistics, and the zero-filter argument does not hold. |
| **The filter or γ is *exactly* zero and β ≤ 0.** | Exact floating-point comparison on the weights. | A value that is tiny but not zero is reported as *decayed to ~0*, never certified. |

!!! warning "What the certificate is not"
    The certificate is a statement about one wiring pattern under the assumptions above. It does not apply to transformers, GELU/SiLU activations, LayerNorm, or a BatchNorm whose output joins a skip connection before the ReLU. On those models you get [alarms](alarms.md), loss checks and [the record](record.md), but no dead-unit proof.

## The other channel states

Not every stuck channel is provably dead. The monitor and `check` report four states, from strongest to weakest evidence:

| State | Meaning | Evidence |
|---|---|---|
| **Certified dead** | Zero filter or γ, β ≤ 0, ReLU directly after, no momentum. | Proof (Theorem 2). |
| **Constant (flat)** | Zero filter or γ, but β > 0, or a condition for the certificate is missing. The channel outputs one value for every input and carries no information. | Exact, from weights. Not necessarily permanent. |
| **Decayed to ~0** | Filter or γ below 10⁻⁵ × the layer's median, β ≤ 0, ReLU after. What a dead channel looks like after weight decay has shrunk it for a while. | From weights. Not exact, so not certified. |
| **Silent** (monitor) / **dormant** (`check`) | The monitor saw no positive output on any input for `patience` checks in a row. `check` estimates dormancy from β ≤ −3·|γ|. | Observation or estimate. Labelled as one. |

## Prior art

Dead and dormant units are not a new observation. ReDo (Sokar et al., *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*, ICML 2023) detects and resets dormant neurons, and network slimming (Liu et al., ICCV 2017) deliberately pushes BatchNorm γ toward 0 to prune channels. `ormas-monitor` claims the certificate (a proof, with its assumptions checked at run time) and the packaging, not the discovery.

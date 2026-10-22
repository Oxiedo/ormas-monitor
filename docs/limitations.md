# Limitations

Stated in one place so they do not drift.

## Scope of the certificate

- **The certificate needs Conv/Linear → BatchNorm → ReLU** (or Linear → ReLU with no BatchNorm), with an `nn.ReLU` or `nn.ReLU6` *module*. A functional `F.relu` cannot be seen by a hook, so those channels are reported as constant, never certified.
- **No proofs for transformers or LLMs.** LayerNorm, GELU, SiLU and attention are outside Theorem 2. Those models get alarms, loss checks and the record.
- **A skip connection before the ReLU removes the certificate.** In ResNet-style blocks, the BatchNorm whose output is added to the identity path is not certified.
- **The wiring check confirms that a ReLU receives the layer's unmodified output. It does not confirm that the ReLU is the only consumer of that output.** If the same tensor also feeds another branch, a gradient could reach the channel through that branch. **Status: OPEN.**
- **Exact zero only.** Channels decayed to ~0 are flagged, never certified.
- **Momentum matters.** While the optimizer holds first-moment momentum for a channel, it is not certified. Adam momentum revived a zeroed layer in testing.
- **SGD and AdamW are the optimizers the argument covers.** An optimizer that adds noise, or changes weights other than through gradients and decoupled decay, is outside it.

## The monitor

- **It never reads your data.** It cannot find mislabelled examples, data leakage or distribution shift.
- **Alarms say where, not why.** A vanishing gradient in one layer may be caused by another.
- **Thresholds are design choices, not tuned optima** (status: DESIGNED). Noisy runs may need a higher `sigma` or `warmup`.
- **Checks are sampled.** With `check_every=50`, an event between two checks is seen at the next one, or not at all if it is transient.
- **Behind the Hugging Face Trainer, gradients are seen after clipping** (`max_grad_norm`), because the Trainer's only pre-step event runs after clipping. Exploding gradients may be capped before the monitor sees them. See the [guide](guides/huggingface.md).
- **The Hugging Face Trainer has no exception hook.** Call `callback.close()` in a `finally:` block to close the record after a crash.
- **The Hugging Face callback reads the loss the model returns.** If `compute_loss` builds the loss outside the model, call `callback.note_loss(loss)`, or the run has no loss checks.
- **With several Lightning optimizers, only `optimizer_index` is monitored.**
- **Integrations are tested on CPU with one Python version.** The Lightning and Hugging Face callbacks have not run under DDP, on CUDA or on Python 3.9; the MLflow sink has been tested against a stand-in, not a real MLflow server. Status: OPEN.
- **Repair restores capacity, not what was learnt,** and it changes the training trajectory.

## Hardware and scale

- **All measured numbers come from one Apple-silicon Mac** on CIFAR-10, with one seed. **Not yet tested on NVIDIA/Linux, multi-GPU or at scale.** A GPU stress test is planned.
- **Distributed support (DDP/FSDP) is covered by CPU unit tests only.** Status: OPEN.
- **Overhead on large models is not measured.**

## `check`

- **A state dict does not record the wiring.** Without `--arch`, dead-channel findings say "if a ReLU follows", and pairing a BatchNorm with its Conv relies on names and file order.
- **Dormant is an estimate,** from the BatchNorm shift and scale alone.
- **`--trust` loads pickles,** which can run code. It is off by default.

## The record

- **It is evidence, not a compliance certificate.** No regulator has reviewed it.
- **It is written by the training process,** so it can only be as honest as that process. See the [threat model](concepts/record.md#threat-model).
- **An unsigned record can be rewritten** by anyone with write access, and still verify.
- **Timestamps come from the local clock.**

## Not claimed

- **Not ORMAS.** The ORMAS training architecture (local heads, three signals, seven diagnoses) is not in this package.
- **Not the first.** Dead and dormant-unit detection and resets exist in research: ReDo (Sokar et al., ICML 2023); network slimming pushes BatchNorm γ toward 0 on purpose. We claim the certificate and the packaging, not the discovery.
- **No uploads.** Nothing leaves the machine unless you configure a sink that sends data.

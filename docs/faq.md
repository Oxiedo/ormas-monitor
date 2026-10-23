# FAQ

### Does it work on my transformer or LLM?

Partly. You get per-layer gradient alarms against each layer's own baseline, loss-spike and non-finite checks, and the training record. You do **not** get the dead-channel certificate: it needs BatchNorm → ReLU. See [Hugging Face Trainer](guides/huggingface.md#what-you-get-on-a-transformer).

### How is this different from watching gradient norms in TensorBoard or W&B?

A dashboard shows you curves; you decide what is abnormal. The monitor compares each layer with its own history and writes an event when one leaves its range, so the record names the layer and the step. For BN→ReLU channels it adds what a curve cannot: a proof that the channel can never recover. It also sends its events to those dashboards if you want: see [Sinks](guides/sinks.md).

### Is "certified dead" just a heuristic with a confident name?

No. It is a mathematical statement (Theorem 2 of the DeepMath 2026 paper) that holds under stated assumptions, and the monitor checks those assumptions at run time before it issues the certificate: the wiring from the autograd graph, an exactly zero gradient at that check, the optimizer and its momentum, weight decay, and BatchNorm's training mode. The checks have known gaps, listed on the same page. When an assumption fails, or a value is only *near* zero, the channel is reported under a weaker label. See [Dead channels](concepts/dead-channels.md).

### Will repair hurt my model?

Repair is off by default. When on, it re-initialises only channels that are stuck (certified dead, constant or decayed; silent ones too with `repair="all"`), and it changes the training trajectory. In the CIFAR-10 demo it recovered 76.77% against a healthy 78.84% (one seed, Apple M-series). See [Repair](concepts/repair.md) for when not to use it.

### Does it slow training down?

**+3.6%** median epoch time at the default `check_every=50`, measured on one Apple-silicon Mac with a small CNN. Not yet measured on NVIDIA GPUs or large models. Raise `check_every` to spend less.

### Does it send anything anywhere?

No, unless you add a sink that does (W&B, a remote MLflow, a webhook). The core makes no network calls; the only other network use is `ormas-monitor demo` downloading CIFAR-10 once. See [Security and privacy](security-and-privacy.md).

### Does it make us compliant with the EU AI Act?

No. The record is evidence for your technical documentation, not a compliance certificate, and no regulator has reviewed it. See [For leaders](for-leaders.md#how-it-relates-to-ai-act-style-documentation).

### Is this ORMAS?

No. ORMAS is Oxiedo's research training architecture. This package contains none of it and needs no change to your model. See [About](about.md).

### Weren't dead neurons already known?

Yes. ReDo (Sokar et al., ICML 2023) detects and resets dormant neurons, and network slimming pushes BatchNorm γ toward zero on purpose. What `ormas-monitor` adds is the certificate, with its assumptions checked at run time, and the packaging: monitor, checkpoint check, record and reports.

### `check` says "if a ReLU directly follows". Why the hedge?

A saved state dict contains tensors, not the wiring. The check cannot see whether a ReLU follows a BatchNorm or a skip connection joins first. Pass `--arch torchvision:<name>` to use the real wiring. A weights file also cannot show the optimizer's momentum, which the live monitor checks before it certifies anything. See the [findings catalogue](reference/check-findings.md).

### My published model has "decayed" channels. Is it broken?

Probably not broken, but it carries capacity it never uses. In our scan, 19 of 21 torchvision BN→ReLU models had some. Whether to prune or retrain is your call; `check` only tells you where they are.

### Which versions are supported?

Declared: Python 3.9 and later, PyTorch 2.3 and later. Run so far on Python 3.13 with PyTorch 2.11 on macOS. The CI workflow is set up to test Linux, macOS and Windows with Python 3.9 and 3.12, CPU only, but has not run yet. **Status: OPEN.**

### How do I cite it?

See [`CITATION.cff`](https://github.com/Oxiedo/ormas-monitor/blob/main/CITATION.cff) or the citation in the [README](https://github.com/Oxiedo/ormas-monitor#citation).

### Where do I get help?

Open an issue on GitHub. If a finding looks wrong, use the "A finding looks wrong" issue template. Or email raadh@oxiedo.com.

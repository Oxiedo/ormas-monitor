# Why use it

Training dashboards show you charts. Research tools such as Cockpit and ReDo, and small libraries that warn about dead ReLUs, tell you a layer *looks* wrong. `ormas-monitor` adds the things those tools leave to you.

## 1. Alerts you can safely automate

A heuristic alarm ("this neuron looks dormant") can be wrong, so nobody lets it change a run or block a release without a human looking. The dead-channel **certificate** is different. It is issued only when a channel provably receives no gradient and the conditions for it to stay dead hold:

- training-mode BatchNorm;
- a ReLU as the only consumer;
- a covered optimizer with no momentum for the channel;
- an exact-zero gradient on that check.

When a condition fails, the monitor says so (`certificate_refused`) instead of guessing. See [Dead channels and the certificate](concepts/dead-channels.md).

That is what makes these safe to wire in:

- **Repair without a human watching:** with `repair=True`, proven-dead channels are re-initialised at the next check.
- **Release gates:** `ormas-monitor check model.pt --fail-on warning` exits non-zero when it finds dead channels. Put it in CI or in front of your model registry. See [check as a CI gate](guides/ci-gate.md).
- **Fine-tune checks:** `ormas-monitor check finetuned.pt --against base.pt` reports which channels died between the two files.

## 2. Find the dead weight your model already carries

Channels that output nothing still cost compute on every forward pass, in training and in production.

`check` reads the weights of torchvision's ImageNet models, using each model's real wiring. It found channels whose filter or BatchNorm scale has decayed to ~0 with a non-positive shift in 19 of the 21 models that use BatchNorm → ReLU. Examples: 292 of 5,504 checked channels in VGG-19-BN (5.3%), and 7,667 of 77,536 in DenseNet-169 (9.9%). Full table: [Results](results.md).

**They really are dead weight.** We switched off every flagged channel and ran each model on 256 real photos (CIFAR-10 test images upscaled to 224 px). The logits did not change at all: maximum change 0.0, and the same top-1 answer on 100% of images. This held for ResNet-18, ResNet-50, VGG-19-BN, DenseNet-169, MnasNet-1.0 and MobileNet-V2.

As a control, we switched off the *same number* of randomly chosen live channels in ResNet-18, VGG-19-BN and MnasNet (128 images). The answers changed on 98–100% of images. Reproduce with `python examples/reproduce/zero_flagged_channels.py --data DIR`; add `--images 128 --control` for the control.

**And removing them saves real compute.** In VGG-19-BN we cut the 292 flagged channels out of the network. The cut model gave the same top-1 answer on 100% of 64 real photos, needed **18.6% less compute** per image, and ran **12–18% faster** on a laptop CPU (three runs). See [Results §6](results.md#6-removing-dead-channels-makes-a-model-smaller-and-faster). Savings depend on the model: VGG-19-BN carries many dead channels, ResNet-50 almost none.

## 3. A record of what happened in training

Every alarm, proof and repair is written to a hash-chained diary that you can seal and sign with an Ed25519 key, then verify with one command. `ormas-monitor diff` compares two runs: "what changed since the run we approved?" The record is evidence for your own technical documentation. It does not make anyone compliant with any regulation.

## 4. Two readers, two reports

Engineers get a layer-by-layer report. A CEO, CFO or compliance lead gets a one-page summary: what happened, how much of the run was spent with part of the model stuck, what was done, and whether the record can be trusted. See [For leaders](for-leaders.md).

## 5. Nothing to trust us with

It is open source (Apache-2.0) and runs on your machine. The core makes no network calls and never reads your training data. Model files load with `weights_only=True`. Try it in one command on a file you already have.

## Where it fits best, and where it does not

| Fits well | Why |
|---|---|
| Vision models (ResNet, VGG, MobileNet, DenseNet and similar) | The certificate covers Conv/Linear → BatchNorm → ReLU, the building block these use |
| Teams fine-tuning vision models | One command shows what a fine-tune broke |
| Regulated vision work (for example medical imaging) | A signed record of training problems and fixes |

| Weaker fit today | Why |
|---|---|
| Transformers and LLMs | They get alarms, loss checks and the record, but no dead-channel proof |
| Multi-GPU and NVIDIA clusters | Not yet tested there. See [Limitations](limitations.md) |

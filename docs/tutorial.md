# Tutorial: five minutes, start to finish

This is a real session: a fresh Python environment, install, and every command below run as shown. The output is pasted unedited, apart from the hash fingerprints. Your numbers will differ slightly with your hardware and torch version.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install "ormas-monitor[safetensors,signing,demo]"
ormas-monitor --version
# ormas-monitor 0.2.1
```

## 2. Check a model file you already have

Here that is torchvision's ImageNet ResNet-18, plus a copy where 40 filters of one layer were zeroed, standing in for a fine-tune that went wrong:

```text
$ ormas-monitor check resnet18_finetuned.pt --against resnet18.pt
resnet18_finetuned.pt: 11,699,112 parameters in 122 tensors (pytorch, float32, int64); wiring: name-based heuristic; 6 findings in 0.1 s
  [warning] 40 channels died between the two checkpoints
  [warning] 40 of 256 channels in layer `layer3.0.conv1` (Conv2d, BatchNorm `layer3.0.bn1`) are dead (zero filter or scale, shift <= 0) if a ReLU directly follows `layer3.0.bn1`
  [warning] 7 of 64 channels in layer `conv1` (Conv2d, BatchNorm `bn1`) are effectively dead if a ReLU directly follows `bn1`
  [info] 1 of 64 channels in layer `conv1` (Conv2d, BatchNorm `bn1`) are probably dormant if a ReLU directly follows `bn1`
  [info] 3 of 64 channels of BatchNorm `bn1` never vary (running variance ~0)
  [info] 100 of 102 tensors are exactly unchanged
  note: Wiring not known, so BatchNorm findings say "if a ReLU directly follows". Pass --arch torchvision:<name> or --arch hf for exact wiring.
report: resnet18_finetuned.pt.health.html
exit 0: no critical findings (3 warnings; use --fail-on warning to fail on them)
```

As a release gate, the same check fails the build:

```text
$ ormas-monitor check resnet18_finetuned.pt --fail-on warning --quiet; echo "exit code $?"
exit code 1
```

## 3. Watch a training run

Add four lines to an ordinary training loop. The full script is [`examples/walkthrough_train.py`](https://github.com/Oxiedo/ormas-monitor/blob/main/examples/walkthrough_train.py). To show what a failure looks like, it knocks out two conv layers at epoch 3 on purpose.

```python
from ormas_monitor import Monitor                        # 1

mon = Monitor(model, opt, repair=True, out="my_run")     # 2
for ...:
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward()
    mon.step(loss)                                       # 3: after backward, before opt.step()
    opt.step()
mon.close()                                              # 4: writes the reports
```

Real output (10,000 CIFAR-10 images, CPU, torch 2.11):

```text
$ python walkthrough_train.py
epoch 0: train accuracy 57.1%
[ormas-monitor] step 150: layer 4 (Linear): 5 of 256 channels silent for several checks (not proven dead)
epoch 1: train accuracy 62.7%
[ormas-monitor] step 200: layer 4 (Linear): 170 of 256 channels silent for several checks (not proven dead)
epoch 2: train accuracy 71.5%
--- epoch 3: two conv layers knocked out (simulated failure)
[ormas-monitor] step 250: layer 1.0 (Conv2d): 64 of 64 channels stuck (55 certified dead, 9 constant)
[ormas-monitor] step 250: layer 2.0 (Conv2d): 128 of 128 channels stuck (128 certified dead)
[ormas-monitor] step 250: layer 4 (Linear): 110 of 256 channels silent for several checks (not proven dead)
[ormas-monitor] step 300: layer 1.0: repaired 55 stuck channels (re-initialised)
[ormas-monitor] step 300: layer 2.0: repaired 128 stuck channels (re-initialised)
epoch 3: train accuracy 44.9%
[ormas-monitor] step 350: layer 1.0: all channels active again
[ormas-monitor] step 350: layer 2.0: all channels active again
epoch 4: train accuracy 62.8%
[ormas-monitor] step 450: layer 4 (Linear): 138 of 256 channels silent for several checks (not proven dead)
epoch 5: train accuracy 69.5%
[ormas-monitor] report written to my_run/report.html
[ormas-monitor] report written to my_run/summary.html
[ormas-monitor] report written to my_run/summary.md
```

How to read it:

- **Stuck** channels are proven from the weights:
    - *certified dead* means they can never recover under this training;
    - *constant* means they output the same value for every input.
- **Silent** channels produced no positive output for several checks. They are probably unused, but that is not proven.
- With `repair=True`, the proven-dead channels were re-initialised at the next check, and training recovered from 44.9% to 69.5% in two epochs.

## 4. Open the reports

- `my_run/summary.html` is one page in plain language, for anyone: what happened, how much of the run was spent stuck, what was done, and whether the record can be trusted.
- `my_run/report.html` is the engineer's view: a timeline, layer by layer.

## 5. Sign the record, then try to tamper with it

```text
$ ormas-monitor keygen
private key: ormas-signing.pem (mode 0600; keep it secret, ideally off the training machine)
public key:  ormas-signing.pub.pem (give this to whoever verifies your runs)

$ ormas-monitor sign my_run --key ormas-signing.pem
sealed: my_run/record.json
signed: my_run/record.sig

$ ormas-monitor verify my_run --key ormas-signing.pub.pem
ormas-monitor verify my_run  (16 events)
  [ok  ] JSON lines              16 events parsed
  [ok  ] genesis link            first event links to the genesis hash
  [ok  ] hash chain              all 16 hashes recompute and link
  [ok  ] sequence numbers        seq runs 0..15
  [ok  ] run_start / run_end     run_start is first and run_end is last: the run finished
  [ok  ] event schema            all 16 events match record schema 1
  [ok  ] metrics hash            sha256 4b07fe41c376... as recorded in run_end
  [ok  ] seal (record.json)      diary.jsonl, metrics.jsonl and the diary head match the seal of 2026-10-23T12:42:52Z
  [ok  ] signature (record.sig)  valid Ed25519 signature over record.json by key 5293b3d9522a...
PASSED: 9 of 9 checks passed
```

Now edit one line of `my_run/diary.jsonl` (here, the `step` of event 5) and verify again:

```text
$ ormas-monitor verify my_run --key ormas-signing.pub.pem
  [FAIL] hash chain              event 5 (seq 5, channels): its hash does not match its contents, so it was edited after it was written
  [FAIL] seal (record.json)      diary.jsonl changed after sealing
FAILED: 7 of 9 checks passed (2 fail)
```

The edit is caught at the exact line. A record shows what the monitor wrote. It is evidence for your documentation, not a compliance certificate.

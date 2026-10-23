# For leaders

*For a CEO, CFO, CTO or compliance lead deciding whether a team should use `ormas-monitor`, or reading what it produced.*

## In plain words

Training an AI model is like teaching a very large team of tiny workers. Each part of the model slowly learns its job. Sometimes some of those parts **die**: they stop working and never come back. Training carries on and the bill keeps running, but those parts contribute nothing. Usually nobody notices, because training dashboards only show the overall score.

`ormas-monitor` is like a doctor sitting beside the model while it trains:

1. **It watches every part** while training runs.
2. **When a part dies, it proves it,** instead of only saying "this looks wrong". The proof comes from peer-reviewed mathematics (a paper accepted at DeepMath 2026), so the alert can be acted on automatically.
3. **It can fix it,** bringing those parts back so the model keeps learning.
4. **It keeps a sealed diary** of what happened. The diary can be signed, so nobody can quietly edit it afterwards.
5. **It writes you a one-page summary** in plain language: what happened, how much of the run was affected, what was done, and whether the record can be trusted.

It can also check a finished model. On 19 of 21 popular public image models it found parts that do nothing at all, up to 1 in 10 in one of them. In the six models we tested, switching those parts off did not change a single answer on 256 test photos, so they are weight the model carries for nothing. See [Why use it](why.md).

**To see what you would receive,** open one of the real sample summaries on GitHub, for example [`cifar10_watch_only/summary.html`](https://github.com/Oxiedo/ormas-monitor/tree/main/sample_reports/cifar10_watch_only) (download it and open it in a browser).

## What it is

`ormas-monitor` is a free, open-source (Apache-2.0) tool that your engineers add to a PyTorch training run with two lines of code or one callback. While the model trains, it watches each layer of the network against that layer's own normal behaviour, names the layer that failed, and, for one common kind of network, proves when part of the model has died and can never recover. It can optionally repair that part. It writes a tamper-evident record of what it saw, and a plain-language summary for you. It runs on your own machines and sends nothing anywhere unless your team configures it to (the optional demo downloads a public dataset once).

## Why it matters

A training run can fail silently: the loss curve still moves, but part of the model has stopped learning. That capacity is paid for in compute and delivers nothing. In our CIFAR-10 demonstration, after two layers of a trained network were broken, the run with the monitor only watching sat at chance accuracy (10.00%) for all 12 remaining epochs, while the run with repair recovered to 76.77% (healthy: 78.84%). That is one seed on one Apple-silicon Mac, a small model and a deliberately broken network; exact percentages shift with software version and hardware. It shows the mechanism, not what to expect on your models.

## What the summary report tells you

At the end of a run the monitor writes `summary.html` and `summary.md` next to the engineer report. One page, in plain language, under these headings:

- **A one-line verdict**, for example "Healthy", "Problem found and repaired" or "Problem, not repaired". If the monitor was wired in wrongly and never saw the model learn, it says **Not monitored** (or **Partly monitored**) instead of healthy; with no check at all, **Unknown**.
- **What happened:** which layers had problems, when, and whether any part was *proven* dead rather than suspected. "Stuck" means the weights show a channel is dead, constant or shrunk to almost zero; "silent" means it gave no output on the batches checked, which is not proof and is reported separately.
- **What it cost:** how many training steps, and how much wall-clock time, part of the model was stuck (silent channels are not counted). This is counted from the first check that saw the problem, so it may have started up to one check interval earlier. A dollar figure appears only if your team supplies a GPU-hour price (`ormas-monitor report RUN --gpu-hour-cost PRICE --gpus N`); it is labelled **ESTIMATE** and shows its formula (stuck hours × GPUs × price). Without a price it prints no dollars. It never guesses one.
- **What was done:** what was repaired, where and when.
- **Can you trust this record?** Whether the record verifies, and whether it is signed and by which key.
- **What this report does not tell you:** the same limits as on this page.
- **Words used here:** a short glossary.

Every number in the summary is computed from the run's record. Where the record cannot support a number, the summary says "unknown" rather than print one.

## What it costs

| | |
|---|---|
| **Licence** | Free. Apache-2.0, including commercial use. |
| **Engineering time** | Two lines in a training loop, or one callback for Lightning or the Hugging Face Trainer. The checkpoint check needs no code at all. |
| **Compute** | Measured: **+3.6%** median epoch time at the default setting, on one Apple-silicon Mac with a small CNN. **Not yet measured** on NVIDIA GPUs, multi-GPU or large models. |
| **Data exposure** | None by default. The core makes no network calls and never reads training data. |

## How it relates to AI-Act-style documentation

Regulations such as the EU AI Act (Regulation (EU) 2024/1689) require providers of some AI systems to keep technical documentation, which includes describing how a model was trained. A training record can be one piece of evidence in that documentation: it shows, with a hash chain and an optional signature, what the monitor observed during training, which checks ran, and every intervention made.

Be precise about the limits:

- **Using `ormas-monitor` does not make anything compliant** with the EU AI Act or any other regulation. The record is evidence, not a certificate.
- **No regulator, auditor or notified body has reviewed** this record format.
- **The record covers only what the monitor observes:** layer gradients, channel states, loss, and repairs. It says nothing about your data, its provenance, bias, evaluation, or the deployed system.
- **The record is written by the training process.** It shows that the record has not been edited since it was written (or signed); it does not prove the training happened as described. See the [threat model](concepts/record.md#threat-model).
- This page is our reading of the regulation's general shape, not legal advice. Ask counsel what your obligations are.

## What it does not do

- It is **not ORMAS**, Oxiedo's research training architecture. It needs no change to your model.
- It gives **no proofs for transformers or large language models.** On those it provides layer alarms, loss checks and the record only.
- It **never reads your data**, so it cannot find mislabelled or bad examples.
- It is **not the first tool to notice dead units.** Research methods such as ReDo detect and reset them. What is specific here is the certificate, with its assumptions checked, and the packaging.
- It has **not yet been tested on NVIDIA/Linux, multi-GPU or at scale.** A GPU stress test is planned.

Full list: [Limitations](limitations.md).

## Questions to ask your team

1. Which of our models use BatchNorm → ReLU (where the certificate applies), and which are transformers (where only alarms apply)?
2. Would we run it watch-only, or allow repair? Repair changes the model being trained.
3. Who holds the signing key? A key held by a reviewer, not the training machine, makes a signature mean more.
4. Have we added any sink that sends data off our machines?

Contact: raadh@oxiedo.com.

## Next step

Ask an engineer on your team to run one command on a model you already have (`ormas-monitor check model.pt`) and show you the report; it takes minutes. If you would like help, or want your team to shape what comes next, see [Work with us](work-with-us.md) or email [raadh@oxiedo.com](mailto:raadh@oxiedo.com).

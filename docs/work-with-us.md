# Work with us

`ormas-monitor` is free and open source, and it will stay that way. Oxiedo is now looking for its **first design partners**: a few teams who train their own models and want help getting more out of their training runs.

## What you get

- **Help setting it up.** We will help you add the monitor to your training code or your CI, and read the first reports with you.
- **A review of your runs.** Send us the reports (never your data or your model) and we will go through what the monitor found and what it means for your team.
- **A say in what we build next.** Design partners decide which of the planned features come first.
- **Early access.** You use the team features before anyone else, free for the duration of the partnership.

## What we ask

- A 30-minute call to understand how your team trains models today and where runs go wrong.
- Honest feedback after you have tried it on a real run.
- If it helps, permission to say so, anonymously if you prefer.

## Who it suits best today

- **Teams training or fine-tuning vision models** (ResNet, VGG, MobileNet, DenseNet and similar). These are the networks where the dead-channel proof applies.
- **Teams shipping models to phones, cameras or other devices,** where every unused channel costs speed and battery.
- **Regulated teams, for example in medical imaging,** that need a record of how a model was trained.

If you train transformers or large language models, talk to us too. Today you get the alarms and the record, but not the proof, and that is exactly the kind of gap we want partners to shape.

## Where this is going

**Oxiedo's goal: every model trains healthy, ships lean, and comes with a record anyone can verify, automatically.**

| From (how it usually is) | To (with Oxiedo) |
|---|---|
| You find out a run failed when the bill arrives | Failures are caught, and fixed, while training runs |
| Models ship carrying dead weight nobody can see | Dead parts are proven and removed before release |
| "How was this model trained?" has no good answer | Every model has a signed record of its training |

The free tool is the first piece. What comes next, with honest status:

| Product | What it is | Status |
|---|---|---|
| **ormas-monitor** (free, open source) | Finds, proves and repairs dead parts during training; checks model files; signed records and reports | **Available now** |
| **Oxiedo Lean** | One command that proves and removes dead weight, returning a smaller, faster model and a savings report | **Next.** Shown by hand on VGG-19-BN (18.6% less compute, same answers); not yet automated |
| **Oxiedo Guard** (team edition) | Every training run across a team watched and auto-repaired, release gates, "compare to the approved run", on your own servers | DESIGNED |
| **Oxiedo Record** | Training records formatted for model-risk reviews and AI-regulation technical documentation | DESIGNED |
| Transformer support | Proofs beyond the BatchNorm → ReLU building block | OPEN (research) |
| NVIDIA and multi-GPU testing | The test suite runs on Linux, macOS and Windows CPUs today | PLANNED |

## Get in touch

Email **[raadh@oxiedo.com](mailto:raadh@oxiedo.com?subject=ormas-monitor%20design%20partner)** with the subject "design partner", and tell us in two lines what you train and what goes wrong. You will hear back from the founder, Rokib Al Dhin Raadh, not a sales team.

Found a bug, or a finding that looks wrong? [Open an issue on GitHub](https://github.com/Oxiedo/ormas-monitor/issues). That helps just as much.

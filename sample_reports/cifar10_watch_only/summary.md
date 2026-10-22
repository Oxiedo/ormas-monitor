# CIFAR-10 dead-layer demo (watch)

*Training run summary · StandardCNN · 2026-10-23 00:10 · watch-only mode*

> **PROBLEM, NOT REPAIRED: Part of the model stopped learning (detected at step 400) and was still stuck when the run ended.**

## What happened

The monitor checked the model every 50 steps over 5,083 training steps (3 min 30 s). At step 400 it found 2 layers (conv2.0 and conv3.0) with stuck channels; 192 channels were proven dead, meaning they could never learn again on their own. They were still stuck when the run ended. The training loss (the error the model is trained to reduce) jumped once, first at step 400. The learning signal of 8 layers fell to zero or moved far outside its own normal range (9 warnings, first at step 450).

## What it cost

- 4,683 of 5,083 training steps (92%) ran with part of the model stuck.
- 3 min 17 s of wall-clock time (94% of 3 min 30 s).

_Counted from the first check that saw the problem; it may have started up to 50 steps earlier. Time comes from the record's own timestamps._

## What was done

Nothing. The monitor ran in watch-only mode: it recorded the problem but did not change the model. With repair switched on it re-initialises stuck channels automatically.

## Can you trust this record?

Partly: all 24 entries of the record check out, so no line was edited, removed or reordered after it was written, but it is not signed, so it cannot show who produced it or rule out the whole file being rewritten.

## What this report does not tell you

- It does not make anyone compliant with the EU AI Act or any other rule. The record is evidence for technical documentation, not a certificate.
- It does not measure how accurate or fit for purpose the model is. A run with no problems found can still produce a poor model.
- It never reads the training data, so it cannot find mislabelled or harmful examples.
- Proofs that a part of the model is dead apply only to one common building block (convolution or linear layer, then batch normalisation, then ReLU). Transformers and large language models get warnings only, never proofs.
- The tool has so far been tested on one Apple-silicon Mac, not yet on NVIDIA/Linux, multiple GPUs or at large scale.

## Words used here

- **Layer**: One stage of the neural network. Layers have internal names such as conv2.0.
- **Channel**: One small unit inside a layer. A layer has tens to thousands of them.
- **Stuck**: A channel that, judging from its weights, outputs nothing useful: proven dead, a constant, or shrunk to almost zero.
- **Proven dead**: A mathematical certificate that a channel can never learn again on its own, however long training continues.
- **Warning (alarm)**: A layer's learning signal moved far outside that layer's own normal range.
- **Tamper-evident record**: The log of everything the monitor saw. Each line carries a fingerprint of the line before it, so edits show.

Technical detail: `report.html` (engineer report) and the record itself, `diary.jsonl`.
Generated on this machine by ormas-monitor; nothing was uploaded.

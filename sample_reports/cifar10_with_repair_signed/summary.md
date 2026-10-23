# CIFAR-10 dead-layer demo (repair)

*Training run summary · StandardCNN · 2026-10-23 09:16 · repair on*

> **PROBLEM FOUND AND REPAIRED: Part of the model stopped learning (detected at step 400); the monitor repaired 192 channels by step 450. After step 450, no check found a channel proven dead, constant or decayed to ~0. 204 channels were still silent at the end (likely unused, not proven dead).**

## What happened

The monitor checked the model every 50 steps over 5,083 training steps (3 h 30 min). At step 400 it found stuck channels in 2 layers, `conv2.0` (Conv2d) and `conv3.0` (Conv2d); 192 channels were proven dead, meaning they could never learn again on their own. By step 450 those channels had been repaired. At the last check (step 5,050), 204 channels were silent in layer `classifier.0` (Linear): no positive output for several checks in a row. Such channels are likely unused, not proven dead, and are not counted as stuck. The training loss (the error the model is trained to reduce) jumped 3 times, first at step 400. The learning signal of 8 layers fell to zero or moved far outside its own normal range (8 warnings, first at step 450).

## What it cost

- 50 of 5,083 training steps (under 1%) ran with part of the model stuck.
- 15 min 19 s of wall-clock time (7% of 3 h 30 min).

_Counted from the first check that saw the problem; it may have started up to 50 steps earlier. Time comes from the record's own timestamps._

## What was done

The monitor repaired 192 channels in 2 layers (at step 450): it reset them to fresh starting values and cleared the optimizer's memory of them, so training could use them again. No limit was enforced on how much a repair could change the layer. At the end of the run no channel was proven dead, constant or decayed to ~0. Silent channels are left alone in this mode; repair="all" would reset them too.

## Can you trust this record?

Yes: all 37 entries of the record check out and the record is signed by the key with fingerprint deb089ff782e3437…, verified with that key, so nothing was edited, removed or rewritten after signing (a signature shows who vouched for the record, not that training happened as recorded).

## What this report does not tell you

- It does not make anyone compliant with the EU AI Act or any other rule. The record is evidence for technical documentation, not a certificate.
- It does not measure how accurate or fit for purpose the model is. A run with no problems found can still produce a poor model.
- It never reads the training data, so it cannot find mislabelled or harmful examples.
- Proofs that a part of the model is dead apply only to one common building block (convolution or linear layer, then batch normalisation, then ReLU). Transformers and large language models get warnings only, never proofs.
- The tool is tested automatically on Linux, macOS and Windows, but not yet on NVIDIA GPUs, multiple GPUs or at large scale; its measured results come from one Apple-silicon Mac.

## Words used here

- **Layer**: One stage of the neural network. Layers have internal names such as conv2.0.
- **Channel**: One small unit inside a layer. A layer has tens to thousands of them.
- **Stuck**: A channel that, judging from its weights, outputs nothing useful: proven dead, a constant, or shrunk to almost zero.
- **Proven dead**: A mathematical certificate that a channel can never learn again on its own, however long training continues.
- **Silent**: A channel that gave no positive output for several checks in a row. Often normal; it is not proof that the channel is dead, and it is not counted as stuck.
- **Warning (alarm)**: A layer's learning signal moved far outside that layer's own normal range.
- **Repair**: Resetting stuck channels to fresh starting values so training can use them again. Only happens if switched on.
- **Tamper-evident record**: The log of everything the monitor saw. Each line carries a fingerprint of the line before it, so edits show.
- **Signature**: A digital seal made with a private key. It shows who vouched for the record and detects a record rewritten from scratch by anyone who does not hold the private key.

Technical detail: `report.html` (engineer report) and the record itself, `diary.jsonl`.
Generated on this machine by ormas-monitor; nothing was uploaded.

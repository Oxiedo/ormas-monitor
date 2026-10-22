# CIFAR-10 dead-layer demo (repair)

*Training run summary · StandardCNN · 2026-10-23 00:13 · repair on*

> **PROBLEM FOUND AND REPAIRED: Part of the model stopped learning (detected at step 400); the monitor repaired it at step 450, and no part was stuck after that.**

## What happened

The monitor checked the model every 50 steps over 5,083 training steps (3 min 56 s). At step 400 it found 2 layers (conv2.0 and conv3.0) with stuck channels; 192 channels were proven dead, meaning they could never learn again on their own. By step 450 no layer was stuck any more. The training loss (the error the model is trained to reduce) jumped 3 times, first at step 400. The learning signal of 8 layers fell to zero or moved far outside its own normal range (8 warnings, first at step 450).

## What it cost

- 50 of 5,083 training steps (under 1%) ran with part of the model stuck.
- 3.2 s of wall-clock time (1% of 3 min 56 s).

_Counted from the first check that saw the problem; it may have started up to 50 steps earlier. Time comes from the record's own timestamps._

## What was done

The monitor repaired 192 channels in 2 layers (at step 450): it reset them to fresh starting values and cleared the optimizer's memory of them, so training could use them again. No limit was enforced on how much a repair could change the layer. No layer was stuck at the end of the run.

## Can you trust this record?

Yes: all 37 entries of the record check out and the record is signed by the key with fingerprint d8798457870b88e1…, verified with that key, so nothing was edited, removed or rewritten after signing (a signature shows who vouched for the record, not that training happened as recorded).

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
- **Repair**: Resetting stuck channels to fresh starting values so training can use them again. Only happens if switched on.
- **Tamper-evident record**: The log of everything the monitor saw. Each line carries a fingerprint of the line before it, so edits show.
- **Signature**: A digital seal made with a private key. It shows who vouched for the record and detects a record rewritten from scratch.

Technical detail: `report.html` (engineer report) and the record itself, `diary.jsonl`.
Generated on this machine by ormas-monitor; nothing was uploaded.

# Large models and `layers=`

By default the monitor watches every module that owns trainable parameters. On a model with thousands of modules that is more detail than you need, and each check costs a little more. Restrict it with `layers=`, one `fnmatch` pattern or a list of them, matched against module names from `model.named_modules()`:

```python
mon = Monitor(model, optimizer, layers=["encoder.layer.*.attention.*", "classifier"])
mon = Monitor(model, optimizer, layers="features.*")
```

Only matching modules get hooks, baselines and channel checks. A pattern list that matches no layer raises `ValueError` and shows a few real names to help you fix it. The patterns are written to the record in `run_start.config.layers`, so a reader can see what was and was not watched.

Wrappers such as `torch.compile` and DDP are unwrapped first, so patterns use the names of your own model.

## Cost

The monitor does nothing on unsampled steps. On a sampled step it:

- reads one gradient norm per watched layer, stacked on the device and copied to the host once;
- records the per-channel maximum activation of watched Conv/Linear/BatchNorm outputs during that step's forward pass.

Measured overhead is **+3.6%** median epoch time at `check_every=50` on the CIFAR-10 CNN (one seed, Apple M-series). It has not been measured on large models. **Status: OPEN.** To spend less, raise `check_every` or narrow `layers=`.

## Checking large checkpoints

`ormas-monitor check` reads sharded safetensors or `.bin` indexes (`*.index.json`) and Hugging Face model folders, and reads tensors one at a time rather than building the model. PyTorch zip-format files are memory-mapped. On one Apple-silicon Mac, a 1.78 GiB sharded fp16 checkpoint was checked in 7.95 s with a 220 MB peak footprint, against 2,039 MB for the 0.1 loader (one run; see [Results](../results.md#4-checkpoint-check-time-and-memory)). Not measured on larger checkpoints or other platforms.

## What applies to transformers

Alarms, loss checks and the record. Not the certificate: see [What you get on a transformer](huggingface.md#what-you-get-on-a-transformer). `check` adds findings that are relevant to transformers, such as normalisation scales at ~0 and embedding rows that never moved; see the [findings catalogue](../reference/check-findings.md).

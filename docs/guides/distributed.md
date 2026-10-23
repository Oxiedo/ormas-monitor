# Distributed training

!!! warning "Not yet tested on multi-GPU hardware"
    Distributed support is covered by two-process CPU tests (gloo) that run on Linux and macOS, not Windows. It has not been run on a multi-GPU machine or at scale. **Status: OPEN.** Please report what you see.

Create the monitor on **every rank**, the same way, with the wrapped model. The monitor detects `torch.distributed` and the wrapper itself.

```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
mon = Monitor(model, optimizer, out="ormas_run")
```

## What each rank does

- **Only rank 0 writes files, prints and calls sinks.** Every rank runs the checks, so all ranks make the same decisions, but there is one record, one `metrics.jsonl` and one set of reports. On other ranks, `close()` returns `None`.
- **Gradient alarms** use the gradients after DDP has averaged them, so every rank sees the same norms.
- **The `manifest` event** records the distributed mode (`ddp`, `distributed` or `fsdp`), so a reader of the record knows how the run was set up.

## DistributedDataParallel

Channel checks, the certificate and repair all work, with two adjustments:

- **Repair is identical on every rank.** A repair re-initialises weights outside the optimizer; if ranks drew different random filters, the replicas would silently diverge. Under `torch.distributed` the new filters come from a generator seeded from (`repair_seed`, step, layer); `repair_seed=None` is treated as `0`. See [MonitorConfig](../reference/config.md).
- **Repair targets only channels detected from the weights** (certified, constant, decayed), which are the same on every rank. With `repair="all"`, silent channels are **not** repaired, because activations differ between ranks; the `manifest` event says so.

## FullyShardedDataParallel

Under FSDP (or DTensor parameters) each rank holds a shard of each flattened parameter, so whole filters cannot be read:

- **Per-channel scans, the certificate and repair are disabled.** `repair=True` raises `ValueError`.
- **Gradient alarms, loss checks and the record still work.** The `manifest` event records why channel checks are off.

## SyncBatchNorm

`SyncBatchNorm` is treated like `BatchNorm2d`.

## Launch

Nothing special: `torchrun --nproc-per-node 4 train.py`.

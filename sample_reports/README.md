# Sample reports

These are real outputs of `ormas-monitor` 0.2. Open the `.html` files in a browser; nothing needs installing.

They come from `ormas-monitor demo --full`: the paper's CIFAR-10 CNN trained for 8 epochs, then two of its three conv layers zeroed with the optimizer state cleared, then trained for 12 more epochs from that identical state. One seed (0), on an Apple M-series GPU (MPS) with torch 2.11; exact percentages vary with torch version and hardware.

| Folder or file | What it shows |
|---|---|
| `cifar10_watch_only/` | The monitor only watches. The two layers are proven dead at the next check and the run stays at **10.00%** (chance). Includes `summary.html` (one page, plain language), `report.html` (engineer view), and the raw record (`diary.jsonl`, `metrics.jsonl`). |
| `cifar10_with_repair_signed/` | The same break with `repair=True`: 192 channels re-initialised, back to **76.77%** (78.84% before the break). The record is sealed and signed. Check it yourself with `ormas-monitor verify cifar10_with_repair_signed --key cifar10_with_repair_signed/signing-key.pub.pem`. |
| `diff_watch_vs_repair.html` | `ormas-monitor diff` of the two runs: what changed between them, layer by layer. |
| `checkpoint_check/resnet18_broken_vs_original.html` | `ormas-monitor check` on torchvision's ImageNet ResNet-18 with 40 filters of one layer zeroed, compared with the original (`--against`, `--arch torchvision:resnet18`). |

The signing key is a demo key made for these samples. A signature shows the files are unchanged since signing; it says nothing about who trained the model.

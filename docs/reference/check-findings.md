# `check` findings catalogue

Every finding `ormas-monitor check` can report, what it means, and how strong the evidence is. Each finding in the JSON output has `severity`, `kind`, `title` and `detail`, and usually `layer`, `batchnorm` and `channels`.

## Severities

| Severity | Meaning | Default exit code |
|---|---|---|
| `critical` | The file is broken: the model cannot produce finite output. | `1` |
| `warning` | Capacity is dead or effectively dead, or something died since the baseline. | `0` (see `--fail-on`) |
| `info` | Worth knowing; often harmless. | `0` |

## Finding kinds

| Kind | Severity | What it means | Evidence |
|---|---|---|---|
| `nonfinite` | critical | One or more tensors contain NaN or Inf. | Exact. |
| `missing_tensors` | critical | A sharded checkpoint's index names tensors that are not in its shards. | Exact. |
| `arch_partial_match` | warning | With `--arch`: the architecture matches the file only in part (some BatchNorm layers are in one but not the other). Under half matching on either side is an error instead (exit 2). | Exact count of matching names. |
| `newly_dead` | warning | With `--against`: channels that are certified dead now and were not in the baseline. | Exact, under the same conditions as `dead_channels`. |
| `dead_channels` | warning | Incoming filter (with zero or no bias) or BatchNorm scale exactly zero, BatchNorm shift ≤ 0: the BatchNorm outputs a constant ≤ 0, so a ReLU after it is closed for every input and passes no gradient (Theorem 2). | **Exact** for the weights. Without `--arch`, the title says "if a ReLU directly follows". Whether the channels stay dead in further training also depends on what a weights file does not hold: the optimizer and its momentum, BatchNorm in training mode, and the ReLU being the BatchNorm's *only* consumer (`--arch` tracing confirms that a ReLU receives the unmodified output, not that nothing else reads it). The live monitor checks all of these; see [Dead channels](../concepts/dead-channels.md). |
| `decayed_channels` | warning | Filter or BatchNorm scale below 10⁻⁵ × the layer's median (or a zero filter behind a nonzero bias), shift ≤ 0: they output about 0 for every input. What a dead channel looks like after weight decay has shrunk it. | From weights. Not exact, so **not** certified. |
| `zero_tensor` | warning | A weight tensor is entirely zero (biases, running means and norm scales excluded). | Exact. |
| `norm_scale_zero` | warning | LayerNorm/RMSNorm/GroupNorm scale entries at ~0: that feature leaves the norm as a constant for every input. Skipped for families that store 1 + weight (e.g. Gemma), recognised from `config.json`. | From weights. **Not** proven permanent: gradient can still reach the scale. |
| `identical_weights` | warning | With `--against`: every tensor is bit-for-bit unchanged; no training step changed the model, or both paths point to the same weights. | Exact. |
| `flat_channels` | info | Filter or scale exactly zero but shift > 0, or the BatchNorm output joins a skip connection before any ReLU: the channel outputs a constant and carries no information. | Exact. |
| `dormant_channels` | info | BatchNorm shift ≤ −3 × \|scale\|: if a ReLU follows, these fire on under ~0.13% of positions when normalised inputs are roughly normal. | **Estimate.** Labelled as one. |
| `collapsed_variance` | info | A BatchNorm's running variance is ~0 (below 10⁻¹²) for some channels: whatever reaches them is the same for every input. Usually a dead filter upstream. | From running statistics. |
| `zero_filters` | info | A Conv/Linear without a BatchNorm has output units whose weights are all zero; each outputs only its bias. | Exact. |
| `frozen_tensors` | info | With `--against`: some tensors are exactly unchanged. Expected for frozen layers (a frozen backbone, adapter-only training); if you meant to train them, they received no updates. | Exact. |
| `embedding_rows_unchanged` | info | With `--against`: rows of an embedding that never moved, typically tokens absent from the fine-tuning data. Usually harmless. | Exact. |

## How the check finds the wiring

A state dict holds tensors, not the graph. The check pairs each BatchNorm with the Conv/Linear that feeds it by name (`bn1` ↔ `conv1`, `layer.1` ↔ `layer.0`) and, failing that, by file order and matching size. Without `--arch` it assumes a BatchNorm feeds a ReLU unless its name contains `downsample`, `shortcut`, `skip` or `residual`, and qualifies its findings accordingly. With `--arch torchvision:<name>`, or `--arch hf` for a Hugging Face image model (built from `config.json` with random weights, no download; text models fall back to the heuristic and say why), it builds the real network once (on PyTorch's `meta` device where it can: shapes only, no weights; otherwise on the CPU), runs one forward pass, and uses the exact set of BatchNorms whose output a ReLU module receives unmodified. It does not check that the ReLU is the only consumer of that output. If under half of the file's BatchNorm layers are in the model (or of the model's in the file), the check stops with an error naming the mismatch (exit 2); a smaller mismatch adds an `arch_partial_match` warning. The terminal summary and the `wiring` key of the result say which method was used.

## The diff (`--against`)

With `--against`, the result also contains `diff`:

| Key | Meaning |
|---|---|
| `changed` | Every floating-point tensor with its relative change ‖new − old‖ / ‖old‖ and absolute change, largest first. |
| `unchanged` | Tensors that did not move at all. A layer you meant to train and that never moved is worth a look. |
| `added`, `removed`, `reshaped` | Tensors present in only one file, or with a different shape. |
| `newly_dead` | Per layer, how many channels are certified dead now and were not before. |
| `dtype_changed`, `embedding_rows_unchanged` | Tensors whose dtype changed, and per embedding the rows that never moved. |

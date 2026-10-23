# Results

Every number on this page was measured on **one Apple M-series Mac**: training results with seed 0 on its GPU (MPS), torch 2.11 and the default monitor settings, checkpoint timings on one run. None has been measured on NVIDIA/Linux, multi-GPU or at scale. The real reports are in [`sample_reports/`](https://github.com/Oxiedo/ormas-monitor/tree/main/sample_reports).

## 1. The dead-layer experiment

The experiment from the DeepMath 2026 poster:

- **Network:** the paper's StandardCNN for CIFAR-10: three Conv-BN-ReLU blocks and an MLP head (620,810 parameters), trained with AdamW (lr 1e-3, weight decay 1e-2).
- **The break:** after training, conv2 and conv3 are zeroed (filters and biases), and the optimizer state is cleared, as in the paper's harness.
- **The comparison:** from that identical state, one run where the monitor only watches, and one where it also repairs.

| Setting | Healthy | Watch only | With repair |
|---|---:|---:|---:|
| **Full:** 50,000 images, 8 epochs, then 12 after the break | 78.84% | **10.00%** for all 12 epochs | **76.77%** (192 channels repaired) |
| **Quick:** 10,000 images, 4 + 4 epochs | 61.76% | **10.00%** | **59.12%** (190 channels repaired) |

The quick numbers were reproduced exactly with ormas-monitor 0.2.0 on 2026-10-23 (about 50 s, CIFAR-10 already downloaded).

Test accuracy on the 10,000 CIFAR-10 test images. 10.00% is chance for 10 classes.

- **Watch only:** at the first check after the break, the monitor certified every channel of both layers as dead in the full run (in the quick run, 62 of conv2's 64 channels; the other 2 had a positive shift and were reported constant), then logged exactly zero gradient for every conv layer. The run then stayed at chance, as Theorem 2 says it must.
- **With repair:** the monitor re-initialised the proven-dead channels one check later and the network recovered, though not to full accuracy: the repaired channels start from scratch.

**Reproduce:**

```bash
pip install "ormas-monitor[demo]"
ormas-monitor demo --full --data DIR     # full; or: python examples/dead_layer_demo.py --data DIR
ormas-monitor demo --data DIR            # quick
```

Expect different numbers on other hardware and other torch versions, even with a fixed seed: a fresh install that resolved torch 2.14.1 got 61.96% / 10.00% / 62.67% (191 repaired) on the quick demo. The qualitative result has held so far: watch-only stays at chance, repair recovers most of the accuracy.

## 2. Overhead

| | Median epoch time |
|---|---:|
| Plain training | 12.12 s |
| Monitored, `check_every=50` | 12.55 s |
| **Overhead** | **+3.6%** |

Same CNN, full CIFAR-10 training set, alternating plain and monitored epochs; the first epoch of each is discarded as warm-up.

**Reproduce:** `python examples/overhead.py --data DIR --check-every 50`

The checks run only on sampled steps, so `check_every` trades detail for speed. Overhead on large models is **not yet measured**.

## 3. Published models carry dead channels

`ormas-monitor check` on 22 torchvision ImageNet models, with each model's real wiring (so skip connections are handled). **19 of the 21 that use BatchNorm → ReLU** contain channels **decayed to ~0**: the filter or BatchNorm scale is under 10⁻⁵ × the layer's median, and the shift is ≤ 0. EfficientNet-B0 has no ReLU, so it is out of scope.

| Model | Channels after BN→ReLU | Decayed to ~0 |
|---|---:|---:|
| resnet18 | 1,984 | 7 |
| resnet50 | 7,616 | 1 |
| resnet152 | 24,000 | 8 |
| vgg16_bn | 4,224 | 19 |
| vgg19_bn | 5,504 | 292 (5.3%) |
| mobilenet_v2 | 15,552 | 18 |
| mnasnet1_0 | 17,232 | 734 (4.3%) |
| densenet121 | 40,800 | 2,242 (5.5%) |
| densenet169 | 77,536 | 7,667 (9.9%) |

The full table for all 22 models is in [`scan_results/scan.md`](https://github.com/Oxiedo/ormas-monitor/blob/main/scan_results/scan.md).

On 64 real images (CIFAR-10 test images upscaled to 224 px), these channels output **exactly 0** in ResNet-18, VGG-19-BN and MnasNet, and at most 1.5 × 10⁻⁶ in DenseNet-169.

None of these channels is *exactly* zero, so none is certified. Shrinking BatchNorm scales toward zero is also a known pruning technique (network slimming), so the point is not that such channels exist; it is that one command finds them.

**Reproduce:** `python examples/scan_public_models.py` (downloads about 1.5 GB of weights; needs torchvision).

**Reproduce the 64-image activation check:**

```bash
python examples/reproduce/decayed_channels_on_images.py --data DIR
```

Output on the reference machine: the flagged channels' largest post-ReLU output is 0.00e+00 in resnet18, vgg19_bn and mnasnet1_0, and 1.54e-06 in densenet169. The median peak of the other channels in the same layers is 1.52, 0.82, 0.93 and 0.27 respectively.

## 4. Checkpoint check: time and memory

`ormas-monitor check` reads tensors one at a time instead of loading the whole file. One run on one Apple-silicon Mac; memory is the macOS peak footprint (memory the process owns, as Activity Monitor shows it):

| Checkpoint | Time | Peak footprint |
|---|---:|---:|
| ResNet-50 (98 MB), streamed | 0.32 s | 186 MB |
| Synthetic fp16 transformer, sharded safetensors (1.78 GiB), streamed | 7.95 s | 220 MB |
| The same, with the 0.1 loader (everything in memory) | 10.85 s | 2,039 MB |
| `--against` between two 1.78 GiB copies, streamed | 20.8 s | 238 MB |

**Reproduce:** `python -m ormas_monitor.checkpoint.bench` (ResNet-50 plus a synthetic folder of about 2 GB, written to a temporary directory and deleted afterwards; `--gb 0.5 --resnet ""` for a smaller run). Each measurement runs in a fresh subprocess. The peak footprint is macOS-only; on other platforms the script reports peak RSS, which overstates what the check needs because it counts file pages the OS can drop.

## 5. The paper's own network

Before any damage, the monitor flagged **160 of the 256 hidden units** in the StandardCNN's classifier layer as silent. Checked afterwards, they never fire on any of the 10,000 CIFAR-10 test images. The paper's baseline network had been carrying those silent units all along.

**Reproduce:** the silent-unit flag is in both demo reports (`classifier.0`, step 150). Reproduce the check over all images with:

```bash
python examples/reproduce/classifier_dead_units.py --data DIR
```

It trains the 8 pre-lesion epochs, then counts the units that never fire. Output on the reference machine: 160 of 256 never positive on the 10,000 test images, and 159 of 256 on the 50,000 training images.

## 6. Removing dead channels makes a model smaller and faster

`check` flagged 292 dead channels in torchvision's ImageNet VGG-19-BN. These are channels whose BatchNorm output is a constant ≤ 0 before a ReLU, so they always output 0. We cut them out of the network: their filters, their BatchNorm entries, and the matching inputs of the next layer. Then we compared the cut model with the original on 64 real photos (CIFAR-10 test images upscaled to 224 px).

| | Original | Dead channels removed |
|---|---:|---:|
| Top-1 answers | — | **identical on 100% of photos** (largest logit difference 1.3 × 10⁻⁵, floating-point rounding) |
| Compute per image | 19.63 GMACs | 15.98 GMACs (**18.6% less**) |
| Parameters | 143.7M | 142.2M |
| CPU latency, batch 16 | — | **12–18% faster** (three runs on an Apple-silicon laptop: 11.7%, 12.7%, 18.4%) |

The compute saving is much larger than the share of channels (5.3%) because most of the dead channels sit in the early layers, which run at high resolution. In one 224 × 224 layer, 20 of 64 channels are dead.

Reproduce: `python examples/reproduce/prune_vgg19.py --data DIR`. The script contains the removal code; `ormas-monitor` itself finds and proves the channels but does not yet remove them for you.

How much a given model gains depends entirely on how many dead channels it carries. VGG-19-BN carries many; torchvision's ResNet-50 has one. Run `ormas-monitor check` on your model to see.

## What these results do not show

- Anything on NVIDIA GPUs, multi-GPU, or models larger than these.
- Anything about transformers or LLMs: the certificate does not apply to them.
- How often dead layers occur in real training runs. The demo breaks a network deliberately.
- Variance across seeds or runs. Every number above is one seed, and the timings are one run.

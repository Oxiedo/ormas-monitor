"""Run the checkpoint check over torchvision's ImageNet-pretrained CNNs.

For each model it builds the real network once to find which BatchNorms feed a ReLU directly
(so skip connections are handled correctly), then checks the published weights.

Run:  python examples/scan_public_models.py [--out scan_results]   (downloads ~1.5 GB of weights)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision

from ormas_monitor.checkpoint import check
from ormas_monitor.structure import relu_followers

MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d",
          "wide_resnet50_2", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "densenet121",
          "densenet169", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
          "shufflenet_v2_x1_0", "mnasnet1_0", "regnet_x_400mf", "regnet_y_400mf",
          "regnet_y_1_6gf", "efficientnet_b0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scan_results")
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rows = []
    for name in a.models:
        model = torchvision.models.get_model(name, weights="DEFAULT").eval()
        follow = relu_followers(model, torch.randn(1, 3, 224, 224))
        res = check(model.state_dict(), relu_after=follow)
        layers = [l for l in res["layers"] if l["batchnorm"] in follow]
        row = {"model": name, "parameters": res["parameters"], "bn_feeding_relu": len(layers),
               "channels": sum(l["channels"] for l in layers),
               "certified": sum(len(l["certified"]) for l in layers),
               "constant": sum(len(l["flat"]) for l in res["layers"]),
               "decayed": sum(len(l.get("decayed", [])) for l in layers),
               "dormant": sum(len(l.get("dormant", [])) for l in layers),
               "worst_layer": max(layers, key=lambda l: len(l.get("dormant", [])) / l["channels"],
                                  default={}).get("batchnorm")}
        rows.append(row)
        print(f"{name:20s} {row['bn_feeding_relu']:4d} BN->ReLU layers  {row['channels']:7,d} channels  "
              f"certified dead {row['certified']:4d}  constant {row['constant']:4d}  decayed {row['decayed']:4d}  "
              f"dormant {row['dormant']:5d} ({100 * row['dormant'] / max(row['channels'], 1):.2f}%)", flush=True)
    (out / "scan.json").write_text(json.dumps({"torchvision": torchvision.__version__, "rows": rows}, indent=1))
    lines = ["| Model | BN→ReLU layers | Channels | Certified dead | Constant | Decayed to ~0 | Dormant (est.) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['bn_feeding_relu']} | {r['channels']:,} | {r['certified']} | {r['constant']} | "
              f"{r['decayed']} | "
              f"{r['dormant']:,} ({100 * r['dormant'] / max(r['channels'], 1):.2f}%) |" for r in rows]
    (out / "scan.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'scan.md'}")


if __name__ == "__main__":
    main()

# Flagged channels on real photos

2,000 ImageNetV2 photos, the same fixed sample for every model. One Apple M2 Mac, torch 2.11, published `DEFAULT` weights.

Flagged means decayed to ~0 (filter or BatchNorm scale under 10⁻⁵ of the layer median, shift ≤ 0). None is exactly zero, so none is certified.

| Model | Channels after BN→ReLU | Flagged | Share | Largest flagged output | Typical peak of other channels | Answers changed when switched off | Compute saved if cut |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 1,984 | 7 | 0.35% | 0 | 3.32 | not needed | not measured |
| resnet34 | 3,840 | 7 | 0.18% | 0 | 3.77 | not needed | not measured |
| resnet50 | 7,616 | 1 | 0.01% | 0 | 13.44 | not needed | not measured |
| resnet101 | 16,320 | 1 | 0.01% | 0 | 7.87 | not needed | not measured |
| resnet152 | 24,000 | 8 | 0.03% | 0 | 7.24 | not needed | not measured |
| resnext50_32x4d | 15,168 | 6 | 0.04% | 0 | 7.72 | not needed | not measured |
| wide_resnet50_2 | 15,168 | 0 | 0.00% | n/a | n/a | n/a | not measured |
| vgg11_bn | 2,752 | 32 | 1.16% | 0 | 4.95 | not needed | 5.6% |
| vgg13_bn | 2,944 | 91 | 3.09% | 0 | 2.12 | not needed | 17.8% |
| vgg16_bn | 4,224 | 19 | 0.45% | 0 | 3.71 | not needed | 2.3% |
| vgg19_bn | 5,504 | 292 | 5.31% | 0 | 1.58 | not needed | 18.6% |
| densenet121 | 40,800 | 2,242 | 5.50% | 5.5e-06 | 0.66 | 0 | not measured |
| densenet169 | 77,536 | 7,667 | 9.89% | 6.2e-06 | 0.48 | 0 | not measured |
| mobilenet_v2 | 15,552 | 18 | 0.12% | 0 | 5.44 | not needed | not measured |
| mobilenet_v3_large | 912 | 0 | 0.00% | n/a | n/a | n/a | not measured |
| mobilenet_v3_small | 336 | 30 | 8.93% | 0 | 3.17 | not needed | not measured |
| shufflenet_v2_x1_0 | 5,630 | 77 | 1.37% | 0 | 1.62 | not needed | not measured |
| mnasnet1_0 | 17,232 | 734 | 4.26% | 0 | 1.76 | not needed | not measured |
| regnet_x_400mf | 12,192 | 3 | 0.02% | 0 | 19.50 | not needed | not measured |
| regnet_y_400mf | 8,528 | 2 | 0.02% | 0 | 13.54 | not needed | not measured |
| regnet_y_1_6gf | 16,640 | 1 | 0.01% | 0 | 15.21 | not needed | not measured |

"Not needed": the flagged channels never fired on these photos, so switching them off cannot change an answer.

Compute saved is an operation count, and is only worked out for the VGG models, where the cut is simple.

# About

## Oxiedo

`ormas-monitor` is built by Rokib Al Dhin Raadh at [Oxiedo](https://oxiedo.com). It is released under the Apache-2.0 licence.

Contact: raadh@oxiedo.com.

## The paper

The dead-channel certificate is Theorem 2 of:

> R. A. D. Raadh. *Self-Repair as a Bounded Disturbance: Input-to-State Stability of Neural Network Training Dynamics.* DeepMath 2026, The Ohio State University, 29–30 October 2026. Accepted as a poster.

The paper studies self-repair during training as a disturbance to the training dynamics. One of its results is that a Conv/Linear → BatchNorm → ReLU channel with a zero filter (or zero BatchNorm scale) and a non-positive shift sits in an *absorbing dead set*: gradient descent cannot leave it, and neither can any repair whose size is proportional to the weights. Only a discrete jump, such as a re-initialisation, gets out. The CIFAR-10 dead-layer experiment in [Results](results.md) is the experiment from the poster.

## ORMAS, and how this tool relates to it

ORMAS is Oxiedo's research training architecture: a network design with local heads, three health signals and seven diagnoses, studied in Oxiedo's research. **`ormas-monitor` is not ORMAS and contains none of it.** The name reflects where the work came from, not what the package does.

`ormas-monitor` is a separate, smaller tool. It needs no architecture change: it attaches to an ordinary PyTorch model from outside, with hooks. It packages one result from that research context (the certificate) together with general-purpose monitoring, a checkpoint check, a training record and reports.

The ORMAS research results, including its accuracy numbers, are not claims about this tool and are not repeated here.

## Prior art

Dead and dormant units are well studied. ReDo (Sokar et al., *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*, ICML 2023) detects and resets dormant neurons. Network slimming (Liu et al., *Learning Efficient Convolutional Networks through Network Slimming*, ICCV 2017) drives BatchNorm scales toward zero to prune channels. `ormas-monitor` claims the certificate, checked at run time against its assumptions, and the packaging; not the discovery.

## Citing

See [`CITATION.cff`](https://github.com/Oxiedo/ormas-monitor/blob/main/CITATION.cff).

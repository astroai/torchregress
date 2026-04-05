As of April 2026, the strongest **general-purpose test-time formulation for tabular data under distribution shift** looks to be **tabular-specific test-time adaptation that jointly handles label shift and feature/covariate shift**, rather than generic vision-style entropy minimization alone. The clearest current front-runner in that direction is **PFT3A (Prior-Free Tabular Test-Time Adaptation, ICLR 2026 poster)**. It is explicitly designed for the realistic setting where you have **no source data and no prior knowledge at test time**, and it combines three pieces: **class-prior estimation** to correct label shift, **robust feature alignment** to cope with feature shift, and **representative subspace exploration** to suppress redundant or misleading features. The paper reports that this beats prior tabular TTA baselines, including FTAT, across its TableShift-based evaluations. ([OpenReview][1])

If I had to compress that into one sentence: the best current recipe is **“estimate target label proportions at test time, then adapt only on high-confidence / well-aligned structure, while constraining adaptation in feature space.”** That is the common pattern behind the newest stronger methods for tables. 

A good way to place the recent methods is this:

* **AdapTable (2024)**: early tabular-specific TTA; mainly **output-probability correction** via estimated target label distribution plus uncertainty calibration. Good conceptual step, but narrower than the later methods. ([OpenReview][2])
* **TabLog (ICML 2024)**: introduces a more structured formulation using **logic-rule representations** and assumes the logical structure remains invariant under shift; competitive or better than earlier tabular TTA methods, but its inductive bias is stronger and less “drop-in general” than the newer prior-free methods. ([Proceedings of Machine Learning Research][3])
* **FTAT (AAAI 2025)**: the first strong “fully test-time adaptation for tables” baseline; it explicitly tackles **label distribution optimization**, **covariate-shift adaptation**, and adaptation sensitivity, and reports the best average performance across three backbones in its benchmark. ([arXiv][4])
* **PFT3A (ICLR 2026)**: the most compelling current upgrade, because it removes FTAT’s dependence on source priors and improves average results over FTAT on both TabTransformer and FT-Transformer settings shown in the paper. For example, with TabTransformer it reports average accuracy / balanced accuracy / F1 of **68.59 / 67.42 / 72.65** versus FTAT’s **64.25 / 65.57 / 66.38**; with FT-Transformer it reports **68.01 / 67.51 / 74.16** versus **64.47 / 66.31 / 67.29**. 

For **streaming or temporally drifting deployment**, the most interesting recent variant is **OT3A (2025/2026 OpenReview submission)**. Its formulation is **online** rather than batch/offline: it uses **high-confidence, domain-consistent pseudo-labels** to estimate and correct target label shift, then applies **self-training plus entropy minimization** as batches arrive. I would treat it as the leading candidate when your deployment is sequential and nonstationary, though I would still place slightly more weight on PFT3A overall because its framing is cleaner and better grounded in the current tabular-TTA line. ([OpenReview][5])

If your base model is specifically **TabPFN or a tabular foundation model**, there is also a newer specialized answer: **DistPFN / DistPFN-T**. That is not a full adaptation scheme in the same sense; it is a **test-time posterior adjustment** for **label shift**, with no retraining or architecture change. It is attractive because it is lightweight and was evaluated on **250+ OpenML datasets**, but it is narrower: it is about **label-shift correction for TabPFN-like models**, not broad mixed-shift robustness for arbitrary tabular backbones. ([OpenReview][6])

On the robustness side, two benchmark results matter. First, **TableShift** remains the core benchmark for natural distribution shifts in tabular ML; it shows that robustness gains are possible, but they often trade off against in-distribution accuracy, and label-distribution changes are strongly tied to the observed shift gaps. Second, **TabFSBench (ICML 2025)** shows that **feature shifts remain a major unsolved weakness**: most tabular models struggle there, and performance degradation tracks the importance of shifted features. So even the best current TTA formulation is not “solved robustness”; it is better described as the best available compromise under mixed shifts. ([OpenReview][7])

My bottom line:

**Best current overall formulation:** **PFT3A-style prior-free tabular TTA**
**Why:** it is the most complete current answer to realistic tabular shift, because it treats **label shift + feature shift + source-free deployment** jointly. 

**Best formulation if deployment is an online stream:** **OT3A-style high-confidence online adaptation**. 

**Best formulation if you are using TabPFN specifically and mostly fear label shift:** **DistPFN-T**. ([OpenReview][6])

For our kind of work, we could currently prototype this stack:

1. strong tabular backbone,
2. **test-time class-prior estimation / posterior correction**,
3. **confidence-filtered pseudo-label updates**,
4. **feature-subspace alignment or masking**,
5. conservative update schedule to avoid collapse.

That is where the field appears to be converging. 

[1]: https://openreview.net/forum?id=BgSDPE24pa "Prior-free Tabular Test-time Adaptation | OpenReview"
[2]: https://openreview.net/forum?id=ws0F5NTzGw "AdapTable: Test-Time Adaptation for Tabular Data via Shift-Aware Uncertainty Calibrator and Label Distribution Handler | OpenReview"
[3]: https://proceedings.mlr.press/v235/ren24b.html "TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules"
[4]: https://arxiv.org/abs/2412.10871?utm_source=chatgpt.com "Fully Test-time Adaptation for Tabular Data"
[5]: https://openreview.net/forum?id=TA4R2GYA50 "Online Test-Time Adaptation in Tabular Data with Minimal High-Certainty Samples | OpenReview"
[6]: https://openreview.net/forum?id=vlpAgjkw39 "DistPFN: Test-Time Posterior Adjustment for Tabular Foundation Models under Label Shift | OpenReview"
[7]: https://openreview.net/forum?id=XYxNklOMMX "Benchmarking Distribution Shift in Tabular Data with TableShift | OpenReview"

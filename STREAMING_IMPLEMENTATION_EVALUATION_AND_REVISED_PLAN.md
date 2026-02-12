# Evaluation and Revised Plan: Uncertainty, Scalability, and Shift Robustness

Date: 2026-02-12
Status: Proposed replacement for `STREAMING_IMPLEMENTATION_PLAN.md`

## 1. Executive Evaluation

The current streaming plan is ambitious, but only partially aligned with the stated product focus.

### 1.1 Scorecard

| Dimension | Score (1-10) | Why |
|---|---:|---|
| Alignment to uncertainty + robustness | 5 | It includes drift and online ideas, but most effort is in generic training wrappers and non-core model families. |
| Scalability value to users | 6 | Targets real pain points (batch size, AMP), but overlaps heavily with PyTorch/Lightning/Accelerate. |
| PyTorch ecosystem fit | 6 | Uses familiar concepts, but several proposals conflict with current best practice (`torch.amp`, safe checkpoint APIs, functional per-sample grads). |
| Scientist adoption likelihood | 4 | Scientists typically prefer thin, composable tools inside existing training stacks over a new framework layer. |
| Delivery risk | 7 | Scope is too wide and mixes unrelated domains (regression, online learning infrastructure, RL quantile architectures). |

### 1.2 High-Impact Issues in Current Plan

1. Scope drift away from core identity.
The plan introduces large training infrastructure and RL-oriented quantile architectures that are not central to uncertainty-aware regression workflows.

2. Redundant abstractions.
`IncrementalOptimizer`, custom mixed-precision trainers, and custom dataloaders duplicate functionality available in PyTorch and ecosystem trainers.

3. Technical implementation risks.
- Gradient accumulation proposal can mis-handle normalization across variable-sized batches.
- Per-sample gradient proposal omits the functional model pattern (`functional_call`).
- AMP proposal uses deprecated `torch.cuda.amp.*` APIs instead of `torch.amp.*`.
- Checkpointing proposal monkeypatches `forward`, which is brittle.

4. Adoption risk from complexity.
Adding many new classes increases cognitive load and maintenance burden before proving measurable uncertainty/robustness gains.

## 2. Ecosystem Reality Check (What to optimize for)

### 2.1 PyTorch-native baselines are now strong

- Data streaming and online ingestion are expected through `Dataset`, `IterableDataset`, and `DataLoader`.
- Distributed scalability should default to DDP/FSDP-compatible patterns.
- Mixed precision should use `torch.amp.autocast(...)` and `torch.amp.GradScaler(...)`.
- `torch.compile` is mainstream but has warmup overhead and requires practical guidance.

### 2.2 Ecosystem adoption pattern

In practice, scientist users tend to combine:
- core PyTorch loops,
- Lightning or Accelerate for orchestration,
- domain-specific losses/metrics for scientific validity.

`torchregress` should therefore win on domain signal quality (uncertainty under shift) and compatibility, not on replacing training frameworks.

## 3. Revised Strategy

## Principle

Build the thinnest possible infrastructure needed to maximize three outcomes:

1. Better uncertainty quality under distribution shift.
2. Scalability without forcing a new training framework.
3. Fast adoption by scientists through drop-in APIs, reference recipes, and benchmark evidence.

## 4. Revised Implementation Plan

### Phase 0 (1-2 weeks): Scope correction and contracts

Deliverables:
1. Freeze/de-scope non-core items from the prior plan:
- IQN / FQF / QR-DQN / MQRNN
- custom `StreamingDataLoader`
- custom `IncrementalOptimizer`
- monkeypatch-based checkpointing utility
2. Publish API contracts for new additions:
- no mandatory training loop abstractions
- compatibility with plain PyTorch, Lightning, and Accelerate
- mask/weight semantics remain consistent with `BaseLoss`

Acceptance criteria:
1. Design note merged with explicit non-goals and integration guarantees.
2. API signatures reviewed for backward compatibility.

### Phase 1 (2-4 weeks): Shift-aware uncertainty evaluation (core value)

Deliverables:
1. New metrics/reporting module focused on shift robustness:
- rolling calibration error
- coverage drift over time windows
- conditional coverage by uncertainty bins and feature bins
- interval width vs. realized error diagnostics
2. Add a `shift_report(...)` utility that combines OOD scores and calibration metrics into one structured output.
3. Add examples for common scientific workflows (tabular + time-indexed batches).

Acceptance criteria:
1. Unit tests for all metric computations and edge cases.
2. Integration tests validating report outputs on synthetic covariate shift.
3. Docs page with interpretation guidance and failure modes.

### Phase 2 (3-5 weeks): Online calibration and robust adaptation

Deliverables:
1. Add online/streaming conformal recalibration utilities:
- rolling-window recalibration
- optional exponential forgetting
- drift-triggered recalibration hooks
2. Add robust loss scheduling utilities (e.g., adaptive Huber/density weighting schedule) for non-stationary data.
3. Add monitor helpers for delayed labels (common in scientific pipelines).

Acceptance criteria:
1. Simulation benchmark: maintain target coverage under induced shift.
2. Clear latency/memory profile for online updates.
3. Reproducible notebook and script in `examples/`.

### Phase 3 (2-3 weeks): Scalability integration utilities (thin layer only)

Deliverables:
1. Add lightweight helpers, not trainer replacements:
- gradient accumulation helper that is DDP-safe
- AMP helper updated to `torch.amp`
- optional `torch.compile` helper wrappers with caveat docs
2. Provide framework interop recipes:
- plain PyTorch loop
- Lightning recipe
- Accelerate recipe

Acceptance criteria:
1. Tested on CPU + single GPU + multi-GPU-compatible code path.
2. No duplicate optimizer/dataloader stacks added to the public API.
3. Performance docs include warmup and throughput methodology.

### Phase 4 (3-4 weeks): Scientist adoption package

Deliverables:
1. Benchmarks that reflect real adoption criteria:
- uncertainty calibration under in-distribution and shifted data
- robustness to outliers and heteroscedastic noise
- throughput/memory tradeoffs for scalable training
2. "Method selection" guide for scientists:
- which method for coverage guarantees vs decomposition
- when to use conformal, ensembles, evidential, quantiles
3. Publication-ready figures and scripts for reproducibility.

Acceptance criteria:
1. Benchmark suite runs from one command and emits comparable tables/plots.
2. Docs contain actionable recommendations and anti-patterns.
3. At least two end-to-end examples demonstrate shift detection + recalibration.

## 5. Proposed Initial Backlog (First 6 PRs)

1. PR1: Design doc + de-scope update + migration notes.
2. PR2: `metrics/shift.py` with rolling/conditional calibration metrics.
3. PR3: `metrics/shift_report.py` integrating OOD + calibration summaries.
4. PR4: Online conformal recalibration utility module + tests.
5. PR5: AMP/accumulation/compile compatibility helpers (thin API).
6. PR6: Benchmark scripts + docs + scientist-facing method guide.

## 6. De-scoped Items (for now)

These are explicitly postponed unless future evidence shows demand from core users:

1. Distributional RL quantile architectures (IQN, FQF, QR-DQN).
2. Full custom training framework abstraction.
3. Custom dataloader stack replacing `torch.utils.data`.
4. Automatic batch-size finder and invasive memory patching utilities.

## 7. Risks and Mitigations

1. Risk: Scope re-expands during implementation.
Mitigation: PR template requires stating which of the three pillars each change improves.

2. Risk: Added metrics without decision value.
Mitigation: every metric must map to a concrete operational action (recalibrate, retrain, reject, escalate).

3. Risk: Scalability helpers become framework clones.
Mitigation: enforce "no custom Trainer" rule and keep APIs functional/utilitarian.

## 8. Success Metrics (for this roadmap)

Product metrics:
1. Coverage error under shift decreases by >= 30% after recalibration.
2. OOD/shift event detection lead time improves versus current baseline.
3. No regression in existing uncertainty decomposition APIs.

Adoption metrics:
1. Time-to-first-use from install to calibrated interval workflow <= 20 minutes.
2. At least 3 examples mirror real scientific setups (tabular, time-indexed stream, heteroscedastic).
3. Minimal integration friction with Lightning/Accelerate/PyTorch-native loops.

## 9. External References informing this revision

1. PyTorch `torch.utils.data` documentation: [https://docs.pytorch.org/docs/stable/data.html](https://docs.pytorch.org/docs/stable/data.html)
2. PyTorch AMP (`torch.amp`) and deprecations: [https://docs.pytorch.org/docs/2.9/amp.html](https://docs.pytorch.org/docs/2.9/amp.html)
3. PyTorch per-sample gradients tutorial (`torch.func`): [https://docs.pytorch.org/tutorials/intermediate/per_sample_grads](https://docs.pytorch.org/tutorials/intermediate/per_sample_grads)
4. PyTorch checkpointing API: [https://docs.pytorch.org/docs/stable/checkpoint](https://docs.pytorch.org/docs/stable/checkpoint)
5. PyTorch `torch.compile` tutorial: [https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)
6. PyTorch DDP docs: [https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
7. TorchData status note (DataPipes/DataLoader2 deprecation direction): [https://docs.pytorch.org/data/0.9/](https://docs.pytorch.org/data/0.9/)
8. Lightning Trainer (current stable): [https://lightning.ai/docs/pytorch/stable/common/trainer.html](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
9. Accelerate gradient accumulation: [https://huggingface.co/docs/accelerate/main/en/usage_guides/gradient_accumulation](https://huggingface.co/docs/accelerate/main/en/usage_guides/gradient_accumulation)


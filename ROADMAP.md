# Roadmap

Current version: `0.1.0` (unreleased).  
Tests: 2723 pass, 89% coverage. Docs: build clean.

---

## v0.1.0 — Initial PyPI release

The version string exists; the release is not shipped. These milestones get it out the door.

### M1 — CI/CD infrastructure

- [ ] **ci.yml** — Run ruff, mypy, pytest on every PR/push (3.12–3.14)
- [ ] **release.yml** — Trusted Publishing to PyPI on annotated tags (flow documented, workflow not present)
- [ ] Badges in README match reality

### M2 — Lint & type gate clean

- [ ] `ruff check src/torchregress` — zero errors (currently 42, ~20 auto-fixable)
- [ ] `ruff format --check src/torchregress` — clean (23 files drift)
- [ ] `mypy src/torchregress` — zero errors (currently 19, mostly `Dict`/import issues in `ot_conformal.py`)
- [ ] Add `--strict` flag or equivalent to CI gate

### M3 — Docs & examples review

- [x] **Example smoke test** — All 63 examples parse (compileall clean; SyntaxWarnings are raw LaTeX strings, not errors).
- [x] **Example-code sync** — Orphan references removed; docs index and API pages updated.
- [~] **Docstring contract audit** — Added docstrings to most-used items (`GaussianNLLLoss.forward`, `crps_gaussian`). Remaining gaps tagged as ponytail debt.
- [x] **README example snippets** — All 3 snippets compiled and run; fixed `crps_gaussian` return type, `QuantileLoss`→`MultiQuantileLoss`, `SplitConformal` API.
- [x] **Cross-reference integrity** — `zensical build` passes with zero issues.
- [x] **Deleted examples** — 2 orphan deleted files had no references; 2 (`propensity_tail`, `self_agreement`) had cross-references — all updated. Regenerated `method_catalog`, `comparative_evidence_matrix`, `real_data_recommendation_guide`.
- [x] Clean up `__pycache__/` from the `examples/` directory

### M4 — Release logistics

- [ ] CHANGELOG.md (keep it simple: one `## [0.1.0]` section)
- [ ] Verify `./scripts/release/prepare_release.sh` passes end-to-end
- [ ] Push tag `v0.1.0` and confirm PyPI publish

### Stretch (nice but non-blocking)

- [ ] Drop remaining `np.ndarray` type annotations from `metrics/point.py` and `metrics/interval.py` where `convert_to_tensor` accepts both (requires API audit to avoid breaking user code)

---

## v0.2.0 — GPU & hardening

- [ ] **CUDA CI runner** — Add `runs-on: [self-hosted, gpu]` or `nvidia` matrix entry to CI so GPU tests run on every push
- [ ] **`@pytest.mark.cuda`** — Tag every test that exercises device-specific behaviour (device placement, tensor creation, NCCL ops) and add an `--cuda` marker filter
- [ ] **Mixed-device regression suite** — Test that every loss and metric produces bitwise-identical results on CPU vs CUDA (same seed, same inputs)
- [ ] **Automatic device fixture** — A `device` fixture in `conftest.py` that yields `torch.device("cuda")` when available, falls back to CPU. All parameterized tests consume it
- [ ] **CUDA OOM guard** — Graceful skip (not crash) when GPU memory is insufficient for a given batch size
- [ ] **MPS (Apple Silicon) consideration** — Document whether the project intends to support `mps`; if yes, add a smoke-test column

---

- [ ] Coverage > 92% (target untested paths in `test_time/transport.py`, `losses/imbalanced.py`)
- [ ] Property-based tests for distribution CRPS invariants
- [ ] Benchmarks for all loss functions (smoke exists, expand to regression tests)
- [ ] Full mypy strict coverage for `tests/` directory

---

## v0.3.0 — Documentation & examples

- [ ] Jupyter notebook examples for every method category in README
- [ ] API reference completeness audit (no undocumented public symbols)
- [ ] Decision-tree guide for method selection (mermaid flowchart, wired into `index.md`)
- [ ] Docstring consistency pass: all `Parameters`/`Returns` sections follow numpy-style

---

## Future (no commitment)

| Area | Ideas |
|------|-------|
| **Losses** | Skew-t, mixture of experts, copula losses, deep-kernel GP |
| **Metrics** | Variogram, scoring-rule suites, skill scores |
| **Ensemble** | Snapshot ensembles, SWAG, Laplace approximation |
| **Inference** | MCMC-based uncertainty via Pyro/BFGS bridges |
| **Test-time** | Online conformal, adaptive calibration, prototype methods |
| **Performance** | torch.compile support, vmap for ensemble heads, CUDA graphs |
| **Interop** | sklearn `Pipeline` / `GridSearchCV` adapters, MLflow tracking |

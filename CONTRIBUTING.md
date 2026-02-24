# Contributing to TorchRegress

## Development Setup

We use `uv` and `pixi` for environment management.

### Installation

```bash
uv sync
```

### Running Tests

To run all tests:
```bash
uv run pytest
```

To run a specific test file:
```bash
uv run pytest tests/losses/test_eiv.py
```

### Code Quality

We use `pre-commit` to ensure code quality.

```bash
pre-commit install
pre-commit run --all-files
```

## EIV Loss Implementation Notes

- Always use `torch.double` when performing `gradcheck` on EIV losses.
- Analytical EIV losses use second-order derivatives; ensure your model is twice-differentiable if you need gradients of the EIV loss.
- For stochastic EIV losses (Monte Carlo, Ensemble), use finite-gradient checks instead of `gradcheck`.

"""Comparison helper seeding delegates to shared reproducibility utilities."""

from __future__ import annotations

from torchregress.comparison import set_comparison_seed


def test_set_comparison_seed_uses_set_all_seeds(monkeypatch) -> None:
    calls: list[int] = []

    def _record(seed: int) -> None:
        calls.append(seed)

    monkeypatch.setattr("torchregress.comparison.set_all_seeds", _record)
    set_comparison_seed(42)
    assert calls == [42]

"""Unit tests for ``tools.benchmarks.compare_against_baseline``.

Two layers:
  - ``_parse_bench_output`` regex tests (lock down the median-line format).
  - End-to-end main() flow test (stash -> bench -> pop -> bench -> JSON
    report) with mocked git + bench calls, so the whole tool is
    regression-tested without running real benchmarks.
"""

import json
import subprocess
import sys

import pytest

from tools.benchmarks.compare_against_baseline import _parse_bench_output


def test_parses_us_iter_format() -> None:
    """``bench_gaussian.py`` prints ``us/iter``; the regex must accept it."""
    stdout = """
=== test | B=1024 D=5 CPU float32 | iters=100 repeats=5 ===
--- GaussianCRPSLoss ---
  forward only                           median  106.26 us/iter  (min 105.00, max 110.00)
  forward + backward                     median  251.29 us/iter  (min 248.00, max 255.00)
"""
    parsed = _parse_bench_output(stdout)
    assert parsed == {
        "forward only": 106.26,
        "forward + backward": 251.29,
    }


def test_parses_bare_us_format() -> None:
    """``profile_mvn.py`` / ``profile_compile.py`` print bare ``us``."""
    stdout = """
=== test | B=1024 D=5 CPU float32 | iters=100 repeats=5 ===
  cov + jitter (in-place clone + diag embed)  median    6.80 us  (min 6.50, max 7.10)
  cholesky (B, D, D)                          median   82.42 us  (min 80.00, max 85.00)
  default fwd                                 median  166.25 us  (min 160.00, max 170.00)
"""
    parsed = _parse_bench_output(stdout)
    assert parsed == {
        "cov + jitter (in-place clone + diag embed)": 6.80,
        "cholesky (B, D, D)": 82.42,
        "default fwd": 166.25,
    }


def test_handles_empty_input() -> None:
    assert _parse_bench_output("") == {}


def test_returns_empty_when_no_median_rows() -> None:
    """Header lines and section banners do not contain a 'median' row."""
    stdout = """
=== test | B=1024 D=5 CPU float32 | iters=100 repeats=5 ===
--- Section timings ---
--- Full forward ---
"""
    assert _parse_bench_output(stdout) == {}


def test_ignores_median_in_narration_not_followed_by_number() -> None:
    """A bare word 'median' in a non-row line should not be captured."""
    stdout = """
The median runtime is competitive.
  forward only                           median  123.45 us/iter  (min 120.00, max 130.00)
"""
    parsed = _parse_bench_output(stdout)
    assert parsed == {"forward only": 123.45}


def test_handles_mixed_us_and_us_iter_in_one_blob() -> None:
    """A combined / synthetic report should parse both unit forms."""
    stdout = """
  section A                              median  10.00 us  (min 9, max 11)
  section B                              median  20.00 us/iter  (min 19, max 21)
"""
    parsed = _parse_bench_output(stdout)
    assert parsed == {"section A": 10.00, "section B": 20.00}


def test_handles_decimal_label_widths() -> None:
    """The label group is non-greedy, so multi-decimal medians parse cleanly."""
    stdout = "  x  median  0.001 us  (min 0.001, max 0.001)\n"
    assert _parse_bench_output(stdout) == {"x": 0.001}


# ---------------------------------------------------------------------------
# End-to-end main() flow tests (mocked git + bench calls).
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Replace every subprocess call with a recording fake.

    The compare tool needs ``subprocess.run`` to do two things:
      1. ``git stash push`` / ``git stash pop`` (recorded but not executed).
      2. ``git rev-parse`` for the head/branch state (return fake strings).
    The fake below does both.  Real benchmarks are never invoked because
    the test also patches ``_run_bench``.
    """
    git_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        git_calls.append(list(cmd))
        if cmd[:2] == ["git", "rev-parse"]:
            stdout = "deadbeef\n" if cmd[2] == "HEAD" else "test-branch\n"
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    from tools.benchmarks import compare_against_baseline as cab

    monkeypatch.setattr(cab, "_run", fake_run)
    monkeypatch.setattr(cab, "_run_quiet", fake_run)
    return git_calls


def test_end_to_end_stash_bench_pop_report(tmp_path, monkeypatch, fake_subprocess) -> None:
    """Full main() flow: stash, bench 'before', pop, bench 'after', write report."""
    from tools.benchmarks import compare_against_baseline as cab

    output_dir = tmp_path / "reports" / "benchmarks"
    output_dir.mkdir(parents=True)
    bench_calls: list[str] = []

    def fake_run_bench(name: str, label: str) -> dict[str, float]:
        bench_calls.append(name)
        # Return slightly different medians on the second invocation so
        # delta_pct is non-zero and easy to assert.
        before = {"median-A": 100.0, "median-B": 200.0}
        after = {"median-A": 90.0, "median-B": 220.0}
        return before if "before" in label else after

    monkeypatch.setattr(cab, "_run_bench", fake_run_bench)
    monkeypatch.setattr(cab, "_has_uncommitted_changes", lambda paths: True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_against_baseline",
            "--benchmarks",
            "fake_a",
            "fake_b",
            "--output-dir",
            str(output_dir),
            "--label",
            "test-e2e",
        ],
    )

    rc = cab.main()
    assert rc == 0

    # Stash push + pop were both issued.
    stash_pushes = [c for c in fake_subprocess if "stash" in c and "push" in c]
    stash_pops = [c for c in fake_subprocess if "stash" in c and "pop" in c]
    assert len(stash_pushes) == 1, f"expected 1 stash push, got {stash_pushes}"
    assert len(stash_pops) == 1, f"expected 1 stash pop, got {stash_pops}"

    # Each benchmark ran twice (before + after).
    assert bench_calls.count("fake_a") == 2
    assert bench_calls.count("fake_b") == 2

    # Report structure on disk.
    report_path = output_dir / "compare_test-e2e.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["label"] == "test-e2e"
    assert report["has_changes"] is True
    assert "fake_a" in report["benchmarks"]
    assert "fake_b" in report["benchmarks"]

    a = report["benchmarks"]["fake_a"]
    assert a["before"] == {"median-A": 100.0, "median-B": 200.0}
    assert a["after"] == {"median-A": 90.0, "median-B": 220.0}
    # (-10% and +10%)
    assert a["delta_pct"]["median-A"] == pytest.approx(-10.0)
    assert a["delta_pct"]["median-B"] == pytest.approx(10.0)


def test_end_to_end_no_stash_path(tmp_path, monkeypatch, fake_subprocess) -> None:
    """``--no-stash`` skips the stash/pop dance; only 'after' is populated."""
    from tools.benchmarks import compare_against_baseline as cab

    output_dir = tmp_path / "reports" / "benchmarks"
    output_dir.mkdir(parents=True)
    bench_calls: list[str] = []

    def fake_run_bench(name: str, label: str) -> dict[str, float]:
        bench_calls.append(name)
        return {"median": 50.0}

    monkeypatch.setattr(cab, "_run_bench", fake_run_bench)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_against_baseline",
            "--benchmarks",
            "fake_a",
            "--output-dir",
            str(output_dir),
            "--label",
            "test-no-stash",
            "--no-stash",
        ],
    )

    rc = cab.main()
    assert rc == 0

    # No git stash calls at all.
    assert not any("stash" in c for c in fake_subprocess)
    # Bench ran only once.
    assert bench_calls.count("fake_a") == 1

    report = json.loads((output_dir / "compare_test-no-stash.json").read_text())
    a = report["benchmarks"]["fake_a"]
    assert a["before"] is None
    assert a["after"] == {"median": 50.0}
    # delta_pct is omitted when there is no before.
    assert "delta_pct" not in a


def test_end_to_exits_early_when_no_changes(tmp_path, monkeypatch, fake_subprocess) -> None:
    """No uncommitted changes + no ``--no-stash`` -> exit 1, no bench runs."""
    from tools.benchmarks import compare_against_baseline as cab

    output_dir = tmp_path / "reports" / "benchmarks"
    output_dir.mkdir(parents=True)
    bench_calls: list[str] = []

    monkeypatch.setattr(cab, "_run_bench", lambda *a, **k: bench_calls.append(a[0]) or {})
    monkeypatch.setattr(cab, "_has_uncommitted_changes", lambda paths: False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_against_baseline",
            "--benchmarks",
            "fake_a",
            "--output-dir",
            str(output_dir),
            "--label",
            "test-noop",
        ],
    )

    rc = cab.main()
    assert rc == 1
    assert bench_calls == []  # type: ignore[comparison-overlap]
    # No report should be written.
    assert list(output_dir.iterdir()) == []

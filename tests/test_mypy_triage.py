from tools import mypy_triage


def test_parse_and_summarize_mypy_output_by_package_and_code() -> None:
    text = "\n".join(
        [
            'torchregress/metrics/ood.py:55: error: "Tensor" not callable  [operator]',
            (
                "torchregress/metrics/ood.py:59: error: "
                'No overload variant of "cat" matches argument type "Tensor | Module"  '
                "[call-overload]"
            ),
            'torchregress/ensemble/swag.py:124: error: "Tensor" not callable  [operator]',
            (
                'torchregress/viz/utils.py:33: error: Library stubs not installed for "seaborn"  '
                "[import-untyped]"
            ),
            (
                "torchregress/health.py:10: error: "
                "Function is missing a return type annotation  [no-untyped-def]"
            ),
            "note: this line should be ignored",
        ]
    )

    report = mypy_triage.build_report(text)
    summary = report["summary"]

    assert summary["total_errors"] == 5
    assert summary["packages"]["metrics"] == 2
    assert summary["packages"]["ensemble"] == 1
    assert summary["packages"]["viz"] == 1
    assert summary["packages"]["root"] == 1
    assert summary["error_codes"]["operator"] == 2
    assert summary["error_codes"]["call-overload"] == 1

    first = report["errors"][0]
    assert first["path"] == "torchregress/metrics/ood.py"
    assert first["package"] == "metrics"

"""Regression tests for committed tutorial chart assets.

The documentation deliberately uses committed SVGs so Read the Docs does not
need to generate Plotly images during the build.  Keep both the assets and their
embeds protected: a broad documentation rewrite should not be able to silently
remove the charts while leaving the SVG files orphaned.
"""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATED = _REPO_ROOT / "docs" / "source" / "_static" / "generated"

_EXPECTED_CHARTS = {
    "docs/tutorials/getting_started.md": (
        "getting_started_aggregate_cdf.svg",
    ),
    "docs/tutorials/distributions_guide.md": (
        "distributions_guide_cdf.svg",
        "distributions_guide_histogram.svg",
    ),
    "docs/tutorials/coupling_groups_and_copulas.md": (
        "copula_scatter_plots.svg",
        "copula_rank_scatter.svg",
        "copula_value_scatter.svg",
    ),
    "docs/tutorials/risk_measures_and_allocation.md": (
        "risk_measure_weights.svg",
        "xol_pricing_curve.svg",
    ),
}


def test_documentation_chart_assets_are_committed_and_embedded() -> None:
    """Every intended tutorial chart must exist and remain embedded."""
    for tutorial_path, chart_names in _EXPECTED_CHARTS.items():
        tutorial = _REPO_ROOT / tutorial_path
        content = tutorial.read_text(encoding="utf-8")

        for chart_name in chart_names:
            asset = _GENERATED / chart_name
            assert asset.is_file(), f"Missing committed chart asset: {asset}"
            assert asset.stat().st_size > 0, f"Empty committed chart asset: {asset}"
            assert chart_name in content, (
                f"{tutorial_path} no longer embeds {chart_name}. "
                "If this chart is intentionally removed, update the regression "
                "test and remove the orphaned generated asset in the same change."
            )

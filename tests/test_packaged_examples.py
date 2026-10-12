"""Verify runnable examples and supporting datasets in installed package."""

from importlib import resources, util
from pathlib import Path


def test_example_modules_discoverable() -> None:
    """Every executable example has an importable module in PAL."""
    root = Path(__file__).resolve().parents[1] / "examples"
    for example in root.glob("*.py"):
        if example.name != "typecheck_exports.py":
            assert util.find_spec(f"pal.examples.{example.stem}") is not None


def test_example_csv_resources() -> None:
    """Example data can be located from the installed package."""
    root = resources.files("pal")
    for parts in [
        ("data", "catastrophes", "cat_ylt.csv"),
        ("data", "reserve_risk", "claims_triangle.csv"),
        ("data", "proteus_scenario_generator", "Economics_USD_Inflation_Inflation Index.csv"),
    ]:
        assert root.joinpath(*parts).is_file()
    assert resources.files("pal.examples").joinpath("data", "property_exposures.csv").is_file()


def test_odp_and_property_example_imports() -> None:
    """Examples expose their reusable model and data path."""
    from pal.examples.example_odp_reserve_risk import ODPModel
    from pal.examples.example_property_exposure_rating import DATA_PATH

    assert ODPModel.__name__ == "ODPModel"
    assert DATA_PATH.is_file()

"""Guard against omitting examples and data from distribution wheels."""
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


def test_wheel_contains_examples_and_csv_files(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(tmp_path), str(root)], check=True, capture_output=True, text=True)
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as z:
        names = set(z.namelist())
    expected = {f"pal/examples/{p.name}" for p in (root / "examples").glob("*.py") if p.name != "typecheck_exports.py"}
    expected |= {"pal/examples/__init__.py", "pal/examples/data/property_exposures.csv", "pal/data/catastrophes/cat_ylt.csv", "pal/data/reserve_risk/claims_triangle.csv", "pal/data/proteus_scenario_generator/Economics_USD_Inflation_Inflation Index.csv"}
    assert expected <= names, sorted(expected - names)

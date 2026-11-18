"""Guard against omitting examples and data from distribution wheels."""

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_wheel_contains_examples_and_csv_files(tmp_path: Path) -> None:
    """Build a wheel in isolation and inspect its packaged contents."""
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(tmp_path),
        str(root),
    ]
    # The command uses the current interpreter and a fixed local repository path.
    result = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
    assert result.returncode == 0, f"Wheel build failed:\\n{result.stdout}\\n{result.stderr}"

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as wheel_file:
        names = set(wheel_file.namelist())

    expected = {
        f"pal/examples/{example.name}"
        for example in (root / "examples").glob("*.py")
        if example.name != "typecheck_exports.py"
    }
    expected |= {
        "pal/examples/__init__.py",
        "pal/examples/data/property_exposures.csv",
        "pal/data/catastrophes/cat_ylt.csv",
        "pal/data/reserve_risk/claims_triangle.csv",
        "pal/data/proteus_scenario_generator/Economics_USD_Inflation_Inflation Index.csv",
    }
    assert expected <= names, sorted(expected - names)

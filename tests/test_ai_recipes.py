"""API discovery, packaging and executable smoke tests for PAL recipes."""

from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path

import pytest

from pal import api

RECIPES = api.recipes()


def test_recipe_catalog_has_unique_json_compatible_entries() -> None:
    """Every shipped recipe has a stable key and importable public metadata."""
    assert len(RECIPES) >= 7
    assert len({entry["id"] for entry in RECIPES}) == len(RECIPES)
    assert json.loads(json.dumps(RECIPES)) == RECIPES
    assert all(entry["module"].startswith("pal.examples.recipe_") for entry in RECIPES)
    assert all(entry["caveats"] for entry in RECIPES)


def test_recipe_search_is_useful_to_unfamiliar_assistants() -> None:
    """Search by actuarial intention, not only by an exact recipe identifier."""
    assert api.recipes("price excess of loss layer")[0]["id"] == "price-xol-layer"
    assert api.recipes("censored policy limit")[0]["id"] == "fit-censored-large-losses"
    assert api.recipes("sobol monte carlo")[0]["id"] == "use-rqmc-sampling"
    assert api.recipes("gamma fit")[0]["id"] == "fit-gamma-severity"
    assert api.recipes(limit=2) == RECIPES[:2]
    assert api.recipes("totally_unrelated_query_12345") == []


def test_recipe_lookup_is_non_executing_and_handles_errors() -> None:
    """Reading recipe code must have no simulation or fitting side effects."""
    item = api.recipe("price-xol-layer")
    assert item["id"] == "price-xol-layer"
    assert "from pal.contracts import XoL" in item["code"]
    assert "def main()" in item["code"]
    assert ast.parse(item["code"])
    assert api.recipe("PRICE-XOL-LAYER") == item

    with pytest.raises(ValueError, match="Unknown PAL recipe"):
        api.recipe("no-such-recipe")
    with pytest.raises(ValueError, match="limit must be at least 1"):
        api.recipes(limit=0)


@pytest.mark.parametrize("entry", RECIPES, ids=[item["id"] for item in RECIPES])
def test_packaged_recipe_executes_as_standalone_module(entry: dict[str, object], capsys: pytest.CaptureFixture[str]) -> None:
    """Every recommended snippet must run using the shipped PAL distribution."""
    details = api.recipe(str(entry["id"]))
    ast.parse(str(details["code"]))
    runpy.run_module(str(entry["module"]), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_evaluation_prompts_are_distinct_and_have_objective_rubrics() -> None:
    """The benchmark task set must stay machine-readable and reviewable."""
    root = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / "evals" / "pal_modeling_tasks.json").read_text(encoding="utf-8"))
    assert len(tasks) >= 10
    assert len({task["id"] for task in tasks}) == len(tasks)
    for task in tasks:
        assert task["prompt"]
        assert task["required_behaviour"]
        assert task["failure_modes"]

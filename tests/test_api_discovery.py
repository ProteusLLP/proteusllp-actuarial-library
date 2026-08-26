"""Tests for the installed-package API discovery helpers."""

import json

import pytest

from pal import api, distributions
from pal.variables import StochasticScalar


def test_api_module_is_exposed_at_top_level() -> None:
    import pal

    assert pal.api is api
    assert "pal.distributions" in api.modules()
    assert "pal.variables" in api.modules()
    assert "pal.stochastic_scalar" not in api.modules()


def test_distribution_catalog_contains_supported_aliases() -> None:
    entries = api.catalog("distributions")
    names = {entry["name"] for entry in entries}

    assert "Gamma" in names
    assert "Empirical" in names
    assert "HyperExponential" in names
    assert "MultivariateNormal" in names


def test_variables_catalog_exposes_stochastic_scalar() -> None:
    entries = api.catalog("variables")
    names = {entry["name"] for entry in entries}

    assert "ProteusVariable" in names
    assert "StochasticScalar" in names
    assert api.describe("StochasticScalar")["qualified_name"] == "pal.variables.StochasticScalar"
    assert api.describe(StochasticScalar)["qualified_name"] == "pal.variables.StochasticScalar"


def test_catalog_excludes_imported_implementation_names() -> None:
    names = {entry["name"] for entry in api.catalog("distributions")}

    assert "np" not in names
    assert "special" not in names
    assert "typing" not in names


def test_catalog_is_json_serialisable() -> None:
    json.dumps(api.catalog())


def test_describe_accepts_bare_qualified_and_object_names() -> None:
    bare = api.describe("Gamma")
    qualified = api.describe("pal.distributions.Gamma")
    by_object = api.describe(distributions.Gamma)

    assert bare["qualified_name"] == "pal.distributions.Gamma"
    assert qualified["qualified_name"] == bare["qualified_name"]
    assert by_object["qualified_name"] == bare["qualified_name"]
    assert bare["kind"] == "class"
    assert bare["signature"]
    assert bare["doc"]
    assert any(method["name"] == "generate" for method in bare["methods"])


def test_describe_output_is_json_serialisable() -> None:
    json.dumps(api.describe("Gamma"))


def test_search_finds_names_and_methods() -> None:
    gamma_results = api.search("gamma")
    percentile_results = api.search("percentile")

    assert gamma_results[0]["name"] == "Gamma"
    stochastic_scalar = next(entry for entry in percentile_results if entry["name"] == "StochasticScalar")
    assert stochastic_scalar["qualified_name"] == "pal.variables.StochasticScalar"


def test_search_supports_multiple_terms_and_limits() -> None:
    results = api.search("inverse gaussian", limit=2)

    assert 1 <= len(results) <= 2
    assert all("gaussian" in (entry["name"] + " " + entry["summary"]).lower() for entry in results)


def test_invalid_module_and_name_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="Unknown PAL API module"):
        api.catalog("not_a_module")

    with pytest.raises(ValueError, match="No public PAL API object"):
        api.describe("DefinitelyNotAPalObject")

    with pytest.raises(ValueError, match="limit must be at least 1"):
        api.search("gamma", limit=0)

"""Runtime discovery helpers for PAL's public API.

The functions in this module make an installed PAL package easy to interrogate
without access to the source tree or online documentation. Outputs contain only
ordinary Python and JSON-compatible values so coding assistants and other tools
can consume them directly.
"""

from __future__ import annotations

import importlib
import importlib.resources
import inspect
import re
import typing as t

__all__ = ["catalog", "describe", "modules", "recipe", "recipes", "search"]

_PUBLIC_MODULES: tuple[str, ...] = (
    "config",
    "contracts",
    "copulas",
    "couplings",
    "distributions",
    "frequency_severity",
    "maths",
    "multivariate_distributions",
    "risk_measures",
    "stats",
    "types",
    "variables",
)

_DISTRIBUTION_ALIASES = {
    "Empirical",
    "HyperExponential",
    "Dirichlet",
    "GeneralizedDirichlet",
    "InverseWishart",
    "InvertedDirichlet",
    "InvertedGeneralizedDirichlet",
    "MatrixDistributionBase",
    "Multinomial",
    "MultivariateDistributionBase",
    "MultivariateNormal",
    "MultivariateStudentsT",
    "Wishart",
}

_PUBLIC_OBJECTS = {"config": {"config"}}


def modules() -> list[str]:
    """Return the PAL modules included in runtime API discovery."""
    return [f"pal.{name}" for name in _PUBLIC_MODULES]


def _module(name: str) -> t.Any:
    return importlib.import_module(f"pal.{name}")


def _is_public_member(module_name: str, name: str, obj: t.Any) -> bool:
    if name.startswith("_"):
        return False

    module = _module(module_name)
    explicit = getattr(module, "__all__", None)
    if explicit is not None:
        return name in explicit

    if name in _PUBLIC_OBJECTS.get(module_name, set()):
        return True
    if module_name == "distributions" and name in _DISTRIBUTION_ALIASES:
        return True

    defining_module = getattr(obj, "__module__", None)
    return defining_module == f"pal.{module_name}"


def _kind(obj: t.Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    return "object"


def _safe_signature(obj: t.Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _summary(obj: t.Any) -> str:
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def _entry(module_name: str, name: str, obj: t.Any) -> dict[str, t.Any]:
    return {
        "name": name,
        "qualified_name": f"pal.{module_name}.{name}",
        "module": f"pal.{module_name}",
        "kind": _kind(obj),
        "signature": _safe_signature(obj),
        "summary": _summary(obj),
    }


def catalog(module: str | None = None) -> list[dict[str, t.Any]]:
    """Return a compact, JSON-serialisable catalogue of PAL's public API.

    Args:
        module: Optional module name such as ``"distributions"`` or
            ``"pal.distributions"``. If omitted, all discoverable public
            modules are included.
    """
    if module is None:
        module_names = _PUBLIC_MODULES
    else:
        module_name = module.removeprefix("pal.")
        if module_name not in _PUBLIC_MODULES:
            raise ValueError(f"Unknown PAL API module {module!r}. Available modules: {', '.join(_PUBLIC_MODULES)}")
        module_names = (module_name,)

    entries: list[dict[str, t.Any]] = []
    for module_name in module_names:
        module_obj = _module(module_name)
        for name, obj in inspect.getmembers(module_obj):
            if _is_public_member(module_name, name, obj):
                entries.append(_entry(module_name, name, obj))

    entries.sort(key=lambda item: t.cast(str, item["qualified_name"]).lower())
    return entries


def _resolve_name(name: str) -> tuple[str, str, t.Any]:
    normalised = name.removeprefix("pal.")

    if "." in normalised:
        module_name, member_name = normalised.split(".", 1)
        if module_name not in _PUBLIC_MODULES:
            raise ValueError(f"Unknown PAL API module {module_name!r}")
        module_obj = _module(module_name)
        if not hasattr(module_obj, member_name):
            raise ValueError(f"No public PAL API object named {name!r}")
        obj = getattr(module_obj, member_name)
        if not _is_public_member(module_name, member_name, obj):
            raise ValueError(f"{name!r} is not part of the discoverable public API")
        return module_name, member_name, obj

    matches = [entry for entry in catalog() if t.cast(str, entry["name"]).lower() == normalised.lower()]
    if not matches:
        raise ValueError(f"No public PAL API object named {name!r}")
    if len(matches) > 1:
        candidates = ", ".join(t.cast(str, entry["qualified_name"]) for entry in matches)
        raise ValueError(f"Ambiguous PAL API name {name!r}; use one of: {candidates}")

    match = matches[0]
    module_name = t.cast(str, match["module"]).removeprefix("pal.")
    member_name = t.cast(str, match["name"])
    return module_name, member_name, getattr(_module(module_name), member_name)


def _parameters(obj: t.Any) -> list[dict[str, t.Any]]:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return []

    parameters: list[dict[str, t.Any]] = []
    for parameter in signature.parameters.values():
        parameters.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name.lower(),
                "annotation": None if parameter.annotation is inspect.Parameter.empty else repr(parameter.annotation),
                "default": None if parameter.default is inspect.Parameter.empty else repr(parameter.default),
                "required": parameter.default is inspect.Parameter.empty
                and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
            }
        )
    return parameters


def _methods(obj: t.Any) -> list[dict[str, t.Any]]:
    if not inspect.isclass(obj):
        return []

    methods: list[dict[str, t.Any]] = []
    for name, member in inspect.getmembers(obj):
        if name.startswith("_") or not callable(member):
            continue
        methods.append(
            {
                "name": name,
                "signature": _safe_signature(member),
                "summary": _summary(member),
            }
        )
    return methods


def describe(obj_or_name: t.Any) -> dict[str, t.Any]:
    """Describe one PAL API object using JSON-serialisable metadata.

    ``obj_or_name`` may be an object itself, a bare public name such as
    ``"Gamma"``, or a qualified name such as ``"distributions.Gamma"`` or
    ``"pal.distributions.Gamma"``.
    """
    if isinstance(obj_or_name, str):
        module_name, name, obj = _resolve_name(obj_or_name)
    else:
        obj = obj_or_name
        matches = []
        for entry in catalog():
            entry_module = t.cast(str, entry["module"]).removeprefix("pal.")
            entry_name = t.cast(str, entry["name"])
            if getattr(_module(entry_module), entry_name) is obj:
                matches.append(entry)
        if not matches:
            raise ValueError("Object is not part of PAL's discoverable public API")
        match = matches[0]
        module_name = t.cast(str, match["module"]).removeprefix("pal.")
        name = t.cast(str, match["name"])

    entry = _entry(module_name, name, obj)
    entry.update(
        {
            "doc": inspect.getdoc(obj) or "",
            "parameters": _parameters(obj),
            "methods": _methods(obj),
        }
    )
    return entry



# The shipped Python recipe modules are the single source of executable code.
# This index holds only discovery metadata, not duplicated example snippets.
_RECIPE_INDEX: tuple[dict[str, t.Any], ...] = (
    {
        "id": "fit-gamma-severity",
        "title": "Fit a Gamma claim severity distribution",
        "summary": "Estimate Gamma shape and scale by maximum likelihood from uncensored claims.",
        "keywords": ["fit", "fitting", "gamma", "claims", "loss", "severity", "mle", "maximum likelihood"],
        "module": "pal.examples.recipe_gamma_fit",
        "caveats": ["theta is a scale, not a rate", "plug-in estimates ignore parameter uncertainty"],
    },
    {
        "id": "fit-censored-large-losses",
        "title": "Fit large losses with reporting thresholds and policy limits",
        "summary": "Fit pre-limit LogNormal severity using left truncation and right censoring.",
        "keywords": ["fit", "fitting", "large loss", "policy limit", "threshold", "censoring", "truncation", "mle"],
        "module": "pal.examples.recipe_censored_severity",
        "caveats": ["capped payments are censored", "tail extrapolation and parameter uncertainty matter"],
    },
    {
        "id": "simulate-frequency-severity",
        "title": "Simulate frequency and severity and aggregate annual losses",
        "summary": "Generate Poisson claim counts and LogNormal claims; derive aggregate and occurrence results.",
        "keywords": ["frequency", "severity", "compound", "poisson", "simulate", "aggregate", "occurrence"],
        "module": "pal.examples.recipe_frequency_severity",
        "caveats": ["claims and simulations have different axes"],
    },
    {
        "id": "price-xol-layer",
        "title": "Price an excess of loss reinsurance layer",
        "summary": "Simulate claim recoveries for an occurrence XoL layer and estimate annual expected loss.",
        "keywords": ["price", "pricing", "reinsurance", "xol", "excess", "loss", "layer", "attachment", "limit"],
        "module": "pal.examples.recipe_xol_layer",
        "caveats": ["layer terms apply per event", "simulated expected recovery is not a commercial premium"],
    },
    {
        "id": "model-copula-dependence",
        "title": "Model correlated risks with a Gaussian copula",
        "summary": "Reorder marginal simulations and preserve derived loss variables through coupling groups.",
        "keywords": ["correlation", "dependence", "copula", "gaussian", "portfolio", "coupling", "reordering"],
        "module": "pal.examples.recipe_copula_portfolio",
        "caveats": ["a copula changes dependence, not marginal values"],
    },
    {
        "id": "apply-stochastic-inflation",
        "title": "Inflate claim severities with a stochastic scenario factor",
        "summary": "Apply one inflation factor per simulation to all the claims in that simulation.",
        "keywords": ["inflation", "stochastic", "parameters", "frequency", "severity", "events", "coupling"],
        "module": "pal.examples.recipe_stochastic_inflation",
        "caveats": ["scenario-level inflation is not drawn independently for each claim"],
    },
    {
        "id": "use-rqmc-sampling",
        "title": "Use randomized quasi-Monte Carlo with a distribution",
        "summary": "Switch between ordinary Monte Carlo and scrambled Sobol RQMC sampling.",
        "keywords": ["rqmc", "qmc", "monte carlo", "sobol", "sampling", "simulation"],
        "module": "pal.examples.recipe_rqmc_sampling",
        "caveats": ["independent scrambles are needed to estimate RQMC uncertainty"],
    },
)

_RECIPE_STOPWORDS = frozenset({"a", "an", "and", "by", "for", "how", "in", "my", "of", "on", "the", "to", "using", "with"})


def recipes(query: str | None = None, limit: int = 20) -> list[dict[str, t.Any]]:
    """Find executable actuarial recipes by modelling intention.

    Return JSON-compatible metadata, ordered by keyword relevance. An empty
    query lists all recipes. Retrieve a full, runnable source with recipe(id).
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    terms = [
        word
        for word in re.findall(r"[a-z0-9]+", (query or "").lower())
        if word not in _RECIPE_STOPWORDS
    ]
    ranked: list[tuple[int, str, dict[str, t.Any]]] = []
    for item in _RECIPE_INDEX:
        haystack = " ".join(
            (item["id"], item["title"], item["summary"], *item["keywords"], *item["caveats"])
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if terms and score == 0:
            continue
        if terms and all(term in haystack for term in terms):
            score += 5
        ranked.append((-score, item["id"], item))
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    return [dict(item) for _, _, item in ranked[:limit]]


def recipe(identifier: str) -> dict[str, t.Any]:
    """Return metadata and packaged source code for one runnable recipe.

    The module is read from the installed package, so this also works in
    a wheel installation without access to GitHub or a source checkout.
    No example code is executed by the lookup.
    """
    key = identifier.strip().lower()
    for item in _RECIPE_INDEX:
        if item["id"] == key:
            filename = item["module"].rsplit(".", 1)[-1] + ".py"
            source = importlib.resources.files("pal.examples").joinpath(filename).read_text(encoding="utf-8")
            return {**item, "code": source}
    available = ", ".join(item["id"] for item in _RECIPE_INDEX)
    raise ValueError(f"Unknown PAL recipe {identifier!r}. Available recipes: {available}")


def search(query: str, limit: int = 20) -> list[dict[str, t.Any]]:
    """Search PAL's public API by name, signature, methods and documentation.

    Results favour exact and prefix name matches, followed by textual matches in
    names, signatures, summaries, full docstrings and public method names.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    query_lower = query.strip().lower()
    terms = [term for term in re.split(r"\s+", query_lower) if term]
    if not terms:
        return []

    ranked: list[tuple[int, str, dict[str, t.Any]]] = []
    for entry in catalog():
        module_name = t.cast(str, entry["module"]).removeprefix("pal.")
        name = t.cast(str, entry["name"])
        obj = getattr(_module(module_name), name)
        method_text = " ".join(f"{method['name']} {method['summary']}" for method in _methods(obj))
        signature = t.cast(str | None, entry["signature"]) or ""
        haystack = " ".join(
            [
                name,
                t.cast(str, entry["qualified_name"]),
                signature,
                t.cast(str, entry["summary"]),
                inspect.getdoc(obj) or "",
                method_text,
            ]
        ).lower()

        if not all(term in haystack for term in terms):
            continue

        lower_name = name.lower()
        score = 0
        if lower_name == query_lower:
            score += 100
        elif lower_name.startswith(query_lower):
            score += 50
        elif query_lower in lower_name:
            score += 25
        score += sum(5 for term in terms if term in lower_name)
        ranked.append((-score, t.cast(str, entry["qualified_name"]).lower(), entry))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in ranked[:limit]]

"""AICc-weighted model averaging over PAL univariate distributions.

Model uncertainty is a categorical choice of distribution family. Parameter
uncertainty, when requested, is sampled *within* each candidate family using
the fitted joint asymptotic covariance. Neither is a Bayesian posterior.
"""

from __future__ import annotations

import typing as t
from copy import copy
from dataclasses import dataclass

import numpy as np

from ._maths import scalar_or_array, to_backend, xp
from .config import config
from .couplings import couple_stochastic_results, validate_stochastic_axes
from .distributions import DiscreteDistributionBase, DistributionBase
from .fitting import FitResult, UncertaintyMethod
from .qmc import _inverse_transform_uniforms
from .stochastic_scalar import StochasticScalar


def _raw(value: t.Any) -> t.Any:
    """Extract the underlying value from a stochastic input.

    Args:
        value: Value to inspect or evaluate.
    """
    return value.values if isinstance(value, StochasticScalar) else value


def _model_indices(weights: np.ndarray, n_sims: int, rng: t.Any) -> StochasticScalar:
    """Draw one categorical model per simulation using PAL MC/RQMC uniforms.

    Args:
        n_sims: Number of simulations to generate.
        rng: Random number generator supplying the simulation draws.
        weights: Component weights or observation weights, according to the operation.
    """
    uniform = to_backend(_inverse_transform_uniforms(rng, n_sims))
    cumulative = to_backend(np.cumsum(weights)[:-1])
    return StochasticScalar(xp.searchsorted(cumulative, uniform, side="right"))


class WeightedDistribution(DistributionBase):
    """Probability mixture of PAL univariate distributions.

    The weights are non-negative and automatically normalised. Without a
    scenario-specific model choice, each call to generate draws a model for
    each future observation. With sample_models, the chosen family is held
    fixed for each scenario, including across repeated generate calls.

    Attributes:
        distributions: Component distributions in the same order as the model weights.
        weights: Normalized probabilities assigned to the component distributions.
    """

    def __init__(
        self,
        distributions: t.Sequence[DistributionBase],
        weights: t.Sequence[float] | np.ndarray,
        *,
        model_indices: StochasticScalar | None = None,
    ) -> None:
        """Validate and normalise candidate models and weights.

        Args:
            distributions: Component distributions used in the mixture.
            model_indices: Component index selected for each simulation.
            weights: Component weights or observation weights, according to the operation.
        """
        members = tuple(distributions)
        probabilities = np.asarray(weights, dtype=float)
        if not members or any(not isinstance(d, DistributionBase) for d in members):
            raise TypeError("distributions must be a nonempty sequence of PAL distributions")
        if probabilities.shape != (len(members),):
            raise ValueError("weights must have one element per distribution")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.any(probabilities > 0):
            raise ValueError("weights must be finite, non-negative and have a positive sum")
        kinds = [isinstance(d, DiscreteDistributionBase) for d in members]
        if any(kind != kinds[0] for kind in kinds):
            raise ValueError("Cannot mix discrete and continuous distributions")
        self.distributions = members
        self.weights = probabilities / probabilities.sum()
        # Flat stochastic parameters use PAL's standard scenario -> event
        # expansion; nested distribution objects are only family templates.
        self._component_parameter_names = tuple(tuple(d._params) for d in members)
        parameters = {
            f"component_{i}_{name}": value
            for i, distribution in enumerate(members)
            for name, value in distribution._params.items()
        }
        if model_indices is not None:
            parameters["model_indices"] = model_indices
        super().__init__(**parameters)

        if model_indices is not None:
            if model_indices.values.dtype.kind not in "iu":
                raise TypeError("model_indices must have an integer dtype")
            if bool(xp.any((model_indices.values < 0) | (model_indices.values >= len(members)))):
                raise ValueError("model_indices contains an invalid component index")
            validate_stochastic_axes(self._sources(), n_sims=model_indices.n_sims)

    @classmethod
    def fit(
        cls,
        observations: t.Sequence[t.Any] | np.ndarray,
        families: t.Sequence[type[DistributionBase]],
        *,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.NONE,
        per_family_options: t.Mapping[type[DistributionBase], t.Mapping[str, t.Any]] | None = None,
        model_weights: t.Sequence[float] | t.Mapping[type[DistributionBase], float] | None = None,
        **fit_options: t.Any,
    ) -> ModelAveragingResult:
        """Fit candidates to identical data and average by finite AICc scores.

        Shared fitting options include weights, censored and threshold when
        supported by each candidate. per_family_options supplies any differing
        family-specific fitting options, such as a Binomial trial count. A
        candidate that fails to fit, or whose estimator is nonregular, stops the
        averaging with its error rather than being dropped silently.

        Args:
            families: Candidate distribution classes to fit.
            fit_options: Shared fitting options supplied to every candidate.
            model_weights: User-specified model probabilities; omitted weights use information criteria.
            observations: Observed values used to estimate the distribution parameters.
            per_family_options: Fitting options overriding shared settings for individual candidates.
            uncertainty_method: Optional method for estimating joint parameter uncertainty.
        """
        candidates = tuple(families)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("families must be nonempty and contain no duplicates")
        if any(not isinstance(f, type) or not issubclass(f, DistributionBase) for f in candidates):
            raise TypeError("families must contain PAL distribution classes")
        from .distributions import GPD, MBBEFD, Categorical, HyperGeometric, Pareto, Uniform

        if any(f in (Categorical, HyperGeometric, MBBEFD, Uniform) for f in candidates):
            raise ValueError("AICc model averaging does not support nonregular or mixed-measure families")
        options = per_family_options or {}
        if any(f not in candidates for f in options):
            raise ValueError("per_family_options contains a family not in candidates")
        if any(
            key in {"weights", "censored", "threshold"} for family_options in options.values() for key in family_options
        ):
            raise ValueError("Observation, censoring and truncation options must be shared across all families")
        fitted: list[FitResult[t.Any]] = []
        for family in candidates:
            kwargs = {**fit_options, **options.get(family, {})}
            if "uncertainty_method" in kwargs:
                raise TypeError("Specify uncertainty_method only once")
            fit = t.cast(t.Any, family).fit(observations, uncertainty_method=uncertainty_method, **kwargs)
            if not fit.converged:
                raise RuntimeError(f"{family.__name__} fit did not converge")
            # AICc penalises each parameter by 2 because a regular maximum-likelihood
            # estimate adds about one chi-squared degree of freedom. An estimated
            # support endpoint has no such penalty, which is why Uniform is excluded.
            if family is Pareto and fit.threshold <= 0:
                raise ValueError("Pareto without a threshold estimates its lower endpoint, which AICc does not support")
            if family is GPD and not np.isclose(fit.parameters["loc"], fit.threshold, rtol=0, atol=1e-12):
                raise ValueError("A GPD with free location estimates a support endpoint, which AICc does not support")
            fitted.append(fit)

        scores = np.array([f.aicc for f in fitted], dtype=float)
        if len({(f.n_observations, f.n_censored, f.threshold) for f in fitted}) != 1:
            raise ValueError("Candidate fits must use matching sample, censoring and truncation")
        # AICc is comparable only for models fitted to the same observations
        # using the same likelihood measure, not a mixture of counts and losses.
        if model_weights is None:
            if not np.all(np.isfinite(scores)):
                raise ValueError("All candidates need finite AICc; check sample size and integer frequency weights")
            differences = -0.5 * (scores - scores.min())
            selected_weights = np.exp(differences)
        elif isinstance(model_weights, t.Mapping):
            if set(model_weights) != set(candidates):
                raise ValueError("model_weights mapping must contain exactly the fitted families")
            selected_weights = np.array([model_weights[family] for family in candidates], dtype=float)
        else:
            selected_weights = np.asarray(model_weights, dtype=float)
        ensemble = cls([fit.distribution for fit in fitted], selected_weights)
        return ModelAveragingResult(
            distribution=ensemble,
            fits=tuple(fitted),
            weights=ensemble.weights.copy(),
        )

    def lev(self, limit: float | np.ndarray) -> float | np.ndarray:
        """Return the probability-weighted limited expected value of the mixture.

        Args:
            limit: Policy limit or array of limits in the same units as the loss.
        """
        if self._sources():
            raise ValueError("lev requires fixed component parameters and model weights")
        result = np.zeros(np.asarray(limit).shape, dtype=float)
        for weight, component in zip(self.weights, self._components(), strict=True):
            if weight > 0:
                result += weight * np.asarray(component.lev(limit))
        return float(result) if result.ndim == 0 else result

    @property
    def model_indices(self) -> StochasticScalar | None:
        """Selected distribution index for each simulation, when sampled."""
        return t.cast(StochasticScalar | None, self._params.get("model_indices"))

    def _sources(self, extra: t.Any = None) -> tuple[StochasticScalar, ...]:
        """Collect the stochastic parameter sources used by the mixture.

        Args:
            extra: Additional stochastic inputs participating in the coupling operation.
        """
        sources = [p for p in self._params.values() if isinstance(p, StochasticScalar)]
        if isinstance(extra, StochasticScalar):
            sources.append(extra)
        return tuple(sources)

    def _components(self) -> tuple[DistributionBase, ...]:
        """Rebuild effective parameters from PAL's standard flat parameter store.

        FrequencySeverityModel copies a distribution and expands its _params
        to claim indices. Reading parameters here supports that generic
        mechanism without any compound-model-specific code.
        """
        result: list[DistributionBase] = []
        for i, distribution in enumerate(self.distributions):
            parameters = {name: self._params[f"component_{i}_{name}"] for name in self._component_parameter_names[i]}
            if all(parameters[name] is distribution._params[name] for name in parameters):
                result.append(distribution)
            else:
                effective = copy(distribution)
                effective._params = parameters
                result.append(effective)
        return tuple(result)

    def _evaluate_selected(self, method: str, argument: t.Any) -> t.Any:
        """Evaluate the distribution selected on each aligned scenario path.

        Args:
            argument: Probability-function input used to preserve simulation coupling.
            method: Sampling method used for the random draws.
        """
        if self.model_indices is None:
            raise RuntimeError("No scenario model has been selected")
        sources = self._sources(argument)
        validate_stochastic_axes(sources, n_sims=self.model_indices.n_sims)
        selected = self.model_indices.values
        values = xp.zeros(self.model_indices.n_sims, dtype=float)
        for i, distribution in enumerate(self._components()):
            evaluated = xp.asarray(to_backend(_raw(getattr(distribution, method)(argument))))
            if method == "cdf":
                evaluated = xp.clip(evaluated, 0.0, 1.0)
            values = xp.where(selected == i, evaluated, values)
        result = StochasticScalar(values)
        couple_stochastic_results((result,), sources, n_sims=self.model_indices.n_sims)
        return result

    def cdf(self, x: t.Any) -> t.Any:
        """Model-averaged CDF, or conditional CDF after choosing models.

        Args:
            x: Evaluation point, or stochastic evaluation points aligned with the model simulations.
        """
        if self.model_indices is not None:
            return self._evaluate_selected("cdf", x)
        sources = self._sources(x)
        n_sims = validate_stochastic_axes(sources)
        result: t.Any = 0.0
        for weight, component in zip(self.weights, self._components(), strict=False):
            if weight:
                probability = xp.clip(xp.asarray(to_backend(_raw(component.cdf(x)))), 0.0, 1.0)
                result = result + weight * probability
        if n_sims is not None:
            output = StochasticScalar(xp.broadcast_to(result, (n_sims,)).copy())
            couple_stochastic_results((output,), sources, n_sims=n_sims)
            return output
        return scalar_or_array(result)

    def invcdf(self, u: t.Any) -> t.Any:
        """Mixture quantile via bracketed bisection, with GPU-compatible arrays.

        Args:
            u: Cumulative probability in the interval from zero to one.
        """
        if self.model_indices is not None:
            return self._evaluate_selected("invcdf", u)
        sources = self._sources(u)
        n_sims = validate_stochastic_axes(sources)
        probabilities = xp.asarray(to_backend(_raw(u)))
        if bool(xp.any(~xp.isfinite(probabilities) | (probabilities < 0) | (probabilities > 1))):
            raise ValueError("u must contain finite probabilities between zero and one")

        # The mixture quantile lies between the smallest and largest component
        # quantiles at the same probability. No guess at a support bound needed.
        internal = xp.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
        components = self._components()
        # PAL inverse-CDF implementations accept backend arrays at runtime,
        # although the public DistributionParameter alias is scalar-oriented.
        brackets = [xp.asarray(to_backend(_raw(d.invcdf(t.cast(t.Any, internal))))) for d in components]
        left = brackets[0]
        right = brackets[0]
        for bracket in brackets[1:]:
            left = xp.minimum(left, bracket)
            right = xp.maximum(right, bracket)
        if bool(xp.any(~xp.isfinite(left) | ~xp.isfinite(right))):
            raise ValueError("Cannot bracket the mixture quantile for these parameters")
        if isinstance(self.distributions[0], DiscreteDistributionBase):
            # The discrete quantile is the smallest integer with CDF >= p.
            left = xp.floor(left)
            right = xp.ceil(right)
            for _ in range(70):
                middle = xp.floor(left + 0.5 * (right - left))
                cumulative = to_backend(_raw(self.cdf(StochasticScalar(middle) if n_sims is not None else middle)))
                left = xp.where(cumulative < internal, middle + 1, left)
                right = xp.where(cumulative >= internal, middle, right)
            result = left
        else:
            for _ in range(70):
                middle = left + 0.5 * (right - left)
                cumulative = to_backend(_raw(self.cdf(StochasticScalar(middle) if n_sims is not None else middle)))
                left = xp.where(cumulative < internal, middle, left)
                right = xp.where(cumulative >= internal, middle, right)
            result = left + 0.5 * (right - left)
        if bool(xp.any(probabilities == 0)):
            endpoints = [to_backend(_raw(d.invcdf(0.0))) for d in components]
            bottom = endpoints[0]
            for endpoint in endpoints[1:]:
                bottom = xp.minimum(bottom, endpoint)
            result = xp.where(probabilities == 0, bottom, result)
        if bool(xp.any(probabilities == 1)):
            endpoints = [to_backend(_raw(d.invcdf(1.0))) for d in components]
            top = endpoints[0]
            for endpoint in endpoints[1:]:
                top = xp.maximum(top, endpoint)
            result = xp.where(probabilities == 1, top, result)
        if n_sims is not None:
            output = StochasticScalar(xp.broadcast_to(result, (n_sims,)).copy())
            couple_stochastic_results((output,), sources, n_sims=n_sims)
            return output
        return scalar_or_array(result)

    def sample_models(self, n_sims: int | None = None, rng: t.Any = None) -> WeightedDistribution:
        """Choose a common model per scenario for reuse across future losses.

        Args:
            n_sims: Number of simulations to generate.
            rng: Random number generator supplying the simulation draws.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if not isinstance(n_sims, int) or isinstance(n_sims, bool) or n_sims < 1:
            raise ValueError("n_sims must be a positive integer")
        rng = config.rng if rng is None else rng
        if rng is None:
            raise RuntimeError("No PAL random generator is configured")
        sources = self._sources()
        validate_stochastic_axes(sources, n_sims=n_sims)
        choices = _model_indices(self.weights, n_sims, rng)
        couple_stochastic_results((choices,), sources, n_sims=n_sims)
        return WeightedDistribution(self.distributions, self.weights, model_indices=choices)

    def generate(self, n_sims: int | None = None, rng: t.Any = None) -> StochasticScalar:
        """Draw one loss per scenario, retaining model/parameter coupling.

        Args:
            n_sims: Number of simulations; None uses the configured simulation count.
            rng: Random number generator; omitted values use the configured generator.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if not isinstance(n_sims, int) or isinstance(n_sims, bool) or n_sims < 1:
            raise ValueError("n_sims must be a positive integer")
        rng = config.rng if rng is None else rng
        if rng is None:
            raise RuntimeError("No PAL random generator is configured")
        sources = self._sources()
        validate_stochastic_axes(sources, n_sims=n_sims)
        chosen = self.model_indices if self.model_indices is not None else _model_indices(self.weights, n_sims, rng)
        generated = [distribution.generate(n_sims, rng) for distribution in self._components()]
        values = generated[0].values
        for i in range(1, len(generated)):
            values = xp.where(chosen.values == i, generated[i].values, values)
        result = StochasticScalar(values)
        couple_stochastic_results((result,), (chosen, *generated, *sources), n_sims=n_sims)
        return result


@dataclass(frozen=True)
class ModelAveragingResult:
    """Candidate MLE fits, normalised AICc weights, and the point-estimate mixture.

    Attributes:
        distribution: Weighted distribution constructed from the fitted candidates.
        fits: Individual candidate fits and their likelihood diagnostics.
        weights: Normalized probabilities assigned to the component distributions.
    """

    distribution: WeightedDistribution
    fits: tuple[FitResult[t.Any], ...]
    weights: np.ndarray

    @property
    def aicc(self) -> dict[type[DistributionBase], float]:
        """Return the corrected Akaike information criteria of the candidate fits."""
        return {type(f.distribution): f.aicc for f in self.fits}

    @property
    def model_weights(self) -> dict[type[DistributionBase], float]:
        """Return the model probabilities associated with the candidate fits."""
        return {type(f.distribution): float(w) for f, w in zip(self.fits, self.weights, strict=False)}

    def sample_distribution(self, n_sims: int | None = None, rng: t.Any = None) -> WeightedDistribution:
        """Select one model and one joint parameter vector per scenario.

        Model identity and parameter draws are retained for repeated generation
        on the returned distribution. If a fit had no parameter uncertainty,
        that candidate retains its fixed maximum-likelihood parameters.

        Args:
            n_sims: Number of simulations to generate.
            rng: Random number generator supplying the simulation draws.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if not isinstance(n_sims, int) or isinstance(n_sims, bool) or n_sims < 1:
            raise ValueError("n_sims must be a positive integer")
        rng = config.rng if rng is None else rng
        if rng is None:
            raise RuntimeError("No PAL random generator is configured")
        components = [
            fit.uncertainty.sample_distribution(n_sims, rng) if fit.uncertainty is not None else fit.distribution
            for fit in self.fits
        ]
        return WeightedDistribution(components, self.weights).sample_models(n_sims, rng)

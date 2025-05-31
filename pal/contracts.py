from .config import xp as np
from .frequency_severity import FreqSevSims
from dataclasses import dataclass
from .variables import StochasticScalar


@dataclass
class ContractResults:
    """A class to hold the recoveries and reinstatement premiums resulting from applying a reinsurance contract to a
    set of claims."""

    def __init__(
        self,
        recoveries: FreqSevSims,
        reinstatement_premium: StochasticScalar | None = None,
    ):
        """
        Create a new contract results object.

        Args:
            recoveries (FreqSevSims): Object representing the recoveries.
            reinstatement_premium (np.ndarray | None): Optional Array representing the reinstatement premium.
        """
        self.recoveries = recoveries
        self.reinstatement_premium = reinstatement_premium


class XoL:
    """Represents an excess of loss reinsurance contract"""

    def __init__(
        self,
        name: str,
        limit: float,
        excess: float,
        premium: float,
        reinstatement_cost: list[float] | None = None,
        aggregate_limit: float | None = None,
        aggregate_deductible: float | None = None,
        franchise: float | None = None,
        reverse_franchise: float | None = None,
    ):
        """
        Initialize a new XoL layer.

        Args:
            name (str): The name of the XoL layer.
            limit (float): The limit of coverage.
            excess (float): The excess amount.
            premium (float): The premium amount.
            reinstatement_cost (list[float] | None, optional): The reinstatement cost as a fraction of the base
                premium, by reinstatement. Defaults to None.
            aggregate_limit (float, optional): The aggregate limit. Defaults to None.
            aggregate_deductible (float, optional): The aggregate deductible. Defaults to None.
            franchise (float, optional): The franchise amount. Defaults to None.
            reverse_franchise (float, optional): The reverse franchise amount. Defaults to None.
        """
        self.name = name
        self.limit = limit
        self.excess = excess
        self.aggregate_limit: float = (
            aggregate_limit if aggregate_limit is not None else np.inf
        )
        self.premium = premium
        self.aggregate_deductible: float = (
            aggregate_deductible if aggregate_deductible is not None else 0
        )
        self.franchise: float = franchise if franchise is not None else 0
        self.reverse_franchise: float = (
            reverse_franchise if reverse_franchise is not None else np.inf
        )
        self.summary: dict[str, float] = {}
        self.num_reinstatements = (
            aggregate_limit / limit - 1 if aggregate_limit is not None else None
        )
        self.reinstatement_premium_cost = (
            np.array(reinstatement_cost) if reinstatement_cost is not None else None
        )

    def apply(self, claims: FreqSevSims) -> ContractResults:
        """
        Apply the XoL contract to a set of claims.

        Args:
            claims (FreqSevSims): The simulated claims to apply the contract to.

        Returns:
            ContractResults: The results of applying the contract.

        Calculation of the recoveries from the excess of loss contract:

        Firstly, the effect of any franchise or reverse franchise is calculated on the individual losses.

        losses post franchise = loss if loss >= franchise and loss<= reverse franchise

        Next the individual losses to the layer are calculated:

        layer_loss = min(max(losses post franchise - excess, 0), limit)

        Then the aggregate layer losses before aggregate limit and deductible are calculated.
        The aggregate limit and deductible are then applied to get the aggregate recoveries for the layer:

        aggregate_recoveries = min(max(aggregate_layer_losses - aggregate_deductible, 0), aggregate_limit)

        The aggregate recoveries are then allocated back to the individual losses in proportion to the individual recoveries before aggregate limit and deductible.

        The reinstatement premium is calculated as the sum of the reinstatement premium cost multiplied by the number of reinstatements used.
        The number of reinstatements used is calculated as the minimum of the aggregate recoveries divided by the limit and the number of reinstatements available (which is the aggregate limit divided by the occurrence limit, less one).
        """
        # apply franchise
        if self.franchise != 0 or self.reverse_franchise != np.inf:
            is_in_window = (claims >= self.franchise) & (
                claims < self.reverse_franchise
            )
            claims = np.where(
                is_in_window,
                claims,
                0,
            )

        individual_recoveries_pre_aggregate: FreqSevSims = np.minimum(
            np.maximum(claims - self.excess, 0), self.limit
        )
        if self.aggregate_limit == np.inf and self.aggregate_deductible == 0:
            self.calc_summary(claims, individual_recoveries_pre_aggregate.aggregate())
            return ContractResults(individual_recoveries_pre_aggregate)
        aggregate_limit = (
            self.aggregate_limit if self.aggregate_limit is not None else np.inf
        )
        aggregate_deductible = (
            self.aggregate_deductible if self.aggregate_deductible is not None else 0
        )
        aggregate_recoveries_pre_agg = individual_recoveries_pre_aggregate.aggregate()

        aggregate_recoveries = np.minimum(
            np.maximum(aggregate_recoveries_pre_agg - aggregate_deductible, 0),
            aggregate_limit,
        )
        non_zero_recoveries = aggregate_recoveries != 0
        ratio = np.ones(aggregate_recoveries_pre_agg.values.shape)
        np.putmask(
            ratio,
            non_zero_recoveries.values,
            np.divide(
                aggregate_recoveries.values[non_zero_recoveries.values],
                aggregate_recoveries_pre_agg.values[non_zero_recoveries.values],
            ),
        )

        recoveries = individual_recoveries_pre_aggregate * ratio
        results = ContractResults(recoveries)
        if (
            self.reinstatement_premium_cost is not None
            and self.num_reinstatements is not None
        ):
            cumulative_reinstatement_cost = np.cumsum(self.reinstatement_premium_cost)
            limits_used = aggregate_recoveries / self.limit
            reinstatements_used = np.minimum(limits_used, self.num_reinstatements)
            reinstatements_used_full = StochasticScalar(
                np.floor(reinstatements_used).values.astype(int)
            )
            reinstatements_used_fraction = (
                reinstatements_used - reinstatements_used_full
            )
            reinstatement_number = np.maximum(reinstatements_used_full - 1, 0)
            reinstatement_premium_proportion = self.reinstatement_premium_cost[
                reinstatement_number.values
            ] * reinstatements_used_fraction + np.where(
                reinstatements_used_full.values > 0,
                cumulative_reinstatement_cost[reinstatement_number.values],
                0,
            )
            reinstatement_premium = reinstatement_premium_proportion * self.premium
            results.reinstatement_premium = reinstatement_premium
        self.calc_summary(claims, aggregate_recoveries)
        return results

    def calc_summary(
        self, gross_losses: FreqSevSims, aggregate_recoveries: StochasticScalar
    ):
        """
        Calculate a summary of the losses to the layer. The results are stored in the summary attribute of the layer.

        The summary includes the mean and standard deviation of the recoveries, the probability of attachment, the probability of vertical exhaustion and the probability of horizontal exhaustion.

        Args:
            gross_losses (FreqSevSims): Object representing the gross losses.
            aggregate_recoveries (StochasticScalar): Array of aggregate recoveries.

        Returns:
            None

        """
        mean = aggregate_recoveries.mean()
        sd = aggregate_recoveries.std()
        count = (aggregate_recoveries > 0).ssum()
        vertical_exhaust: FreqSevSims = np.maximum(
            gross_losses - self.limit + self.excess, 0
        )
        aggregate_vertical_exhaust = vertical_exhaust.aggregate()

        v_count = (aggregate_vertical_exhaust > 0).ssum()
        h_count = (aggregate_recoveries >= self.aggregate_limit).ssum()

        self.summary = {
            "mean": mean,
            "std": sd,
            "prob_attach": count / aggregate_recoveries.n_sims,
            "prob_vert_exhaust": v_count / len(gross_losses.values),
            "prob_horizonal_exhaust": (
                h_count / aggregate_recoveries.n_sims
                if self.aggregate_limit is not None
                else 0
            ),
        }

    def print_summary(self):
        """Print a summary of the losses to the layer

        >>> layer.print_summary()
        Layer Name : Layer 1
        Mean Recoveries:  100000.0
        SD Recoveries:  0.0
        Probability of Attachment:  0.0
        Probability of Vertical Exhaustion:  0.0
        Probability of Horizontal Exhaustion:  0.0

        """

        print("Layer Name : {}".format(self.name))
        print("Mean Recoveries: ", self.summary["mean"])
        print("SD Recoveries: ", self.summary["std"])
        print("Probability of Attachment: ", self.summary["prob_attach"]),

        print(
            "Probability of Vertical Exhaustion: ", self.summary["prob_vert_exhaust"]
        ),
        print(
            "Probability of Horizontal Exhaustion: ",
            self.summary["prob_horizonal_exhaust"],
        ),
        print("")


class XoLTower:
    """Represents a tower of excess of loss reinsurance contracts."""

    def __init__(
        self,
        limit: list[float],
        excess: list[float],
        premium: list[float],
        name: list[str] | None = None,
        reinstatement_cost: list[list[float] | None] | None = None,
        aggregate_deductible: list[float | None] | None = None,
        aggregate_limit: list[float | None] | None = None,
        franchise: list[float | None] | None = None,
        reverse_franchise: list[float | None] | None = None,
    ):
        """
        Create an XoL Tower.

        Args:
            limit (list[float]): A list of limits for each layer.
            excess (list[float]): A list of excesses for each layer.
            premium (list[float]): The premium for each layer.
            name (list[str], optional): A list of names for each layer. Defaults to None.
            reinstatement_cost (list[list[float]] | None, optional): A list of reinstatement costs for each reinstatement for each layer. Defaults to None.
            aggregate_deductible (list[float], optional): The aggregate deductible. Defaults to None.
            aggregate_limit (list[float], optional): The aggregate limit. Defaults to None.
            franchise (list[float], optional): The franchise amount. Defaults to None.
            reverse_franchise (list[float], optional): The reverse franchise amount. Defaults to None.
        """

        self.limit = limit
        self.excess = excess
        self.aggregate_limit = aggregate_limit
        self.aggregate_deductible = aggregate_deductible
        self.franchise = franchise
        self.reverse_franchise = reverse_franchise
        self.n_layers = len(limit)
        self.layers = [
            XoL(
                "Layer {}".format(i + 1) if name is None else name[i],
                limit[i],
                excess[i],
                premium[i],
                reinstatement_cost[i] if reinstatement_cost is not None else None,
                aggregate_limit[i] if aggregate_limit is not None else None,
                aggregate_deductible[i] if aggregate_deductible is not None else None,
                franchise[i] if franchise is not None else None,
                reverse_franchise[i] if reverse_franchise is not None else None,
            )
            for i in range(self.n_layers)
        ]

    def apply(self, claims: FreqSevSims) -> ContractResults:
        """
        Applies the XoL Tower to a set of claims.

        Parameters:
            claims (FreqSevSims): The set of claims to apply the XoL Tower to.

        Returns:
            ContractResults: The results of applying the XoL Tower to the claims.
        """
        recoveries = claims.copy() * 0
        reinstatement_premium = StochasticScalar(np.zeros(claims.n_sims))
        for layer in self.layers:
            layer_results = layer.apply(claims)
            recoveries += layer_results.recoveries
            reinstatement_premium += (
                layer_results.reinstatement_premium
                if layer_results.reinstatement_premium is not None
                else 0
            )

        return ContractResults(recoveries, reinstatement_premium)

    def print_summary(self):
        """Print a summary of the program losses"""
        for layer in self.layers:
            layer.print_summary()

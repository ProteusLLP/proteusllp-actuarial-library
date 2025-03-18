from pcm.variables import StochasticScalar
from pcm.couplings import ProteusStochasticVariable
from pcm.config import xp as np
import pandas as pd


class SimEventId:
    """A class to represent a mapping of sim_no to event_id."""

    def __init__(self, sim_no, sim_event_id, n_sims: int):
        super().__init__()
        self.sim_no = sim_no
        self.values = sim_event_id
        self.reverse_mapping = {
            event_id: sim_no for sim_no, event_id in zip(sim_no, sim_event_id)
        }
        self.n_sims = n_sims

    def __getitem__(self, item):
        return self.values[item]

    def get_sim_no(self, event_id):
        return np.array(
            [self.reverse_mapping[e] for e in event_id]
        )  # need a faster map here

    def _reorder_sims(self, ordering):
        self.sim_no = ordering[self.sim_no]


class SimEventLossTable(ProteusStochasticVariable):

    def __init__(
        self, sim_event_id: SimEventId, loss, master_sim_event_table: SimEventId
    ):
        self.sim_event_id = sim_event_id
        self.loss = loss
        self.master_sim_event_table = master_sim_event_table
        self.n_sims = master_sim_event_table.n_sims

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, master_yet: SimEventId):
        return cls(
            SimEventId(
                master_yet.get_sim_no(df["SimEventId"].values),
                df["SimEventId"].values,
                n_sims=master_yet.n_sims,
            ),
            df["Loss"].values,
            master_yet,
        )

    def _reorder_sims(self, new_order):
        self.master_sim_event_table._reorder_sims(new_order)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ) -> "SimEventLossTable":
        """Override the __array_ufunc__ method means that you can apply standard numpy functions"""
        inputs_list = list(
            (
                x.loss
                if isinstance(x, SimEventLossTable)
                else x  # promote an input ndarray to match the simulation index
            )
            for x in inputs
        )
        # if any of the inputs are StochasticScalars, we need to match the simulation index
        for i, x in enumerate(inputs_list):
            if isinstance(x, StochasticScalar):
                inputs_list[i] = x.values[self.sim_event_id.sim_no]
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x.values for x in out)
        result = SimEventLossTable(
            self.sim_event_id,
            getattr(ufunc, method)(*inputs_list, **kwargs),
            self.master_sim_event_table,
        )

        return result

    def __add__(self, other):
        if isinstance(other, SimEventLossTable):
            # combine the two tables
            # look at the union of the event_ids, etc
            assert (
                self.master_sim_event_table == other.master_sim_event_table
            ), "Cannot add two tables with different master sim event tables"
            event_ids = np.unique(
                np.concatenate((self.sim_event_id.values, other.sim_event_id.values))
            )
            sim_nos = self.master_sim_event_table.get_sim_no(event_ids)
            # create a new table with the combined event_ids
            loss = np.zeros(len(event_ids))
            loss[np.searchsorted(event_ids, self.sim_event_id.values)] = self.loss
            np.add.at(
                loss, np.searchsorted(event_ids, other.sim_event_id.values), other.loss
            )
            return SimEventLossTable(
                SimEventId(sim_nos, event_ids, self.master_sim_event_table.n_sims),
                loss,
                master_sim_event_table=self.master_sim_event_table,
            )
        elif isinstance(other, float | int):
            return SimEventLossTable(
                self.sim_event_id, self.loss + other, self.master_sim_event_table
            )
        elif isinstance(other, StochasticScalar):
            return SimEventLossTable(
                self.sim_event_id,
                self.loss + other[self.sim_event_id.sim_no],
                self.master_sim_event_table,
            )
        else:
            raise ValueError(f"Cannot add SimEventLossTable to {type(other)} type")

    def __sub__(self, other):
        if isinstance(other, SimEventLossTable):
            # combine the two tables
            # look at the union of the event_ids, etc
            assert (
                self.master_sim_event_table == other.master_sim_event_table
            ), "Cannot add two tables with different master sim event tables"
            event_ids = np.unique(
                np.concatenate((self.sim_event_id.values, other.sim_event_id.values))
            )
            sim_nos = self.master_sim_event_table.get_sim_no(event_ids)
            # create a new table with the combined event_ids
            loss = np.zeros(len(event_ids))
            loss[np.searchsorted(event_ids, self.sim_event_id.values)] = self.loss
            np.subtract.at(
                loss, np.searchsorted(event_ids, other.sim_event_id.values), other.loss
            )
            return SimEventLossTable(
                SimEventId(sim_nos, event_ids, self.master_sim_event_table.n_sims),
                loss,
                master_sim_event_table=self.master_sim_event_table,
            )
        elif isinstance(other, float | int):
            return SimEventLossTable(
                self.sim_event_id, self.loss - other, self.master_sim_event_table
            )
        elif isinstance(other, StochasticScalar):
            return SimEventLossTable(
                self.sim_event_id,
                self.loss - other[self.sim_event_id.sim_no],
                self.master_sim_event_table,
            )
        else:
            raise ValueError(f"Cannot add SimEventLossTable to {type(other)} type")

    def __gt__(self, other):
        if isinstance(other, float | int):
            return SimEventLossTable(
                self.sim_event_id,
                self.loss > other,
                self.master_sim_event_table,
            )
        elif isinstance(other, StochasticScalar):
            return SimEventLossTable(
                self.sim_event_id,
                self.loss > other[self.sim_event_id.sim_no],
                self.master_sim_event_table,
            )
        else:
            raise ValueError(f"Cannot compare SimEventLossTable to {type(other)} type")

    def _rle__(self, other):
        return self.__gt__(other)

    def __le__(self, other):
        if isinstance(other, float | int):
            return SimEventLossTable(
                self.sim_event_id,
                self.loss <= other,
                self.master_sim_event_table,
            )
        else:
            raise ValueError(f"Cannot compare SimEventLossTable to {type(other)} type")

    def _rgt__(self, other):
        return self.__le__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SimEventLossTable):
            # combine the two tables
            # look at the union of the event_ids, etc
            assert (
                self.master_sim_event_table == other.master_sim_event_table
            ), "Cannot add two tables with different master sim event tables"
            event_ids = np.union1d(self.sim_event_id.values, other.sim_event_id.values)
            sim_nos = self.master_sim_event_table.get_sim_no(event_ids)
            # create a new table with the combined event_ids
            loss = np.ones(len(event_ids))
            np.multiply.at(
                loss, np.searchsorted(event_ids, self.sim_event_id.values), self.loss
            )
            np.multiply.at(
                loss, np.searchsorted(event_ids, other.sim_event_id.values), other.loss
            )
            return SimEventLossTable(
                SimEventId(sim_nos, event_ids),
                loss,
                master_sim_event_table=self.master_sim_event_table,
            )
        elif isinstance(other, float | int):
            return SimEventLossTable(
                self.sim_event_id, self.loss * other, self.master_sim_event_table
            )
        elif isinstance(other, StochasticScalar):
            return SimEventLossTable(
                self.sim_event_id,
                self.loss * other[StochasticScalar(self.sim_event_id.sim_no)].values,
                self.master_sim_event_table,
            )
        else:
            return other.__rmul__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def aggregate(self) -> StochasticScalar:
        """Aggregates the losses by sim_no."""
        result = np.zeros(self.n_sims)
        np.add.at(
            result,
            self.sim_event_id.sim_no,
            self.loss,
        )
        return StochasticScalar(result)

    def __repr__(self):
        return f"SimEventLossTable(\nSimEventIds{self.sim_event_id.values}\nLoss {self.loss})"

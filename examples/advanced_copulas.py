import numpy as np
from examples.utilities import rank_scatter_plot

from pal import ProteusVariable, copulas, distributions

lobs = ["Motor", "Property", "Liability", "Specialty"]
losses = ProteusVariable(
    "lob",
    {
        lob: distributions.Gamma(alpha=alpha, theta=theta).generate()
        for lob, alpha, theta in zip(lobs, [2.0, 1.0, 4.0, 0.5], [0.5, 0.75, 0.25, 2])
    },
)
tail_dependence_matrix = np.array([[1, 0.5, 0.2, 0.1], [0.5, 1, 0.3, 0.2], [0.2, 0.3, 1, 0.4], [0.1, 0.2, 0.4, 1]])
copulas.HuslerReissCopula.from_tail_dependence_matrix(tail_dependence_matrix).apply(losses)
rank_scatter_plot(losses).show()

"""Proteus styling for Plotly figures.

Importing :mod:`pal` registers the ``proteus`` template and makes it the
Plotly default. Applications can still select another template explicitly
with ``fig.update_layout(template="plotly_white")``.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

PROTEUS_NAVY = "#001a64"
PROTEUS_BLUE = "#1d4ed8"
PROTEUS_SKY = "#4aa3df"
PROTEUS_ORANGE = "#f59e0b"
PROTEUS_TEAL = "#0f766e"

PROTEUS_COLORWAY = [PROTEUS_NAVY, PROTEUS_BLUE, PROTEUS_SKY, PROTEUS_ORANGE, PROTEUS_TEAL]

proteus_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=PROTEUS_COLORWAY,
        font={"family": "Arial, sans-serif", "color": "#0f172a"},
        title={"font": {"size": 22, "color": PROTEUS_NAVY}, "x": 0.02, "xanchor": "left"},
        legend={"bgcolor": "rgba(255,255,255,0.8)"},
        margin={"l": 70, "r": 40, "t": 70, "b": 65},
        xaxis={"showline": True, "linecolor": "#cbd5e1", "gridcolor": "#e5eaf2", "zeroline": False},
        yaxis={"showline": True, "linecolor": "#cbd5e1", "gridcolor": "#e5eaf2", "zeroline": False},
        hoverlabel={"bgcolor": "white", "font": {"color": "#0f172a"}},
        annotations=[
            {
                "text": "PROTEUS",
                "xref": "paper",
                "yref": "paper",
                "x": 1,
                "y": -0.16,
                "xanchor": "right",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 10, "color": PROTEUS_NAVY},
                "opacity": 0.45,
            }
        ],
    )
)

pio.templates["proteus"] = proteus_template
pio.templates.default = "proteus"

__all__ = [
    "PROTEUS_BLUE",
    "PROTEUS_COLORWAY",
    "PROTEUS_NAVY",
    "PROTEUS_ORANGE",
    "PROTEUS_SKY",
    "PROTEUS_TEAL",
    "proteus_template",
]

"""Proteus styling for Plotly figures.

Importing :mod:`pal` registers the ``proteus`` template and makes it the
Plotly default. Applications can still select another template explicitly
with ``fig.update_layout(template="plotly_white")``.
"""

from __future__ import annotations

import base64

import plotly.graph_objects as go
import plotly.io as pio

PROTEUS_NAVY = "#001a64"
PROTEUS_BLUE = "#1d4ed8"
PROTEUS_SKY = "#4aa3df"
PROTEUS_ORANGE = "#f59e0b"
PROTEUS_TEAL = "#0f766e"

PROTEUS_COLORWAY = [PROTEUS_NAVY, PROTEUS_BLUE, PROTEUS_SKY, PROTEUS_ORANGE, PROTEUS_TEAL]
_PROTEUS_MARK = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#001a64"/><path d="M17 47V17h16.5c8.8 0 14.5 5.2 14.5 13s-5.7 13-14.5 13H25v4h-8Zm8-12h8c4.5 0 7-1.8 7-5s-2.5-5-7-5h-8v10Z" fill="#fff"/><path d="M48 47h-8l8-8v8Z" fill="#5b8def"/></svg>"""
_PROTEUS_MARK_URI = "data:image/svg+xml;base64," + base64.b64encode(_PROTEUS_MARK.encode()).decode()

proteus_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=PROTEUS_COLORWAY,
        font={"family": "Arial, sans-serif", "color": "#0f172a"},
        title={"font": {"size": 22, "color": PROTEUS_NAVY}, "x": 0.02, "xanchor": "left"},
        legend={"bgcolor": "rgba(255,255,255,0.8)"},
        margin={"l": 70, "r": 40, "t": 70, "b": 90},
        xaxis={"showline": True, "linecolor": "#cbd5e1", "gridcolor": "#e5eaf2", "zeroline": False},
        yaxis={"showline": True, "linecolor": "#cbd5e1", "gridcolor": "#e5eaf2", "zeroline": False},
        hoverlabel={"bgcolor": "white", "font": {"color": "#0f172a"}},
        annotations=[
        ],
    )
)


def add_proteus_branding(fig: go.Figure) -> go.Figure:
    """Add a subtle, export-safe Proteus mark to a Plotly figure."""
    fig.add_layout_image(
        source=_PROTEUS_MARK_URI,
        xref="paper",
        yref="paper",
        x=0.86,
        y=-0.135,
        sizex=0.035,
        sizey=0.07,
        xanchor="left",
        yanchor="middle",
        sizing="contain",
        layer="above",
    )
    fig.add_annotation(
        text="PROTEUS",
        xref="paper",
        yref="paper",
        x=0.995,
        y=-0.135,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font={"size": 11, "color": PROTEUS_NAVY},
        opacity=0.8,
    )
    return fig

pio.templates["proteus"] = proteus_template
pio.templates.default = "proteus"

__all__ = [
    "PROTEUS_BLUE",
    "PROTEUS_COLORWAY",
    "PROTEUS_NAVY",
    "PROTEUS_ORANGE",
    "PROTEUS_SKY",
    "PROTEUS_TEAL",
    "add_proteus_branding",
    "proteus_template",
]

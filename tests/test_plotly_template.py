"""Tests for PAL's default Plotly styling."""

import plotly.graph_objects as go
import plotly.io as pio

from pal import PROTEUS_COLORWAY, PROTEUS_NAVY


def test_proteus_is_the_default_plotly_template():
    assert pio.templates.default == "proteus"
    figure = go.Figure()
    assert figure.layout.template.layout.title.font.color == PROTEUS_NAVY
    assert figure.layout.template.layout.annotations[0].text == "PROTEUS"


def test_proteus_template_exposes_brand_palette():
    assert list(pio.templates["proteus"].layout.colorway) == PROTEUS_COLORWAY

"""Tests for PAL's default Plotly styling."""

import plotly.graph_objects as go
import plotly.io as pio

from pal import PROTEUS_COLORWAY, PROTEUS_NAVY, add_proteus_branding


def test_proteus_is_the_default_plotly_template():
    assert pio.templates.default == "proteus"
    figure = go.Figure()
    assert figure.layout.template.layout.title.font.color == PROTEUS_NAVY
    add_proteus_branding(figure)
    assert figure.layout.annotations[0].text == "PROTEUS"
    assert figure.layout.images[0].source.startswith("data:image/svg+xml;base64,")


def test_proteus_template_exposes_brand_palette():
    assert list(pio.templates["proteus"].layout.colorway) == PROTEUS_COLORWAY

"""Tests for paramham.plotting."""

import matplotlib as mpl

from paramham.plotting import COL_W, COLORS, FULL_W, H_COL, _savefig, set_pub_style


def test_colors_keys():
    assert "GT" in COLORS
    assert "ID" in COLORS
    assert "FD" in COLORS
    assert "ENV" in COLORS


def test_size_constants():
    assert COL_W > 0
    assert FULL_W > COL_W
    assert H_COL > 0


def test_set_pub_style():
    set_pub_style()
    assert mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False


def test_set_pub_style_grid():
    set_pub_style(grid=True)
    assert mpl.rcParams["axes.grid"] is True
    set_pub_style(grid=False)
    assert mpl.rcParams["axes.grid"] is False


def test_set_pub_style_base_size():
    set_pub_style(base_size=10)
    assert mpl.rcParams["font.size"] == 10
    assert mpl.rcParams["axes.labelsize"] == 11


def test_savefig(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "sub" / "test.png"
    _savefig(fig, path)
    assert path.exists()
    assert path.stat().st_size > 0
    plt.close(fig)

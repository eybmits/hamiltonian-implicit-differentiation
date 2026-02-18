"""Tests for paramham.io."""

from paramham.io import parse_float_list, parse_int_list, parse_str_list, write_csv, write_tex_table


def test_parse_int_list():
    assert parse_int_list("1,2,3") == [1, 2, 3]


def test_parse_int_list_empty():
    assert parse_int_list("") == []
    assert parse_int_list(None) == []


def test_parse_float_list():
    result = parse_float_list("1.5,2.0,3.7")
    assert result == [1.5, 2.0, 3.7]


def test_parse_float_list_empty():
    assert parse_float_list("") == []


def test_parse_str_list():
    assert parse_str_list("a, b, c") == ["a", "b", "c"]


def test_write_csv(tmp_path):
    rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    path = tmp_path / "out.csv"
    write_csv(path, rows)
    content = path.read_text()
    assert "x,y" in content
    assert "1,2" in content
    assert "3,4" in content


def test_write_csv_empty(tmp_path):
    write_csv(tmp_path / "empty.csv", [])
    assert not (tmp_path / "empty.csv").exists()


def test_write_csv_custom_fieldnames(tmp_path):
    rows = [{"x": 1, "y": 2}]
    path = tmp_path / "out.csv"
    write_csv(path, rows, fieldnames=["y", "x"])
    content = path.read_text()
    assert content.startswith("y,x")


def test_write_tex_table(tmp_path):
    rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    path = tmp_path / "table.tex"
    write_tex_table(path, rows, columns=["a", "b"], header=["A", "B"])
    content = path.read_text()
    assert r"\begin{tabular}" in content
    assert "A & B" in content
    assert r"\end{tabular}" in content

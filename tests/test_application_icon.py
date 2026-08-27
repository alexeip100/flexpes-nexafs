from pathlib import Path


def test_icon_is_bundled_as_package_data():
    root = Path(__file__).resolve().parents[1]
    icon = root / "src" / "flexpes_nexafs" / "assets" / "flexpes_xas_icon.ico"
    assert icon.is_file()
    assert icon.stat().st_size > 0


def test_pyproject_includes_ico_package_data():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"assets/*.ico"' in text

from __future__ import annotations

"""Application icon resource helpers."""

from importlib import resources

from PyQt5.QtGui import QIcon  # type: ignore


_ICON_PACKAGE = "flexpes_nexafs.assets"
_ICON_NAME = "flexpes_xas_icon.ico"


def application_icon() -> QIcon:
    """Return the bundled FlexPES XAS application icon.

    The icon is loaded from package data so installed applications do not depend
    on the source tree or the user's original icon-file location.
    """
    try:
        icon_path = resources.files(_ICON_PACKAGE).joinpath(_ICON_NAME)
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon
    except Exception:
        pass
    return QIcon()

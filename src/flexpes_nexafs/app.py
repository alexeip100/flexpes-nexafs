"""Application entry point.

Important: the rest of this package currently uses **PyQt5** (including the
Matplotlib Qt backend import in ``ui.py``). If both PyQt6 and PyQt5 are
installed, importing PyQt6 here would create a *PyQt6* QApplication, while the
UI constructs *PyQt5* widgets. That mismatch triggers the classic runtime
crash:

    QWidget: Must construct a QApplication before a QWidget

So we deliberately prefer PyQt5 here.
"""

from __future__ import annotations

try:
    from PyQt5 import QtWidgets  # type: ignore
    from PyQt5 import QtCore  # type: ignore
except Exception as exc:  # pragma: no cover
# The UI code in this package imports PyQt5 directly, so PyQt5 is required.
    raise ImportError(
        "PyQt5 is required to run flexpes_nexafs (UI modules import PyQt5)."
    ) from exc


def main():
    import sys
    import os

    # Keep the GUI at the same baseline scale on HiDPI / 4K screens as on
    # the reference 1920x1080 setup. These settings must be applied before the
    # QApplication is created.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")

    try:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
    except Exception:
        pass

    try:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_Use96Dpi, True)
    except Exception:
        pass

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# Import the main UI only after a QApplication exists.
    from .ui import MainWindow

# Force a consistent cross-platform Qt style. / (Fusion exists in both Qt5 and Qt6.)
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

# Use the style's standard palette as the default (light) palette.
    try:
        app.setPalette(app.style().standardPalette())
    except Exception:
        pass

# Make fonts a bit larger for readability across platforms
    try:
        f = app.font()
        ps = int(f.pointSize())
        if ps > 0:
            f.setPointSize(ps + 2)
        else:
# Fallback for pixel-sized fonts
            px = int(f.pixelSize())
            if px > 0:
                f.setPixelSize(px + 2)
        app.setFont(f)
    except Exception:
        pass


    win = MainWindow()
    win.show()

# Preload the decomposition UI and its heavy dependencies shortly after startup / so the first click on the "PCA" button feels responsive. This runs...
    def _preload_decomposition() -> None:
        try:
# Heavy imports (NumPy wheels already present, but sklearn/pandas can take time)
            import pandas  # noqa: F401
            import sklearn  # noqa: F401

# Import the decomposition UI module to warm up its import graph
            from .decomposition import legacy  # noqa: F401
        except Exception:
# Never fail startup due to preload
            pass

    try:
        QtCore.QTimer.singleShot(200, _preload_decomposition)
    except Exception:
        pass

# Qt5 uses exec_(), Qt6 uses exec()
    try:
        sys.exit(app.exec())
    except Exception:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()

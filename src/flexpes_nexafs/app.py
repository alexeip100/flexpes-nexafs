try:
    from PyQt6 import QtWidgets  # type: ignore
except Exception:
    from PyQt5 import QtWidgets  # type: ignore

from .ui import MainWindow


def main():
    import sys

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Force a consistent cross-platform Qt style.
    # (Fusion exists in both Qt5 and Qt6.)
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

    # Qt5 uses exec_(), Qt6 uses exec()
    try:
        sys.exit(app.exec())
    except Exception:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
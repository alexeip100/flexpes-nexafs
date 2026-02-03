"""Help browser widget used for the Help->Usage dialog.

Moved out of flexpes_nexafs.plotting to keep plotting logic focused.
"""

from PyQt5.QtWidgets import QTextBrowser, QTextEdit, QMenu
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextOption


class HelpBrowser(QTextBrowser):
    """QTextBrowser subclass that keeps text wrapping responsive on resize.

    This variant uses a FixedPixelWidth wrap mode and updates the wrap
    width on each resize event. This tends to behave consistently across
    different Qt / PyQt builds and platforms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
# Wrap at a fixed pixel width that we'll update on resize
        self.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.setWordWrapMode(QTextOption.WordWrap)
# We want wrapping instead of horizontal scrolling
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

# Quick navigation (right-click) to headings in the help text.
        self._help_anchors = []  # list[(level:int, title:str, anchor_id:str)]
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_help_context_menu)

    def setHtml(self, html: str) -> None:  # type: ignore[override]
        """Set HTML and extract heading anchors for fast navigation."""
        super().setHtml(html)
        try:
            import re, html as _html

            def _strip_tags(s: str) -> str:
                s = re.sub(r"<[^>]+>", "", s)
                return _html.unescape(s).strip()

            anchors = []
            # Collect H1/H2/H3 headings with ids.
            for mh in re.finditer(
                r"<h([123])[^>]*id=\"([^\"]+)\"[^>]*>(.*?)</h\1>",
                html,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                level = int(mh.group(1))
                title = _strip_tags(mh.group(3))
                anchor = mh.group(2)
                if title and anchor:
                    anchors.append((level, title, anchor))
            self._help_anchors = anchors
        except Exception:
            self._help_anchors = []

    def _show_help_context_menu(self, pos):
        """Show a context menu with a "Go to" section index."""
        try:
            from functools import partial
            menu = self.createStandardContextMenu()
            if getattr(self, "_help_anchors", None):
                nav = QMenu("Go to", menu)
                for level, title, anchor in self._help_anchors:
                    disp = ("    " + title) if level == 2 else title
                    nav.addAction(disp, partial(self.scrollToAnchor, anchor))
                # Put navigation on top.
                if menu.actions():
                    first = menu.actions()[0]
                    menu.insertMenu(first, nav)
                    menu.insertSeparator(first)
                else:
                    menu.addMenu(nav)
            menu.exec_(self.mapToGlobal(pos))
        except Exception:
            # Fallback: show the default context menu.
            try:
                self.createStandardContextMenu().exec_(self.mapToGlobal(pos))
            except Exception:
                pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the wrap column in sync with the viewport width
        try:
            self.setLineWrapColumnOrWidth(self.viewport().width())
        except Exception:
            # If anything goes wrong, we just fall back to default behavior.
            pass

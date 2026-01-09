from flexpes_nexafs.utils.sorting import parse_entry_number
from .data import lookup_energy
# Auto-generated/maintained PlottingMixin (post-split)

import h5py
import numpy as np
from .compat import trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.ioff()
try:
    from .widgets.curve_item import CurveListItemWidget
except Exception:
    CurveListItemWidget = None
try:
    from .widgets.curve_item import CurveTreeWidgetItem
except Exception:
    CurveTreeWidgetItem = None
    
from datetime import datetime    

from PyQt5.QtWidgets import (
    QTextEdit,
    QApplication, QFileDialog, QTreeWidget, QTreeWidgetItem, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTabWidget, QCheckBox, QComboBox,
    QSpinBox, QMessageBox, QSizePolicy, QDialog, QListWidgetItem, QInputDialog, QTextBrowser, QMenu, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor, QTextOption

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
            # Collect H1/H2 headings with ids.
            for mh in re.finditer(
                r"<h([12])[^>]*id=\"([^\"]+)\"[^>]*>(.*?)</h\1>",
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


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Use the processing helpers from the split module
from . import processing


# --- Help / Usage loader (Markdown -> HTML) ----------------------------------
def _pkg_root_path():
    try:
        from pathlib import Path
        return Path(__file__).resolve().parent
    except Exception:
        return None

def _read_help_markdown():
    root = _pkg_root_path()
    if not root:
        return None
    md_path = root / "docs" / "help.md"
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception:
        return None

def _basic_md_to_html(md: str) -> str:
    """Very small Markdown->HTML fallback.

    This version is careful not to treat every single line break in the
    source file as a separate paragraph. Instead, consecutive non-empty
    lines that are not list items or headings are merged into one
    paragraph, so that soft-wrapped text from the editor behaves like a
    normal flowing paragraph in the help window.

    It supports:
      - #, ##, ### headings
      - unordered lists starting with "- " or "* "
      - ordered lists starting with "1. ", "2. ", ... "9. "
      - **bold**, *italic*, and `inline code`
    """

    import html, re

    # Escape HTML first, then re-introduce simple formatting

    text = html.escape(md)

    # inline code

    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # bold / italic

    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

    # Now process line-by-line with simple block structure

    import re as _re


    def _slugify(s: str) -> str:

        s = (s or "").strip().lower()

        s = _re.sub(r"[\s_]+", "-", s)

        s = _re.sub(r"[^a-z0-9\-]+", "", s)

        s = _re.sub(r"-{2,}", "-", s).strip("-")

        return s or "section"


    _used_ids = {}

    lines_out = []

    in_ul = False

    in_ol = False

    para_buf = []  # accumulate plain paragraph lines

    def flush_para():

        nonlocal para_buf

        if para_buf:

            # Join soft-wrapped lines with spaces into one logical paragraph

            lines_out.append(f"<p>{' '.join(para_buf).strip()}</p>")

            para_buf = []

    for raw in text.splitlines():

        line = raw.strip()

        if not line:

            # Blank line: end any running paragraph, leave lists open

            flush_para()

            continue

        # Headings

        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):

            flush_para()

            if in_ul:

                lines_out.append("</ul>"); in_ul = False

            if in_ol:

                lines_out.append("</ol>"); in_ol = False

            if line.startswith("# "):

                _title = line[2:].strip()
                _id = _slugify(_title)
                _used_ids[_id] = _used_ids.get(_id, 0) + 1
                if _used_ids[_id] > 1:
                    _id = f"{_id}-{_used_ids[_id]}"
                lines_out.append(f'<h1 id="{_id}">{_title}</h1>')

            elif line.startswith("## "):

                _title = line[3:].strip()
                _id = _slugify(_title)
                _used_ids[_id] = _used_ids.get(_id, 0) + 1
                if _used_ids[_id] > 1:
                    _id = f"{_id}-{_used_ids[_id]}"
                lines_out.append(f'<h2 id="{_id}">{_title}</h2>')

            else:

                lines_out.append(f"<h3>{line[4:].strip()}</h3>")

            continue

        # Unordered list item

        if line.startswith("- ") or line.startswith("* "):

            flush_para()

            if in_ol:

                lines_out.append("</ol>"); in_ol = False

            if not in_ul:

                lines_out.append("<ul>"); in_ul = True

            item = line[2:].strip()

            lines_out.append(f"<li>{item}</li>")

            continue

        # Ordered list item (1. 2. ... 9.)

        if any(line.startswith(f"{n}. ") for n in range(1, 10)):

            flush_para()

            if in_ul:

                lines_out.append("</ul>"); in_ul = False

            if not in_ol:

                lines_out.append("<ol>"); in_ol = True

            dot = line.find('.')

            item = line[dot+1:].strip()

            lines_out.append(f"<li>{item}</li>")

            continue

        # Otherwise part of a normal paragraph: accumulate

        para_buf.append(line)

    # Flush trailing paragraph and lists

    flush_para()

    if in_ul:

        lines_out.append("</ul>")

    if in_ol:

        lines_out.append("</ol>")

    return "\n".join(lines_out)
def get_usage_html() -> str:
    """Return the Help->Usage content as HTML from docs/help.md."""
    md = _read_help_markdown()
    if not md:
        return "<p><b>Help file not found.</b></p>"
    try:
        import markdown
        # "toc" adds stable id attributes to headings (used for in-text links and
        # the HelpBrowser right-click "Go to" menu).
        return markdown.markdown(md, extensions=["tables","fenced_code","sane_lists","toc"])  # type: ignore
    except Exception:
        return _basic_md_to_html(md)




class PlottingMixin:
    def shorten_label(self, hdf5_path: str) -> str:
        """Return a compact label like 'TEY in entry0001' from an HDF5 dataset path.
        Heuristics follow the old app_window.py behavior."""
        try:
            path = (hdf5_path or "").strip("/")
            tokens = path.split("/") if path else []
            last = tokens[-1] if tokens else path
            label = last
            low = (last or "").lower()
            if "ch1" in low:
                label = "TEY"
            elif "ch3" in low:
                label = "PEY"
            elif "roi2_dtc" in low or "roi2" in low:
                label = "TFY"
            elif "roi1_dtc" in low or "roi1" in low:
                label = "PFY"
            entry = ""
            for t in tokens:
                if t.startswith("entry"):
                    entry = t
                    break
            return f"{label} in {entry}" if entry else label
        except Exception:
            return str(hdf5_path)

    # ------------ Waterfall ------------
    def _filter_empty_plot_data(self):
        """Drop entries with zero-length arrays to avoid reduction errors."""
        try:
            self.plot_data = {k: v for k, v in self.plot_data.items() if getattr(v, "size", 0) > 0}
        except Exception:
            pass

    def on_waterfall_toggled(self, state):
        checked = (state == Qt.Checked)
        self.waterfall_slider.setEnabled(checked)
        self.waterfall_spin.setEnabled(checked)
        self.recompute_waterfall_layout()

    def on_waterfall_slider_changed(self, value):
        param = value / 100.0
        self.waterfall_spin.blockSignals(True)
        self.waterfall_spin.setValue(param)
        self.waterfall_spin.blockSignals(False)
        self.recompute_waterfall_layout()

    def on_waterfall_spin_changed(self, dvalue):
        ival = int(round(dvalue * 100))
        self.waterfall_slider.blockSignals(True)
        self.waterfall_slider.setValue(ival)
        self.waterfall_slider.blockSignals(False)
        self.recompute_waterfall_layout()

    def apply_waterfall_shift(self):
        """Apply/update Waterfall offsets (Uniform step only)."""
        if not getattr(self, "plotted_lines", None):
            return

        enabled = False
        try:
            enabled = bool(getattr(self, "waterfall_checkbox", None) and self.waterfall_checkbox.isChecked())
        except Exception:
            enabled = False

        # Restore originals first so Waterfall is always applied on clean data
        try:
            self.restore_original_line_data()
        except Exception:
            pass

        # Build ordered visible keys (respect the plotted list order if present)
        plotted_keys_in_order = []
        if hasattr(self, "plotted_list"):
            try:
                for i in range(self.plotted_list.count()):
                    item = self.plotted_list.item(i)
                    widget = self.plotted_list.itemWidget(item)
                    if widget:
                        key = getattr(widget, "key", None)
                        line = self.plotted_lines.get(key) if key else None
                        if line is not None and line.get_visible():
                            plotted_keys_in_order.append(key)
            except Exception:
                plotted_keys_in_order = []
        if not plotted_keys_in_order:
            try:
                for k, line in self.plotted_lines.items():
                    if line is not None and line.get_visible():
                        plotted_keys_in_order.append(k)
            except Exception:
                plotted_keys_in_order = []

        if (not plotted_keys_in_order) or (not enabled):
            # No Waterfall: just rescale and redraw
            try:
                self.rescale_plotted_axes()
                if hasattr(self, "canvas_plotted"):
                    self.canvas_plotted.draw()
            except Exception:
                pass
            return

        # Uniform step parameter (0..1) multiplied by global y-range
        try:
            k = float(self.waterfall_spin.value())
        except Exception:
            k = 0.0

        # Compute global y-range from restored originals
        try:
            ymins, ymaxs = [], []
            for key in plotted_keys_in_order:
                line = self.plotted_lines[key]
                y = np.asarray(line.get_ydata(), dtype=float)
                m = np.isfinite(y)
                if np.any(m):
                    ymins.append(float(np.min(y[m])))
                    ymaxs.append(float(np.max(y[m])))
            if not ymins or not ymaxs:
                self.rescale_plotted_axes()
                return
            global_range = max(ymaxs) - min(ymins)
            if not np.isfinite(global_range) or global_range <= 0:
                global_range = 1.0
            step = k * global_range
        except Exception:
            step = 0.0

        # Apply offsets in order
        try:
            for idx, key in enumerate(plotted_keys_in_order):
                line = self.plotted_lines[key]
                y = np.asarray(line.get_ydata(), dtype=float)
                line.set_ydata(y + idx * step)
        except Exception:
            pass

        try:
            self.rescale_plotted_axes()
            if hasattr(self, "canvas_plotted"):
                self.canvas_plotted.draw()
        except Exception:
            pass
    def restore_original_line_data(self):
        """Restore each line to the unshifted data we stored earlier."""
        for key, line in getattr(self, "plotted_lines", {}).items():
            if key in self.original_line_data:
                x_orig, y_orig = self.original_line_data[key]
                line.set_xdata(x_orig)
                line.set_ydata(y_orig)
        if hasattr(self, "canvas_plotted"):
            self.canvas_plotted.draw()

    def get_line_color_for_key(self, key, ax):
        for line in ax.get_lines():
            if getattr(line, "dataset_key", None) == key:
                return line.get_color()
        return None

    # ------------ Plotted tab helpers ------------
    def pass_to_plotted_no_clear(self):
        if not getattr(self, "plot_data", None):
            return

        visible_keys = [key for key in self.plot_data if self.raw_visibility.get(key, False)]
        if self.chk_sum.isChecked():
            if not visible_keys:
                return
            key = visible_keys[0]
        else:
            if len(visible_keys) != 1:
                # Multi-pass is allowed only for group Auto BG with subtraction enabled.
                allow_multi = False
                try:
                    mode = str(getattr(self, "combo_bg").currentText()) if hasattr(self, "combo_bg") else ""
                    cb = getattr(self, "chk_group_bg", None)
                    if (len(visible_keys) >= 2 and str(mode) in ("Automatic", "Auto") and cb is not None and cb.isEnabled() and cb.isChecked()
                        and getattr(self, "chk_show_without_bg", None) is not None and self.chk_show_without_bg.isChecked()
                        and (not self.chk_sum.isChecked())):
                        allow_multi = True
                except Exception:
                    allow_multi = False

                if not allow_multi:
                    QMessageBox.warning(self, "Warning", "Please select exactly one dataset, or enable 'Sum'.")
                    return

                # --- Multi-pass: pass all selected curves at once ---
                try:
                    deg = int(self.combo_poly.currentText())
                except Exception:
                    deg = 2
                try:
                    pre = float(self.spin_preedge.value()) / 100.0
                except Exception:
                    pre = 0.12

                # Post-normalization mode
                norm_mode = "None"
                try:
                    if hasattr(self, "combo_post_norm"):
                        # In Group BG mode the combobox is disabled but still reflects the enforced mode ("Area").
                        if (getattr(self, "chk_group_bg", None) is not None and self.chk_group_bg.isChecked()):
                            norm_mode = str(self.combo_post_norm.currentText())
                        elif self.combo_post_norm.isEnabled():
                            norm_mode = str(self.combo_post_norm.currentText())
                except Exception:
                    norm_mode = "None"

                # Group backgrounds (fallback handled inside)
                _x_common, bgs = self._compute_group_auto_backgrounds(visible_keys, deg=deg, pre_edge_percent=pre)
                if bgs is None:
                    bgs = {}

                # If Area post-normalization is active, equalize the *area-normalized jump* across the group
                # by adding a zero-area tilt term to the per-spectrum background.
                try:
                    m = str(norm_mode).strip().lower()
                    _equalize_jump = m in ("area", "area=1")
                except Exception:
                    _equalize_jump = False

                if _equalize_jump and len(visible_keys) >= 2:
                    # Coupled group constraint for Area post-normalization:
                    # - pre-edge baseline after BG subtraction is set to 0 for each spectrum
                    # - jump after BG subtraction and Area-normalization is equalized across the group
                    # - optionally, pre-edge slope after BG subtraction is also matched across the group
                    try:
                        items_simple = {}
                        for key in visible_keys:
                            parts = key.split("##", 1)
                            if len(parts) != 2:
                                continue
                            abs_path, hdf5_path = parts
                            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
                            y_data = self.plot_data.get(key)
                            if y_data is None:
                                continue
                            x_data = lookup_energy(self, abs_path, parent, len(y_data))
                            y_proc = processing.apply_normalization(self, abs_path, parent, y_data)
                            mlen = min(len(x_data), len(y_proc))
                            if mlen < 3:
                                continue
                            x_use = np.asarray(x_data[:mlen], dtype=float).ravel()
                            y_use = np.asarray(y_proc[:mlen], dtype=float).ravel()
                            bg = bgs.get(key)
                            if bg is None:
                                bg = self._apply_automatic_bg_new(x_use, y_use, deg=deg, pre_edge_percent=pre, do_plot=False)
                            bg = np.asarray(bg, dtype=float).ravel()[:mlen]
                            items_simple[key] = (x_use, y_use, bg)

                        _match_slope = False
                        try:
                            scb = getattr(self, "chk_group_bg_slope", None)
                            if scb is not None and scb.isEnabled() and scb.isChecked():
                                _match_slope = True
                        except Exception:
                            _match_slope = False

                        adjusted, msg = self._group_equalize_area_jump_and_zero_preedge(
                            items_simple, pre=float(pre), match_preedge_slope=bool(_match_slope)
                        )
                        for k, (_x_u, _y_u, bg_u) in adjusted.items():
                            bgs[k] = bg_u
                        if msg:
                            try:
                                self.statusBar().showMessage(msg, 6000)
                            except Exception:
                                pass
                    except Exception:
                        pass

                added = 0
                skipped = 0

                for key in visible_keys:
                    storage_key = key
                    if storage_key in self.plotted_curves:
                        skipped += 1
                        continue

                    # Build x/y (normalized, bg-subtracted, post-normalized)
                    parts = key.split("##", 1)
                    if len(parts) != 2:
                        skipped += 1
                        continue
                    abs_path, hdf5_path = parts
                    parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
                    y_data = self.plot_data.get(key)
                    if y_data is None:
                        skipped += 1
                        continue
                    x_data = lookup_energy(self, abs_path, parent, len(y_data))
                    y_proc = processing.apply_normalization(self, abs_path, parent, y_data)
                    mlen = min(len(x_data), len(y_proc))
                    if mlen <= 1:
                        skipped += 1
                        continue
                    main_x = np.asarray(x_data[:mlen], dtype=float).ravel()
                    main_y = np.asarray(y_proc[:mlen], dtype=float).ravel()

                    bg = bgs.get(key)
                    if bg is None:
                        bg = self._apply_automatic_bg_new(main_x, main_y, deg=deg, pre_edge_percent=pre, do_plot=False)
                    bg = np.asarray(bg, dtype=float).ravel()[:mlen]
                    subtracted = main_y - bg
                    try:
                        from . import processing as _p
                        subtracted = _p._proc_safe_post_normalize(self, main_x, subtracted, norm_mode)
                    except Exception:
                        pass
                    main_y = subtracted

                    # Build label + metadata
                    source_file = abs_path
                    source_entry = hdf5_path
                    detector_name = ""
                    path_parts = hdf5_path.split("/")
                    if len(path_parts) >= 3 and path_parts[1] == "measurement":
                        entry = path_parts[0]
                        channel_name = path_parts[-1]
                        origin_label = f"{entry}: {channel_name}"
                        detector_name = channel_name
                    else:
                        origin_label = hdf5_path
                        detector_name = path_parts[-1] if path_parts else hdf5_path

                    metadata_parts = []
                    try:
                        nm = str(norm_mode).lower()
                        if nm and nm != "none":
                            metadata_parts.append(nm)
                    except Exception:
                        pass
                    if self.chk_normalize.isChecked():
                        metadata_parts.append("normalized")
                    metadata_parts.append("bg subtracted")
                    if metadata_parts:
                        origin_label += " (" + ", ".join(metadata_parts) + ")"

                    if not hasattr(self, "plotted_metadata") or not isinstance(getattr(self, "plotted_metadata", None), dict):
                        self.plotted_metadata = {}
                    self.plotted_metadata[storage_key] = {
                        "detector": detector_name or "",
                        "source_file": source_file or "",
                        "source_entry": source_entry or "",
                        "post_normalization": norm_mode,
                        "is_reference": False,
                    }

                    # Add to plot + list
                    self.custom_labels[storage_key] = None
                    line, = self.plotted_ax.plot(main_x, main_y, label="<select curve name>")
                    self.plotted_curves.add(storage_key)
                    self.plotted_lines[storage_key] = line
                    self.original_line_data[storage_key] = (np.asarray(main_x).copy(), np.asarray(main_y).copy())

                    item = QListWidgetItem()
                    try:
                        item.setData(Qt.UserRole, storage_key)
                    except Exception:
                        pass
                    CurveListItemWidget = globals().get("CurveListItemWidget", None)
                    if CurveListItemWidget:
                        widget = CurveListItemWidget(origin_label, line.get_color(), storage_key)
                        widget.colorChanged.connect(self.change_curve_color)
                        widget.visibilityChanged.connect(self.change_curve_visibility)
                        widget.styleChanged.connect(self.change_curve_style)
                        if hasattr(widget, "removeRequested"):
                            widget.removeRequested.connect(self.on_curve_remove_requested)
                        if hasattr(widget, "addToLibraryRequested"):
                            try:
                                self.on_add_to_library_requested
                            except AttributeError:
                                pass
                            else:
                                widget.addToLibraryRequested.connect(self.on_add_to_library_requested)
                        item.setSizeHint(widget.sizeHint())
                        self.plotted_list.addItem(item)
                        self.plotted_list.setItemWidget(item, widget)
                    else:
                        item.setText(origin_label)
                        self.plotted_list.addItem(item)

                    added += 1

                if added:
                    try:
                        self.data_tabs.setCurrentIndex(2)
                    except Exception:
                        pass
                    self.update_legend()

                if skipped and added == 0:
                    QMessageBox.warning(self, "Warning", "No new curves were passed (they may already be present in Plotted Data).")
                return

            key = visible_keys[0]

        # Use synthetic key for summed curve so it doesn't collide with first component
        storage_key = f"SUMMED#{getattr(self, '_sum_serial', 0) + 1}" if self.chk_sum.isChecked() else key

        if storage_key in self.plotted_curves:
            QMessageBox.warning(self, "Warning", "This curve is already passed for plotting.")
            return

        main_x, main_y = self.compute_main_curve()
        if main_x is None or main_y is None:
            return

        # Post-normalization mode (used for metadata and potential library export)
        norm_mode = "None"
        try:
            if hasattr(self, "combo_post_norm") and self.combo_post_norm.isEnabled():
                norm_mode = str(self.combo_post_norm.currentText())
        except Exception:
            norm_mode = "None"

        bg_subtracted = False
        if self.chk_show_without_bg.isChecked():
            background = self._compute_background(main_x, main_y)
            subtracted = main_y - background
            try:
                # IMPORTANT:
                # Do NOT bind a local name called `processing` in this method.
                # This method also uses the module-level `processing` import earlier
                # (e.g. for multi-pass). A local import like `from . import processing`
                # would make `processing` a local variable for the entire function and
                # can trigger UnboundLocalError on code paths that reference it before
                # the local import executes.
                from . import processing as _p
                subtracted = _p._proc_safe_post_normalize(self, main_x, subtracted, norm_mode)
            except Exception:
                pass
            main_y = subtracted
            bg_subtracted = True

        # Parse the key into source file / entry information
        source_file = ""
        source_entry = ""
        detector_name = ""
        parts = key.split("##", 1)
        if self.chk_sum.isChecked():
            origin_label = "Summed Curve"
        else:
            if len(parts) == 2:
                source_file = parts[0]
                path = parts[1]
                source_entry = path
                # Typical path looks like "entryXXXX/measurement/<channel_name>".
                # We want to drop the redundant "measurement" and show
                # "entryXXXX: <channel_name>" in the Plotted Data list.
                path_parts = path.split("/")
                if len(path_parts) >= 3 and path_parts[1] == "measurement":
                    entry = path_parts[0]
                    channel_name = path_parts[-1]
                    origin_label = f"{entry}: {channel_name}"
                    detector_name = channel_name
                else:
                    origin_label = path
                    detector_name = path_parts[-1] if path_parts else path
            else:
                origin_label = key

        metadata_parts = []
        if bg_subtracted and hasattr(self, "combo_post_norm") and self.combo_post_norm.isEnabled():
            try:
                nm = self.combo_post_norm.currentText().lower()
                if nm != "none":
                    metadata_parts.append(nm)
            except Exception:
                pass
        if self.chk_normalize.isChecked():
            metadata_parts.append("normalized")
        if self.chk_sum.isChecked():
            metadata_parts.append("summed")
        if bg_subtracted:
            metadata_parts.append("bg subtracted")
        if metadata_parts:
            origin_label += " (" + ", ".join(metadata_parts) + ")"

        # Metadata bookkeeping for potential library export
        if not hasattr(self, "plotted_metadata") or not isinstance(getattr(self, "plotted_metadata", None), dict):
            self.plotted_metadata = {}
        self.plotted_metadata[storage_key] = {
            "detector": detector_name or "",
            "source_file": source_file or "",
            "source_entry": source_entry or "",
            "post_normalization": norm_mode,
            "is_reference": False,
        }

        # Add the curve to the plotted axes and list
        self.custom_labels[storage_key] = None
        line, = self.plotted_ax.plot(main_x, main_y, label="<select curve name>")
        self.plotted_curves.add(storage_key)
        self.plotted_lines[storage_key] = line

        # Store original data so we can revert it for Waterfall
        self.original_line_data[storage_key] = (np.asarray(main_x).copy(), np.asarray(main_y).copy())

        item = QListWidgetItem()
        # Store curve key in the item for robust reordering
        try:
            item.setData(Qt.UserRole, storage_key)
        except Exception:
            pass
            pass

        # Avoid circular import with UI; gracefully degrade if custom widget is unavailable.
        CurveListItemWidget = globals().get("CurveListItemWidget", None)
        if CurveListItemWidget:
            widget = CurveListItemWidget(origin_label, line.get_color(), storage_key)
            widget.colorChanged.connect(self.change_curve_color)
            widget.visibilityChanged.connect(self.change_curve_visibility)
            widget.styleChanged.connect(self.change_curve_style)
            # Allow removing a plotted curve from the list/plot.
            if hasattr(widget, "removeRequested"):
                widget.removeRequested.connect(self.on_curve_remove_requested)
            # Allow adding the curve to the reference library, if LibraryMixin is present.
            if hasattr(widget, "addToLibraryRequested"):
                try:
                    self.on_add_to_library_requested  # type: ignore[attr-defined]
                except AttributeError:
                    pass
                else:
                    widget.addToLibraryRequested.connect(self.on_add_to_library_requested)
            item.setSizeHint(widget.sizeHint())
            self.plotted_list.addItem(item)
            self.plotted_list.setItemWidget(item, widget)
        else:
            item.setText(origin_label)
            self.plotted_list.addItem(item)

        self.data_tabs.setCurrentIndex(2)
        self.update_legend()
        if self.chk_sum.isChecked():
            self._sum_serial = getattr(self, "_sum_serial", 0) + 1
    def _add_reference_curve_to_plotted(self, storage_key, x, y, label, meta=None):
        """
        Internal helper to add a reference spectrum (from the library) directly to
        the Plotted Data plot and list.

        This bypasses the Raw/Processed panels and does not depend on plot_data.
        """
        try:
            import numpy as _np
        except Exception:
            import numpy as _np

        if not hasattr(self, "plotted_curves"):
            self.plotted_curves = set()
        if not hasattr(self, "plotted_lines"):
            self.plotted_lines = {}
        if not hasattr(self, "original_line_data"):
            self.original_line_data = {}
        if not hasattr(self, "custom_labels"):
            self.custom_labels = {}
        if not hasattr(self, "plotted_list") or self.plotted_list is None:
            return

        # Normalize arrays
        try:
            x_arr = _np.asarray(x).ravel()
            y_arr = _np.asarray(y).ravel()
        except Exception:
            return
        if x_arr.size == 0 or y_arr.size == 0:
            return
        n = min(int(x_arr.size), int(y_arr.size))
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]

        # Register curve
        self.custom_labels[storage_key] = str(label)
        line, = self.plotted_ax.plot(x_arr, y_arr, label=str(label))
        self.plotted_curves.add(storage_key)
        self.plotted_lines[storage_key] = line
        self.original_line_data[storage_key] = (x_arr.copy(), y_arr.copy())

        # Metadata bookkeeping (mark as reference)
        if not hasattr(self, "plotted_metadata") or not isinstance(getattr(self, "plotted_metadata", None), dict):
            self.plotted_metadata = {}
        m = dict(meta or {})
        m.setdefault("detector", str(m.get("detector", "") or ""))
        m.setdefault("source_file", str(m.get("source_file", "") or ""))
        m.setdefault("source_entry", str(m.get("source_entry", "") or ""))
        m.setdefault("post_normalization", str(m.get("post_normalization", "") or "None"))
        m["is_reference"] = bool(m.get("is_reference", True))
        m["is_imported"] = bool(m.get("is_imported", False))
        self.plotted_metadata[storage_key] = m

        # Create list item with curve controls
        item = QListWidgetItem()
        # Store curve key in the item for robust reordering
        try:
            item.setData(Qt.UserRole, storage_key)
        except Exception:
            pass

        CurveListItemWidget = globals().get("CurveListItemWidget", None)
        origin_label = str(label)
        if CurveListItemWidget:
            widget = CurveListItemWidget(origin_label, line.get_color(), storage_key)
            widget.colorChanged.connect(self.change_curve_color)
            widget.visibilityChanged.connect(self.change_curve_visibility)
            widget.styleChanged.connect(self.change_curve_style)
            if hasattr(widget, "removeRequested"):
                widget.removeRequested.connect(self.on_curve_remove_requested)
            # Reference curves should not be added to library again
            if hasattr(widget, "set_add_to_library_enabled"):
                widget.set_add_to_library_enabled(False)
            item.setSizeHint(widget.sizeHint())
            self.plotted_list.addItem(item)
            self.plotted_list.setItemWidget(item, widget)
        else:
            item.setText(origin_label)
            self.plotted_list.addItem(item)

        # Ensure Plotted Data tab is visible and legend updated
        try:
            self.data_tabs.setCurrentIndex(2)
        except Exception:
            pass
        try:
            self.update_legend()
        except Exception:
            pass

    def on_grid_toggled(self, index):
        """Handle change of grid density from combo box."""
        try:
            self._apply_grid_mode()
        except Exception:
            pass


    def _apply_grid_mode(self, mode_text=None):
        """Apply grid mode ('None', 'Coarse', 'Fine', 'Finest') to plotted_ax.

        'Coarse' reproduces the original checkbox behavior (major-grid only).
        'Fine' uses 1 minor division (2× finer), 'Finest' uses 5 divisions
        (~5× finer). Minor grid lines are shown less prominently than major
        grid lines."""
        ax = getattr(self, 'plotted_ax', None)
        if ax is None:
            return

        # Determine mode from combo if not provided
        if mode_text is None:
            try:
                combo = getattr(self, 'grid_mode_combo', None)
                if combo is not None:
                    mode_text = combo.currentText()
            except Exception:
                mode_text = None
        if not mode_text:
            mode_text = 'None'
        mode = str(mode_text).strip().lower()

        # Start from a clean grid state
        try:
            ax.grid(False, which='both')
        except Exception:
            pass

        if mode == 'none':
            # No grid at all, turn off minor ticks too
            try:
                ax.minorticks_off()
            except Exception:
                pass
        else:
            # Always show major grid lines for any non-'None' mode, as solid lines
            try:
                ax.grid(True, which='major', linestyle='-')
            except Exception:
                pass

            # 'Coarse' -> only major grid (no extra minor divisions)
            if mode == 'coarse':
                try:
                    ax.minorticks_off()
                except Exception:
                    pass
            else:
                # 'Fine' and 'Finest' -> add minor grid with more divisions,
                # drawn as softer dashed lines between the major grid lines.
                n = 2 if mode == 'fine' else 5
                try:
                    ax.minorticks_on()
                    ax.xaxis.set_minor_locator(AutoMinorLocator(n))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(n))
                    # Minor grid only; keep major grid style unchanged
                    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.6)
                except Exception:
                    pass

        # Redraw canvas if available
        try:
            if hasattr(self, 'canvas_plotted'):
                self.canvas_plotted.draw()
        except Exception:
            pass
    def toggle_plotted_legend(self, visible: bool):
        """Show or hide the legend on the Plotted Data axes."""
        try:
            self.plotted_legend_visible = bool(visible)
        except Exception:
            self.plotted_legend_visible = bool(visible)

        ax = getattr(self, "plotted_ax", None)
        leg = None
        if ax is not None:
            try:
                leg = ax.get_legend()
            except Exception:
                leg = None

        if leg is not None:
            # Just toggle visibility of the existing legend; position is preserved
            try:
                leg.set_visible(self.plotted_legend_visible)
            except Exception:
                pass
        else:
            # No legend yet: create one if we are turning it on
            if self.plotted_legend_visible and hasattr(self, "update_legend"):
                try:
                    self.update_legend()
                except Exception:
                    pass

        if hasattr(self, "canvas_plotted"):
            try:
                self.canvas_plotted.draw()
            except Exception:
                pass


    def _get_plotted_legend_mode(self) -> str:
        """Return the current legend mode for Plotted Data."""
        # Prefer the UI combo box if present
        try:
            cb = getattr(self, "legend_mode_combo", None)
            if cb is not None:
                return str(cb.currentText() or "User-defined")
        except Exception:
            pass
        try:
            return str(getattr(self, "plotted_legend_mode", "User-defined") or "User-defined")
        except Exception:
            return "User-defined"


    def set_plotted_legend_mode(self, mode: str):
        """Set legend mode: None / User-defined / Entry number."""
        try:
            self.plotted_legend_mode = str(mode or "User-defined")
        except Exception:
            self.plotted_legend_mode = "User-defined"

        m = str(self._get_plotted_legend_mode() or "User-defined").strip().lower()
        ax = getattr(self, "plotted_ax", None)
        if ax is None:
            return

        # Remove legend entirely in 'None' mode
        if m in ("none", "off", "no"):
            try:
                self.plotted_legend_visible = False
            except Exception:
                pass
            try:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
            except Exception:
                pass
            try:
                if hasattr(self, "canvas_plotted"):
                    self.canvas_plotted.draw()
            except Exception:
                pass
            return

        # Otherwise ensure legend exists and is visible
        try:
            self.plotted_legend_visible = True
        except Exception:
            pass
        try:
            if hasattr(self, "update_legend"):
                self.update_legend()
        except Exception:
            pass


    def toggle_plotted_annotation(self, visible: bool):
        """Show or hide the the annotation text box on the Plotted Data axes."""
        ax = getattr(self, "plotted_ax", None)
        if ax is None:
            return

        ann = getattr(self, "plotted_annotation", None)

        if visible:
            # Create annotation if it doesn't exist yet
            if ann is None:
                text = getattr(self, "plotted_annotation_text", "") or "Right-click to edit"
                try:
                    ann = ax.text(
                        0.02,
                        0.98,
                        text,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        bbox=dict(boxstyle="round", fc="w", alpha=0.4),
                    )
                    ann.set_picker(True)
                    try:
                        ann.set_draggable(True)
                    except Exception:
                        pass
                    # Ensure drag event handlers are connected once
                    try:
                        if not hasattr(self, "_annot_drag_cids"):
                            cids = []
                            if hasattr(self, "canvas_plotted"):
                                canvas = self.canvas_plotted
                                cids.append(canvas.mpl_connect("button_press_event", self._on_annotation_press))
                                cids.append(canvas.mpl_connect("motion_notify_event", self._on_annotation_motion))
                                cids.append(canvas.mpl_connect("button_release_event", self._on_annotation_release))
                            self._annot_drag_cids = tuple(cids)
                    except Exception:
                        pass
                    self.plotted_annotation = ann
                except Exception:
                    ann = None
                    self.plotted_annotation = None
            else:
                try:
                    ann.set_visible(True)
                except Exception:
                    pass
        else:
            if ann is not None:
                try:
                    ann.set_visible(False)
                except Exception:
                    pass

        try:
            if hasattr(self, "canvas_plotted"):
                self.canvas_plotted.draw()
        except Exception:
            pass
    def _on_annotation_press(self, event):
        """Start dragging the annotation text (left mouse button on the annotation)."""
        ax = getattr(self, "plotted_ax", None)
        ann = getattr(self, "plotted_annotation", None)
        if ax is None or ann is None:
            return
        if event.inaxes is not ax:
            return
        # Only react to left mouse button
        if getattr(event, "button", None) not in (1,):
            return
        contains, _ = ann.contains(event)
        if not contains:
            return
        try:
            # Current annotation position in display coordinates
            pos = ann.get_position()
            disp = ax.transAxes.transform(pos)
            self._annot_drag_active = True
            self._annot_drag_offset = (disp[0] - event.x, disp[1] - event.y)
        except Exception:
            self._annot_drag_active = False
            self._annot_drag_offset = None

    def _on_annotation_motion(self, event):
        """Update annotation position while dragging."""
        if not getattr(self, "_annot_drag_active", False):
            return
        ax = getattr(self, "plotted_ax", None)
        ann = getattr(self, "plotted_annotation", None)
        if ax is None or ann is None:
            return
        if event.inaxes is not ax:
            return
        try:
            dx, dy = self._annot_drag_offset
        except Exception:
            dx, dy = 0.0, 0.0
        try:
            new_disp = (event.x + dx, event.y + dy)
            new_axes = ax.transAxes.inverted().transform(new_disp)
            ann.set_position(new_axes)
            if hasattr(self, "canvas_plotted"):
                self.canvas_plotted.draw_idle()
        except Exception:
            pass

    def _on_annotation_release(self, event):
        """Finish dragging the annotation."""
        if getattr(self, "_annot_drag_active", False):
            self._annot_drag_active = False
            self._annot_drag_offset = None
    def clear_plotted_data(self):
        self.plotted_ax.clear()
        self.plotted_ax.set_xlabel("Photon energy (eV)")
        self.plotted_ax.set_ylabel("XAS intensity (arb. units)")
        self.canvas_plotted_fig.tight_layout()
        self.canvas_plotted.draw()
        self.plotted_curves.clear()
        self.plotted_lines.clear()
        self.plotted_list.clear()
        self.custom_labels.clear()
        self.original_line_data.clear()  # Waterfall originals

        # Reset Waterfall controls
        try:
            self.waterfall_checkbox.setChecked(False)
        except Exception:
            pass
        self.waterfall_slider.setValue(0)
        self.waterfall_spin.setValue(0.00)

        # Reset annotation: remove any existing annotation and uncheck the checkbox
        try:
            ann = getattr(self, "plotted_annotation", None)
            if ann is not None:
                try:
                    ann.remove()
                except Exception:
                    pass
            self.plotted_annotation = None
            try:
                self.plotted_annotation_text = "Right-click to edit"
            except Exception:
                self.plotted_annotation_text = "Right-click to edit"
        except Exception:
            pass
        try:
            if hasattr(self, "chk_show_annotation") and self.chk_show_annotation is not None:
                # This will also update the internal visible flag via the slot
                self.chk_show_annotation.setChecked(False)
        except Exception:
            pass

        # Re-apply grid mode after clearing (keep user selection)
        try:
            mode = "None"
            if hasattr(self, "grid_mode_combo") and self.grid_mode_combo is not None:
                mode = str(self.grid_mode_combo.currentText())
            self._apply_grid_mode(mode)
        except Exception:
            pass

    def recursive_uncheck(self, item, col):
        if not item:
            return
        if item.data(col, Qt.UserRole):
            item.setCheckState(col, Qt.Unchecked)
        for i in range(item.childCount()):
            self.recursive_uncheck(item.child(i), col)

    # ------------ Tabs & right-panel trees ------------
    def on_tab_changed(self, index):
        if index == 0:
            self.update_plot_raw()
            if hasattr(self, "raw_tree"):
                self.raw_tree.update()
        elif index == 1:
            self.update_plot_processed()
            if hasattr(self, "proc_tree"):
                self.proc_tree.update()

    def on_legend_pick(self, event):
        """
        Handle pick events on the Plotted Data canvas.

        - If the picked artist is the annotation text object, open a dialog to edit it.
        - If the picked artist is a legend text entry, open a dialog to rename the curve.
        """
        artist = getattr(event, "artist", None)

        # 1) Annotation editing
        ann = getattr(self, "plotted_annotation", None)
        if ann is not None and artist is ann:
            mouse = getattr(event, 'mouseevent', None)
            button = getattr(mouse, 'button', None)
            # Only open edit dialog on right-click; left-click is for dragging
            if button not in (3,):
                return

            try:
                old_text = ann.get_text() or ""
            except Exception:
                old_text = ""
            new_text, ok = QInputDialog.getText(
                self,
                "Edit annotation",
                "Annotation text:",
                text=old_text,
            )
            if ok:
                try:
                    text = str(new_text)
                except Exception:
                    text = old_text
                try:
                    ann.set_text(text)
                except Exception:
                    pass
                try:
                    self.plotted_annotation_text = text
                except Exception:
                    self.plotted_annotation_text = text
                try:
                    if hasattr(self, "canvas_plotted"):
                        self.canvas_plotted.draw()
                except Exception:
                    pass
            return

        # 2) Legend text renaming (only in 'User-defined' mode)
        try:
            mode = str(self._get_plotted_legend_mode() or "User-defined").strip().lower()
        except Exception:
            mode = "user-defined"
        if mode not in ("user-defined", "user", "custom"):
            return

        ax = getattr(self, "plotted_ax", None)
        if ax is None:
            return
        leg = ax.get_legend()
        if leg is None:
            return

        texts = list(leg.get_texts() or [])
        if artist not in texts:
            # Ignore picks that are not legend text
            return

        if hasattr(artist, "get_text"):
            old_text = artist.get_text()
            new_text, ok = QInputDialog.getText(
                self,
                "Rename Curve",
                "New legend name:",
                text=old_text,
            )
            if ok and new_text:
                # Map the clicked legend text to the corresponding plotted curve.
                # We use the legend's current ordering (stored when the legend is rebuilt)
                # so renaming is unambiguous even when multiple curves share the same
                # placeholder label (<select curve name>).
                key_to_rename = None
                try:
                    idx_text = texts.index(artist)
                except Exception:
                    idx_text = None
                try:
                    legend_keys = getattr(self, "_legend_keys_in_order", None)
                    if idx_text is not None and isinstance(legend_keys, (list, tuple)) and 0 <= idx_text < len(legend_keys):
                        key_to_rename = str(legend_keys[idx_text])
                except Exception:
                    key_to_rename = None

                # Fallback: old text matching (legacy behavior)
                if key_to_rename is None:
                    try:
                        for k, ln in getattr(self, "plotted_lines", {}).items():
                            current_label = getattr(self, "custom_labels", {}).get(k)
                            if (current_label is None and old_text == "<select curve name>") or (current_label == old_text):
                                key_to_rename = k
                                break
                    except Exception:
                        key_to_rename = None

                if key_to_rename is not None:
                    try:
                        self.custom_labels[key_to_rename] = str(new_text)
                    except Exception:
                        pass
                    try:
                        ln = getattr(self, "plotted_lines", {}).get(key_to_rename)
                        if ln is not None:
                            ln.set_label(str(new_text))
                    except Exception:
                        pass

                self.update_legend()


    def _legend_label_from_entry_number(self, key: str, line=None) -> str:
        """Derive a short legend label from an entry identifier.

        Example: 'entry6567/...' or 'entry6567: TEY' -> '6567'.
        If no entry number can be detected, falls back to an existing label.
        """
        import re
        candidates = []
        try:
            meta = getattr(self, "plotted_metadata", {}).get(key, {}) if hasattr(self, "plotted_metadata") else {}
            if isinstance(meta, dict):
                candidates.append(meta.get("source_entry") or "")
                candidates.append(meta.get("source_file") or "")
        except Exception:
            pass
        try:
            if line is not None and hasattr(line, "get_label"):
                candidates.append(line.get_label() or "")
        except Exception:
            pass
        candidates.append(str(key or ""))

        for txt in candidates:
            try:
                s = str(txt)
            except Exception:
                continue
            if not s:
                continue
            # Try the first path segment if present
            seg = s.split("/", 1)[0]
            m = re.search(r"entry\s*(\d+)", seg, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"entry\s*(\d+)", s, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        # Fallback
        try:
            if line is not None and hasattr(line, "get_label"):
                return str(line.get_label() or key)
        except Exception:
            pass
        return str(key)


    def _keep_legend_inside_axes(self, ax, leg, pad: float = 0.01) -> None:
        """Ensure the legend stays fully inside the Axes without shrinking the plot.

        Matplotlib's draggable legend stores the placement via bbox_to_anchor (often using
        an "upper left" reference point). When legend text changes (e.g. switching legend
        mode), the legend box can grow and protrude outside the Axes, and tight_layout may
        shrink the Axes to make room. Here we (1) exclude the legend from layout and (2)
        nudge its anchor point so the legend bbox fits inside [0, 1]x[0, 1] in Axes coords.
        """
        if ax is None or leg is None:
            return
        # Do not let tight_layout / constrained_layout resize the Axes because of the legend
        try:
            leg.set_in_layout(False)
        except Exception:
            pass

        try:
            fig = ax.figure
            canvas = getattr(fig, "canvas", None)
            if canvas is None:
                return
            # Need a draw to get correct text extents
            try:
                canvas.draw()
            except Exception:
                return
            renderer = canvas.get_renderer()
            bbox_disp = leg.get_window_extent(renderer=renderer)
            bbox_ax = bbox_disp.transformed(ax.transAxes.inverted())

            dx = 0.0
            dy = 0.0
            # Horizontal
            if bbox_ax.x0 < pad:
                dx = pad - bbox_ax.x0
            elif bbox_ax.x1 > 1.0 - pad:
                dx = (1.0 - pad) - bbox_ax.x1
            # Vertical
            if bbox_ax.y0 < pad:
                dy = pad - bbox_ax.y0
            elif bbox_ax.y1 > 1.0 - pad:
                dy = (1.0 - pad) - bbox_ax.y1

            if dx != 0.0 or dy != 0.0:
                # Re-anchor using the legend's current upper-left corner (axes coords)
                ul_x = bbox_ax.x0 + dx
                ul_y = bbox_ax.y1 + dy
                try:
                    # Use 'upper left' anchoring to interpret bbox_to_anchor as UL corner
                    try:
                        leg.set_loc("upper left")
                    except Exception:
                        leg._loc = 2
                    leg.set_bbox_to_anchor((ul_x, ul_y), transform=ax.transAxes)
                except Exception:
                    return

                try:
                    canvas.draw()
                except Exception:
                    pass
        except Exception:
            return
    def update_legend(self):
        """Rebuild the legend so its order follows the Plotted list (visible curves only)."""
        try:
            # Legend mode: None / User-defined / Entry number
            try:
                mode = str(self._get_plotted_legend_mode() or "User-defined").strip().lower()
            except Exception:
                mode = "user-defined"

            # Remove legend entirely in 'None' mode
            if mode in ("none", "off", "no"):
                ax = getattr(self, "plotted_ax", None)
                if ax is not None:
                    try:
                        leg = ax.get_legend()
                        if leg is not None:
                            leg.remove()
                    except Exception:
                        pass
                try:
                    self.plotted_legend_visible = False
                except Exception:
                    pass
                try:
                    if hasattr(self, "canvas_plotted"):
                        self.canvas_plotted.draw()
                except Exception:
                    pass
                return

            order = []
            if hasattr(self, "plotted_list") and self.plotted_list is not None:
                from .widgets.curve_item import CurveListItemWidget
                for row in range(self.plotted_list.count()):
                    item = self.plotted_list.item(row)
                    if item is None:
                        continue
                    # Retrieve the storage key from the item (set in pass_to_plotted_no_clear / _add_reference_curve_to_plotted)
                    key = item.data(Qt.UserRole)
                    if key is not None:
                        key = str(key)

                    # If Qt has detached the row widget (e.g. after drag/drop), recreate it
                    widget = self.plotted_list.itemWidget(item)
                    if widget is None and key:
                        line = getattr(self, "plotted_lines", {}).get(key)
                        if line is not None:
                            origin_label = line.get_label() or key
                            widget = CurveListItemWidget(origin_label, line.get_color(), key)
                            widget.colorChanged.connect(self.change_curve_color)
                            widget.visibilityChanged.connect(self.change_curve_visibility)
                            widget.styleChanged.connect(self.change_curve_style)
                            if hasattr(widget, "removeRequested"):
                                widget.removeRequested.connect(self.on_curve_remove_requested)
                            # For reference curves from the library, disable "Add to library" button again
                            meta = getattr(self, "plotted_metadata", {}).get(key, {}) if hasattr(self, "plotted_metadata") else {}
                            if meta.get("is_reference") and hasattr(widget, "set_add_to_library_enabled"):
                                widget.set_add_to_library_enabled(False)
                            item.setSizeHint(widget.sizeHint())
                            self.plotted_list.setItemWidget(item, widget)

                    if key:
                        order.append(key)

            ax = getattr(self, "plotted_ax", None)
            existing_leg = None
            saved_loc = getattr(self, "_plotted_legend_loc", None)
            saved_bbox = getattr(self, "_plotted_legend_bbox", None)
            if ax is not None:
                try:
                    existing_leg = ax.get_legend()
                except Exception:
                    existing_leg = None
            if existing_leg is not None:
                try:
                    saved_loc = getattr(existing_leg, "_loc", saved_loc)
                except Exception:
                    pass
                try:
                    saved_bbox = existing_leg.get_bbox_to_anchor()
                except Exception:
                    pass
            try:
                self._plotted_legend_loc = saved_loc
            except Exception:
                self._plotted_legend_loc = saved_loc
            try:
                self._plotted_legend_bbox = saved_bbox
            except Exception:
                self._plotted_legend_bbox = saved_bbox

            handles, labels = [], []
            legend_keys_in_order = []  # keys corresponding 1:1 with handles/labels
            entry_mode = mode in ("entry number", "entry", "entry-number", "entry_id", "entry id", "id")
            for key in order:
                line = getattr(self, "plotted_lines", {}).get(key)
                if line is None or (hasattr(line, "get_visible") and not line.get_visible()):
                    continue

                handles.append(line)
                legend_keys_in_order.append(key)

                lbl = None
                if entry_mode:
                    # IMPORTANT: Do NOT overwrite the underlying Line2D label in 'Entry number' mode.
                    # We only change what is displayed in the legend.
                    try:
                        lbl = self._legend_label_from_entry_number(key, line)
                    except Exception:
                        lbl = None
                else:
                    # User-defined mode: use stored custom labels when present,
                    # otherwise show the placeholder.
                    try:
                        lbl = getattr(self, "custom_labels", {}).get(key)
                    except Exception:
                        lbl = None

                    if not lbl:
                        try:
                            current = line.get_label()
                        except Exception:
                            current = None

                        # If the label got corrupted earlier (e.g. overwritten by entry number),
                        # restore to placeholder unless the user explicitly set a label.
                        try:
                            entry_lbl = self._legend_label_from_entry_number(key, line)
                        except Exception:
                            entry_lbl = None

                        if (not current) or (str(current).strip() == ""):
                            lbl = "<select curve name>"
                        else:
                            s = str(current).strip()
                            # Treat as corrupted if it looks like a pure entry number and matches what we'd derive.
                            if entry_lbl and s == str(entry_lbl).strip() and s.isdigit():
                                lbl = "<select curve name>"
                            else:
                                lbl = s

                    # In user-defined mode we keep the underlying label in sync,
                    # because other parts of the UI rely on it.
                    try:
                        line.set_label(lbl)
                    except Exception:
                        pass

                if not lbl:
                    lbl = "<select curve name>"
                labels.append(lbl)

            if not hasattr(self, "plotted_ax"):
                # Nothing to draw the legend on
                if hasattr(self, "canvas_plotted"):
                    try:
                        self.canvas_plotted.draw()
                    except Exception:
                        pass
                return

            ax = self.plotted_ax
            leg = None
            if handles:
                # Recreate legend at its previous location if known
                saved_loc = getattr(self, "_plotted_legend_loc", None)

                if saved_loc is not None:
                    # Preserve the user's dragged legend placement when possible.
                    # Matplotlib's draggable legend often stores placement via bbox_to_anchor.
                    saved_bbox = getattr(self, "_plotted_legend_bbox", None)
                    if saved_bbox is not None:
                        try:
                            # Use a plain tuple in Axes coordinates when possible
                            b = getattr(saved_bbox, "bounds", None)
                            if b is not None:
                                leg = ax.legend(handles, labels, loc=saved_loc, bbox_to_anchor=b, bbox_transform=ax.transAxes)
                            else:
                                leg = ax.legend(handles, labels, loc=saved_loc, bbox_to_anchor=saved_bbox, bbox_transform=ax.transAxes)
                        except Exception:
                            leg = ax.legend(handles, labels, loc=saved_loc)
                    else:
                        leg = ax.legend(handles, labels, loc=saved_loc)
                else:
                    leg = ax.legend(handles, labels)
                try:
                    if leg:
                        # Prevent tight_layout from shrinking the axes to "make room" for the legend
                        # (we keep the plot size constant and instead nudge the legend to fit inside the axes).
                        try:
                            leg.set_in_layout(False)
                        except Exception:
                            pass
                        leg.set_draggable(True)
                        for t in leg.get_texts():
                            # Smaller picker tolerance reduces accidental selection of a neighbour entry.
                            try:
                                t.set_picker(5)
                            except Exception:
                                t.set_picker(True)
                except Exception:
                    pass

            # Store legend ordering for unambiguous renaming in on_legend_pick()
            try:
                self._legend_keys_in_order = list(legend_keys_in_order)
            except Exception:
                self._legend_keys_in_order = legend_keys_in_order
            try:
                self._plotted_legend = leg
            except Exception:
                self._plotted_legend = leg

            # Apply visibility flag so that hiding the legend keeps its position
            try:
                show_legend = getattr(self, "plotted_legend_visible", True)
            except Exception:
                show_legend = True
            if leg is not None:
                try:
                    leg.set_visible(bool(show_legend))
                except Exception:
                    pass

                # Keep the legend fully inside the Axes without shrinking the plot area
                try:
                    self._keep_legend_inside_axes(ax, leg)
                except Exception:
                    pass

            if hasattr(self, "canvas_plotted_fig"):
                try:
                    self.canvas_plotted_fig.tight_layout()
                except Exception:
                    pass
            if hasattr(self, "canvas_plotted"):
                try:
                    self.canvas_plotted.draw()
                except Exception:
                    pass
        except Exception:
            pass
    def visible_curves_count(self):
        return sum(1 for _k, visible in getattr(self, "raw_visibility", {}).items() if visible)

    def update_pass_button_state(self):
        try:
            cond_sum = bool(getattr(self, "chk_sum", None) and self.chk_sum.isChecked())
            vc = self.visible_curves_count()

            # Group Auto-BG multi-pass condition:
            # - multiple curves are selected (checked)
            # - sum is OFF
            # - Automatic BG
            # - background subtraction is ON
            # - group background checkbox is ON
            group_cond = False
            try:
                if (not cond_sum) and vc >= 2:
                    mode = str(getattr(self, "combo_bg").currentText()) if hasattr(self, "combo_bg") else ""
                    if str(mode) in ("Automatic", "Auto") and getattr(self, "chk_show_without_bg", None) is not None and self.chk_show_without_bg.isChecked():
                        cb = getattr(self, "chk_group_bg", None)
                        if cb is not None and cb.isEnabled() and cb.isChecked():
                            group_cond = True
            except Exception:
                group_cond = False

            if getattr(self, "pass_button", None):
                self.pass_button.setEnabled(bool(cond_sum or vc == 1 or group_cond))
        except Exception:
            pass

    def _on_group_bg_checkbox_toggled(self, state):
        """Handle Group BG toggle.

        New UX (v2.1.1+):
        - The checkbox becomes *checkable* as soon as >=2 curves are selected.
        - When checked by the user, Group BG *forces* the required math settings:
          BG mode = Automatic, Subtract BG = ON, Post-normalization = Area.
        - While Group BG stays checked, BG mode and Area post-normalization stay fixed
          (controls are locked).
        """
        checked = False
        try:
            checked = (state == Qt.Checked)
        except Exception:
            checked = bool(state)

        # Remember user's choice (only meaningful while in multi-selection)
        try:
            self._group_bg_user_choice = bool(checked)
        except Exception:
            pass

        # Enter/exit Group BG mode (force/restore UI + processing settings)
        try:
            self._set_group_bg_mode(bool(checked))
        except Exception:
            # If anything goes wrong, still attempt a redraw.
            pass

        try:
            self.update_plot_processed()
        except Exception:
            pass

    def _set_group_bg_mode(self, enabled: bool):
        """Force (or restore) processing controls for Group BG mode.

        When enabled, the required settings are applied and controls are locked:
        - BG mode: Automatic
        - Subtract BG: checked
        - Post-normalization: Area

        When disabled, previously stored user settings (if available) are restored.
        """
        # Lazy-import Qt safely (already imported at module top in most builds)
        try:
            _QtChecked = Qt.Checked
        except Exception:
            _QtChecked = 2

        cb = getattr(self, "chk_group_bg", None)
        if cb is None:
            return

        if enabled:
            # Save previous settings once
            if getattr(self, "_group_bg_prev_settings", None) is None:
                prev = {}
                try:
                    prev["bg_text"] = str(self.combo_bg.currentText())
                    prev["bg_enabled"] = bool(self.combo_bg.isEnabled())
                except Exception:
                    prev["bg_text"] = "None"
                    prev["bg_enabled"] = True
                try:
                    prev["subtract"] = bool(self.chk_show_without_bg.isChecked())
                    prev["subtract_enabled"] = bool(self.chk_show_without_bg.isEnabled())
                except Exception:
                    prev["subtract"] = False
                    prev["subtract_enabled"] = True
                try:
                    prev["post_norm_text"] = str(self.combo_post_norm.currentText())
                    prev["post_norm_enabled"] = bool(self.combo_post_norm.isEnabled())
                except Exception:
                    prev["post_norm_text"] = "None"
                    prev["post_norm_enabled"] = False
                self._group_bg_prev_settings = prev

            # Allow the user to toggle "Subtract BG" for visual inspection in Group BG mode.
            # We only auto-check it the first time Group BG is enabled; afterwards we preserve user choice.
            try:
                first_enable = not bool(getattr(self, "_group_bg_mode_active", False))
                self._group_bg_mode_active = True
            except Exception:
                first_enable = True

            # Force required settings without triggering cascaded updates
            try:
                self.combo_bg.blockSignals(True)
                self.chk_show_without_bg.blockSignals(True)
                self.combo_post_norm.blockSignals(True)
            except Exception:
                pass
            try:
                # BG mode -> Automatic
                try:
                    idx = int(self.combo_bg.findText("Auto"))
                    if idx >= 0:
                        self.combo_bg.setCurrentIndex(idx)
                    else:
                        self.combo_bg.setCurrentText("Auto")
                except Exception:
                    pass

                # Subtract BG -> ON (only on first enable)
                if "first_enable" in locals() and first_enable:
                    try:
                        self.chk_show_without_bg.setChecked(True)
                    except Exception:
                        pass

                # Post-norm -> Area
                try:
                    idxn = int(self.combo_post_norm.findText("Area"))
                    if idxn >= 0:
                        self.combo_post_norm.setCurrentIndex(idxn)
                    else:
                        self.combo_post_norm.setCurrentText("Area")
                except Exception:
                    pass
            finally:
                try:
                    self.combo_bg.blockSignals(False)
                    self.chk_show_without_bg.blockSignals(False)
                    self.combo_post_norm.blockSignals(False)
                except Exception:
                    pass

            # Lock controls that must stay fixed in Group BG mode
            try:
                self.combo_bg.setEnabled(False)
            except Exception:
                pass
            try:
                self.chk_show_without_bg.setEnabled(True)  # keep enabled for visual inspection in Group BG mode
            except Exception:
                pass
            try:
                self.combo_post_norm.setEnabled(False)
            except Exception:
                pass

        else:
            # Restore previous settings (if we have them)
            prev = getattr(self, "_group_bg_prev_settings", None)

            # Unlock first, then restore values, then restore enabled states.
            try:
                self.combo_bg.setEnabled(True)
            except Exception:
                pass
            try:
                self.chk_show_without_bg.setEnabled(True)
            except Exception:
                pass
            try:
                self.combo_post_norm.setEnabled(True)
            except Exception:
                pass

            if prev:
                try:
                    self.combo_bg.blockSignals(True)
                    self.chk_show_without_bg.blockSignals(True)
                    self.combo_post_norm.blockSignals(True)
                except Exception:
                    pass
                try:
                    # BG mode
                    try:
                        txt = str(prev.get("bg_text", "None"))
                        idx = int(self.combo_bg.findText(txt))
                        if idx >= 0:
                            self.combo_bg.setCurrentIndex(idx)
                        else:
                            self.combo_bg.setCurrentText(txt)
                    except Exception:
                        pass

                    # Subtract BG
                    try:
                        self.chk_show_without_bg.setChecked(bool(prev.get("subtract", False)))
                    except Exception:
                        pass

                    # Post-norm
                    try:
                        txtn = str(prev.get("post_norm_text", "None"))
                        idxn = int(self.combo_post_norm.findText(txtn))
                        if idxn >= 0:
                            self.combo_post_norm.setCurrentIndex(idxn)
                        else:
                            self.combo_post_norm.setCurrentText(txtn)
                    except Exception:
                        pass
                finally:
                    try:
                        self.combo_bg.blockSignals(False)
                        self.chk_show_without_bg.blockSignals(False)
                        self.combo_post_norm.blockSignals(False)
                    except Exception:
                        pass

                # Restore enabled states (post-norm enable depends on subtract in normal mode,
                # so we restore exactly what the user had before entering Group BG).
                try:
                    self.combo_bg.setEnabled(bool(prev.get("bg_enabled", True)))
                except Exception:
                    pass
                try:
                    self.chk_show_without_bg.setEnabled(bool(prev.get("subtract_enabled", True)))
                except Exception:
                    pass
                try:
                    self.combo_post_norm.setEnabled(bool(prev.get("post_norm_enabled", False)))
                except Exception:
                    pass

            # Mark Group BG as inactive so Subtract BG can be user-toggled on next activation.
            try:
                self._group_bg_mode_active = False
            except Exception:
                pass

            try:
                self._group_bg_prev_settings = None
            except Exception:
                self._group_bg_prev_settings = None

    def _on_group_bg_slope_checkbox_toggled(self, state):
        """Remember user choice for group pre-edge slope matching and update the processed plot."""
        try:
            self._group_bg_slope_user_choice = (state == Qt.Checked)
        except Exception:
            pass
        try:
            self.update_plot_processed()
        except Exception:
            pass

    def _update_group_bg_checkbox_state(self, visible_curves: int, mode_text: str):
        """Enable/disable the Group BG checkbox.

        UX goals (2025-12):
        - The checkbox is *checkable* whenever >=2 curves are selected, regardless of current BG/Norm settings.
        - It is NOT auto-checked. Group BG must be an explicit user choice.
        - When selection drops back to 0/1, Group BG is turned off and the checkbox is disabled.

        Math logic is enforced elsewhere:
        - When the user checks Group BG, we force: Automatic BG + Subtract BG + Area normalization
          and lock these controls while Group BG stays checked.
        """
        cb = getattr(self, "chk_group_bg", None)
        if cb is None:
            return
        # Track transitions into/out of multi-selection
        prev_multi = bool(getattr(self, "_group_bg_prev_multi", False))
        multi_now = (int(visible_curves) >= 2)

        if not multi_now:
            # Leaving multi-selection: ensure we exit Group BG mode cleanly.
            try:
                if cb.isChecked():
                    # Avoid recursion via the checkbox signal.
                    try:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                    finally:
                        try:
                            cb.blockSignals(False)
                        except Exception:
                            pass
                    try:
                        self._set_group_bg_mode(False)
                    except Exception:
                        pass
            except Exception:
                pass

            # Reset remembered choice
            try:
                setattr(self, "_group_bg_user_choice", None)
            except Exception:
                pass

            try:
                cb.setEnabled(False)
            except Exception:
                pass
        else:
            # Entering / staying in multi-selection: checkbox is always available.
            try:
                cb.setEnabled(True)
            except Exception:
                pass

            # Do NOT auto-check when transitioning from 1 -> 2+. Keep unchecked unless the user ticks it.
            if (not prev_multi) and multi_now:
                try:
                    cb.blockSignals(True)
                    cb.setChecked(False)
                finally:
                    try:
                        cb.blockSignals(False)
                    except Exception:
                        pass
                try:
                    setattr(self, "_group_bg_user_choice", False)
                except Exception:
                    pass

        try:
            setattr(self, "_group_bg_prev_multi", bool(multi_now))
        except Exception:
            pass

        # Keep the related 'Match pre-edge slope' checkbox in sync
        try:
            self._update_group_bg_slope_checkbox_state(visible_curves=int(visible_curves), mode_text=str(mode_text))
        except Exception:
            pass

    def _update_group_bg_slope_checkbox_state(self, visible_curves: int, mode_text: str):
        """Enable/disable the 'Match pre-edge slope' checkbox.

        This option is only meaningful when:
        - BG mode is Automatic
        - >=2 curves are selected
        - Group background checkbox is enabled & checked

        Default is OFF (unchecked). If the user manually toggles it while in multi-selection,
        keep their choice until selection drops back to 0/1.
        """
        cb = getattr(self, "chk_group_bg_slope", None)
        if cb is None:
            return

        try:
            mode = str(mode_text or "")
        except Exception:
            mode = ""

        # Determine whether group BG is currently usable/active
        group_cb = getattr(self, "chk_group_bg", None)
        group_ok = bool(group_cb is not None and group_cb.isEnabled() and group_cb.isChecked())

        prev_multi = bool(getattr(self, "_group_bg_slope_prev_multi", False))
        user_choice = getattr(self, "_group_bg_slope_user_choice", None)

        multi_now = (int(visible_curves) >= 2)
        usable_now = multi_now and (str(mode) in ("Automatic", "Auto")) and group_ok

        if not multi_now:
            user_choice = None
            try:
                setattr(self, "_group_bg_slope_user_choice", None)
            except Exception:
                pass

        try:
            cb.blockSignals(True)
            cb.setEnabled(bool(usable_now))
            if not usable_now:
                cb.setChecked(False)
            else:
                # Default is OFF when entering multi-selection.
                if (not prev_multi) and multi_now and user_choice is None:
                    cb.setChecked(False)
                    setattr(self, "_group_bg_slope_user_choice", False)
                elif user_choice is not None:
                    cb.setChecked(bool(user_choice))
        finally:
            try:
                cb.blockSignals(False)
            except Exception:
                pass

        try:
            setattr(self, "_group_bg_slope_prev_multi", bool(multi_now))
        except Exception:
            pass

    def show_about_info(self):
        info_text = (
            f"Software Version: {self.VERSION_NUMBER}\n"
            f"Date: {self.CREATION_DATETIME}\n\n"
            "License: MIT\n"
            "Created by: Alexei Preobrajenski\n\n"
            "This is a Python-based GUI for browsing, pre-processing and plotting Near-Edge X-ray Absorption Fine Structure (NEXAFS) spectra stored in HDF5 files, "
            "as collected at the FlexPES beamline (MAX IV Laboratory)."
        )
        QMessageBox.information(self, "About FlexPES NEXAFS Plotter", info_text)

    def show_usage_info(self):
        """Show the Help->Usage dialog populated from docs/help.md.

        The help text is stored in the package as Markdown and converted
        to HTML via :func:`get_usage_html` so that only a single source
        of truth needs to be maintained.
        """
        try:
            usage_html = get_usage_html()
        except Exception:
            usage_html = "<p><b>Help text could not be loaded.</b></p>"

        dlg = QDialog(self)
        dlg.setWindowTitle("Usage – FlexPES NEXAFS Plotter")
        dlg.resize(900, 650)
        dlg.setSizeGripEnabled(True)
        # Enable maximize button on the dialog window.
        try:
            dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowMaximizeButtonHint)
        except Exception:
            pass

        layout = QVBoxLayout(dlg)

        # --- Controls row: font size ---------------------------------
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)
        controls_row.addWidget(QLabel("Font size:"))
        font_spin = QSpinBox()
        font_spin.setRange(8, 28)
        font_spin.setSingleStep(1)
        # Default font size for Usage window
        font_spin.setValue(18)
        font_spin.setToolTip("Change the font size used in this help window.")
        controls_row.addWidget(font_spin)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        splitter = QSplitter(Qt.Horizontal, dlg)

        # Table of contents (clickable headings)
        toc = QTreeWidget()
        toc.setHeaderHidden(True)
        toc.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # Make TOC items stand out (bold + blue)
        # Styling for the table-of-contents (left) list
        def _set_toc_style(px: int):
            try:
                px = int(px)
            except Exception:
                px = 14
            # Keep TOC items blue + bold, and ensure selected items remain readable
            # across light/dark themes.
            try:
                toc.setStyleSheet(
                    "QTreeWidget {"
                    f" font-size: {px}px;"
                    " }"
                    "QTreeWidget::item { color: #0066CC; font-weight: 700; }"
                    "QTreeWidget::item:selected { color: palette(text); }"
                )
            except Exception:
                pass
            # Belt-and-suspenders: make sure the widget's font is bold as well.
            try:
                f = toc.font()
                f.setBold(True)
                toc.setFont(f)
            except Exception:
                pass

        _set_toc_style(font_spin.value())

        toc.setMinimumWidth(200)
        toc.setMaximumWidth(320)

        # Main help browser
        browser = HelpBrowser()
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser.setStyleSheet("font-size: 18px;")
        browser.setHtml(usage_html)

        # Build TOC from extracted heading anchors (H1/H2)
        try:
            toc.clear()
            current_h1 = None
            for level, title, anchor in getattr(browser, "_help_anchors", []) or []:
                item = QTreeWidgetItem([title])
                item.setData(0, Qt.UserRole, anchor)
                if level == 1:
                    toc.addTopLevelItem(item)
                    current_h1 = item
                else:
                    if current_h1 is None:
                        toc.addTopLevelItem(item)
                    else:
                        current_h1.addChild(item)
            toc.expandAll()

            def _jump_to_section(item, _col=0):
                anchor = item.data(0, Qt.UserRole)
                if anchor:
                    browser.scrollToAnchor(str(anchor))

            toc.itemClicked.connect(_jump_to_section)
        except Exception:
            pass

        splitter.addWidget(toc)
        splitter.addWidget(browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 660])

        layout.addWidget(splitter)

        # Wire up font size changes (apply to both TOC and browser for a consistent look)
        def _apply_help_font(px: int):
            try:
                px = int(px)
            except Exception:
                px = 14
            try:
                browser.setStyleSheet(f"font-size: {px}px;")
            except Exception:
                pass
            try:
                _set_toc_style(px)
            except Exception:
                pass

        font_spin.valueChanged.connect(_apply_help_font)

        dlg.exec_()


    def clear_all_except_plotted(self):
        if hasattr(self, "tree"):
            for i in range(self.tree.topLevelItemCount()):
                self.recursive_uncheck(self.tree.topLevelItem(i), 1)
        if hasattr(self, "raw_tree"):
            for i in range(self.raw_tree.topLevelItemCount()):
                self.recursive_uncheck(self.raw_tree.topLevelItem(i), 0)

        self.raw_tree_reset = False
        self.raw_visibility.clear()
        self.region_states.clear()
        self.proc_region_states.clear()
        self.plot_data.clear()
        self.energy_cache.clear()
        self.cb_all_tey.setChecked(False)
        self.cb_all_pey.setChecked(False)
        self.cb_all_tfy.setChecked(False)
        self.cb_all_pfy.setChecked(False)
        self.combo_bg.setCurrentIndex(0)
        self.spin_preedge.setValue(5)
        self.chk_normalize.setChecked(False)
        self.combo_norm.setEnabled(False)

        # Pick default I0 channel according to channel mapping (beamline profile).
        defaults = ["b107a_em_03_ch2", "b107a_em_04_ch2", "Pt_No"]
        try:
            cc = getattr(self, "channel_config", None)
            if cc is not None:
                cands = cc.get_candidates("I0")
                if cands:
                    defaults = list(cands)
        except Exception:
            pass
        for default in defaults:
            idx = self.combo_norm.findText(str(default))
            if idx != -1:
                self.combo_norm.setCurrentIndex(idx)
                break
        fm = self.combo_norm.fontMetrics()
        self.combo_norm.setMinimumWidth(fm.boundingRect(self.combo_norm.currentText()).width() + 20)

        self.chk_sum.setChecked(False)
        self.chk_show_without_bg.setChecked(False)
        self.reset_manual_mode()
        self.scalar_display_raw.setText("")
        self.update_plot_raw()
        self.update_plot_processed()
        self.update_pass_button_state()
        self.raw_tree.repaint()

    def clear_all(self):
        if hasattr(self, "tree"):
            for i in range(self.tree.topLevelItemCount()):
                self.recursive_uncheck(self.tree.topLevelItem(i), 1)
        self.plot_data.clear()
        self.energy_cache.clear()
        self.cb_all_tey.setChecked(False)
        self.cb_all_pey.setChecked(False)
        self.cb_all_tfy.setChecked(False)
        self.cb_all_pfy.setChecked(False)
        self.region_states.clear()
        self.proc_region_states.clear()
        self.update_plot_raw()
        self.update_plot_processed()
        self.combo_bg.setCurrentIndex(0)
        self.chk_normalize.setChecked(False)
        self.chk_sum.setChecked(False)
        self.chk_show_without_bg.setChecked(False)
        self.reset_manual_mode()
        self.scalar_display_raw.setText("")
        self.plotted_ax.clear()
        self.plotted_ax.set_xlabel("Photon energy (eV)")
        self.plotted_ax.set_ylabel("XAS intensity (arb. units)")
        self.canvas_plotted_fig.tight_layout()
        self.canvas_plotted.draw()
        # Reset annotation state when clearing everything
        try:
            ann = getattr(self, "plotted_annotation", None)
            if ann is not None:
                try:
                    ann.remove()
                except Exception:
                    pass
            self.plotted_annotation = None
            try:
                self.plotted_annotation_text = "Right-click to edit"
            except Exception:
                self.plotted_annotation_text = "Right-click to edit"
        except Exception:
            pass
        try:
            if hasattr(self, "chk_show_annotation") and self.chk_show_annotation is not None:
                self.chk_show_annotation.setChecked(False)
        except Exception:
            pass

        self.plotted_curves.clear()
        self.plotted_lines.clear()
        self.plotted_list.clear()
        self.original_line_data.clear()
        self.update_pass_button_state()

    # ------------ Left tree interactions ------------
    def toggle_plot(self, item, column):
        if column != 0:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        # UserRole payload must be a (abs_path, hdf5_path) tuple
        if not isinstance(data, tuple) or len(data) != 2:
            return
        abs_path, hdf5_path = data
        if abs_path not in self.hdf5_files:
            return

        try:
            with self._open_h5_read(abs_path) as f:
                if hdf5_path in f:
                    ds_obj = f[hdf5_path]
                    if isinstance(ds_obj, h5py.Dataset) and ds_obj.ndim == 1:
                        if getattr(ds_obj, "size", 0) == 0:
                            QMessageBox.warning(
                                self, "Empty dataset",
                                f'The dataset “{hdf5_path}” contains no data and will be ignored.'
                            )
                            item.setCheckState(0, Qt.Unchecked)
                            return

                        combined_label = f"{abs_path}##{hdf5_path}"
                        if item.checkState(0) == Qt.Checked:
                            self.plot_data[combined_label] = ds_obj[()]
                            self.raw_visibility[combined_label] = True
                        else:
                            if combined_label in self.plot_data:
                                del self.plot_data[combined_label]
                            self.raw_visibility[combined_label] = False

            self._filter_empty_plot_data()
            self.update_plot_raw()
            self.update_plot_processed()
            self.update_pass_button_state()

        except Exception:
            pass

    def display_data(self, item, column):
        if self.data_tabs.currentIndex() != 0:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        abs_path, hdf5_path = data
        if abs_path not in self.hdf5_files:
            return
        try:
            with self._open_h5_read(abs_path) as f:
                if hdf5_path not in f:
                    return
                arr = f[hdf5_path][()]

            # Scalars and 1D
            if isinstance(arr, np.ndarray) and arr.ndim in (0, 1):
                if arr.ndim == 0:
                    arr = arr.item()
                # For 1D, nothing here; the plot is handled elsewhere
                return

            # Bytes → str
            if isinstance(arr, bytes):
                arr = arr.decode("utf-8")

            # Try datetime formatting if string looks like ISO-ish
            if isinstance(arr, str) and "T" in arr and "-" in arr:
                try:
                    dt_obj = datetime.strptime(arr, "%Y-%m-%dT%H:%M:%S.%f")
                    arr = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

            self.scalar_display_raw.setText(str(arr) if not isinstance(arr, float) else f"{arr:.2f}")

        except Exception as e:
            self.scalar_display_raw.setText(f"Error displaying data: {e}")

    # ------------ Raw/Processed plotting ------------
    def plot_curves(self, ax):
        ax.clear()
        for combined_label, y_data in self.plot_data.items():
            if not self.raw_visibility.get(combined_label, True):
                continue
            parts = combined_label.split("##", 1)
            if len(parts) != 2:
                continue
            abs_path, hdf5_path = parts
            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
            x_data = lookup_energy(self, abs_path, parent, len(y_data))
            if getattr(x_data, "size", 0) == 0 or len(y_data) == 0:
                continue
            mlen = min(len(x_data), len(y_data))
            x_use = x_data[:mlen]
            y_use = y_data[:mlen]
            if len(x_use) == 0:
                continue
            line, = ax.plot(x_use, y_use, label=self.shorten_label(hdf5_path))
            line.dataset_key = combined_label
        ax.set_xlabel("Photon energy (eV)")

    def update_plot_raw(self):
        try:
            self._filter_empty_plot_data()
            self.plot_curves(self.raw_ax)
            self.canvas_raw_fig.tight_layout()
            self.canvas_raw.draw()
            self.update_raw_tree()
        except Exception as e:
            print("update_plot_raw error:", e)

    def _plot_multiple_no_bg(self):
        self.proc_ax.clear()
        for combined_label, y_data in self.plot_data.items():
            if not self.raw_visibility.get(combined_label, True):
                continue
            parts = combined_label.split("##", 1)
            if len(parts) != 2:
                continue
            abs_path, hdf5_path = parts
            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
            x_data = lookup_energy(self, abs_path, parent, len(y_data))
            if getattr(x_data, "size", 0) == 0 or len(y_data) == 0:
                continue
            processed_y = processing.apply_normalization(self, abs_path, parent, y_data)
            mlen = min(len(x_data), len(processed_y))
            x_use = x_data[:mlen]
            y_use = processed_y[:mlen]
            if len(x_use) == 0:
                continue
            line, = self.proc_ax.plot(x_use, y_use, label=self.shorten_label(hdf5_path))
            line.dataset_key = combined_label
        self.proc_ax.set_xlabel("Photon energy (eV)")

    def _visible_processed_keys(self):
        """Return list of dataset keys currently visible (checked) in the Processed Data tree."""
        try:
            return [k for k in getattr(self, "plot_data", {}) if getattr(self, "raw_visibility", {}).get(k, False)]
        except Exception:
            return []

    def _compute_group_auto_backgrounds(self, keys, deg: int, pre_edge_percent: float):
        """Compute *per-spectrum* automatic backgrounds for a group.

        This helper exists mainly for the multi-curve "group BG" plotting path.
        It intentionally does **not** force a strictly shared background shape.
        Instead, it computes the same per-spectrum Automatic BG as in the single
        spectrum workflow (using :meth:`_apply_automatic_bg_new` with
        ``do_plot=False``).

        Returns:
            (x_common, backgrounds_dict)
            - x_common is returned only if all curves share the same energy grid.
              Otherwise it is ``None``.
            - backgrounds_dict maps each dataset key to its background array.
        """
        import numpy as np

        if not keys:
            return None, None

        x_list = []
        bgs = {}

        for key in keys:
            parts = key.split("##", 1)
            if len(parts) != 2:
                continue
            abs_path, hdf5_path = parts
            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
            y_data = self.plot_data.get(key)
            if y_data is None:
                continue
            x_data = lookup_energy(self, abs_path, parent, len(y_data))
            y_proc = processing.apply_normalization(self, abs_path, parent, y_data)
            mlen = min(len(x_data), len(y_proc))
            if mlen < 3:
                continue
            x_use = np.asarray(x_data[:mlen], dtype=float).ravel()
            y_use = np.asarray(y_proc[:mlen], dtype=float).ravel()
            bg = self._apply_automatic_bg_new(x_use, y_use, deg=int(deg), pre_edge_percent=float(pre_edge_percent), do_plot=False)
            bgs[key] = np.asarray(bg, dtype=float).ravel()[:mlen]
            x_list.append(x_use)

        if not bgs:
            return None, None

        # Return a common grid only if identical (for optional markers)
        x_common = None
        try:
            x0 = x_list[0]
            same = True
            for xx in x_list[1:]:
                if xx.size != x0.size or (not np.allclose(xx, x0, rtol=0, atol=1e-9, equal_nan=True)):
                    same = False
                    break
            if same:
                x_common = x0
        except Exception:
            x_common = None

        return x_common, bgs

    def _group_equalize_area_jump_and_zero_preedge(self, items_simple: dict, pre: float, match_preedge_slope: bool = False):
        """Adjust *per-spectrum* automatic backgrounds in group mode.

        This routine is used when multiple spectra are selected and the user
        intends to apply **Area** post-normalization. It modifies each spectrum's
        background so that:

        - The **pre-edge baseline after BG subtraction is 0** (per spectrum).
        - The **jump after BG subtraction and Area-normalization** is the same
          across the group (median target).
        - Optionally (when match_preedge_slope=True), the **pre-edge slope after BG subtraction**
          is also made consistent across the group (median target).

        Notes:
            - Backgrounds are not forced to share a common polynomial shape.
            - The correction uses an affine term (linear + constant) per spectrum.
        """
        import numpy as np

        if not items_simple or len(items_simple) < 2:
            return items_simple, None

        # Helper: find pre/post windows on a finite-length array
        def _windows(n: int):
            n = int(n)
            if n < 3:
                return None
            M = max(1, int(float(pre) * n))
            M = min(M, n - 1)
            i_start = max(M + 1, int(0.95 * n))
            if i_start >= n - 1:
                i_start = max(n - 2, 0)
            return M, i_start

        # First pass: compute per-spectrum baseline-zero shift and the current ratio R0 = jump/area.
        ratios = []
        slopes = []  # only used when match_preedge_slope=True
        per = {}
        for key, (x_use, y_use, bg) in items_simple.items():
            x_use = np.asarray(x_use, dtype=float).ravel()
            y_use = np.asarray(y_use, dtype=float).ravel()
            bg = np.asarray(bg, dtype=float).ravel()
            n = min(x_use.size, y_use.size, bg.size)
            if n < 3:
                continue
            x_use = x_use[:n]; y_use = y_use[:n]; bg = bg[:n]
            f = np.isfinite(x_use) & np.isfinite(y_use) & np.isfinite(bg)
            xf = x_use[f]
            if xf.size < 3:
                continue
            yf = y_use[f]
            bgf = bg[f]

            win = _windows(int(xf.size))
            if win is None:
                continue
            M, i_start = win

            # Center x for numerical stability.
            xbar = float(np.nanmean(xf))
            t = xf - xbar

            # Baseline-zero constant for beta=0
            r0 = float(np.nanmean((yf - bgf)[:M]))
            s0 = (yf - bgf) - r0  # pre-edge mean is ~0

            area0 = float(trapezoid(s0, xf))
            if (not np.isfinite(area0)) or abs(area0) < 1e-15:
                continue
            jump0 = float(np.nanmean(s0[i_start:]))  # pre is 0 by construction
            ratio0 = jump0 / area0
            if not np.isfinite(ratio0):
                continue

            # Terms needed for the beta solve under the constraint that pre-edge remains 0.
            mp = float(np.nanmean(t[:M]))
            tt = t - mp  # this ensures the pre-edge mean of the correction term is 0
            T = float(trapezoid(tt, xf))
            P = float(np.nanmean(tt[i_start:]))

            # Optional: pre-edge slope of the BG-subtracted signal (after baseline shift).
            m0 = None
            phi2 = None
            T2 = 0.0
            P2 = 0.0
            if bool(match_preedge_slope):
                try:
                    if int(M) >= 2:
                        m0 = float(np.polyfit(xf[:M], s0[:M], 1)[0])
                    else:
                        m0 = float(np.polyfit(xf[: max(2, int(M))], s0[: max(2, int(M))], 1)[0])
                except Exception:
                    m0 = None

                # A localized pre-edge basis function: w(x)*(x-x_pre_mean)
                # w=1 in pre-edge, then smoothly tapers to 0 well before the post-edge window.
                try:
                    x_pre_mean = float(np.nanmean(xf[:M]))
                    w = np.zeros_like(xf, dtype=float)
                    w[:M] = 1.0
                    # Choose a taper length that (if possible) ends before the post window.
                    k_suggest = max(5, int(0.05 * float(xf.size)))
                    k_max = max(1, int(i_start) - int(M) - 1)
                    K = int(min(k_suggest, k_max))
                    if K < 1:
                        K = 1
                    end = min(int(M) + int(K), int(xf.size) - 1)
                    # Build cosine taper from 1 -> 0 across [M, end]
                    tt_idx = np.arange(0, (end - int(M)) + 1, dtype=float)
                    denomK = float(max(1, (end - int(M))))
                    w[int(M) : end + 1] = 0.5 * (1.0 + np.cos(np.pi * tt_idx / denomK))
                    phi2 = w * (xf - x_pre_mean)
                    T2 = float(trapezoid(phi2, xf))
                    P2 = float(np.nanmean(phi2[i_start:]))
                except Exception:
                    phi2 = None
                    T2 = 0.0
                    P2 = 0.0

                if m0 is not None and np.isfinite(m0):
                    slopes.append(float(m0))

            ratios.append(ratio0)
            per[key] = {
                "n": n,
                "f": f,
                "xf": xf,
                "yf": yf,
                "bgf": bgf,
                "xbar": xbar,
                "r0": r0,
                "mp": mp,
                "s0": s0,
                "area0": area0,
                "jump0": jump0,
                "T": T,
                "P": P,
                "m0": m0,
                "phi2": phi2,
                "T2": T2,
                "P2": P2,
            }

        if len(ratios) < 2:
            return items_simple, None

        R_target = float(np.median(np.asarray(ratios, dtype=float)))

        m_target = None
        if bool(match_preedge_slope):
            try:
                if len(slopes) >= 2:
                    m_target = float(np.median(np.asarray(slopes, dtype=float)))
            except Exception:
                m_target = None

        # Second pass: solve for per-spectrum corrections.
        # Default (match_preedge_slope=False): solve beta (linear) for each spectrum so that (jump/area)
        # matches the target while keeping pre-edge baseline at 0.
        # If match_preedge_slope=True and m_target is available: additionally solve a localized pre-edge
        # correction coefficient so that the pre-edge slope after BG subtraction matches the group target.
        adjusted = {}
        changed = 0
        for key, (x_use, y_use, bg) in items_simple.items():
            if key not in per:
                adjusted[key] = (x_use, y_use, bg)
                continue
            info = per[key]
            x_use = np.asarray(x_use, dtype=float).ravel()[: info["n"]]
            bg = np.asarray(bg, dtype=float).ravel()[: info["n"]]
            f = info["f"]
            xf = info["xf"]
            yf = info["yf"]
            bgf = info["bgf"]
            xbar = float(info["xbar"])
            r0 = float(info["r0"])
            mp = float(info["mp"])
            area0 = float(info["area0"])
            jump0 = float(info["jump0"])
            T = float(info["T"])
            P = float(info["P"])

            # --- Optional: match pre-edge slope as well (3 constraints, 3 DOF: constant + global linear + localized pre-edge linear) ---
            if bool(match_preedge_slope) and (m_target is not None) and (info.get("phi2") is not None) and (info.get("m0") is not None):
                try:
                    A11 = (R_target * T - P)
                    A12 = (R_target * float(info.get("T2", 0.0)) - float(info.get("P2", 0.0)))
                    rhs1 = (R_target * area0 - jump0)
                    rhs2 = (float(info["m0"]) - float(m_target))
                    det = (A11 - A12)
                    if (not np.isfinite(det)) or abs(det) < 1e-15:
                        raise ZeroDivisionError("singular slope/ratio system")
                    u1 = (rhs1 - A12 * rhs2) / det
                    u2 = rhs2 - u1
                    if (not np.isfinite(u1)) or (not np.isfinite(u2)):
                        raise ValueError("non-finite solution")
                    # Constant term that keeps pre-edge baseline at 0
                    u0 = r0 - float(u1) * mp
                    phi2 = np.asarray(info["phi2"], dtype=float).ravel()
                    bg_adj = bg.copy()
                    bg_adj_f = bgf + float(u0) + float(u1) * (xf - xbar) + float(u2) * phi2
                    bg_adj[f] = bg_adj_f
                    adjusted[key] = (x_use, y_use, bg_adj)
                    changed += 1
                    continue
                except Exception:
                    # Fall back to the 2-parameter solution below
                    pass

            # --- Default 2-parameter equalization (baseline + jump/area) ---
            denom = (R_target * T - P)
            if (not np.isfinite(denom)) or abs(denom) < 1e-15:
                # Fallback: only baseline-zero shift
                bg_adj = bg.copy()
                bg_adj[f] = bgf + r0
                adjusted[key] = (x_use, y_use, bg_adj)
                continue

            beta = (R_target * area0 - jump0) / denom
            if not np.isfinite(beta):
                bg_adj = bg.copy()
                bg_adj[f] = bgf + r0
                adjusted[key] = (x_use, y_use, bg_adj)
                continue

            # Constant term that keeps pre-edge baseline at 0 for the chosen beta.
            c = r0 - float(beta) * mp

            bg_adj = bg.copy()
            bg_adj_f = bgf + float(beta) * (xf - xbar) + float(c)
            bg_adj[f] = bg_adj_f
            adjusted[key] = (x_use, y_use, bg_adj)
            changed += 1

        msg = None
        if changed >= 2:
            if bool(match_preedge_slope) and (m_target is not None):
                msg = (
                    f"Group BG: pre-edge=0, equalized jump after Area normalization (target={R_target:.4g}), "
                    f"matched pre-edge slope (target={m_target:.4g})"
                )
            else:
                msg = f"Group BG: pre-edge=0 and equalized jump after Area normalization (target={R_target:.4g})"

        return adjusted, msg

    def _plot_multiple_with_group_auto_bg(self):
        """Plot multiple selected curves with group automatic background fitting."""
        import numpy as np

        self.proc_ax.clear()
        self.reset_manual_mode()
        try:
            if getattr(self, "manual_bg_line", None) is not None:
                self.manual_bg_line.remove()
                self.manual_bg_line = None
        except Exception:
            pass

        keys = self._visible_processed_keys()
        if not keys:
            return

        # Compute group backgrounds; fall back to per-spectrum if group fit is not possible
        try:
            deg = int(self.combo_poly.currentText())
        except Exception:
            deg = 2
        try:
            pre = float(self.spin_preedge.value()) / 100.0
        except Exception:
            pre = 0.12

        # Determine whether background subtraction is active
        subtract = bool(getattr(self, "chk_show_without_bg", None) is not None and self.chk_show_without_bg.isChecked())

        # Read the chosen post-normalization mode (even if the widget is disabled).
        # In group mode we may want to "anticipate" area normalization when drawing BG lines.
        norm_mode = "None"
        try:
            if hasattr(self, "combo_post_norm"):
                norm_mode = str(self.combo_post_norm.currentText())
        except Exception:
            norm_mode = "None"

        # Compute per-spectrum Automatic BG (same as single-spectrum auto), and collect x/y.
        items = {}
        for key in keys:
            parts = key.split("##", 1)
            if len(parts) != 2:
                continue
            abs_path, hdf5_path = parts
            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
            y_data = self.plot_data.get(key)
            if y_data is None:
                continue
            x_data = lookup_energy(self, abs_path, parent, len(y_data))
            y_proc = processing.apply_normalization(self, abs_path, parent, y_data)
            mlen = min(len(x_data), len(y_proc))
            if mlen < 3:
                continue
            x_use = np.asarray(x_data[:mlen], dtype=float).ravel()
            y_use = np.asarray(y_proc[:mlen], dtype=float).ravel()
            bg = self._apply_automatic_bg_new(x_use, y_use, deg=deg, pre_edge_percent=pre, do_plot=False)
            bg = np.asarray(bg, dtype=float).ravel()[:mlen]
            items[key] = (x_use, y_use, bg, hdf5_path)

        if not items:
            return
        # Choose a representative x-grid for the pre-edge marker and dragging.
        # Prefer a common grid if available; otherwise fall back to the first spectrum's grid.
        try:
            # items[key] = (x_use, y_use, bg, hdf5_path)
            _x_first = next(iter(items.values()))[0]
            self._proc_last_x = np.asarray(_x_first, dtype=float)
        except Exception:
            pass

        # Optional marker: show pre-edge boundary if all curves share a common grid.
        try:
            x_common, _bgs = self._compute_group_auto_backgrounds(list(items.keys()), deg=deg, pre_edge_percent=pre)
            if x_common is not None:
                try:
                    if len(x_common) >= 2:
                        self._proc_last_x = np.asarray(x_common, dtype=float)
                except Exception:
                    self._proc_last_x = np.asarray(x_common)
        except Exception:
            pass

        # If user plans to Area-normalize after BG subtraction, equalize the *post-area-normalized jump*
        # across the selected group by adding a per-spectrum zero-area tilt term to the background.
        # This keeps the area (trapz) unchanged while correcting the step height after area normalization.
        try:
            m = str(norm_mode).strip().lower()
            equalize_jump = m in ("area", "area=1")
        except Exception:
            equalize_jump = False

        if equalize_jump and len(items) >= 2:
            # Use a coupled correction so that each spectrum has zero pre-edge baseline after BG subtraction,
            # and the jump after Area-normalization is the same across the selected group.
            try:
                simple = {k: (v[0], v[1], v[2]) for k, v in items.items()}
                _match_slope = False
                try:
                    scb = getattr(self, "chk_group_bg_slope", None)
                    if scb is not None and scb.isEnabled() and scb.isChecked():
                        _match_slope = True
                except Exception:
                    _match_slope = False

                adjusted, msg = self._group_equalize_area_jump_and_zero_preedge(
                    simple, pre=float(pre), match_preedge_slope=bool(_match_slope)
                )
                for k, (x_u, y_u, bg_u) in adjusted.items():
                    if k in items:
                        _x0, _y0, _bg0, hdf5_path = items[k]
                        items[k] = (x_u, y_u, bg_u, hdf5_path)
                if msg:
                    try:
                        self.statusBar().showMessage(msg, 6000)
                    except Exception:
                        pass
            except Exception:
                pass

        # Plot all selected curves
        for key, (x_use, y_use, bg, hdf5_path) in items.items():
            if x_use.size == 0:
                continue
            if subtract:
                yy = y_use - bg
                try:
                    from . import processing as _p
                    yy = _p._proc_safe_post_normalize(self, x_use, yy, norm_mode)
                except Exception:
                    pass
                line, = self.proc_ax.plot(x_use, yy, label=self.shorten_label(hdf5_path))
            else:
                line, = self.proc_ax.plot(x_use, y_use, label=self.shorten_label(hdf5_path))
                self.proc_ax.plot(x_use, bg, linestyle="--", linewidth=1.2, alpha=0.65, color=line.get_color(), label="_bg")
            try:
                line.dataset_key = key
            except Exception:
                pass

    def update_plot_processed(self):
        self.proc_ax.clear()
        visible_curves = sum(1 for key in self.plot_data if self.raw_visibility.get(key, False))
        mode = self.combo_bg.currentText() if hasattr(self, "combo_bg") else "None"
        try:
            self._update_group_bg_checkbox_state(int(visible_curves), str(mode))
        except Exception:
            pass

        # If the user has enabled Group BG, enforce the required settings even if
        # other UI flows (e.g. the multi-selection "no BG controls" view) had
        # previously disabled those widgets.
        try:
            gcb = getattr(self, "chk_group_bg", None)
            if gcb is not None and gcb.isEnabled() and gcb.isChecked():
                self._set_group_bg_mode(True)
                # Refresh local "mode" cache after forcing the combo.
                mode = self.combo_bg.currentText() if hasattr(self, "combo_bg") else mode
        except Exception:
            pass

        group_active = False
        try:
            if visible_curves > 1 and (not self.chk_sum.isChecked()) and str(mode) in ("Automatic", "Auto"):
                cb = getattr(self, "chk_group_bg", None)
                if cb is not None and cb.isEnabled() and cb.isChecked():
                    group_active = True
        except Exception:
            group_active = False

        if visible_curves > 1 and not self.chk_sum.isChecked() and not group_active:
            # Default behavior: multiple curves without sum -> show only normalized curves, no BG controls.
            self._toggle_bg_widgets(False)
            self._plot_multiple_no_bg()
            self.proc_ax.figure.tight_layout()
            self.canvas_proc.draw()
            self.update_pass_button_state()
            self.update_proc_tree()
            return
        else:
            self._toggle_bg_widgets(True)

        # Group automatic background mode for multiple selected curves
        if group_active:
            try:
                self._plot_multiple_with_group_auto_bg()
            except Exception as ex:
                print("Group Auto BG plotting failed:", ex)
                # Fallback to the old behavior
                self._plot_multiple_no_bg()
            try:
                self._draw_preedge_vline(getattr(self, "_proc_last_x", None))
            except Exception:
                pass
            self.proc_ax.set_xlabel("Photon energy (eV)")
            try:
                self.proc_ax.figure.tight_layout()
            except Exception:
                pass
            try:
                self.canvas_proc.draw()
            except Exception:
                pass
            self.update_pass_button_state()
            self.update_proc_tree()
            return

        main_x, main_y = self.compute_main_curve()
        if main_x is not None and main_y is not None:
            self.proc_ax.plot(main_x, main_y, label="Main curve")

        new_sum_state = self.chk_sum.isChecked()
        new_norm_state = self.chk_normalize.isChecked()
        new_len = len(main_y) if main_y is not None else 0
        mode = self.combo_bg.currentText()

        # If we previously were in Manual BG mode, ensure we fully exit manual mode
        # when switching back to Automatic/None. Otherwise the mouse handlers will
        # keep treating clicks as manual-anchor drags and block the Auto pre-edge marker.
        if mode != "Manual" and getattr(self, "manual_mode", False):
            self.reset_manual_mode()

        if mode == "Manual" and main_y is not None:
            if (not self.manual_mode or self.manual_poly_degree is None or
                (self.manual_poly_degree != int(self.combo_poly.currentText())) or
                (new_sum_state != self.last_sum_state) or
                (new_norm_state != self.last_normalize_state) or
                (new_len != self.last_plot_len)):
                self.reset_manual_mode()
                self.init_manual_mode(main_x, main_y)

        self.last_sum_state = new_sum_state
        self.last_normalize_state = new_norm_state
        self.last_plot_len = new_len

        if str(mode) in ("Automatic", "Auto") and main_x is not None and main_y is not None:
            self.spin_preedge.setEnabled(True)
            self._apply_automatic_bg_new(
                main_x, main_y,
                deg=int(self.combo_poly.currentText()),
                pre_edge_percent=float(self.spin_preedge.value())/100.0,
                ax=self.proc_ax, do_plot=True
            )
            self.reset_manual_mode()
            if self.manual_bg_line is not None:
                self.manual_bg_line.remove()
                self.manual_bg_line = None
        elif mode == "Manual" and main_x is not None and main_y is not None:
            self.spin_preedge.setEnabled(False)
            self._apply_manual_bg(main_x, main_y)
        else:
            self.spin_preedge.setEnabled(True)
            self.reset_manual_mode()
            if self.manual_bg_line is not None:
                self.manual_bg_line.remove()
                self.manual_bg_line = None

        if self.chk_show_without_bg.isChecked() and main_x is not None and main_y is not None:
            self._show_subtracted_only(mode, main_x, main_y)

        try:
            if str(mode) in ("Automatic", "Auto") and main_x is not None:
                import numpy as _np
                self._proc_last_x = _np.asarray(main_x)
                self._draw_preedge_vline(self._proc_last_x)
        except Exception:
            pass

        self.proc_ax.set_xlabel("Photon energy (eV)")
        self.proc_ax.figure.tight_layout()
        self.canvas_proc.draw()
        self.update_pass_button_state()
        self.update_proc_tree()


    # ------------ Pre-edge marker (Processed Data) ------------
    def _draw_preedge_vline(self, x):
        """Draw or update the vertical pre-edge boundary marker on the Processed plot."""
        import numpy as np
        if x is None:
            return
        try:
            x = np.asarray(x, dtype=float)
            if x.size < 2:
                return
        except Exception:
            return

        # Compute index from current pre-edge %
        try:
            pre = float(getattr(self, "spin_preedge").value()) / 100.0
        except Exception:
            pre = 0.1
        pre = max(0.001, min(0.999, pre))
        idx = int(pre * len(x))
        idx = max(1, min(idx, len(x) - 1))
        x0 = float(x[idx])

        # Remove old marker if axes got cleared
        ln = getattr(self, "proc_preedge_line", None)
        if ln is not None:
            try:
                if ln.axes is not self.proc_ax:
                    ln = None
            except Exception:
                ln = None

        if ln is None:
            try:
                ln = self.proc_ax.axvline(x0, linestyle=":", linewidth=1.2, alpha=0.8)
                # Make the line reliably pickable/draggable across backends and HiDPI.
                # Prefer using the artist's built-in hit-testing instead of pixel-space heuristics.
                try:
                    # Enable picking and enlarge pick radius (in points).
                    ln.set_picker(True)
                    if hasattr(ln, "set_pickradius"):
                        ln.set_pickradius(10)
                    else:
                        # Older API fallback
                        ln.set_picker(10)
                except Exception:
                    pass
                self.proc_preedge_line = ln
            except Exception:
                return
        else:
            try:
                ln.set_xdata([x0, x0])
            except Exception:
                pass

    def _on_preedge_vline_press(self, event):
        """Start dragging the pre-edge marker (Auto BG only).

        Uses the artist's hit-testing (Line2D.contains) which is robust across
        backends and HiDPI scaling.
        """
        import numpy as np
        try:
            # Only for Automatic background mode
            if getattr(self, "combo_bg", None) is None:
                return
            if self.combo_bg.currentText() != "Automatic":
                return
            if event is None:
                return
            if getattr(event, "button", None) not in (1,):
                return
            # Prefer not to hard-require event.inaxes: on some backends/HiDPI
            # configurations the press can arrive with inaxes=None even though
            # the click was visually on the axes.
            if getattr(self, "proc_ax", None) is None:
                return

            ln = getattr(self, "proc_preedge_line", None)
            if ln is None:
                return

            # Prefer robust artist hit-testing (accounts for HiDPI/backend details)
            try:
                contains, _info = ln.contains(event)
                if bool(contains):
                    self._dragging_preedge = True
                    return
            except Exception:
                pass

            # Fallback: data/pixel-space tolerance (in case contains() fails)
            try:
                x0 = float(ln.get_xdata()[0])
                ax = self.proc_ax
                xlim = ax.get_xlim()
                xr = abs(float(xlim[1]) - float(xlim[0]))
                # generous tolerance: 3% of the visible range, but at least a few grid steps
                tol = max(0.03 * xr, 1e-6)
                x = getattr(self, "_proc_last_x", None)
                if x is not None:
                    x = np.asarray(x, dtype=float)
                    if x.size >= 2:
                        dx = np.nanmedian(np.diff(x))
                        if np.isfinite(dx) and dx > 0:
                            tol = max(tol, 3.0 * float(dx))
                xevt = None
                if getattr(event, "xdata", None) is not None and np.isfinite(event.xdata):
                    xevt = float(event.xdata)
                elif getattr(event, "x", None) is not None:
                    # Convert display x (pixels) to data x
                    try:
                        y0, y1 = ax.get_ylim()
                        ymid = 0.5 * (float(y0) + float(y1))
                        ydisp = float(ax.transData.transform((0.0, ymid))[1])
                        xevt = float(ax.transData.inverted().transform((float(event.x), ydisp))[0])
                    except Exception:
                        xevt = None
                if xevt is None:
                    return
                if abs(float(xevt) - x0) <= tol:
                    self._dragging_preedge = True
            except Exception:
                return
        except Exception:
            return

    
    def _on_preedge_vline_pick(self, event):
        """Start dragging the pre-edge marker when the marker line is picked."""
        try:
            if getattr(self, "combo_bg").currentText() != "Automatic":
                return
            ln = getattr(self, "proc_preedge_line", None)
            if ln is None:
                return
            if getattr(event, "artist", None) is not ln:
                return
            # Only left mouse button
            me = getattr(event, "mouseevent", None)
            if me is not None and getattr(me, "button", None) not in (1,):
                return
            self._dragging_preedge = True
        except Exception:
            return

    def _on_preedge_vline_motion(self, event):
        """Drag handler for the pre-edge marker."""
        import numpy as np
        if not getattr(self, "_dragging_preedge", False):
            return
        try:
            ax = getattr(self, "proc_ax", None)
            if ax is None:
                return
            # event.xdata can be None on some backend/HiDPI combos; fall back to pixel->data conversion.
            x_new = None
            if getattr(event, "xdata", None) is not None and np.isfinite(event.xdata):
                x_new = float(event.xdata)
            elif getattr(event, "x", None) is not None:
                try:
                    y0, y1 = ax.get_ylim()
                    ymid = 0.5 * (float(y0) + float(y1))
                    # Get display y at the midpoint, then invert transform for the mouse x.
                    ydisp = float(ax.transData.transform((0.0, ymid))[1])
                    x_new = float(ax.transData.inverted().transform((float(event.x), ydisp))[0])
                except Exception:
                    x_new = None
            if x_new is None:
                return
            x = getattr(self, "_proc_last_x", None)
            if x is None:
                return
            x = np.asarray(x, dtype=float)
            if x.size < 2:
                return
            # Find nearest index
            idx = int(np.argmin(np.abs(x - x_new)))
            idx = max(1, min(idx, len(x) - 1))
            pre_pct = 100.0 * (idx / float(len(x)))
            pre_pct = max(0.1, min(99.9, pre_pct))
            spin = getattr(self, "spin_preedge", None)
            if spin is None:
                return
            # Avoid tiny oscillations
            if abs(float(spin.value()) - pre_pct) >= 0.1:
                spin.setValue(pre_pct)
            # Move marker immediately for responsiveness
            try:
                ln = getattr(self, "proc_preedge_line", None)
                if ln is not None:
                    ln.set_xdata([float(x[idx]), float(x[idx])])
                    self.canvas_proc.draw_idle()
            except Exception:
                pass
        except Exception:
            return

    def _on_preedge_vline_release(self, event):
        """Stop dragging the pre-edge marker."""
        try:
            self._dragging_preedge = False
        except Exception:
            pass

    # ------------ Misc helpers ------------
    def _toggle_bg_widgets(self, enabled: bool):
        """Enable/disable BG-related widgets for the Processed Data panel.

        Note: in **Group BG** mode, BG mode (combo_bg) and BG subtraction
        (chk_show_without_bg) are intentionally locked by :meth:`_set_group_bg_mode`.
        This helper respects that lock so that other UI flows (e.g. multiple-selection
        hiding) don't inadvertently re-enable those controls.
        """
        group_locked = False
        try:
            gcb = getattr(self, "chk_group_bg", None)
            group_locked = bool(gcb is not None and gcb.isChecked())
        except Exception:
            group_locked = False

        try:
            self.combo_bg.setEnabled(bool(enabled) and (not group_locked))
        except Exception:
            pass
        try:
            self.combo_poly.setEnabled(bool(enabled))
        except Exception:
            pass
        try:
            self.spin_preedge.setEnabled(bool(enabled))
        except Exception:
            pass
        try:
            self.chk_show_without_bg.setEnabled(bool(enabled))  # allow toggling even in Group BG mode
        except Exception:
            pass

    def _lookup_energy(self, abs_path, parent, length):
        if length < 1:
            return np.array([])
        if abs_path not in self.hdf5_files:
            return np.arange(length)

        cache_key = f"{abs_path}##{parent}"
        if cache_key in self.energy_cache:
            x_data, _ = self.energy_cache[cache_key]
            return x_data if x_data is not None else np.arange(length)

        x_data = None
        try:
            with self._open_h5_read(abs_path) as f:
                if parent:
                    pcap = f"{parent}/pcap_energy_av"
                    mono = f"{parent}/mono_traj_energy"
                    if pcap in f:
                        x_data = f[pcap][()]
                    elif mono in f:
                        x_data = f[mono][()]
        except Exception:
            x_data = None

        if x_data is None:
            x_data = np.arange(length)
        self.energy_cache[cache_key] = (x_data, False)
        return x_data

    def _apply_normalization(self, abs_path, parent, y_data):
        """Divide by the chosen I0 channel – open file briefly (non-locking)."""
        if self.chk_normalize.isChecked():
            norm_channel = self.combo_norm.currentText()
            norm_path = f"{parent}/{norm_channel}" if parent else norm_channel
            try:
                with self._open_h5_read(abs_path) as f:
                    if norm_path in f:
                        try:
                            norm_data = f[norm_path][()]
                            if getattr(norm_data, "size", 0) == 0 or getattr(y_data, "size", 0) == 0:
                                return y_data.copy()
                            safe = np.divide(
                                y_data,
                                norm_data,
                                out=np.zeros_like(y_data, dtype=float),
                                where=norm_data != 0
                            )
                            return safe
                        except Exception as ex:
                            print("Normalisation error:", ex)
            except Exception:
                pass
        return y_data

    def _apply_automatic_bg_new(self, main_x, main_y, deg=None, pre_edge_percent=None, ax=None, do_plot=True):
        # (unchanged algorithm, but made robust and self-contained)
        import numpy as np
        if deg is None:
            try:
                deg = int(self.combo_poly.currentText())
            except Exception:
                try:
                    deg = int(self.bg_poly_degree_spinbox.value())
                except Exception:
                    deg = 2
        if pre_edge_percent is None:
            try:
                pre_edge_percent = float(self.spin_preedge.value()) / 100.0
            except Exception:
                try:
                    pre_edge_percent = float(self.pre_edge_spinbox.value()) / 100.0
                except Exception:
                    pre_edge_percent = 0.20
        if ax is None:
            for name in ("proc_ax", "ax_processed", "ax_proc", "ax2", "ax"):
                if hasattr(self, name):
                    cand = getattr(self, name)
                    if hasattr(cand, "plot"):
                        ax = cand
                        break

        main_x = np.asarray(main_x, dtype=float).ravel()
        main_y = np.asarray(main_y, dtype=float).ravel()
        N = len(main_x)
        if N == 0 or N != len(main_y):
            return np.zeros_like(main_y, dtype=float)

        finite = np.isfinite(main_x) & np.isfinite(main_y)
        x_all = main_x[finite]
        y_all = main_y[finite]
        Nf = len(x_all)
        if Nf < 3:
            background = np.zeros_like(main_y, dtype=float)
            if N > 0 and np.isfinite(main_y[0]):
                background[0] = main_y[0]
            if do_plot and ax is not None:
                if hasattr(self, "_auto_bg_vline") and getattr(self._auto_bg_vline, "axes", None) is ax:
                    try:
                        self._auto_bg_vline.remove()
                    except Exception:
                        pass
                self._auto_bg_vline = None
                if hasattr(self, "_auto_bg_line") and getattr(self._auto_bg_line, "axes", None) is ax:
                    try:
                        self._auto_bg_line.remove()
                    except Exception:
                        pass
                self._auto_bg_line = None
            return background

        idx_end_finite = max(1, int(pre_edge_percent * Nf))
        idx_end_finite = min(idx_end_finite, Nf - 1)
        M = idx_end_finite

        deg_eff = int(max(0, deg))
        deg_eff = min(deg_eff, max(0, min(M, Nf) - 1))

        x_min = float(x_all[0]); x_max = float(x_all[-1])
        span = x_max - x_min
        if span <= 0:
            span = 1.0

        x_prime = 2.0 * (x_all - x_min) / span - 1.0
        x_pre = x_prime[:M]; y_pre = y_all[:M]
        x_end_prime = x_prime[-1]

        i_start = max(M + 1, int(0.95 * Nf))
        if i_start >= Nf - 1:
            i_start = max(Nf - 2, 0)
        x_tail = x_all[i_start:]; y_tail = y_all[i_start:]
        if len(x_tail) >= 2:
            slope_final, _ = np.polyfit(x_tail, y_tail, 1)
        else:
            slope_final = 0.0
        slope_prime = slope_final * (span / 2.0)

        A = np.zeros((M, deg_eff + 1), dtype=float)
        for i in range(M):
            A[i, :] = np.array([x_pre[i] ** (deg_eff - k) for k in range(deg_eff + 1)], dtype=float)
        b = y_pre.copy()

        deriv_row = np.zeros(deg_eff + 1, dtype=float)
        for k in range(deg_eff + 1):
            power = deg_eff - k
            if power > 0:
                deriv_row[k] = power * (x_end_prime ** (power - 1))

        row_norms = np.linalg.norm(A, axis=1)
        w = np.nanmedian(row_norms) if row_norms.size else 1.0
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        A2 = np.vstack([A, w * deriv_row[None, :]])
        b2 = np.concatenate([b, [w * slope_prime]])

        try:
            coeffs, *_ = np.linalg.lstsq(A2, b2, rcond=None)
            if not np.all(np.isfinite(coeffs)):
                raise np.linalg.LinAlgError("Non-finite coefficients")
            x_prime_full = 2.0 * (main_x - x_min) / span - 1.0
            background = np.zeros_like(main_y, dtype=float)
            for i in range(N):
                xx = x_prime_full[i]
                val = 0.0
                for kk in range(deg_eff + 1):
                    power = deg_eff - kk
                    val += coeffs[kk] * (xx ** power)
                background[i] = val
        except Exception as ex:
            print("Automatic BG (constrained) failed -> fallback:", ex)
            idx_end_plot = max(1, int(pre_edge_percent * N))
            idx_end_plot = min(idx_end_plot, N - 1)
            x_fit = main_x[:idx_end_plot]; y_fit = main_y[:idx_end_plot]
            msk = np.isfinite(x_fit) & np.isfinite(y_fit)
            x_fit = x_fit[msk]; y_fit = y_fit[msk]
            if len(x_fit) == 0:
                background = np.zeros_like(main_y)
            else:
                x0 = float(x_fit[0]); x1 = float(x_fit[-1])
                span_fb = max(1.0, x1 - x0)
                xp = 2.0 * (x_fit - x0) / span_fb - 1.0
                deg_fb = min(int(deg), max(0, len(xp) - 1))
                p = np.polyfit(xp, y_fit, deg_fb)
                xp_full = 2.0 * (main_x - x0) / span_fb - 1.0
                background = np.polyval(p, xp_full)

        if np.isfinite(main_y[0]):
            background[0] = main_y[0]

        if do_plot and ax is not None:
            idx_end_plot = max(1, int(pre_edge_percent * N))
            idx_end_plot = min(idx_end_plot, N - 1)

            def _alive(artist):
                return (artist is not None) and (getattr(artist, "axes", None) is ax)

            if not hasattr(self, "_auto_bg_vline"): self._auto_bg_vline = None
            if not hasattr(self, "_auto_bg_line"): self._auto_bg_line = None

            if not _alive(self._auto_bg_vline):
                try:
                    self._auto_bg_vline = ax.axvline(main_x[idx_end_plot], linestyle="--", linewidth=1.0, alpha=0.6, label="_auto_preedge")
                except Exception:
                    self._auto_bg_vline = None
            else:
                try:
                    x0 = main_x[idx_end_plot]
                    self._auto_bg_vline.set_xdata([x0, x0])
                except Exception:
                    self._auto_bg_vline = None

            if not _alive(self._auto_bg_line):
                try:
                    (self._auto_bg_line,) = ax.plot(main_x, background, linestyle="--", linewidth=1.5, label="_auto_bg")
                except Exception:
                    self._auto_bg_line = None
            else:
                try:
                    self._auto_bg_line.set_data(main_x, background)
                except Exception:
                    self._auto_bg_line = None
            try:
                ax.figure.canvas.draw_idle()
            except Exception:
                pass

        return background

    def change_curve_color(self, key, new_color):
        if key in self.plotted_lines:
            self.plotted_lines[key].set_color(new_color)
            self.update_legend()

    def change_curve_visibility(self, key, visible):
        if key in self.plotted_lines:
            self.plotted_lines[key].set_visible(visible)
            self.update_legend()
            self.rescale_plotted_axes()

    def change_curve_style(self, key, style, size):
        if key in self.plotted_lines:
            line = self.plotted_lines[key]
            if style == "Solid":
                line.set_linestyle("-")
                line.set_marker("")
                line.set_linewidth(size)
            elif style == "Dashed":
                line.set_linestyle("--")
                line.set_marker("")
                line.set_linewidth(size)
            elif style == "Scatter":
                line.set_linestyle("None")
                line.set_marker("o")
                line.set_markersize(size)
            self.update_legend()
            self.canvas_plotted.draw()

    def on_curve_remove_requested(self, key):
        """Handle request to remove a plotted curve from list and axes.

        Shows a confirmation dialog and, on acceptance, removes the
        corresponding line and updates Waterfall / legend layout.
        """
        # Confirm with the user
        reply = QMessageBox.question(
            self,
            "Remove curve",
            "Do you want to remove this curve?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Ok:
            return

        # Remove from stored structures
        line = self.plotted_lines.pop(key, None) if hasattr(self, "plotted_lines") else None
        if line is not None:
            try:
                line.remove()
            except Exception:
                pass

        if hasattr(self, "plotted_curves"):
            try:
                self.plotted_curves.discard(key)
            except Exception:
                pass

        if hasattr(self, "original_line_data"):
            try:
                self.original_line_data.pop(key, None)
            except Exception:
                pass

        if hasattr(self, "custom_labels"):
            try:
                self.custom_labels.pop(key, None)
            except Exception:
                pass

        # Remove the corresponding list item
        if hasattr(self, "plotted_list"):
            try:
                for i in range(self.plotted_list.count()):
                    item = self.plotted_list.item(i)
                    widget = self.plotted_list.itemWidget(item)
                    if widget and getattr(widget, "key", None) == key:
                        self.plotted_list.takeItem(i)
                        break
            except Exception:
                pass

        # Recompute Waterfall layout (which also rescales axes)
        try:
            if hasattr(self, "recompute_waterfall_layout"):
                self.recompute_waterfall_layout()
            else:
                # Fallback: at least rescale axes and redraw
                if hasattr(self, "rescale_plotted_axes"):
                    self.rescale_plotted_axes()
                if hasattr(self, "canvas_plotted"):
                    self.canvas_plotted.draw()
        except Exception:
            pass

        # Update legend to reflect remaining curves
        try:
            if hasattr(self, "update_legend"):
                self.update_legend()
        except Exception:
            pass

    def rescale_plotted_axes(self):
        x_all, y_all = [], []
        for key, line in self.plotted_lines.items():
            if line.get_visible():
                x = line.get_xdata(); y = line.get_ydata()
                if x is not None and len(x):
                    x_all.extend(x)
                if y is not None and len(y):
                    y_all.extend(y)
        if x_all and y_all:
            xmin, xmax = min(x_all), max(x_all)
            ymin, ymax = min(y_all), max(y_all)
            x_margin = (xmax - xmin) * 0.05 if (xmax - xmin) else 1
            y_margin = (ymax - ymin) * 0.05 if (ymax - ymin) else 1
            self.plotted_ax.set_xlim(xmin - x_margin, xmax + x_margin)
            self.plotted_ax.set_ylim(ymin - y_margin, ymax + y_margin)
            # Re-apply grid mode after rescaling axes
            try:
                self._apply_grid_mode()
            except Exception:
                pass
            self.canvas_plotted_fig.tight_layout()
            self.canvas_plotted.draw()

    # ------------ Right-panel trees (raw/proc) ------------
    def group_datasets(self):
        groups = []
        for key in getattr(self, "plot_data", {}).keys():
            parts = key.split("##", 1)
            if len(parts) != 2:
                continue
            abs_path, hdf5_path = parts
            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
            y_data = self.plot_data[key]
            x_data = getattr(self, "_lookup_energy", lambda *a, **k: np.arange(len(y_data)))(abs_path, parent, len(y_data))
            if getattr(x_data, "size", 0) == 0:
                continue
            min_x = float(np.min(x_data))
            max_x = float(np.max(x_data))
            groups.append((key, min_x, max_x))
        groups.sort(key=lambda t: t[1])
        merged = []
        for key, min_x, max_x in groups:
            if not merged:
                merged.append({'keys': [key], 'min': min_x, 'max': max_x})
            else:
                last = merged[-1]
                if min_x <= last['max']:
                    last['keys'].append(key)
                    last['max'] = max(last['max'], max_x)
                else:
                    merged.append({'keys': [key], 'min': min_x, 'max': max_x})
        return merged

    def update_raw_tree(self):
        # Refresh 'All in channel' combobox if present
        try:
            self._refresh_all_in_channel_combo()
        except Exception:
            pass
        if not hasattr(self, "raw_tree"):
            return
        self.raw_tree.blockSignals(True)
        try:
            self.raw_tree.clear()
            groups = self.group_datasets()
            for idx, group in enumerate(groups):
                region_id = f"region_{idx}"
                region_state = getattr(self, "region_states", {}).get(region_id, Qt.Checked)
                region_item = QTreeWidgetItem([f"Region {idx+1}"])
                region_item.setFlags(region_item.flags() | Qt.ItemIsUserCheckable)
                region_item.setCheckState(0, region_state)
                region_item.setData(0, Qt.UserRole+1, region_id)
                sorted_keys = sorted(group['keys'], key=lambda x: parse_entry_number(x.split("##",1)[1] if "##" in x else ""))
                for key in sorted_keys:
                    parts = key.split("##", 1)
                    label = self.shorten_label(parts[1]) if len(parts) == 2 else key
                    child = QTreeWidgetItem([label])
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    child_state = Qt.Checked if getattr(self, "raw_visibility", {}).get(key, True) else Qt.Unchecked
                    child.setCheckState(0, child_state)
                    child.setData(0, Qt.UserRole, key)
                    color = None
                    try:
                        color = self.get_line_color_for_key(key, self.raw_ax)
                    except Exception:
                        pass
                    if color:
                        pixmap = QPixmap(16, 16); pixmap.fill(QColor(color))
                        child.setIcon(0, QIcon(pixmap))
                    region_item.addChild(child)
                self.raw_tree.addTopLevelItem(region_item)
                region_item.setExpanded(True)
                child_count = region_item.childCount()
                checked_count = sum(region_item.child(i).checkState(0) == Qt.Checked for i in range(child_count))
                if checked_count == 0:
                    region_item.setCheckState(0, Qt.Unchecked)
                elif checked_count == child_count:
                    region_item.setCheckState(0, Qt.Checked)
                else:
                    region_item.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self.raw_tree.blockSignals(False)
            self.raw_tree.update()
            setattr(self, "raw_tree_reset", False)

    def raw_tree_item_changed(self, item, column):
        if not hasattr(self, "raw_tree"):
            return
        self.raw_tree.blockSignals(True)
        try:
            if item.parent() is None:
                state = item.checkState(0)
                region_id = item.data(0, Qt.UserRole+1)
                if region_id is not None:
                    self.region_states[region_id] = state
                for i in range(item.childCount()):
                    child = item.child(i)
                    child.setCheckState(0, state)
                    key = child.data(0, Qt.UserRole)
                    self.raw_visibility[key] = (state == Qt.Checked)
            else:
                key = item.data(0, Qt.UserRole)
                self.raw_visibility[key] = (item.checkState(0) == Qt.Checked)
                parent_item = item.parent()
                child_count = parent_item.childCount()
                checked_count = sum(parent_item.child(i).checkState(0) == Qt.Checked for i in range(child_count))
                if checked_count == 0:
                    parent_item.setCheckState(0, Qt.Unchecked)
                elif checked_count == child_count:
                    parent_item.setCheckState(0, Qt.Checked)
                else:
                    parent_item.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self.raw_tree.blockSignals(False)
        QTimer.singleShot(0, self.update_plot_raw)

    def update_proc_tree(self):
        if not hasattr(self, "proc_tree"):
            return
        self.proc_tree.blockSignals(True)
        try:
            self.proc_tree.clear()
            groups = self.group_datasets()
            for idx, group in enumerate(groups):
                region_id = f"proc_region_{idx}"
                region_item = QTreeWidgetItem([f"Region {idx+1}"])
                region_item.setFlags(region_item.flags() | Qt.ItemIsUserCheckable)
                region_item.setCheckState(0, self.proc_region_states.get(region_id, Qt.Checked))
                region_item.setData(0, Qt.UserRole+1, region_id)
                sorted_keys = sorted(group['keys'], key=lambda x: parse_entry_number(x.split("##",1)[1] if "##" in x else ""))
                for key in sorted_keys:
                    parts = key.split("##", 1)
                    label = self.shorten_label(parts[1]) if len(parts) == 2 else key
                    child = QTreeWidgetItem([label])
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    child_state = Qt.Checked if self.raw_visibility.get(key, True) else Qt.Unchecked
                    child.setCheckState(0, child_state)
                    child.setData(0, Qt.UserRole, key)
                    color = None
                    try:
                        color = self.get_line_color_for_key(key, self.proc_ax)
                    except Exception:
                        pass
                    if color:
                        pixmap = QPixmap(16, 16); pixmap.fill(QColor(color))
                        child.setIcon(0, QIcon(pixmap))
                    region_item.addChild(child)
                self.proc_tree.addTopLevelItem(region_item)
                region_item.setExpanded(True)
                child_count = region_item.childCount()
                checked_count = sum(region_item.child(i).checkState(0) == Qt.Checked for i in range(child_count))
                if checked_count == 0:
                    region_item.setCheckState(0, Qt.Unchecked)
                elif checked_count == child_count:
                    region_item.setCheckState(0, Qt.Checked)
                else:
                    region_item.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self.proc_tree.blockSignals(False)
            self.proc_tree.update()

    def proc_tree_item_changed(self, item, column):
        if not hasattr(self, "proc_tree"):
            return
        self.proc_tree.blockSignals(True)
        try:
            if item.parent() is None:
                state = item.checkState(0)
                region_id = item.data(0, Qt.UserRole+1)
                if region_id is not None:
                    self.proc_region_states[region_id] = state
                for i in range(item.childCount()):
                    child = item.child(i)
                    child.setCheckState(0, state)
                    key = child.data(0, Qt.UserRole)
                    self.raw_visibility[key] = (state == Qt.Checked)
            else:
                key = item.data(0, Qt.UserRole)
                self.raw_visibility[key] = (item.checkState(0) == Qt.Checked)
                parent_item = item.parent()
                child_count = parent_item.childCount()
                checked_count = sum(parent_item.child(i).checkState(0) == Qt.Checked for i in range(child_count))
                if checked_count == 0:
                    parent_item.setCheckState(0, Qt.Unchecked)
                elif checked_count == child_count:
                    parent_item.setCheckState(0, Qt.Checked)
                else:
                    parent_item.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self.proc_tree.blockSignals(False)
        QTimer.singleShot(0, self.update_plot_processed)

    # ------------ Compatibility ------------
    def setGeometry(self, *args, **kwargs):
        """Delegate to the underlying Qt widget implementation.
    
        A previous no-op stub here prevented QMainWindow.setGeometry() from
        taking effect due to MRO (PlottingMixin appears before QMainWindow).
        """
        try:
            return super().setGeometry(*args, **kwargs)
        except Exception:
            return None
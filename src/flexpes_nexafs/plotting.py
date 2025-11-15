from flexpes_nexafs.utils.sorting import parse_entry_number
from .data import lookup_energy
# Auto-generated/maintained PlottingMixin (post-split)

import os
import sys
import time
import csv
import h5py
import numpy as np
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
    QSpinBox, QMessageBox, QSizePolicy, QDialog, QListWidgetItem, QInputDialog, QTextBrowser
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
    Handles # headers, lists, code spans, **bold**, *italic*.
    Prefer 'markdown' module if available.
    """
    import html, re
    text = html.escape(md)

    # inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # bold / italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

    lines = []
    in_ul = False
    in_ol = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            if in_ul:
                lines.append("</ul>"); in_ul = False
            if in_ol:
                lines.append("</ol>"); in_ol = False
            lines.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            if in_ul:
                lines.append("</ul>"); in_ul = False
            if in_ol:
                lines.append("</ol>"); in_ol = False
            lines.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("### "):
            if in_ul:
                lines.append("</ul>"); in_ul = False
            if in_ol:
                lines.append("</ol>"); in_ol = False
            lines.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_ul:
                if in_ol:
                    lines.append("</ol>"); in_ol = False
                lines.append("<ul>"); in_ul = True
            lines.append(f"<li>{line[2:].strip()}</li>")
        elif any(line.startswith(f"{n}. ") for n in range(1, 10)):
            if not in_ol:
                if in_ul:
                    lines.append("</ul>"); in_ul = False
                lines.append("<ol>"); in_ol = True
            item_text = line[line.find('.')+1:].strip()
            lines.append(f"<li>{item_text}</li>")
        elif line == "":
            lines.append("<br>")
        else:
            if in_ul:
                lines.append("</ul>"); in_ul = False
            if in_ol:
                lines.append("</ol>"); in_ol = False
            lines.append(f"<p>{line}</p>")
    if in_ul: lines.append("</ul>")
    if in_ol: lines.append("</ol>")
    return "\n".join(lines)

def get_usage_html() -> str:
    """Return the Help->Usage content as HTML from docs/help.md."""
    md = _read_help_markdown()
    if not md:
        return "<p><b>Help file not found.</b></p>"
    try:
        import markdown
        return markdown.markdown(md, extensions=["tables","fenced_code","sane_lists"])  # type: ignore
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
        """Apply/update Waterfall offsets according to selected mode."""
        if not getattr(self, "plotted_lines", None):
            return

        try:
            mode = self.waterfall_mode_combo.currentText()
        except Exception:
            mode = "Adaptive step" if getattr(self, "waterfall_checkbox", None) and self.waterfall_checkbox.isChecked() else "None"

        # Restore originals first
        try:
            self.restore_original_line_data()
        except Exception:
            pass

        # Build ordered visible keys
        plotted_keys_in_order = []
        if hasattr(self, "plotted_list"):
            for i in range(self.plotted_list.count()):
                item = self.plotted_list.item(i)
                widget = self.plotted_list.itemWidget(item)
                if widget:
                    key = getattr(widget, "key", None)
                    line = self.plotted_lines.get(key) if key else None
                    if line is not None and line.get_visible():
                        plotted_keys_in_order.append(key)

        if not plotted_keys_in_order or mode == "None":
            self.rescale_plotted_axes()
            return

        if mode == "Adaptive step":
            alpha = float(self.waterfall_spin.value())
            prev_max = None
            for key in plotted_keys_in_order:
                line = self.plotted_lines[key]
                xdata = np.asarray(line.get_xdata())
                ydata = np.asarray(line.get_ydata(), dtype=float)
                if xdata.size < 1 or ydata.size < 1:
                    continue
                if prev_max is None:
                    mfin0 = np.isfinite(ydata)
                    prev_max = float(np.max(ydata[mfin0])) if np.any(mfin0) else 0.0
                    continue
                mfin1 = np.isfinite(ydata)
                y0 = float(ydata[mfin1][0]) if np.any(mfin1) else 0.0
                shift_full = (prev_max * 1.02) - y0
                shift_final = alpha * shift_full
                new_y = ydata + shift_final
                line.set_ydata(new_y)
                mfin2 = np.isfinite(new_y)
                if np.any(mfin2):
                    prev_max = float(np.max(new_y[mfin2]))

        elif mode == "Uniform step":
            k = float(self.waterfall_spin.value())  # 0..1
            # Compute global y-range from restored originals
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
            if not np.isfinite(global_range) or global_range <= 1e-15:
                self.rescale_plotted_axes()
                return
            delta = k * global_range
            for idx, key in enumerate(plotted_keys_in_order):
                if idx == 0:
                    continue
                line = self.plotted_lines[key]
                y = np.asarray(line.get_ydata(), dtype=float)
                line.set_ydata(y + idx * delta)

        # Finally rescale/redraw
        self.rescale_plotted_axes()

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
                QMessageBox.warning(self, "Warning", "Please select exactly one dataset, or enable 'Sum'.")
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

        bg_subtracted = False
        if self.chk_show_without_bg.isChecked():
            background = self._compute_background(main_x, main_y)
            subtracted = main_y - background
            norm_mode = self.combo_post_norm.currentText() if self.combo_post_norm.isEnabled() else "None"
            subtracted = processing._proc_safe_post_normalize(self, main_x, subtracted, norm_mode)
            main_y = subtracted
            bg_subtracted = True

        parts = key.split("##", 1)
        if self.chk_sum.isChecked():
            origin_label = "Summed Curve"
        else:
            if len(parts) == 2:
                path = parts[1]
                # Typical path looks like "entryXXXX/measurement/<channel_name>".
                # We want to drop the redundant "measurement" and show
                # "entryXXXX: <channel_name>" in the Plotted Data list.
                path_parts = path.split("/")
                if len(path_parts) >= 3 and path_parts[1] == "measurement":
                    entry = path_parts[0]
                    channel_name = path_parts[-1]
                    origin_label = f"{entry}: {channel_name}"
                else:
                    # Fallback: use the original path if it does not match the
                    # expected pattern.
                    origin_label = path
            else:
                origin_label = key

        metadata_parts = []
        if bg_subtracted and self.combo_post_norm.isEnabled():
            nm = self.combo_post_norm.currentText().lower()
            if nm != "none":
                metadata_parts.append(nm)
        if self.chk_normalize.isChecked():
            metadata_parts.append("normalized")
        if self.chk_sum.isChecked():
            metadata_parts.append("summed")
        if bg_subtracted:
            metadata_parts.append("bg subtracted")
        if metadata_parts:
            origin_label += " (" + ", ".join(metadata_parts) + ")"

        self.custom_labels[storage_key] = None
        line, = self.plotted_ax.plot(main_x, main_y, label="<select curve name>")
        self.plotted_curves.add(storage_key)
        self.plotted_lines[storage_key] = line

        # Store original data so we can revert it for Waterfall
        self.original_line_data[storage_key] = (np.asarray(main_x).copy(), np.asarray(main_y).copy())

        item = QListWidgetItem()
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
        (~5× finer)."""
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
            try:
                ax.minorticks_off()
            except Exception:
                pass
        else:
            # Always show major grid lines for any non-'None' mode
            try:
                ax.grid(True, which='major')
            except Exception:
                pass

            # For 'Coarse' we keep only major grid (original behavior)
            if mode == 'coarse':
                try:
                    ax.minorticks_off()
                except Exception:
                    pass
            else:
                # 'Fine' and 'Finest': add minor grid with more divisions
                n = 2 if mode == 'fine' else 5
                try:
                    ax.minorticks_on()
                    ax.xaxis.set_minor_locator(AutoMinorLocator(n))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(n))
                    ax.grid(True, which='both')
                except Exception:
                    pass

        # Redraw canvas if available
        try:
            if hasattr(self, 'canvas_plotted'):
                self.canvas_plotted.draw()
        except Exception:
            pass

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
        self.waterfall_mode_combo.setCurrentIndex(0)
        self.waterfall_slider.setValue(0)
        self.waterfall_spin.setValue(0.00)

        # Reset Grid option to 'None'
        try:
            if hasattr(self, "grid_mode_combo"):
                self.grid_mode_combo.setCurrentText("None")
            self._apply_grid_mode("None")
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
        if hasattr(event.artist, "get_text"):
            old_text = event.artist.get_text()
            new_text, ok = QInputDialog.getText(self, "Rename Curve", "New legend name:", text=old_text)
            if ok and new_text:
                for key, line in self.plotted_lines.items():
                    current_label = self.custom_labels.get(key)
                    if (current_label is None and old_text == "<select curve name>") or (current_label == old_text):
                        self.custom_labels[key] = new_text
                        break
                self.update_legend()

    def update_legend(self):
        visible_lines = []
        labels = []
        for key, line in self.plotted_lines.items():
            if line.get_visible():
                visible_lines.append(line)
                label = self.custom_labels.get(key) or "<select curve name>"
                line.set_label(label)
                labels.append(label)
        leg = self.plotted_ax.legend(visible_lines, labels)
        if leg:
            leg.set_draggable(True)
            for text in leg.get_texts():
                text.set_picker(True)
        self.canvas_plotted_fig.tight_layout()
        self.canvas_plotted.draw()

    def visible_curves_count(self):
        return sum(1 for _k, visible in getattr(self, "raw_visibility", {}).items() if visible)

    def update_pass_button_state(self):
        try:
            cond = bool(getattr(self, "chk_sum", None) and self.chk_sum.isChecked())
            vc = self.visible_curves_count()
            if getattr(self, "pass_button", None):
                self.pass_button.setEnabled(bool(cond or vc == 1))
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
        dlg.resize(800, 600)
        dlg.setSizeGripEnabled(True)  # optional: shows a size grip in the corner

        layout = QVBoxLayout(dlg)
        browser = HelpBrowser()
        
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser.setLineWrapMode(QTextEdit.WidgetWidth)
        browser.setWordWrapMode(QTextOption.WordWrap)
        
        browser.setStyleSheet("font-size: 14px;")
        # QTextBrowser expects HTML; get_usage_html() already returns HTML
        browser.setHtml(usage_html)
        layout.addWidget(browser)
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

        for default in ["b107a_em_03_ch2", "b107a_em_04_ch2", "Pt_No"]:
            idx = self.combo_norm.findText(default)
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

    def update_plot_processed(self):
        self.proc_ax.clear()
        visible_curves = sum(1 for key in self.plot_data if self.raw_visibility.get(key, False))
        if visible_curves > 1 and not self.chk_sum.isChecked():
            self._toggle_bg_widgets(False)
            self._plot_multiple_no_bg()
            self.proc_ax.figure.tight_layout()
            self.canvas_proc.draw()
            self.update_pass_button_state()
            self.update_proc_tree()
            return
        else:
            self._toggle_bg_widgets(True)

        main_x, main_y = self.compute_main_curve()
        if main_x is not None and main_y is not None:
            self.proc_ax.plot(main_x, main_y, label="Main curve")

        new_sum_state = self.chk_sum.isChecked()
        new_norm_state = self.chk_normalize.isChecked()
        new_len = len(main_y) if main_y is not None else 0
        mode = self.combo_bg.currentText()

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

        if mode == "Automatic" and main_x is not None and main_y is not None:
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
            processing.apply_manual_bg(self, main_x, main_y)
        else:
            self.spin_preedge.setEnabled(True)
            self.reset_manual_mode()
            if self.manual_bg_line is not None:
                self.manual_bg_line.remove()
                self.manual_bg_line = None

        if self.chk_show_without_bg.isChecked() and main_x is not None and main_y is not None:
            self._show_subtracted_only(mode, main_x, main_y)

        self.proc_ax.set_xlabel("Photon energy (eV)")
        self.proc_ax.figure.tight_layout()
        self.canvas_proc.draw()
        self.update_pass_button_state()
        self.update_proc_tree()

    # ------------ Misc helpers ------------
    def _toggle_bg_widgets(self, enabled: bool):
        self.combo_bg.setEnabled(enabled)
        self.combo_poly.setEnabled(enabled)
        self.spin_preedge.setEnabled(enabled)
        self.chk_show_without_bg.setEnabled(enabled)

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
        """No-op stub to avoid AttributeError if viewer is used like a QWidget."""
        try:
            return None
        except Exception:
            return None
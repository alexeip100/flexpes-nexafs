"""Auto-generated ExportMixin extracted from ui.py."""
import os
import sys
import time
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from PyQt5.QtWidgets import (QApplication, QFileDialog, QTreeWidget, QTreeWidgetItem, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTabWidget, QCheckBox, QComboBox, QSpinBox, QMessageBox, QSizePolicy, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor

class ExportMixin:
    def export_ascii_plotted(self):
        # Determine visible curves in the *current* plotted-list order
        visible_keys = []
        try:
            if hasattr(self, "plotted_list") and self.plotted_list is not None:
                for row in range(self.plotted_list.count()):
                    item = self.plotted_list.item(row)
                    if item is None:
                        continue
                    key = item.data(Qt.UserRole)
                    if key in self.plotted_lines and self.plotted_lines[key].get_visible():
                        visible_keys.append(key)
        except Exception:
            visible_keys = []

        # Fallback: use insertion order from plotted_lines if needed
        if not visible_keys:
            visible_keys = [key for key, line in self.plotted_lines.items() if line.get_visible()]
        if not visible_keys:
            QMessageBox.warning(self, "Export ASCII", "No visible curves to export.")
            return
        first_line = self.plotted_lines[visible_keys[0]]
        x_data = first_line.get_xdata()
        min_len = len(x_data)
        curve_names, y_columns = [], []
        for key in visible_keys:
            line = self.plotted_lines[key]
            y = line.get_ydata()
            min_len = min(min_len, len(y))
            label = self.custom_labels.get(key) or "<select curve name>"
            curve_names.append(label)
            y_columns.append(y)
        x_data = x_data[:min_len]
        y_arrays = [np.array(y)[:min_len] for y in y_columns]
        header = ["X"] + curve_names
        file_path, _ = QFileDialog.getSaveFileName(self, "Export ASCII", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for i in range(min_len):
                        row = [f"{x_data[i]:.6g}"] + [f"{col[i]:.6g}" for col in y_arrays]
                        writer.writerow(row)
            except Exception as ex:
                print("Error writing CSV:", ex)

    def export_ascii(self):
        if not self.plot_data:
            return

        # --------------------------------------------------------------
        # 0.  Sanity check: one curve, or summed
        # --------------------------------------------------------------
        visible_keys = [k for k in self.plot_data if self.raw_visibility.get(k, False)]
        num_visible  = len(visible_keys)
        if num_visible > 1 and not self.chk_sum.isChecked():
            QMessageBox.warning(self, "Export Not Possible",
                                "Please sum the curves up first or select only one curve.")
            return

        # --------------------------------------------------------------
        # 1.  Build default file name
        # --------------------------------------------------------------
        any_file   = next(iter(self.hdf5_files), None)
        default_dir = os.path.dirname(any_file) if any_file else ""

        entries = {parts[1].split("/")[0] if len(parts) == 2 else "unknown"
                   for parts in (k.split("##", 1) for k in visible_keys)}
        entries_sorted = sorted(entries)
        if len(entries_sorted) == 1:
            file_id_part = f"entry{entries_sorted[0].replace('entry','')}"
        else:
            file_id_part = "entries" + "_".join(e.replace('entry','') for e in entries_sorted)

        default_name = f"ProcessedScan_for_{file_id_part}.csv"
        default_path = os.path.join(default_dir, default_name)

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save ASCII File", default_path, "CSV Files (*.csv)"
        )
        if not save_path:
            return

        # --------------------------------------------------------------
        # 2.  Get arrays + post-norm mode
        # --------------------------------------------------------------
        arrays = self.prepare_arrays_for_export()
        if arrays is None:
            return

        norm_mode      = arrays.pop("norm_mode", "None")   # remove & keep others intact
        x_array        = arrays["x"]
        y_raw_array    = arrays["y_raw"]
        y_norm_array   = arrays["y_norm"]
        y_bkg_array    = arrays["y_bkgd"]
        y_final_array  = arrays["y_final"]

        # --------------------------------------------------------------
        # 3.  Column 2 / 3 headers depend on “Sum” tickbox
        # --------------------------------------------------------------
        if self.chk_sum.isChecked() and num_visible > 1:
            col2, col3 = "Y_original_sum", "Y_norm_to_I0_sum"
        else:
            col2, col3 = "Y_original", "Y_norm_to_I0"

        # --------------------------------------------------------------
        # 4.  Dynamic header for the last column
        # --------------------------------------------------------------
        if   norm_mode == "Area":
            col5 = "Y_norm_to_I0_without_bckg_norm_to_area"
        elif norm_mode == "Max":
            col5 = "Y_norm_to_I0_without_bckg_norm_to_max"
        elif norm_mode == "Jump":
            col5 = "Y_norm_to_I0_without_bckg_norm_to_jump"
        else:  # "None"
            col5 = "Y_norm_to_I0_without_bckg_no_norm"

        # --------------------------------------------------------------
        # 5.  Write CSV
        # --------------------------------------------------------------
        try:
            with open(save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["X", col2, col3, "Y_bckd", col5])
                for i in range(len(x_array)):
                    writer.writerow([
                        f"{x_array[i]:.6g}",
                        f"{y_raw_array[i]:.6g}",
                        f"{y_norm_array[i]:.6g}",
                        f"{y_bkg_array[i]:.6g}",
                        f"{y_final_array[i]:.6g}",
                    ])
        except Exception as ex:
            print("Error writing CSV:", ex)

    def prepare_arrays_for_export(self):
        # ------------------------------------------------------------------
        # 0.  Guards + fetch raw & main curves
        # ------------------------------------------------------------------
        x_raw, y_raw = self._compute_raw_curve()
        if x_raw is None or y_raw is None:
            return None

        x_norm, y_norm = self.compute_main_curve()
        if x_norm is None or y_norm is None:
            return None

        # ------------------------------------------------------------------
        # 1.  Length alignment
        # ------------------------------------------------------------------
        m = min(len(x_raw), len(x_norm))
        x_out   = x_norm[:m]
        y_raw   = y_raw[:m]
        y_norm  = y_norm[:m]

        # ------------------------------------------------------------------
        # 2.  Background
        # ------------------------------------------------------------------
        y_bkg   = self._compute_background(x_out, y_norm)
        y_final = y_norm - y_bkg

        # ------------------------------------------------------------------
        # 3.  Post-normalisation = contents of the combo box
        # ------------------------------------------------------------------
        norm_mode = self.combo_post_norm.currentText() \
                    if self.combo_post_norm.isEnabled() else "None"

        if   norm_mode == "Max":
            denom = np.max(np.abs(y_final))
            if denom:
                y_final /= denom

        elif norm_mode == "Jump":
            denom = y_final[-1]
            if denom:
                y_final /= denom

        elif norm_mode == "Area":
            area = np.trapz(y_final, x_out)
            if area:
                y_final /= area
        # “None” → leave y_final untouched

        return {
            "x"        : x_out,
            "y_raw"    : y_raw,
            "y_norm"   : y_norm,
            "y_bkgd"   : y_bkg,
            "y_final"  : y_final,
            "norm_mode": norm_mode,   # ← pass the choice upstream
        }

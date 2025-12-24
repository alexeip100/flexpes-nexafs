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
            QMessageBox.warning(self, "Export", "No visible curves to export.")
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
        file_path, _ = QFileDialog.getSaveFileName(self, "Export", "", "CSV Files (*.csv)")
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


    def import_csv_plotted(self):
        """
        Import one or more CSV files (previously exported) and plot them directly
        in the *Plotted Data* panel.

        Expected format:
          - First row: header, with first column being X and remaining columns being Y-series labels.
          - Data rows: numeric values, comma-separated (other common delimiters are also accepted).
        """
        # Choose a reasonable default directory (the directory of any opened HDF5 file, if available)
        any_file = None
        try:
            any_file = next(iter(getattr(self, "hdf5_files", []) or []), None)
        except Exception:
            any_file = None
        default_dir = os.path.dirname(any_file) if any_file else ""

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import CSV",
            default_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_paths:
            return

        def _is_float(s):
            try:
                float(s)
                return True
            except Exception:
                return False

        imported_count = 0
        errors = []

        for fp in file_paths:
            try:
                # --- Read CSV (robust delimiter detection + optional header) ---
                with open(fp, "r", newline="") as f:
                    sample = f.read(4096)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;	")
                    except Exception:
                        dialect = csv.excel  # comma by default

                    reader = csv.reader(f, dialect)
                    rows = []
                    for row in reader:
                        if not row:
                            continue
                        # allow comments
                        if str(row[0]).strip().startswith("#"):
                            continue
                        rows.append([str(c).strip() for c in row])

                if not rows:
                    raise ValueError("empty file")

                first = rows[0]
                has_header = not all(_is_float(c) for c in first if c != "")
                if has_header:
                    header = first
                    data_rows = rows[1:]
                else:
                    header = ["X"] + [f"Y{i}" for i in range(1, max(len(first), 2))]
                    data_rows = rows

                if len(header) < 2:
                    raise ValueError("CSV must have at least two columns (X and one Y)")

                ncols = len(header)

                x_vals = []
                y_vals = [[] for _ in range(ncols - 1)]

                for row in data_rows:
                    if not row:
                        continue
                    if len(row) < 2:
                        continue
                    # pad short rows
                    if len(row) < ncols:
                        row = row + [""] * (ncols - len(row))
                    # parse x
                    try:
                        x = float(row[0])
                    except Exception:
                        continue
                    x_vals.append(x)
                    # parse y's
                    for j in range(1, ncols):
                        try:
                            y = float(row[j])
                        except Exception:
                            y = float("nan")
                        y_vals[j - 1].append(y)

                x_arr = np.asarray(x_vals, dtype=float)
                if x_arr.size < 2:
                    raise ValueError("not enough numeric rows to import")

                file_stem = os.path.splitext(os.path.basename(fp))[0]

                # Add each Y column as a separate curve
                for j, y_list in enumerate(y_vals):
                    y_arr = np.asarray(y_list, dtype=float)
                    if y_arr.size != x_arr.size:
                        n = min(x_arr.size, y_arr.size)
                        x_use = x_arr[:n]
                        y_use = y_arr[:n]
                    else:
                        x_use = x_arr
                        y_use = y_arr

                    mask = np.isfinite(x_use) & np.isfinite(y_use)
                    x_use = x_use[mask]
                    y_use = y_use[mask]
                    if x_use.size < 2:
                        continue

                    col_name = ""
                    try:
                        col_name = str(header[j + 1]).strip()
                    except Exception:
                        col_name = f"Y{j+1}"
                    if not col_name:
                        col_name = f"Y{j+1}"

                    # Labeling logic:
                    # - if file contains multiple Y columns, prefix with file name
                    # - otherwise, just use file name unless the column header is informative
                    if ncols > 2:
                        label = f"{file_stem}: {col_name}"
                    else:
                        # single Y column
                        if has_header and col_name.lower() not in ("y", "y1", "intensity", "intensity1", "value"):
                            label = f"{file_stem}: {col_name}"
                        else:
                            label = file_stem

                    # unique internal key
                    ts = int(time.time() * 1000)
                    storage_key = f"csv##{file_stem}##{col_name}##{ts}##{j}"

                    meta = {
                        "is_reference": False,
                        "is_imported": True,
                        "source": "csv",
                        "file": fp,
                        "column": col_name,
                    }

                    if hasattr(self, "_add_reference_curve_to_plotted"):
                        self._add_reference_curve_to_plotted(storage_key, x_use, y_use, label, meta=meta)
                    else:
                        # Very old fallback: plot directly if helper is unavailable
                        try:
                            line, = self.plotted_ax.plot(x_use, y_use, label=str(label))
                            if not hasattr(self, "plotted_lines"):
                                self.plotted_lines = {}
                            self.plotted_lines[storage_key] = line
                        except Exception:
                            pass

                    imported_count += 1

            except Exception as ex:
                errors.append(f"{os.path.basename(fp)}: {ex}")

        # Rescale and redraw
        try:
            if hasattr(self, "rescale_plotted_axes"):
                self.rescale_plotted_axes()
            else:
                self.plotted_ax.relim()
                self.plotted_ax.autoscale_view()
        except Exception:
            pass
        try:
            if hasattr(self, "canvas_plotted"):
                self.canvas_plotted.draw_idle()
        except Exception:
            pass

        if errors:
            # Keep message short; show first several errors
            msg = "Some files could not be imported:\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                msg += f"\n... ({len(errors) - 10} more)"
            QMessageBox.warning(self, "Import CSV", msg)
        elif imported_count == 0:
            QMessageBox.information(self, "Import CSV", "No curves were imported (no valid numeric data found).")

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

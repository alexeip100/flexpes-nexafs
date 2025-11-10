
"""Auto-generated DataMixin extracted from ui.py."""
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

class DataMixin:
    def _open_h5_read(self, path, retries: int = 3):
        """
        Open HDF5 file for reading with SWMR and retry a few times if busy.
        This helps tolerate concurrent writes by another process.
        """
          
        last_error = None
        for i in range(max(1, int(retries))):
            try:
                # Try modern SWMR-compatible open
                print("trying SWMR=true, locking=false")
                return h5py.File(path, "r", swmr=True, libver="latest", locking=False)
                # return h5py.File(path, "r", libver="latest", locking=False)
            except TypeError as e:
                # Older h5py: swmr/libver not supported; fall back safely
                print(f"trying defaults.  Error: {e}")
                return h5py.File(path, "r")
            except OSError as e:
                # Common transient case: file temporarily locked or being written
                last_error = e
                time.sleep(0.05 * (i + 1))  # short exponential backoff
            except Exception as e:
                last_error = e
                time.sleep(0.05 * (i + 1))
    
        # Final fallback: try without SWMR, even if locked=False only
        try:
            print(f"trying locking=False, {last_error=}")
            return h5py.File(path, "r", locking=False)
        except Exception as e:
            # Give up after retries
            raise last_error or e

    def group_datasets(self):
        groups = []
        for key in self.plot_data.keys():
            parts = key.split("##", 1)
            if len(parts) != 2:
                continue
    
            abs_path, hdf5_path = parts
            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
            y_data = self.plot_data[key]
            x_data = lookup_energy(self, abs_path, parent, len(y_data))
    
            if getattr(x_data, "size", 0) == 0:
                continue
    
            min_x = np.min(x_data)
            max_x = np.max(x_data)
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

    def set_group_visibility(self, filter_str: str, visible: bool):
        """
        Show or hide all 1D datasets across all opened HDF5 files whose HDF5
        relative path contains `filter_str` (typically the channel name).
    
        On visible=True:
            - Read each matching 1D dataset once
            - Store into self.plot_data[abs_path##relpath]
            - Mark self.raw_visibility[...] = True
    
        On visible=False:
            - Remove matching keys from self.plot_data and self.raw_visibility
    
        Notes:
            - Does NOT touch the 'All in channel' combobox or its checkbox.
            - Uses a re-entrancy guard to avoid being called recursively.
            - Assumes helpers/attrs: self.hdf5_files (dict of abs paths),
              self._open_h5_read(path), self.update_plot_raw(), self.update_pass_button_state(),
              and dicts self.plot_data, self.raw_visibility.
        """
        # Re-entrancy guard
        if getattr(self, "_in_set_group_visibility", False):
            return
        self._in_set_group_visibility = True
    
        try:
            files = list(getattr(self, "hdf5_files", {}).keys())
            if not files:
                # Nothing to do
                return
    
            for abs_path in files:
                try:
                    with self._open_h5_read(abs_path) as f:
    
                        def _visit(name, obj):
                            try:
                                import h5py
                                # Only 1D datasets
                                if not isinstance(obj, h5py.Dataset):
                                    return
                                if getattr(obj, "ndim", 0) != 1:
                                    return
                                if getattr(obj, "size", 0) == 0:
                                    return
                                # Require channel substring match in relpath
                                if filter_str and filter_str not in name:
                                    return
    
                                key = f"{abs_path}##{name}"
                                if visible:
                                    # Read y-array; x-array handling is elsewhere (e.g., in plotting)
                                    try:
                                        y = obj[()]
                                    except Exception:
                                        return
                                    self.plot_data[key] = y
                                    self.raw_visibility[key] = True
                                else:
                                    self.plot_data.pop(key, None)
                                    self.raw_visibility.pop(key, None)
    
                            except Exception:
                                # Ignore per-item errors; keep scanning
                                pass
    
                        f.visititems(_visit)
    
                except Exception:
                    # Ignore per-file errors; continue with remaining files
                    continue
    
            # Single refresh at the end for performance and UI stability
            try:
                self.update_plot_raw()
            finally:
                # Keep pass button / other state in sync
                try:
                    self.update_pass_button_state()
                except Exception:
                    pass
    
        finally:
            self._in_set_group_visibility = False


    def update_file_label(self):
        self.file_label.setText("\n".join(self.hdf5_files.keys()) if self.hdf5_files else "No file open")

    def close_file(self):
    # No persistent h5py.File handles are kept; just clear state
        self.hdf5_files.clear()
        self.file_label.setText("No file open")
        self.tree.clear()
        self.plot_data.clear()
        self.energy_cache.clear()
        self.raw_visibility.clear()
        self.update_plot_raw()
        self.update_plot_processed()
        self.combo_bg.setCurrentIndex(0)
        self.combo_poly.setCurrentIndex(0)
        self.chk_normalize.setChecked(False)
        self.chk_sum.setChecked(False)
        self.chk_show_without_bg.setChecked(False)
        self.cb_all_tey.setChecked(False)
        self.cb_all_pey.setChecked(False)
        self.cb_all_tfy.setChecked(False)
        self.cb_all_pfy.setChecked(False)
        self.region_states.clear()
        self.proc_region_states.clear()
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
        self.update_file_label()
        self.update_pass_button_state()
        # Ensure the 'All in channel' combo is populated after opening files
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
        # Refresh 'All in channel' combo and checkbox
        try:
            if hasattr(self, 'combo_all_channel'):
                self.combo_all_channel.clear()
            if hasattr(self, 'cb_all_in_channel'):
                self.cb_all_in_channel.setChecked(False)
            setattr(self, '_last_all_channel_filter', None)
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
        except Exception:
            pass

    def open_file(self):
        dialog = QFileDialog(self, "Open HDF5 File(s)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)          # keep control over size
        dialog.setOption(QFileDialog.DontUseCustomDirectoryIcons, True)  # <<< SPEED-UP: no per-item icon lookups
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("HDF5 Files (*.h5 *.hdf5)")
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setSizeGripEnabled(True)
    
        dialog.setWindowFlags(dialog.windowFlags() | Qt.Window | Qt.WindowMinMaxButtonsHint)
    
        screen_geom = QApplication.primaryScreen().availableGeometry()
        dialog.resize(int(screen_geom.width()*0.60), int(screen_geom.height()*0.60))
        dialog.move(screen_geom.center() - dialog.rect().center())
    
        if dialog.exec_() == QDialog.Accepted:
            file_paths = dialog.selectedFiles()
            if file_paths:
                self.region_states.clear()
                self.proc_region_states.clear()
                for file_path in file_paths:
                    self.load_hdf5_file(os.path.abspath(file_path))
                self.combo_poly.setCurrentIndex(2)
                self.update_file_label()

    def load_hdf5_file(self, abs_path):
        """
        Non-locking: do not keep h5py.File handles open.
        Add a top-level item and a dummy child for lazy expansion.
        """
        try:
            # Mark file as known without keeping it open
            abs_path = os.path.abspath(abs_path)
            self.hdf5_files[abs_path] = True
    
            abs_path = os.path.abspath(abs_path)
            file_item = QTreeWidgetItem([os.path.basename(abs_path)])
            file_item.setData(0, Qt.UserRole, (abs_path, ""))
    
            has_children = False
            try:
                with (self._open_h5_read(abs_path)) as f:
                    has_children = len(f.keys()) > 0
            except Exception:
                has_children = False
    
            if has_children:
                file_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                file_item.addChild(QTreeWidgetItem(["(click to expand)"]))
    
            self.tree.addTopLevelItem(file_item)
            file_item.setExpanded(True)
    
            # Ensure this scans internally with short-lived opens too
            self.populate_norm_channels(abs_path)
            # Refresh 'All in channel' combo after loading this file
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
            # Update 'All in channel' combo now that files are loaded
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
    
        except Exception as e:
            self.file_label.setText(f"Error opening file: {e}")

    def populate_norm_channels(self, abs_path):
        """Populate normalization channels using a short-lived file open (non-locking)."""
        self.combo_norm.clear()
        default_channel = "b107a_em_03_ch2"
        default_index = -1
        try:
            with (self._open_h5_read(abs_path)) as f:
                for key in f.keys():
                    entry = f[key]
                    if "measurement" in entry:
                        meas_group = entry["measurement"]
                        for idx, (ds_name, ds_obj) in enumerate(meas_group.items()):
                            if isinstance(ds_obj, h5py.Dataset) and ds_obj.ndim == 1:
                                self.combo_norm.addItem(ds_name)
                                if ds_name == default_channel:
                                    default_index = idx
                        break
        except Exception:
            pass
    
        if default_index != -1:
            self.combo_norm.setCurrentIndex(default_index)
        else:
            idx = self.combo_norm.findText(default_channel)
            if idx != -1:
                self.combo_norm.setCurrentIndex(idx)

    def load_subtree(self, item):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        abs_path, hdf5_path = data
        try:
            with (self._open_h5_read(abs_path)) as f:
                if hdf5_path == "":
                    if item.childCount() == 1 and item.child(0).text(0) == "(click to expand)":
                        item.removeChild(item.child(0))
                        for key in f.keys():
                            child_item = QTreeWidgetItem([key])
                            child_item.setData(0, Qt.UserRole, (abs_path, key))
                            sub_obj = f[key]
                            if isinstance(sub_obj, h5py.Group) and sub_obj.keys():
                                child_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                                child_item.addChild(QTreeWidgetItem(["(click to expand)"]))
                            elif isinstance(sub_obj, h5py.Dataset) and sub_obj.ndim == 1:
                                child_item.setCheckState(0, Qt.Unchecked)
                                child_item.setData(1, Qt.UserRole, True)
                            item.addChild(child_item)
                    return
    
                if hdf5_path in f:
                    obj = f[hdf5_path]
                    if isinstance(obj, h5py.Group):
                        if item.childCount() == 1 and item.child(0).text(0) == "(click to expand)":
                            item.removeChild(item.child(0))
                            for key in obj.keys():
                                child_item = QTreeWidgetItem([key])
                                child_item.setData(0, Qt.UserRole, (abs_path, f"{hdf5_path}/{key}"))
                                sub_obj = obj[key]
                                if isinstance(sub_obj, h5py.Group) and sub_obj.keys():
                                    child_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                                    child_item.addChild(QTreeWidgetItem(["(click to expand)"]))
                                elif isinstance(sub_obj, h5py.Dataset) and sub_obj.ndim == 1:
                                    child_item.setCheckState(0, Qt.Unchecked)
                                    child_item.setData(1, Qt.UserRole, True)
                                item.addChild(child_item)
        except Exception:
            pass

    def _compute_raw_curve(self):
        if not self.plot_data:
            return None, None
        if self.chk_sum.isChecked():
            sum_y, x_ref = None, None
            for combined_label, y_data in self.plot_data.items():
                if not self.raw_visibility.get(combined_label, True):
                    continue
                parts = combined_label.split("##", 1)
                if len(parts) != 2:
                    continue
                abs_path, hdf5_path = parts
                parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
                x_data = lookup_energy(self, abs_path, parent, len(y_data))
                if sum_y is None:
                    sum_y = y_data.copy()
                    x_ref = x_data
                else:
                    m = min(len(sum_y), len(y_data))
                    sum_y = sum_y[:m] + y_data[:m]
                    x_ref = x_ref[:m]
            return x_ref, sum_y
        else:
            for combined_label, y_data in self.plot_data.items():
                if not self.raw_visibility.get(combined_label, True):
                    continue
                parts = combined_label.split("##", 1)
                if len(parts) == 2:
                    abs_path, hdf5_path = parts
                    parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
                    return (
                        lookup_energy(self, abs_path, parent, len(y_data)),
                        y_data
                    )
        return None, None


def lookup_energy(viewer, abs_path: str, parent: str, length: int):
    cache = getattr(viewer, "energy_cache", None)
    if cache is None: cache = viewer.energy_cache = {}
    key = f"{abs_path}::{parent}"
    if key in cache: return cache[key][0]
    x = None
    try:
        with viewer._open_h5_read(abs_path) as f:
            cands = []
            if parent: cands += [f"{parent}/pcap_energy_av", f"{parent}/mono_traj_energy"]
            else: cands += ["pcap_energy_av","mono_traj_energy"]
            for p in cands:
                try:
                    arr = f[p][...]
                    try: arr = arr.squeeze()
                    except Exception: pass
                    if getattr(arr,"size",0)>0:
                        x = arr; break
                except Exception: continue
    except Exception:
        x = None
    if x is None: x = np.arange(length)
    cache[key]=(x, False)
    return x

    def collect_available_1d_datasets(self):
        """Return a sorted list of unique relative paths for all 1D datasets across opened HDF5 files.
        Relative path is with respect to the group (no leading slash)."""
        import h5py
        rels = set()
        files = list(getattr(self, "hdf5_files", {}).keys()) if hasattr(self, "hdf5_files") else []
        for abs_path in files:
            try:
                with self._open_h5_read(abs_path) as f:
                    def _visit(name, obj):
                        try:
                            if isinstance(obj, h5py.Dataset):
                                shp = tuple(getattr(obj, "shape", ()) or ())
                                if len(shp) == 1:
                                    s = name.lstrip("/")
                                    rels.add(s)
                        except Exception:
                            pass
                    f.visititems(_visit)
            except Exception:
                pass
        out = sorted(rels, key=lambda s: s.lower())
        return out


"""Auto-generated DataMixin extracted from ui.py."""
import os
import time
import h5py
import numpy as np
import logging
logger = logging.getLogger(__name__)
from importlib.resources import files
from PyQt5.QtWidgets import QApplication, QFileDialog, QTreeWidgetItem, QDialog
from PyQt5.QtCore import Qt, QTimer
class DataMixin:
    def _ensure_raw_key_sources(self):
        """Ensure the raw-plot key → sources map exists.

        We use this to keep overlapping selection mechanisms independent:
          - role checkboxes (All TEY/PEY/TFY/PFY)
          - "All in channel" selection
          - manual per-dataset checks in the HDF5 tree

        Each plotted curve key ("abs_path##hdf5_path") can have multiple
        sources. Removing one source should not remove the curve if another
        source still requests it.
        """
        if not hasattr(self, "_raw_key_sources") or not isinstance(getattr(self, "_raw_key_sources"), dict):
            self._raw_key_sources = {}

    def _add_raw_key_source(self, key: str, source: str) -> None:
        self._ensure_raw_key_sources()
        try:
            s = self._raw_key_sources.get(key)
            if not isinstance(s, set):
                s = set()
            s.add(str(source))
            self._raw_key_sources[key] = s
        except Exception:
            pass

    def _remove_raw_key_source(self, key: str, source: str) -> bool:
        """Remove a source from a key. Returns True if the key should be deleted."""
        self._ensure_raw_key_sources()
        try:
            s = self._raw_key_sources.get(key)
            if isinstance(s, set):
                s.discard(str(source))
                if not s:
                    self._raw_key_sources.pop(key, None)
                    return True
                self._raw_key_sources[key] = s
                return False
            # If we had no tracking info, fall back to "delete".
            return True
        except Exception:
            return True

    def _iter_tree_items(self):
        """Yield all QTreeWidgetItems from the main HDF5 structure tree."""
        tree = getattr(self, "tree", None)
        if tree is None:
            return

        def _walk(item):
            yield item
            for i in range(item.childCount()):
                yield from _walk(item.child(i))

        for i in range(tree.topLevelItemCount()):
            yield from _walk(tree.topLevelItem(i))

    def _normalize_hdf5_key(self, hdf5_key):
        """Best-effort normalization of a stored HDF5 key to str.

        This mirrors PlottingMixin._normalize_hdf5_key, but we keep a local
        copy here to avoid cross-mixin import/order issues.
        """
        try:
            if isinstance(hdf5_key, bytes):
                return hdf5_key.decode("utf-8", errors="replace")
            if isinstance(hdf5_key, str):
                return hdf5_key
            if isinstance(hdf5_key, tuple):
                if len(hdf5_key) >= 1 and isinstance(hdf5_key[0], (str, bytes)):
                    first = hdf5_key[0]
                    if isinstance(first, bytes):
                        first = first.decode("utf-8", errors="replace")
                    if all(isinstance(x, (str, bytes)) for x in hdf5_key):
                        parts = []
                        for x in hdf5_key:
                            if isinstance(x, bytes):
                                x = x.decode("utf-8", errors="replace")
                            x = (x or "").strip("/")
                            if x:
                                parts.append(x)
                        if not parts:
                            return ""
                        path = "/".join(parts)
                        if str(first).startswith("/"):
                            path = "/" + path.lstrip("/")
                        return path
                    return str(first)
                return str(hdf5_key)
            return str(hdf5_key)
        except Exception:
            return str(hdf5_key)

    def _uncheck_tree_items_for_filter(self, filter_str: str) -> None:
        """Uncheck matching items in the left HDF5 tree.

        Used to keep the tree state consistent when a bulk selection mechanism
        removes curves.
        """
        tree = getattr(self, "tree", None)
        if tree is None:
            return
        try:
            tree.blockSignals(True)
            for item in self._iter_tree_items() or []:
                try:
                    data = item.data(0, Qt.UserRole)
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    _abs_path, hdf5_path = data
                    hdf5_path = self._normalize_hdf5_key(hdf5_path)
                    if not isinstance(hdf5_path, str):
                        continue
                    if filter_str and filter_str in hdf5_path and item.checkState(0) == Qt.Checked:
                        item.setCheckState(0, Qt.Unchecked)
                except Exception:
                    continue
        finally:
            try:
                tree.blockSignals(False)
            except Exception:
                pass

    def clear_group_visibility(self, filter_str: str) -> None:
        """Remove all curves matching filter_str, regardless of their source.

        Also unchecks matching items in the left tree to keep UI consistent.
        """
        self._ensure_raw_key_sources()
        try:
            for key in list(getattr(self, "plot_data", {}).keys()):
                try:
                    parts = key.split("##", 1)
                    if len(parts) != 2:
                        continue
                    _abs_path, hdf5_path = parts
                    if filter_str and filter_str in hdf5_path:
                        self.plot_data.pop(key, None)
                        self.raw_visibility.pop(key, None)
                        self._raw_key_sources.pop(key, None)
                except Exception:
                    continue
        finally:
            try:
                self._uncheck_tree_items_for_filter(filter_str)
            except Exception:
                pass
            try:
                self.update_plot_raw()
            except Exception:
                pass
            try:
                self.update_pass_button_state()
            except Exception:
                pass
    def _open_h5_read(self, path, retries: int = 3):
        """
        Open HDF5 file for reading with SWMR and retry a few times if busy.
        This helps tolerate concurrent writes by another process.
        """
          
        last_error = None
        for i in range(max(1, int(retries))):
            try:
                # Try modern SWMR-compatible open
                logger.debug("trying SWMR=true, locking=false")
                return h5py.File(path, "r", swmr=True, libver="latest", locking=False)
                # return h5py.File(path, "r", libver="latest", locking=False)
            except TypeError as e:
                # Older h5py: swmr/libver not supported; fall back safely
                logger.debug("trying defaults (TypeError): %s", e)
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
            logger.debug("trying locking=False, last_error=%s", last_error)
            return h5py.File(path, "r", locking=False)
        except Exception as e:
            # Give up after retries
            raise last_error or e

    def group_datasets(self):
        """Group currently loaded 1D datasets into energy 'regions'.

        A region is defined ONLY by the energy start and energy end of the scan
        (derived from the x-axis). Small numerical deviations (< 0.01 eV) in
        start/end must NOT split regions.

        This function is intentionally robust to NaNs in the energy axis
        (pcap_energy_av). If x contains NaNs, we use finite values only.
        If no finite x can be found, we fall back to a simple index axis.

        Returns
        -------
        list of dict
            Each dict has keys: "keys" (list of dataset keys), "min" (region start),
            "max" (region end).
        """
        tol_E = 0.01  # eV

        # Collect (key, E_start, E_end) for every currently loaded curve
        items = []
        for key, y_data in getattr(self, "plot_data", {}).items():
            try:
                parts = str(key).split("##", 1)
                if len(parts) != 2:
                    continue
                abs_path, hdf5_path = parts
                parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""

                if y_data is None:
                    continue
                y_arr = np.asarray(y_data).ravel()
                if y_arr.size == 0:
                    continue

                # Energy axis lookup (prefers pcap_energy_av via lookup_energy)
                try:
                    x_data = lookup_energy(self, abs_path, parent, int(y_arr.size))
                except Exception:
                    x_data = np.arange(int(y_arr.size), dtype=float)

                if getattr(x_data, "size", 0) == 0:
                    continue
                x_arr = np.asarray(x_data).ravel()

                # Match plotted length
                n = int(min(x_arr.size, y_arr.size))
                if n <= 0:
                    continue
                x_use = x_arr[:n]

                # Determine scan start/end using finite x only
                finite = np.isfinite(x_use)
                if not np.any(finite):
                    # No usable energy axis; fall back to index axis for grouping
                    x_use = np.arange(n, dtype=float)
                    finite = np.isfinite(x_use)

                # Use first and last finite x values as endpoints
                i0 = int(np.argmax(finite))
                i1 = int(len(finite) - 1 - np.argmax(finite[::-1]))
                e_start = float(x_use[i0])
                e_end = float(x_use[i1])
                if not (np.isfinite(e_start) and np.isfinite(e_end)):
                    continue
                if e_end < e_start:
                    e_start, e_end = e_end, e_start

                items.append((key, e_start, e_end))
            except Exception:
                continue

        if not items:
            return []

        # ---- Order-independent tolerance clustering (union-find) ----
        n = len(items)
        uf_parent = list(range(n))
        uf_rank = [0] * n

        def uf_find(i):
            while uf_parent[i] != i:
                uf_parent[i] = uf_parent[uf_parent[i]]
                i = uf_parent[i]
            return i

        def uf_union(a, b):
            ra, rb = uf_find(a), uf_find(b)
            if ra == rb:
                return
            if uf_rank[ra] < uf_rank[rb]:
                uf_parent[ra] = rb
            elif uf_rank[ra] > uf_rank[rb]:
                uf_parent[rb] = ra
            else:
                uf_parent[rb] = ra
                uf_rank[ra] += 1

        # Reduce comparisons by bucketing on start energy
        start_bins = {}
        for i, (_, es, ee) in enumerate(items):
            b = int(np.floor(es / tol_E))
            for bb in (b - 1, b, b + 1):
                for j in start_bins.get(bb, []):
                    _, es2, ee2 = items[j]
                    if abs(es - es2) <= tol_E and abs(ee - ee2) <= tol_E:
                        uf_union(i, j)
            start_bins.setdefault(b, []).append(i)

        clusters = {}
        for i, (key, es, ee) in enumerate(items):
            r = uf_find(i)
            clusters.setdefault(r, {"keys": [], "starts": [], "ends": []})
            clusters[r]["keys"].append(key)
            clusters[r]["starts"].append(es)
            clusters[r]["ends"].append(ee)

        out = []
        for c in clusters.values():
            ref_start = float(np.median(c["starts"]))
            ref_end = float(np.median(c["ends"]))
            out.append({"keys": c["keys"], "min": ref_start, "max": ref_end})

        out.sort(key=lambda g: (g.get("min", 0.0), g.get("max", 0.0)))
        return out

    def set_group_visibility(self, filter_str: str, visible: bool, source: str = "group"):
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
        self._ensure_raw_key_sources()

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
                                # Prefer data from the "measurement" group only; ignore "plot_1" and other derived groups.
                                norm = "/" + str(name).strip("/") + "/"
                                if "/measurement/" not in norm:
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
                                    # Only populate y if absent; avoid clobbering existing data.
                                    if key not in self.plot_data:
                                        self.plot_data[key] = y
                                    self.raw_visibility[key] = True
                                    self._add_raw_key_source(key, source)
                                else:
                                    # Remove only this source. Delete the key only if no sources remain.
                                    should_delete = self._remove_raw_key_source(key, source)
                                    if should_delete:
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
        try:
            if hasattr(self, "_raw_key_sources"):
                self._raw_key_sources.clear()
        except Exception:
            pass
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
        QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
        # Refresh 'All in channel' combo and checkbox
        try:
            if hasattr(self, 'combo_all_channel'):
                self.combo_all_channel.clear()
            if hasattr(self, 'cb_all_in_channel'):
                self.cb_all_in_channel.setChecked(False)
            setattr(self, '_last_all_channel_filter', None)
            QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
        except Exception:
            pass

    def close_single_hdf5_file(self, abs_path):
        """
        Close a single HDF5 file and remove its data from internal state,
        without affecting other open files.
        """
        try:
            abs_path = os.path.abspath(abs_path)
        except Exception:
            abs_path = str(abs_path)

        # Validate registry
        if not isinstance(getattr(self, "hdf5_files", None), dict):
            return
        if abs_path not in self.hdf5_files:
            return

        # Remove the file from the registry
        try:
            self.hdf5_files.pop(abs_path, None)
        except Exception:
            pass

        # Remove raw/processed data and visibility flags for this file
        try:
            keys_to_remove = [
                key for key in list(getattr(self, "plot_data", {}).keys())
                if isinstance(key, str) and key.startswith(abs_path + "##")
            ]
        except Exception:
            keys_to_remove = []
        for key in keys_to_remove:
            try:
                self.plot_data.pop(key, None)
            except Exception:
                pass
            try:
                self.raw_visibility.pop(key, None)
            except Exception:
                pass
            try:
                if hasattr(self, "_raw_key_sources") and isinstance(getattr(self, "_raw_key_sources"), dict):
                    self._raw_key_sources.pop(key, None)
            except Exception:
                pass

        # Drop cached energy arrays for this file
        try:
            cache = getattr(self, "energy_cache", None)
            if isinstance(cache, dict):
                for key in list(cache.keys()):
                    if isinstance(key, str) and key.startswith(abs_path + "::"):
                        cache.pop(key, None)
        except Exception:
            pass

        # Remove curves belonging to this file from the Plotted tab
        try:
            plotted_curves = getattr(self, "plotted_curves", None)
            plotted_lines = getattr(self, "plotted_lines", None)
            plotted_list = getattr(self, "plotted_list", None)
            plotted_metadata = getattr(self, "plotted_metadata", {})
            original_line_data = getattr(self, "original_line_data", {})
            custom_labels = getattr(self, "custom_labels", {})

            keys_for_file = set()

            # Prefer explicit metadata if present
            if isinstance(plotted_metadata, dict):
                for key, meta in list(plotted_metadata.items()):
                    try:
                        if meta.get("source_file") == abs_path:
                            keys_for_file.add(key)
                    except Exception:
                        continue

            # Fallback: match on storage key prefix
            if isinstance(plotted_curves, (set, list)):
                for key in list(plotted_curves):
                    if isinstance(key, str) and key.startswith(abs_path + "##"):
                        keys_for_file.add(key)

            for key in keys_for_file:
                # Remove line from axes
                try:
                    if isinstance(plotted_lines, dict):
                        line = plotted_lines.pop(key, None)
                        if line is not None:
                            try:
                                line.remove()
                            except Exception:
                                pass
                except Exception:
                    pass

                # Remove bookkeeping structures
                try:
                    if isinstance(plotted_curves, set):
                        plotted_curves.discard(key)
                    elif isinstance(plotted_curves, list):
                        if key in plotted_curves:
                            plotted_curves.remove(key)
                except Exception:
                    pass
                try:
                    if isinstance(original_line_data, dict):
                        original_line_data.pop(key, None)
                except Exception:
                    pass
                try:
                    if isinstance(custom_labels, dict):
                        custom_labels.pop(key, None)
                except Exception:
                    pass
                try:
                    if isinstance(plotted_metadata, dict):
                        plotted_metadata.pop(key, None)
                except Exception:
                    pass

                # Remove from the list widget
                try:
                    if plotted_list is not None:
                        from PyQt5.QtCore import Qt as _Qt  # local alias to avoid surprises
                        for row in range(plotted_list.count() - 1, -1, -1):
                            item = plotted_list.item(row)
                            key_role = item.data(_Qt.UserRole)
                            widget = plotted_list.itemWidget(item)
                            widget_key = getattr(widget, "key", None) if widget is not None else None
                            if key_role == key or widget_key == key:
                                plotted_list.takeItem(row)
                except Exception:
                    pass
        except Exception:
            pass

        # If no files remain, fall back to the full reset
        try:
            if not self.hdf5_files:
                self.close_file()
                return
        except Exception:
            pass

        # Otherwise, refresh labels and plots
        try:
            self.update_file_label()
        except Exception:
            pass
        try:
            self.update_plot_raw()
        except Exception:
            pass
        try:
            self.update_plot_processed()
        except Exception:
            pass

        # Refresh 'All in channel' controls
        try:
            if hasattr(self, "combo_all_channel"):
                self.combo_all_channel.clear()
            if hasattr(self, "cb_all_in_channel"):
                self.cb_all_in_channel.setChecked(False)
            setattr(self, "_last_all_channel_filter", None)
            QTimer.singleShot(0, getattr(self, "_refresh_all_in_channel_combo", lambda: None))
        except Exception:
            pass

        # Refresh Plotted axes and legend
        try:
            if hasattr(self, "recompute_waterfall_layout"):
                self.recompute_waterfall_layout()
            else:
                if hasattr(self, "rescale_plotted_axes"):
                    self.rescale_plotted_axes()
                if hasattr(self, "canvas_plotted"):
                    self.canvas_plotted.draw()
        except Exception:
            pass
        try:
            if hasattr(self, "update_legend"):
                self.update_legend()
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

        # Try to default to the installed example_data directory, if available
        try:
            example_dir = files("flexpes_nexafs") / "example_data"
            if os.path.isdir(str(example_dir)):
                dialog.setDirectory(str(example_dir))
        except Exception:
            # Fall back to Qt's default directory behaviour if anything goes wrong
            pass

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
            QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
            # Update 'All in channel' combo now that files are loaded
            QTimer.singleShot(0, getattr(self, '_refresh_all_in_channel_combo', lambda: None))
    
        except Exception as e:
            self.file_label.setText(f"Error opening file: {e}")

    def populate_norm_channels(self, abs_path):
        """Populate normalization channels using a short-lived file open (non-locking)."""
        self.combo_norm.clear()
        # Default I0 candidates come from the active channel-mapping profile (beamline).
        # The *first* candidate is treated as the preferred default.
        default_candidates = ["b107a_em_03_ch2", "b107a_em_04_ch2", "Pt_No"]
        try:
            cc = getattr(self, "channel_config", None)
            if cc is not None:
                cands = cc.get_candidates("I0")
                if cands:
                    default_candidates = list(cands)
        except Exception:
            pass

        try:
            with (self._open_h5_read(abs_path)) as f:
                for key in f.keys():
                    entry = f[key]
                    if "measurement" in entry:
                        meas_group = entry["measurement"]
                        for ds_name, ds_obj in meas_group.items():
                            if isinstance(ds_obj, h5py.Dataset) and ds_obj.ndim == 1:
                                self.combo_norm.addItem(ds_name)
                        break
        except Exception:
            pass

        # Prefer exact match in candidate order (first candidate wins).
        for cand in default_candidates:
            cand = str(cand).strip()
            if not cand:
                continue
            idx = self.combo_norm.findText(cand)
            if idx != -1:
                self.combo_norm.setCurrentIndex(idx)
                return

        # Substring fallback (still using candidate order).
        for i in range(self.combo_norm.count()):
            txt = str(self.combo_norm.itemText(i))
            for cand in default_candidates:
                cand = str(cand).strip()
                if cand and cand in txt:
                    self.combo_norm.setCurrentIndex(i)
                    return
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
    """
    Return the energy axis for a given 1D dataset.

    Preference order:
    1) An explicit x/energy dataset under the same parent group, e.g.
       - f"{parent}/x"
       - f"{parent}/energy"
       - f"{parent}/photon_energy"
    2) Fallback to original behaviour using "pcap_energy_av" or "mono_traj_energy"
       either under the same parent or at the file root.
    3) If none found, return a simple arange(length).
    """
    cache = getattr(viewer, "energy_cache", None)
    if cache is None:
        cache = viewer.energy_cache = {}
    key = f"{abs_path}::{parent}"
    if key in cache:
        return cache[key][0]

    x = None
    try:
        with viewer._open_h5_read(abs_path) as f:
            # 1) Explicit x/energy datasets near the parent
            search_roots = []
            if parent:
                search_roots.append(parent)
            else:
                search_roots.append("")
            # Candidate names for the energy axis can be extended via the
            # channel mapping ("Energy" role).
            candidates = ["pcap_energy_av", "mono_traj_energy", "x", "energy", "photon_energy"]
            try:
                cc = getattr(viewer, "channel_config", None)
                if cc is not None:
                    extra = cc.get_candidates("Energy")
                    for name in extra:
                        name = str(name)
                        if name and name not in candidates:
                            candidates.append(name)
            except Exception:
                pass
            for root in search_roots:
                for name in candidates:
                    p = f"{root}/{name}" if root else name
                    try:
                        arr = f[p][...]
                        try:
                            arr = arr.squeeze()
                        except Exception:
                            pass
                        if getattr(arr, "size", 0) > 0:
                            x = arr
                            break
                    except Exception:
                        continue
                if x is not None:
                    break

            # 2) Fallback to original pcap/mono_traj search
            if x is None:
                cands = []
                if parent:
                    cands += [f"{parent}/pcap_energy_av", f"{parent}/mono_traj_energy"]
                else:
                    cands += ["pcap_energy_av", "mono_traj_energy"]
                for p in cands:
                    try:
                        arr = f[p][...]
                        try:
                            arr = arr.squeeze()
                        except Exception:
                            pass
                        if getattr(arr, "size", 0) > 0:
                            x = arr
                            break
                    except Exception:
                        continue
    except Exception:
        x = None

    if x is None:
        x = np.arange(length)
    cache[key] = (x, False)
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
                                    norm = "/" + str(name).strip("/") + "/"
                                    if "/measurement/" not in norm:
                                        return
                                    s = name.lstrip("/")
                                    rels.add(s)
                        except Exception:
                            pass
                    f.visititems(_visit)
            except Exception:
                pass
        out = sorted(rels, key=lambda s: s.lower())
        return out

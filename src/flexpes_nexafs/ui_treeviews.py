"""UI tree view logic (raw/proc curve trees).

This mixin hosts QTreeWidget population + item-changed handling and related helpers.
It is intentionally kept separate from plotting logic to reduce coupling.
"""

from __future__ import annotations

import numpy as np

from .utils.sorting import parse_entry_number

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor, QIcon
from PyQt5.QtWidgets import QTreeWidgetItem


class TreeViewMixin:
    """Mixin for managing the Raw/Processed curve QTreeWidgets."""

    def recursive_uncheck(self, item, col):
        if not item:
            return
        if item.data(col, Qt.UserRole):
            item.setCheckState(col, Qt.Unchecked)
        for i in range(item.childCount()):
            self.recursive_uncheck(item.child(i), col)

    # ------------ Tabs & right-panel trees ------------

    def group_datasets(self):
        """Group currently loaded 1D datasets into "regions" by scan start/end energies.

        Regions are determined ONLY by the start/end of the energy axis (x).
        Small numerical differences (< 0.01 eV) in start/end must not split regions.

        Notes:
        - NaNs in the signal (y) must not influence region limits.
        - This implementation is order-independent (uses union-find clustering),
          avoiding the "greedy clustering" pitfall where earlier region references
          can drift and prevent later merges.
        """
        import numpy as np

        tol_E = 0.01  # eV (10 meV)

        # Collect (key, E_start, E_end)
        items = []
        for key, y_data in getattr(self, "plot_data", {}).items():
            try:
                parts = key.split("##", 1)
                if len(parts) != 2:
                    continue
                abs_path, hdf5_path = parts
                parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""

                if y_data is None:
                    continue
                y_arr = np.asarray(y_data).ravel()
                if y_arr.size == 0:
                    continue

                # Energy lookup (pcap_energy_av preferred inside lookup_energy).
                try:
                    from .data import lookup_energy as _lookup
                    x_data = _lookup(self, abs_path, parent, int(y_arr.size))
                except Exception:
                    x_data = np.arange(y_arr.size)

                if getattr(x_data, "size", 0) == 0:
                    continue
                x_arr = np.asarray(x_data).ravel()

                # Match plotting length.
                n = int(min(x_arr.size, y_arr.size))
                if n <= 0:
                    continue
                x_use = x_arr[:n]

                # Use first/last finite x as scan limits.
                finite = np.isfinite(x_use)
                if not np.any(finite):
                    continue
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

        # ---- Union-Find clustering (order-independent) ----
        n = len(items)
        parent = list(range(n))
        rank = [0] * n

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Bucket by start energy to reduce comparisons.
        # Any pair within tol_E must fall into same or adjacent start-bin.
        start_bins = {}  # bin_id -> list of indices already seen
        for i, (_, es, ee) in enumerate(items):
            b = int(np.floor(es / tol_E))
            # Compare with indices in neighbouring start-bins.
            for bb in (b - 1, b, b + 1):
                for j in start_bins.get(bb, []):
                    _, es2, ee2 = items[j]
                    if abs(es - es2) <= tol_E and abs(ee - ee2) <= tol_E:
                        union(i, j)
            start_bins.setdefault(b, []).append(i)

        # Build clusters
        clusters = {}
        for i, (key, es, ee) in enumerate(items):
            r = find(i)
            clusters.setdefault(r, {"keys": [], "starts": [], "ends": []})
            clusters[r]["keys"].append(key)
            clusters[r]["starts"].append(es)
            clusters[r]["ends"].append(ee)

        out = []
        for c in clusters.values():
            # Robust representative for display
            ref_start = float(np.median(c["starts"]))
            ref_end = float(np.median(c["ends"]))
            out.append({"keys": c["keys"], "min": ref_start, "max": ref_end})

        out.sort(key=lambda g: (g.get("min", 0.0), g.get("max", 0.0)))
        return out

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
            groups = self.group_datasets(include_sums=False)
            for idx, group in enumerate(groups):
                region_id = f"region_{idx}"
                region_state = getattr(self, "region_states", {}).get(region_id, Qt.Checked)
                display = group.get("label")
                if not display:
                    display = f"({group.get('min',0):.3f}–{group.get('max',0):.3f} eV)"
                region_item = QTreeWidgetItem([f"Region {idx+1}  {display}"])
                region_item.setFlags(region_item.flags() | Qt.ItemIsUserCheckable)
                region_item.setCheckState(0, region_state)
                region_item.setData(0, Qt.UserRole+1, region_id)
                sorted_keys = sorted(group['keys'], key=lambda x: parse_entry_number(x.split("##",1)[1] if "##" in x else ""))
                for key in sorted_keys:
                    parts = key.split("##", 1)
                    label = None
                    try:
                        cl = getattr(self, "custom_labels", {}) or {}
                        if key in cl and cl[key]:
                            label = str(cl[key])
                    except Exception:
                        label = None
                    if not label:
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
            groups = self.group_datasets(include_sums=True)
            for idx, group in enumerate(groups):
                region_id = f"proc_region_{idx}"
                display = group.get("label")
                if not display:
                    display = f"({group.get('min',0):.3f}–{group.get('max',0):.3f} eV)"
                region_item = QTreeWidgetItem([f"Region {idx+1}  {display}"])
                region_item.setFlags(region_item.flags() | Qt.ItemIsUserCheckable)
                region_item.setCheckState(0, self.proc_region_states.get(region_id, Qt.Checked))
                region_item.setData(0, Qt.UserRole+1, region_id)
                sorted_keys = sorted(group['keys'], key=lambda x: parse_entry_number(x.split("##",1)[1] if "##" in x else ""))
                for key in sorted_keys:
                    parts = key.split("##", 1)
                    # Use intrinsic curve names (e.g. synthetic summed-group names) when available.
                    label = None
                    try:
                        cdn = getattr(self, "curve_display_names", {}) or {}
                        if key in cdn and cdn[key]:
                            label = str(cdn[key])
                    except Exception:
                        label = None
                    if not label:
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

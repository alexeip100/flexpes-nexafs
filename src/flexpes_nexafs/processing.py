from .data import lookup_energy
"""Auto-generated ProcessingMixin extracted from ui.py."""
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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class ProcessingMixin:
    def _proc_robust_polyfit_on_normalized(self, xs, ys, deg, x_eval):
        import numpy as np
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        x_eval = np.asarray(x_eval, dtype=float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(m):
            return np.zeros_like(x_eval, dtype=float), np.array([], dtype=float)
        xs = xs[m]; ys = ys[m]
        n = xs.size
        try:
            deg_eff = int(deg)
        except Exception:
            deg_eff = 2
        deg_eff = max(1, min(deg_eff, max(1, n - 1)))
        x_min = float(np.nanmin(xs)); x_max = float(np.nanmax(xs))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x0, span = 0.0, 1.0
        else:
            x0 = 0.5 * (x_min + x_max)
            span = x_max - x_min
            if not np.isfinite(span) or span == 0.0:
                span = 1.0
        xp = 2.0 * (xs - x0) / span
        xp_eval = 2.0 * (x_eval - x0) / span
        try:
            p = np.polyfit(xp, ys, deg_eff)
            bg = np.polyval(p, xp_eval)
        except Exception:
            p = np.array([], dtype=float)
            bg = np.zeros_like(x_eval, dtype=float)
        return bg, p

    def recompute_waterfall_layout(self):
        """(Re)apply Waterfall based on combobox mode and slider/spin."""
        # Update controls state based on mode
        mode = None
        try:
            mode = self.waterfall_mode_combo.currentText()
        except Exception:
            mode = "None"
        try:
            if mode == "Adaptive step":
                self.waterfall_slider.setEnabled(True)
                self.waterfall_spin.setEnabled(True)
            elif mode == "Uniform step":
                self.waterfall_slider.setEnabled(True)
                self.waterfall_spin.setEnabled(True)
            else:
                self.waterfall_slider.setEnabled(False)
                self.waterfall_spin.setEnabled(False)
        except Exception:
            pass
        if mode == "None":
            try:
                self.restore_original_line_data()
                self.rescale_plotted_axes()
            except Exception:
                pass
            return
        # Delegate to apply_waterfall_shift which now checks the mode
        try:
            self.apply_waterfall_shift()
        except Exception as ex:
            print("apply_waterfall_shift error:", ex)

    def reset_manual_mode(self):
        self.manual_mode = False
        self.manual_points = []
        self.manual_bg_line = None
        self.manual_poly = None
        self.manual_poly_degree = None
        self.last_sum_state = False
        self.last_normalize_state = False
        self.last_plot_len = 0

    def _compute_background(self, main_x, main_y):
        mode = self.combo_bg.currentText()
        deg = int(self.combo_poly.currentText())
        if mode == "None":
            return np.zeros_like(main_y)
        elif mode == "Manual" and self.manual_points:
            try:
                xs = [pt["x"] for pt in self.manual_points]
                ys = [pt["y"] for pt in self.manual_points]
                background, _ = _proc_robust_polyfit_on_normalized(self, xs, ys, deg, main_x)
                background[0] = main_y[0]
            except Exception as ex:
                print("Error in manual background computation:", ex)
                background = np.zeros_like(main_y)
            return background
        elif mode == "Automatic":
            return self._apply_automatic_bg_new( main_x, main_y, do_plot=False)
        return np.zeros_like(main_y)

    def compute_main_curve(self):
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
                processed_y = apply_normalization(self, abs_path, parent, y_data)
                if sum_y is None:
                    sum_y = processed_y.copy()
                    x_ref = x_data
                else:
                    m = min(len(sum_y), len(processed_y))
                    sum_y = sum_y[:m] + processed_y[:m]
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
                        apply_normalization(self, abs_path, parent, y_data),
                    )
        return None, None

    def _safe_post_normalize(self, x, y, mode):
        """Robust post-normalization on finite data.
        mode: "None" | "Max" | "Jump" | "Area"
        Returns a *copy* of y if scaled, or original y if not applicable.
        """
        import numpy as np
        if mode is None or mode == "None":
            return y
        x = np.asarray(x); y = np.asarray(y, dtype=float)
        mfin = np.isfinite(x) & np.isfinite(y)
        if not np.any(mfin):
            return y
        xf = x[mfin]; yf = y[mfin]
        def _nonzero(val, eps=1e-15):
            return (val is not None) and np.isfinite(val) and (abs(val) > eps)
        try:
            if mode == "Max":
                d = float(np.max(np.abs(yf)))
                if _nonzero(d): return y / d
            elif mode == "Jump":
                d = float(yf[-1])
                if _nonzero(d): return y / d
            elif mode == "Area":
                a = float(np.trapz(yf, xf))
                if _nonzero(a): return y / a
        except Exception as ex:
            print("safe_post_norm error:", ex)
        return y

    def _on_normalize_toggled(self, state):
        self.combo_norm.setEnabled(state == Qt.Checked)
        self.update_plot_processed()

    def _on_bg_subtract_toggled(self, state):
        """Enable/disable post‑normalisation combo and update plot."""
        self.combo_post_norm.setEnabled(state == Qt.Checked)
        self.update_plot_processed()

    def _robust_polyfit_on_normalized(self, xs, ys, deg, x_eval):
        """Stable polynomial fit for manual BG in normalized x ∈ [-1,1]."""
        import numpy as np
        xs = np.asarray(xs, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).ravel()
        x_eval = np.asarray(x_eval, dtype=float).ravel()
        m = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[m], ys[m]
        if xs.size == 0:
            return np.zeros_like(x_eval, dtype=float), np.array([0.0], dtype=float)
        n_uniq = np.unique(xs).size
        deg_eff = int(max(0, min(int(deg), max(0, n_uniq - 1))))
        x0 = float(xs.min()); x1 = float(xs.max())
        span = max(1e-12, x1 - x0)
        xp = 2.0 * (xs - x0) / span - 1.0
        xp_eval = 2.0 * (x_eval - x0) / span - 1.0
        p = np.polyfit(xp, ys, deg_eff)
        bg = np.polyval(p, xp_eval)
        return bg, p

    def _apply_manual_bg(self, main_x, main_y):
        import numpy as np
        if not getattr(self, "manual_points", None):
            return np.zeros_like(main_y)
        try:
            d = int(self.combo_poly.currentText())
        except Exception:
            d = 2
        xs = [pt["x"] for pt in self.manual_points]
        ys = [pt["y"] for pt in self.manual_points]
        background, coeffs = _proc_robust_polyfit_on_normalized(self, xs, ys, d, main_x)
        if np.isfinite(main_y[0]):
            background[0] = main_y[0]
        self.manual_poly = coeffs
    
        # Draw anchors always
        for pt in self.manual_points:
            if pt.get("artist") is not None:
                try: pt["artist"].remove()
                except Exception: pass
                pt["artist"] = None
            marker, = self.proc_ax.plot(pt["x"], pt["y"], marker='o', linestyle='None', markersize=8, markerfacecolor='blue', markeredgecolor='blue', picker=5)
            pt["artist"] = marker
    
        # Show/hide red background line
        hide_bg = getattr(self, "chk_show_without_bg", None) is not None and self.chk_show_without_bg.isChecked()
        if hide_bg:
            if getattr(self, "manual_bg_line", None) is not None:
                try: self.manual_bg_line.remove()
                except Exception: pass
                self.manual_bg_line = None
        else:
            if getattr(self, "manual_bg_line", None) is not None and self.manual_bg_line.axes is self.proc_ax:
                try: self.manual_bg_line.remove()
                except Exception: pass
                self.manual_bg_line = None
            self.manual_bg_line, = self.proc_ax.plot(main_x, background, '--', color='red', label='Background')
        try:
            self.canvas_proc.draw_idle()
        except Exception:
            pass
        return background

    def _show_subtracted_only(self, mode, main_x, main_y):
        self.proc_ax.clear()
        self.manual_bg_line = None
        # compute background
        if mode == "Automatic":
            background = self._apply_automatic_bg_new( main_x, main_y, do_plot=False)
        elif mode == "Manual" and self.manual_points:
            try:
                d = int(self.combo_poly.currentText())
                xs = [pt["x"] for pt in self.manual_points]
                ys = [pt["y"] for pt in self.manual_points]
                background, _ = _proc_robust_polyfit_on_normalized(self, xs, ys, d, main_x)
                background[0] = main_y[0]
            except Exception:
                background = np.zeros_like(main_y)
        else:
            background = np.zeros_like(main_y)

        sub = main_y - background

        # post-normalisation (robust)
        if self.combo_post_norm.isEnabled():
            mode_norm = self.combo_post_norm.currentText()
            sub = _proc_safe_post_normalize(self, main_x, sub, mode_norm)
        # finite-only plotting and autoscale
        import numpy as np
        _mplot = np.isfinite(sub) & np.isfinite(main_x)
        if not np.any(_mplot):
            self.proc_ax.set_xlabel("Photon energy (eV)")
            try:
                self.proc_ax.figure.canvas.draw_idle()
            except Exception:
                pass
            return


        self.proc_ax.plot(np.asarray(main_x)[_mplot], sub[_mplot], label="Background subtracted")
        try:
            self.proc_ax.relim(); self.proc_ax.autoscale_view(); self.proc_ax.figure.tight_layout(); self.canvas_proc.draw_idle()
        except Exception:
            pass
        self.proc_ax.set_xlabel("Photon energy (eV)")

    def init_manual_mode(self, main_x, main_y, auto_bg=None):
        """Prepare manual background mode: anchors + events.
        If auto_bg is None, compute a non-plotted automatic BG for seeding.
        """
        import numpy as np
        self.manual_mode = True
        self._drag_index = None
        self.manual_poly_degree = int(self.combo_poly.currentText())
    
        # Seed auto_bg if not provided
        if auto_bg is None:
            try:
                deg = int(self.combo_poly.currentText())
            except Exception:
                deg = 2
            try:
                pre = float(self.spin_preedge.value()) / 100.0
            except Exception:
                pre = 0.20
            auto_bg = self._apply_automatic_bg_new( 
                main_x, main_y, deg=deg, pre_edge_percent=pre, ax=self.proc_ax, do_plot=False
            )
    
        # Initialize anchors if absent
        if not hasattr(self, "manual_points") or not self.manual_points:
            n_seed = 4
            n = len(main_x)
            idxs = np.linspace(0, max(0, n - 1), n_seed).astype(int) if n > 0 else np.array([], dtype=int)
            self.manual_points = [{"x": float(main_x[i]), "y": float(auto_bg[i]), "artist": None} for i in idxs]
    
        # Clear old artists
        for pt in self.manual_points:
            art = pt.get("artist")
            if art is not None:
                try: art.remove()
                except Exception: pass
            pt["artist"] = None
    
        # Draw blue, pickable anchors (always visible)
        for pt in self.manual_points:
            (ln,) = self.proc_ax.plot([pt["x"]], [pt["y"]], marker="o", markersize=7,
                                      mfc="#66b3ff", mec="k", linestyle="None", zorder=6, label="_manual_anchor")
            try: ln.set_picker(5)
            except Exception: pass
            pt["artist"] = ln
    
        # Draw initial manual BG = auto BG (red dashed). Visibility is handled elsewhere.
        if getattr(self, "manual_bg_line", None) is not None:
            try: self.manual_bg_line.remove()
            except Exception: pass
            self.manual_bg_line = None
        self.manual_bg_line, = self.proc_ax.plot(main_x, auto_bg, "--", color="red", label="Background")
    
        # Connect events
        canvas = self.proc_ax.figure.canvas
        if not hasattr(self, "_mpl_cids"):
            self._mpl_cids = {}
        for key in ("press", "motion", "release"):
            cid = self._mpl_cids.get(key)
            if cid is not None:
                try: canvas.mpl_disconnect(cid)
                except Exception: pass
        self._mpl_cids["press"] = canvas.mpl_connect("button_press_event", self.on_press)
        self._mpl_cids["motion"] = canvas.mpl_connect("motion_notify_event", self.on_motion)
        self._mpl_cids["release"] = canvas.mpl_connect("button_release_event", self.on_release)
    
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def on_press(self, event):
        import numpy as np
        if not getattr(self, "manual_mode", False):
            return
        if getattr(event, "button", None) != 1:
            return
        if event.inaxes is not getattr(self, "proc_ax", None):
            return
        if not getattr(self, "manual_points", None):
            return
        # Cache current y-limits to keep scale stable while dragging
        try:
            self._drag_ylim = tuple(self.proc_ax.get_ylim())
        except Exception:
            self._drag_ylim = None
    
        xs = np.array([pt["x"] for pt in self.manual_points], dtype=float)
        ys = np.array([pt["y"] for pt in self.manual_points], dtype=float)
        if not len(xs) or event.x is None or event.y is None:
            return
        trans = self.proc_ax.transData
        pts_display = trans.transform(np.column_stack([xs, ys]))
        click = np.array([event.x, event.y], dtype=float)
        d2 = np.sum((pts_display - click) ** 2, axis=1)
        i_min = int(np.argmin(d2))
        if d2[i_min] <= 10.0 ** 2:
            self._drag_index = i_min
        else:
            self._drag_index = None

    def on_motion(self, event):
        import numpy as np
        if not getattr(self, "manual_mode", False):
            return
        i = getattr(self, "_drag_index", None)
        if i is None:
            return
        if event.inaxes is not getattr(self, "proc_ax", None):
            return
        if event.ydata is None or not np.isfinite(event.ydata):
            return
        try:
            x_fixed = float(self.manual_points[i]["x"])
        except Exception:
            return
        y_new = float(event.ydata)
        self.manual_points[i]["y"] = y_new
        art = self.manual_points[i].get("artist")
        if art is not None:
            try: art.set_data([x_fixed], [y_new])
            except Exception: pass
        xs = np.array([pt["x"] for pt in self.manual_points], dtype=float)
        ys = np.array([pt["y"] for pt in self.manual_points], dtype=float)
        try:
            deg = int(self.combo_poly.currentText())
        except Exception:
            deg = 2
        x_grid = None
        if getattr(self, "manual_bg_line", None) is not None:
            try:
                x_grid = np.asarray(self.manual_bg_line.get_xdata(), dtype=float)
            except Exception:
                x_grid = None
        if x_grid is None or x_grid.size == 0:
            lines = [ln for ln in self.proc_ax.get_lines() if ln is not getattr(self, "manual_bg_line", None)]
            for ln in lines:
                xd = ln.get_xdata()
                if xd is not None and len(xd):
                    x_grid = np.asarray(xd, dtype=float)
                    break
        if x_grid is None or x_grid.size == 0:
            x_min = np.nanmin(xs) if np.isfinite(xs).any() else 0.0
            x_max = np.nanmax(xs) if np.isfinite(xs).any() else 1.0
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                x_min, x_max = 0.0, 1.0
            x_grid = np.linspace(x_min, x_max, 512)
        try:
            bg, coeffs = _proc_robust_polyfit_on_normalized(self, xs, ys, deg, x_grid)
            self.manual_poly = coeffs
        except Exception:
            try: self.proc_ax.figure.canvas.draw_idle()
            except Exception: pass
            return
        if bg.size and ys.size and np.isfinite(ys[0]):
            bg[0] = ys[0]
        m = np.isfinite(x_grid) & np.isfinite(bg)
        x_plot = x_grid[m] if np.any(m) else np.array([], dtype=float)
        y_plot = bg[m]      if np.any(m) else np.array([], dtype=float)
        if getattr(self, "manual_bg_line", None) is None:
            if x_plot.size and y_plot.size:
                try:
                    (self.manual_bg_line,) = self.proc_ax.plot(x_plot, y_plot, "--", color="red", label="Background")
                except Exception:
                    self.manual_bg_line = None
        else:
            if x_plot.size and y_plot.size:
                try:
                    self.manual_bg_line.set_data(x_plot, y_plot)
                except Exception:
                    try: self.manual_bg_line.remove()
                    except Exception: pass
                    self.manual_bg_line = None
                    try:
                        (self.manual_bg_line,) = self.proc_ax.plot(x_plot, y_plot, "--", color="red", label="Background")
                    except Exception:
                        self.manual_bg_line = None
        ys_fin = ys[np.isfinite(ys)]
        y_candidates = []
        if y_plot.size: y_candidates.append(y_plot)
        if ys_fin.size: y_candidates.append(ys_fin)
        if y_candidates:
            all_y = np.concatenate(y_candidates)
            if all_y.size:
                y_min = float(np.nanmin(all_y)); y_max = float(np.nanmax(all_y))
                if np.isfinite(y_min) and np.isfinite(y_max):
                    # Compute a padded target box from current content
                    if y_max <= y_min:
                        pad_content = max(1e-6, abs(y_max) * 0.05 + 1e-6)
                    else:
                        pad_content = 0.05 * (y_max - y_min)
                    target_low  = y_min - pad_content
                    target_high = y_max + pad_content
    
                    # During drag, only EXPAND y-limits (no sudden zoom-in)
                    curr_low, curr_high = None, None
                    if hasattr(self, "_drag_ylim") and isinstance(self._drag_ylim, tuple):
                        curr_low, curr_high = self._drag_ylim
                    else:
                        try:
                            curr_low, curr_high = self.proc_ax.get_ylim()
                        except Exception:
                            pass
    
                    if curr_low is not None and curr_high is not None:
                        new_low  = min(curr_low, target_low)
                        new_high = max(curr_high, target_high)
                    else:
                        new_low, new_high = target_low, target_high
    
                    # Set limits; guard against identical bounds
                    if np.isfinite(new_low) and np.isfinite(new_high):
                        if new_high <= new_low:
                            new_high = new_low + 1.0
                        try:
                            self.proc_ax.set_ylim(new_low, new_high)
                        except Exception:
                            pass
        
        try: self.proc_ax.figure.canvas.draw_idle()
        except Exception: pass

    def on_release(self, event):
        if not getattr(self, "manual_mode", False):
            return
        if getattr(event, "button", None) != 1:
            return
        self._drag_index = None
        # Restore autoscale after drag ends (smoothly)
        try:
            self.proc_ax.relim()
            self.proc_ax.autoscale_view()
            self.proc_ax.figure.canvas.draw_idle()
        except Exception:
            pass
        # clear cached limits
        self._drag_ylim = None



def apply_normalization(viewer, abs_path: str, parent: str, y_data):
    """Normalize y_data by the selected I0 channel when viewer.chk_normalize is checked.
    Falls back to y_data on any error or if norm channel not found."""
    try:
        if not getattr(viewer, "chk_normalize", None) or not viewer.chk_normalize.isChecked():
            return y_data
        norm_channel = viewer.combo_norm.currentText()
        norm_path = f"{parent}/{norm_channel}" if parent else norm_channel
        with viewer._open_h5_read(abs_path) as f:
            if norm_path in f:
                norm = f[norm_path][()]
                return np.divide(y_data, norm, out=np.zeros_like(y_data, dtype=float), where=norm != 0)
    except Exception:
        pass
    return y_data


def _proc_safe_post_normalize(viewer, x, y, mode: str):
    """Apply an optional post-normalization to array y based on mode.
    Supported (case-insensitive): 'none', 'max', 'max=1', 'jump', 'jump=1', 'area', 'area=1'.
    Returns y unchanged on errors."""
    try:
        if not isinstance(mode, str):
            return y
        m = mode.strip().lower()
        yy = np.asarray(y, dtype=float)
        if yy.size == 0:
            return y

        # Build finite masks
        xf = np.asarray(x, dtype=float) if x is not None else None
        if xf is not None and xf.size:
            n = min(xf.size, yy.size)
            xf = xf[:n]; yy = yy[:n]
        fmask = np.isfinite(yy)
        if xf is not None and xf.size:
            fmask = fmask & np.isfinite(xf)
        yf = yy[fmask]
        xf = xf[fmask] if xf is not None and xf.size else None

        def _nonzero(val, eps=1e-15):
            return (val is not None) and np.isfinite(val) and (abs(val) > eps)

        if m in ("max", "max=1"):
            d = float(np.max(np.abs(yf))) if yf.size else None
            if _nonzero(d):
                return yy / d
            return y

        if m in ("jump", "jump=1"):
            d = float(yf[-1]) if yf.size else None  # divide by last finite y
            if _nonzero(d):
                return yy / d
            return y

        if m in ("area", "area=1"):
            if xf is None or xf.size < 2 or yf.size < 2:
                return y
            area = float(np.trapz(yf, xf))
            if _nonzero(area):
                return yy / area
            return y

        return y  # 'none' or unknown
    except Exception:
        return y



def apply_manual_bg(viewer, x, y):
    """Minimal manual background plot/update using viewer.manual_bg_points.
    If points exist, fit a polynomial of degree from combo_poly (or viewer.manual_poly_degree)
    and draw/update a dashed line on viewer.proc_ax."""
    try:
        pts = getattr(viewer, "manual_bg_points", None) or []
        if len(pts) < 2:
            # Nothing to draw; ensure any old line remains as-is
            return
        xpts = np.array([p[0] for p in pts], dtype=float)
        ypts = np.array([p[1] for p in pts], dtype=float)
        try:
            deg = int(viewer.combo_poly.currentText())
        except Exception:
            deg = int(getattr(viewer, "manual_poly_degree", 2))
        deg = max(0, min(deg, max(0, len(xpts) - 1)))
        coeffs = np.polyfit(xpts, ypts, deg)
        bg = np.polyval(coeffs, np.asarray(x, dtype=float))
        if getattr(viewer, "manual_bg_line", None) is None:
            (viewer.manual_bg_line,) = viewer.proc_ax.plot(x, bg, linestyle="--", linewidth=1.5, label="_manual_bg")
        else:
            viewer.manual_bg_line.set_data(x, bg)
        viewer.proc_ax.figure.canvas.draw_idle()
    except Exception:
        pass


def _proc_robust_polyfit_on_normalized(self, xs, ys, deg, x_eval):
    # free-function alias calling the mixin method (keeps old call sites working)
    return ProcessingMixin._proc_robust_polyfit_on_normalized(self, xs, ys, deg, x_eval)

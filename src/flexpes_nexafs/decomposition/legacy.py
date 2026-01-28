import sys
import os
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QTabWidget, QLabel,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy, QListWidget,
    QListWidgetItem, QDialog, QTextEdit, QDialogButtonBox, QComboBox, QInputDialog, QGroupBox, QFormLayout, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from sklearn.decomposition import PCA, NMF

try:
    from scipy.optimize import nnls as sp_nnls
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -----------------------------
# Utilities / Core algorithms
# -----------------------------

def nnls_solve(A, b):
    """Solve non-negative least squares: min_x ||Ax - b||^2, x >= 0."""
    if HAVE_SCIPY:
        x, _ = sp_nnls(A, b)
        return x
    # Fallback: Tikhonov-regularized least squares + projection
    AtA = A.T @ A + 1e-10 * np.eye(A.shape[1])
    Atb = A.T @ b
    x = np.linalg.solve(AtA, Atb)
    return np.maximum(x, 0.0)


def tikhonov_smooth_matrix(S, lam=0.0):
    """Simple 1D Tikhonov smoothing on each row of S."""
    if lam <= 0.0:
        return S
    k, m = S.shape
    S_sm = S.copy()
    for r in range(k):
        y = S[r]
        if m < 3:
            continue
        main = (1 + 2 * lam) * np.ones(m)
        off = (-lam) * np.ones(m - 1)
        cprime = np.zeros(m - 1)
        dprime = np.zeros(m)
        cprime[0] = off[0] / main[0]
        dprime[0] = y[0] / main[0]
        for i in range(1, m - 1):
            denom = main[i] - off[i - 1] * cprime[i - 1]
            cprime[i] = off[i] / denom
            dprime[i] = (y[i] - off[i - 1] * dprime[i - 1]) / denom
        denom = main[m - 1] - off[m - 2] * cprime[m - 2]
        dprime[m - 1] = (y[m - 1] - off[m - 2] * dprime[m - 2]) / denom
        x = np.zeros(m)
        x[m - 1] = dprime[m - 1]
        for i in range(m - 2, -1, -1):
            x[i] = dprime[i] - cprime[i] * x[i + 1]
        S_sm[r] = np.maximum(x, 0.0)
    return S_sm


def mcr_als(X, k, S_init=None, max_iter=500, tol=1e-7,
            closure=True, smooth=False, smooth_lambda=0.0):
    """
    Basic MCR-ALS with non-negativity (via NNLS) and optional closure + smoothing.

    X: (n_samples, n_energies), non-negative
    Returns C (n,k), S (k,m), err (RMSE)
    """
    n, m = X.shape
    rng = np.random.default_rng(0)
    if S_init is None:
        S = np.maximum(rng.random((k, m)), 1e-12)
    else:
        S = np.maximum(S_init.copy(), 1e-12)
    C = np.zeros((n, k))
    prev = np.inf
    converged = False
    for it in range(max_iter):
        # Update C by row-wise NNLS on S^T
        for i in range(n):
            C[i, :] = nnls_solve(S.T, X[i, :])
        C = np.maximum(C, 0.0)
        if closure:
            rs = C.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            C = C / rs

        # Update S by column-wise NNLS on C
        S_new = np.zeros_like(S)
        for j in range(m):
            S_new[:, j] = nnls_solve(C, X[:, j])
        S = np.maximum(S_new, 0.0)

        if smooth and smooth_lambda > 0.0:
            S = tikhonov_smooth_matrix(S, lam=smooth_lambda)

        Xhat = C @ S
        err = np.sqrt(np.mean((X - Xhat) ** 2))
        if abs(prev - err) < tol:
            converged = True
            break
        prev = err
    return C, S, err, it + 1, converged


def rmse_per_sample(X, Xhat):
    return np.sqrt(((Xhat - X) ** 2).mean(axis=1))


def mean_residual_vs_energy(X, Xhat):
    return (X - Xhat).mean(axis=0)



# -----------------------------
# Anchor-processing helpers (borrowed from this tool)
# -----------------------------

def cumulative_integral(y, x):
    """Cumulative integral of y(x) using trapezoids, returned on same grid.

    Used as a simple monotonic 'integral background' model.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 2:
        return np.zeros_like(y, dtype=float)
    x = x[:n]
    y = y[:n]
    dx = np.diff(x)
    dx = np.where(np.isfinite(dx), dx, 0.0)
    area = 0.5 * (y[:-1] + y[1:]) * dx
    out = np.zeros(n, dtype=float)
    out[1:] = np.cumsum(area)
    # normalize to ~[0,1] (keeps background coefficient interpretable)
    mn = float(np.nanmin(out)) if np.isfinite(np.nanmin(out)) else 0.0
    mx = float(np.nanmax(out)) if np.isfinite(np.nanmax(out)) else 0.0
    rng = mx - mn
    if rng > 0:
        out = (out - mn) / rng
    return out


def gaussian_convolve(y_raw, x_raw, width):
    """Gaussian broadening of y(x) with sigma = width (in same unit as x)."""
    width = float(width)
    if width <= 0:
        return np.asarray(y_raw, dtype=float).copy()
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    if len(x) < 3:
        return np.asarray(y_raw, dtype=float).copy()
    dx = np.median(np.diff(x))
    if not np.isfinite(dx) or dx == 0:
        return np.asarray(y_raw, dtype=float).copy()
    sigma_samples = width / dx
    if sigma_samples <= 0:
        return np.asarray(y_raw, dtype=float).copy()
    half_width = int(4 * sigma_samples)
    if half_width < 1:
        return np.asarray(y_raw, dtype=float).copy()
    kx = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * (kx / sigma_samples) ** 2)
    s = kernel.sum()
    if s <= 0:
        return np.asarray(y_raw, dtype=float).copy()
    kernel /= s
    y_pad = np.pad(y, (half_width, half_width), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def transform_curve_to_grid(x_raw, y_raw, x_target, dx=0.0, factor=1.0, broadening=0.0, background=0.0):
    """Apply (dx, factor, broadening, integral-background) and sample on x_target."""
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    x_target = np.asarray(x_target, dtype=float)

    y = y_raw * float(factor)
    bgp = float(background)
    if bgp != 0.0:
        bg = cumulative_integral(y, x_raw)
        y = y + bgp * bg
    bro = float(broadening)
    if bro > 0.0:
        y = gaussian_convolve(y, x_raw, bro)

    x = x_raw + float(dx)
    o = np.argsort(x)
    x_s = x[o]
    y_s = y[o]
    # Like np.interp usage in the curve-fitting script: clamp outside range
    return np.interp(x_target, x_s, y_s)

# -----------------------------
# Data model
# -----------------------------


class DataModel:
    def __init__(self):
        # Raw per-sample data (each spectrum can in principle have its own energy axis)
        self.energies = []           # list of 1D arrays, length = n_samples
        self.spectra = []            # list of 1D arrays, length = n_samples
        self.sample_labels = []      # list of strings, length = n_samples
        self.active_indices = None   # indices of selected samples

        # Legacy attributes (not used for analysis anymore, kept for backwards compatibility)
        self.energy = None           # common energy axis if all samples share it
        self.X = None                # stacked matrix if all samples share common axis

    @property
    def ready(self):
        return (
            len(self.spectra) > 0
            and len(self.sample_labels) == len(self.spectra)
        )

    def clear(self):
        """Clear all loaded data and derived matrices."""
        self.energies = []
        self.spectra = []
        self.sample_labels = []
        self.active_indices = None
        self.energy = None
        self.X = None

    def load_csv(self, paths):
        """Load one or multiple CSV files.

        Each CSV is expected to have:
            - first column: energy axis (eV)
            - remaining columns: spectra

        All spectra are stored individually and may in general live on
        different energy grids. Compatibility of the energy axes is
        enforced later (in PCA/NMF/MCR) when a specific subset of
        samples is selected for analysis.
        """
        # Normalize to a list of paths
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError("No CSV file selected.")

        energies = []
        spectra = []
        labels = []

        multiple_files = len(paths) > 1

        for path in paths:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                raise ValueError(
                    f"CSV '{os.path.basename(path)}' must have at least two columns: energy + spectra."
                )

            energy = df.iloc[:, 0].to_numpy(dtype=float)
            spec_df = df.iloc[:, 1:]
            base = os.path.splitext(os.path.basename(path))[0]

            for col_name in spec_df.columns.astype(str):
                y = spec_df[col_name].to_numpy(dtype=float)
                energies.append(energy)
                spectra.append(y)
                if multiple_files:
                    labels.append(f"{base} | {col_name}")
                else:
                    labels.append(col_name)

        self.energies = energies
        self.spectra = spectra
        self.sample_labels = labels
        self.active_indices = list(range(len(self.spectra)))

        # If all energy axes are identical, also populate legacy matrix form
        if len(self.energies) > 0:
            e0 = self.energies[0]
            same_grid = True
            for e in self.energies[1:]:
                if e.shape != e0.shape or not np.allclose(e, e0, rtol=1e-6, atol=1e-8):
                    same_grid = False
                    break
            if same_grid:
                self.energy = e0.copy()
                self.X = np.vstack(self.spectra)
            else:
                self.energy = None
                self.X = None

    @property
    def selected_indices(self):
        """Alias for the currently active (checked) spectra indices."""
        return list(self.active_indices) if self.active_indices is not None else []

    @selected_indices.setter
    def selected_indices(self, idx):
        self.active_indices = list(idx) if idx is not None else []

    def get_matrix_for_indices(self, indices):
        """Return a common energy axis and stacked matrix for a subset of samples.

        Parameters
        ----------
        indices : list[int]
            Indices of samples to include.

        Returns
        -------
        energy : 1D np.ndarray
        X : 2D np.ndarray, shape (len(indices), m)

        Raises
        ------
        ValueError
            If the selected samples do not share the same energy axis.
        """
        if not indices:
            raise ValueError("No samples selected.")

        # Reference grid from the first index
        first = indices[0]
        e0 = self.energies[first]
        rows = [self.spectra[first]]

        for idx in indices[1:]:
            e = self.energies[idx]
            if e.shape != e0.shape or not np.allclose(e, e0, rtol=1e-6, atol=1e-8):
                raise ValueError(
                    "Selected samples do not share a common energy axis."
                    "Choose a subset of spectra that live on the same grid."
                )
            rows.append(self.spectra[idx])

        X = np.vstack(rows)
        return e0, X
# -----------------------------
# Matplotlib helpers
# -----------------------------

def new_canvas():
    fig = Figure(figsize=(5, 3), tight_layout=True)
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return fig, canvas


def save_dataframe(parent, df: pd.DataFrame, default_name: str, dialog_title: str):
    """Save a DataFrame to CSV via a QFileDialog (comma-separated)."""
    path, _ = QFileDialog.getSaveFileName(
        parent,
        dialog_title,
        default_name,
        "CSV files (*.csv);;All files (*)",
    )
    if not path:
        return
    try:
        df.to_csv(path, index=False, sep=",")
    except Exception as exc:
        QMessageBox.critical(parent, "Export error", str(exc))



# -----------------------------
# Tabs
# -----------------------------


class DataTab(QWidget):
    """Tab for loading CSV and plotting raw spectra."""
    def __init__(self, model: DataModel):
        super().__init__()
        self.model = model

        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        left_w = QWidget()
        left = QVBoxLayout(left_w)
        splitter.addWidget(left_w)

        # Top row: Open / Clear / Help
        top_row = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV…")
        self.btn_open.clicked.connect(self.open_csv)
        self.btn_clear_all = QPushButton("Clear all")
        self.btn_clear_all.clicked.connect(self.clear_all)
        self.btn_help = QPushButton("Help")
        self.btn_help.clicked.connect(self.show_workflow_help)
        top_row.addWidget(self.btn_open)
        top_row.addWidget(self.btn_clear_all)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_help)
        left.addLayout(top_row)

        self.fig, self.canvas = new_canvas()
        self.ax = self.fig.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas, 1)

        # Right: list of samples
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.list_samples = QListWidget()
        # React to checkbox changes by updating selection + plot
        self.list_samples.itemChanged.connect(self.on_sample_item_changed)
        right.addWidget(QLabel("Samples"))

        # Buttons to quickly (un)check all samples
        btn_group = QHBoxLayout()
        self.btn_check_all = QPushButton("Check all")
        self.btn_check_all.clicked.connect(self.check_all_samples)
        self.btn_uncheck_all = QPushButton("Uncheck all")
        self.btn_uncheck_all.clicked.connect(self.uncheck_all_samples)
        btn_group.addWidget(self.btn_check_all)
        btn_group.addWidget(self.btn_uncheck_all)
        right.addLayout(btn_group)

        right.addWidget(self.list_samples, 1)


    def clear_all(self):
        """Clear all loaded data and reset downstream analysis state."""
        self.model.clear()
        self.list_samples.clear()
        self.ax.clear()
        self.canvas.draw()
        # Re-enable CSV loading (it may be disabled when data is injected from the main app)
        self.btn_open.setEnabled(True)

        # Best-effort: reset other tabs if they expose a reset/clear hook
        mw = self.window()
        for attr in ("tab_pca", "tab_nmf", "tab_mcr", "tab_anchors_cal", "tab_anchors_apply"):
            tab = getattr(mw, attr, None)
            if tab is None:
                continue
            for fn in ("reset", "clear", "clear_results", "on_new_data"):
                if hasattr(tab, fn):
                    try:
                        getattr(tab, fn)()
                    except Exception:
                        pass


    def open_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open CSV",
            "",
            "CSV files (*.csv);;All files (*)",
            options=options,
        )
        if not paths:
            return
        try:
            self.model.load_csv(paths)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self.list_samples.clear()
        for lab in self.model.sample_labels:
            item = QListWidgetItem(str(lab), self.list_samples)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)

        self.update_active_indices()
        self.plot_selected()

    def show_workflow_help(self):
            """Show a high-level help dialog for the recommended decomposition workflow."""
            html = """
            <h2>Recommended workflow (overview)</h2>
            <p>
            This tool supports multivariate decomposition of XAS datasets that show systematic spectral
            changes across a series (angle, temperature, pressure, time, chemical environment, etc.).
            </p>

            <ul>
              <li><b>PCA</b>: estimate the effective dimensionality (how many independent contributions are needed)
                  and identify dominant trends</li>
              <li><b>NMF</b>: obtain a non-negative decomposition that is often closer to physically meaningful
                  spectra and concentrations than raw PCA components</li>
              <li><b>MCR-ALS</b>: iteratively refine components and concentrations under constraints
                  (e.g. non-negativity) and evaluate reconstruction quality</li>
            </ul>

            <h2>Anchor spectra</h2>
            <p>
            Anchors let you incorporate known reference spectra (or robust components extracted from a calibration dataset)
            as fixed or guided basis spectra.
            </p>
            <ul>
              <li><b>Anchor calibration</b>: build and store a set of anchor spectra from a representative dataset</li>
              <li><b>Anchor application</b>: apply calibrated anchors to decompose new experimental datasets</li>
            </ul>

            <p>
            Use the <b>Help</b> buttons in the individual tabs for method- and panel-specific details.
            </p>
            """

            dlg = QDialog(self)
            dlg.setWindowTitle("Workflow help")
            dlg.setSizeGripEnabled(True)

            layout = QVBoxLayout(dlg)

            txt = QTextEdit()
            txt.setReadOnly(True)

            f = txt.font()
            f.setPointSize(f.pointSize() + 5)
            txt.setFont(f)

            txt.setLineWrapMode(QTextEdit.WidgetWidth)
            txt.setAcceptRichText(True)
            txt.setHtml(html)

            layout.addWidget(txt, 1)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok)
            buttons.accepted.connect(dlg.accept)
            layout.addWidget(buttons)

            dlg.resize(860, 600)
            dlg.exec_()

    def plot_all(self):
        """Plot all samples (ignores checkboxes)."""
        self.ax.clear()
        if not self.model.ready:
            self.canvas.draw()
            return
        for i, lab in enumerate(self.model.sample_labels):
            e = self.model.energies[i]
            y = self.model.spectra[i]
            self.ax.plot(e, y, label=str(lab))
        self.ax.set_xlabel("Photon energy (eV)")
        self.ax.set_ylabel("XAS (arb. units)")
        self.ax.legend(fontsize=8, ncol=2)
        self.ax.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas.draw()
    def update_active_indices(self):
        """Update model.active_indices from checkbox states."""
        if not self.model.ready:
            return
        indices = []
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            if item.checkState() == Qt.Checked:
                indices.append(i)
        self.model.active_indices = indices


    def plot_selected(self):
        """Plot only checked samples (model.active_indices)."""
        if not self.model.ready:
            return
        self.update_active_indices()
        idx = self.model.active_indices or []
        self.ax.clear()
        if not idx:
            self.canvas.draw()
            return
        for r in idx:
            e = self.model.energies[r]
            y = self.model.spectra[r]
            self.ax.plot(e, y, label=str(self.model.sample_labels[r]))
        self.ax.set_xlabel("Photon energy (eV)")
        self.ax.set_ylabel("XAS (arb. units)")
        self.ax.legend(fontsize=8, ncol=2)
        self.ax.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas.draw()
    def on_sample_item_changed(self, item):
        """Called when any sample checkbox is toggled (live update)."""
        self.update_active_indices()
        self.plot_selected()

    def check_all_samples(self):
        """Check all sample boxes and update plot once."""
        if not self.model.ready:
            return
        self.list_samples.blockSignals(True)
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            item.setCheckState(Qt.Checked)
        self.list_samples.blockSignals(False)
        self.update_active_indices()
        self.plot_selected()

    def uncheck_all_samples(self):
        """Uncheck all sample boxes and clear plot."""
        if not self.model.ready:
            return
        self.list_samples.blockSignals(True)
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            item.setCheckState(Qt.Unchecked)
        self.list_samples.blockSignals(False)
        self.update_active_indices()
        self.plot_selected()


class BaseAnalysisTab(QWidget):
    """
    Base for PCA / NMF / MCR-ALS tabs.

    Layout:
        - Controls row (auto k + k + method-specific controls + [stretch, Help, Run])
        - Top: components (left) and concentrations (right) with toolbars
        - Bottom: error metrics canvas (no toolbar)
    """
    def __init__(self, model: DataModel, title: str):
        super().__init__()
        self.model = model
        self.title = title
        self.k_last = None
        self.results = {}

        root = QVBoxLayout(self)

        # Controls row
        self.ctrl = QHBoxLayout()
        root.addLayout(self.ctrl)

        # Short status line (shows what happened after "Run")
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #444;")
        root.addWidget(self.lbl_status)

        self.chk_auto_k = QCheckBox("Auto k (PCA ≥99% EVR)")
        self.chk_auto_k.setChecked(False)
        self.chk_auto_k.setToolTip(
            "If enabled, k is chosen automatically from PCA so that the cumulative explained variance ratio (EVR) reaches 99%.\n"
            "This is a practical starting point for many smooth XAS series."
        )
        self.ctrl.addWidget(self.chk_auto_k)

        self.lbl_k = QLabel("k:")
        self.ctrl.addWidget(self.lbl_k)

        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 10)
        self.spin_k.setValue(2)
        self.spin_k.setEnabled(True)
        self.spin_k.setToolTip(
            "Manual number of components (k). Used only when Auto k is disabled.\n"
            "Chemically, k should roughly match the number of independent spectral patterns."
        )
        self.chk_auto_k.toggled.connect(self.spin_k.setDisabled)
        self.ctrl.addWidget(self.spin_k)

        # We will add method-specific controls in subclasses by using self.ctrl.addWidget(...)
        # After that, each subclass calls self.finalize_controls() to add Help + Run.

        # Plots area: components and concentrations side-by-side
        top = QHBoxLayout()
        root.addLayout(top, 2)

        self.fig_comp, self.canvas_comp = new_canvas()
        self.ax_comp = self.fig_comp.add_subplot(111)
        self.toolbar_comp = NavigationToolbar(self.canvas_comp, self)
        comp_box = QVBoxLayout()
        comp_box.addWidget(self.toolbar_comp)
        comp_box.addWidget(self.canvas_comp, 1)

        self.fig_conc, self.canvas_conc = new_canvas()
        self.ax_conc = self.fig_conc.add_subplot(111)
        self.toolbar_conc = NavigationToolbar(self.canvas_conc, self)
        conc_box = QVBoxLayout()
        conc_box.addWidget(self.toolbar_conc)
        conc_box.addWidget(self.canvas_conc, 1)

        top.addLayout(comp_box, 1)
        top.addLayout(conc_box, 1)

        # Error canvas (no toolbar)
        self.fig_err, self.canvas_err = new_canvas()
        self.ax_err = self.fig_err.add_subplot(111)
        root.addWidget(self.canvas_err, 1)

        # Run + Help are created here but added to layout in finalize_controls()
        # Run, Export and Help buttons are created here but added to layout in finalize_controls()
        self.btn_export = QPushButton("Export")
        self.btn_help = QPushButton("Help")
        self.btn_run = QPushButton("Run")
    def finalize_controls(self):
        """Call at end of subclass __init__ to add Export, Help and Run in the correct place."""
        self.ctrl.addStretch(1)
        self.ctrl.addWidget(self.btn_export)
        self.ctrl.addWidget(self.btn_help)
        self.ctrl.addWidget(self.btn_run)
        self.btn_help.clicked.connect(self.show_help)

    def export_results(self):
        """Default export handler – overridden in subclasses.

        If a subclass does not implement export, this will simply show a message.
        """
        if not self.results:
            QMessageBox.warning(self, "No results", f"Run {self.title} before exporting.")
            return
        QMessageBox.information(
            self,
            "Export not implemented",
            f"Export is not implemented for {self.title} tab.",
        )

    def _save_dataframe(self, df: pd.DataFrame, default_name: str, dialog_title: str):
        """Helper to save a DataFrame to CSV via a QFileDialog."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            default_name,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            df.to_csv(path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    # Shared helpers


    def choose_k(self, k_auto):
        if self.chk_auto_k.isChecked():
            return k_auto
        return self.spin_k.value()

    def draw_errors(self, rmse_per_sample_vec, mean_resid_energy,
                    method_name, k):
        self.ax_err.clear()
        x_idx = np.arange(len(rmse_per_sample_vec))
        self.ax_err.plot(
            x_idx, rmse_per_sample_vec, marker='o', linestyle='',
            label="per-sample RMSE"
        )
        self.ax_err.set_xlabel("Sample index")
        self.ax_err.set_ylabel("RMS error")
        self.ax_err.set_title(f"{method_name}: error metrics (k={k})")
        self.ax_err.grid(True, which='both', alpha=0.5, linewidth=0.8)
        self.ax_err.legend()
        self.canvas_err.draw()

    def set_k_default(self, k):
        self.k_last = k
        if self.chk_auto_k.isChecked():
            self.spin_k.setValue(k)

    # Help API: subclasses override get_help_text()

    def get_help_text(self) -> str:
        """Override in subclasses."""
        return "No help text defined."

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{self.title} help")
        dlg.setSizeGripEnabled(True)

        layout = QVBoxLayout(dlg)

        txt = QTextEdit()
        txt.setReadOnly(True)

        # Larger font for readability
        f = txt.font()
        f.setPointSize(f.pointSize() + 5)
        txt.setFont(f)

        # Wrap to widget width and render rich text (HTML with bold headings)
        txt.setLineWrapMode(QTextEdit.WidgetWidth)
        txt.setAcceptRichText(True)
        txt.setHtml(self.get_help_text())

        layout.addWidget(txt, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        dlg.resize(860, 600)
        dlg.exec_()


class PCATab(BaseAnalysisTab):
    def __init__(self, model: DataModel):
        super().__init__(model, "PCA")
        # No extra controls beyond k
        self.finalize_controls()
        self.btn_export.clicked.connect(self.export_results)
        self.btn_run.clicked.connect(self.run_pca)

    def get_help_text(self) -> str:
        return (
            "<b>PCA – Principal Component Analysis</b><br><br>"
            "PCA decomposes the data matrix X into a sum of orthogonal components:<br>"
            "X ≈ scores · loadings + mean. Here X is the matrix of spectra "
            "(one row per sample), the loadings are basis spectra (which may "
            "contain negative values), and the scores tell how strongly each "
            "component contributes to a given sample.<br><br>"
            "<b>Purpose in this script</b><br>"
            "• Estimate the effective rank (number of significant spectral patterns).<br>"
            "• Visualize the dominant trends in the series of spectra.<br><br>"
            "<b>Controls</b><br>"
            "• <b>Auto k (PCA ≥99% EVR)</b> – choose the smallest k such that the "
            "cumulative explained variance ratio (EVR) is ≥ 99%. This is a reasonable "
            "default for smooth XAS series and is limited to the range 2–6 to avoid "
            "overfitting.<br>"
            "• <b>k</b> – manual override for the number of principal components when "
            "Auto k is off. Chemically, k should roughly match the number of "
            "independent spectral patterns (species, trends).<br>"
            "• <b>Run</b> – perform PCA on the currently loaded and selected spectra.<br><br>"
            "<b>Plots</b><br>"
            "• <b>Left</b> – PCA loadings (basis spectra) offset by the mean spectrum. "
            "These are mathematical components; they can have negative values and are "
            "not directly species spectra.<br>"
            "• <b>Right</b> – PCA scores versus sample index. Smooth or monotonic "
            "trends in scores often indicate gradual changes such as oxidation state "
            "or composition along the series.<br>"
            "• <b>Bottom</b> – per-sample RMSE (root mean squared error) between the "
            "original spectra and the PCA reconstruction using k components. Lower "
            "RMSE means a better description of the dataset by those k components."
        )


    def run_pca(self):
        if not self.model.ready:
            QMessageBox.warning(self, "No data", "Load a CSV on the Data tab first.")
            return
        # Use only checked samples from Data tab
        idx_sel = (
            self.model.active_indices
            if self.model.active_indices is not None
            else list(range(len(self.model.sample_labels)))
        )
        if len(idx_sel) == 0:
            QMessageBox.warning(
                self,
                "No samples selected",
                "Check at least one sample on the Data tab before running the analysis.",
            )
            return

        # Build matrix only from samples that share a common energy axis
        try:
            e, X = self.model.get_matrix_for_indices(idx_sel)
        except ValueError as err:
            QMessageBox.warning(
                self,
                "Incompatible energy axes",
                str(err),
            )
            return

        # Handle missing values (NaNs) consistently (overlap trimming + optional interpolation)
        try:
            from flexpes_nexafs.utils.nan_policy import prepare_matrix_with_nan_policy
        except Exception:
            prepare_matrix_with_nan_policy = None

        if prepare_matrix_with_nan_policy is not None:
            labels = [self.model.sample_labels[i] for i in idx_sel]
            cleaned = prepare_matrix_with_nan_policy(
                self,
                e,
                X,
                labels,
                action_label="PCA",
            )
            if cleaned is None:
                return
            e, X = cleaned


        n, m = X.shape

        # Center data
        X_mean = X.mean(axis=0)
        Xc = X - X_mean

        # Fit PCA with as many components as possible but capped at 12
        pca = PCA(n_components=min(12, n, m), svd_solver="full", random_state=0)
        scores = pca.fit_transform(Xc)
        loadings = pca.components_
        evr = pca.explained_variance_ratio_

        # Number of components actually available
        max_k = len(evr)
        if max_k == 0:
            QMessageBox.warning(
                self,
                "PCA error",
                "PCA did not return any components. Please check your data.",
            )
            return

        # Auto-k logic with safe handling for small max_k
        if max_k == 1:
            # With a single spectrum (or effectively rank-1 data),
            # there is only one PCA component.
            k_auto = 1
        else:
            cum = np.cumsum(evr)
            k_auto = int(np.searchsorted(cum, 0.99) + 1)
            # Keep previous behaviour: 2–6 components, but never exceed max_k
            k_auto = max(2, min(6, k_auto, max_k))

        # Decide k (auto/manual) and clamp to max_k
        k = self.choose_k(k_auto)
        self.set_k_default(k_auto)

        if k > max_k:
            # Inform the user and clamp k
            QMessageBox.warning(
                self,
                "k reduced",
                f"Requested k = {k}, but PCA only produced {max_k} component(s).\n"
                f"Using k = {max_k} instead."
            )
            k = max_k

        # Status
        self.lbl_status.setText(
            f"PCA: n={n} spectra, k={k} (auto k={k_auto}), EVR@k={np.sum(evr[:k]):.3f}"
        )

        # Reconstruction with the chosen k
        Xhat = (scores[:, :k] @ loadings[:k, :]) + X_mean
        rmse_ps = rmse_per_sample(X, Xhat)
        resid_mean = mean_residual_vs_energy(X, Xhat)

        # Components (loadings offset by mean)
        self.ax_comp.clear()
        for i in range(k):
            self.ax_comp.plot(e, loadings[i] + X_mean, label=f"PC {i+1}")
        self.ax_comp.set_xlabel("Photon energy (eV)")
        self.ax_comp.set_ylabel("Loading (offset by mean)")
        self.ax_comp.set_title("PCA components")
        self.ax_comp.legend(fontsize=8)
        self.ax_comp.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_comp.draw()

        # Scores (concentrations-like)
        self.ax_conc.clear()
        for i in range(k):
            self.ax_conc.plot(
                np.arange(n), scores[:, i],
                marker='o', linestyle='', label=f"PC{i+1}"
            )
        self.ax_conc.set_xlabel("Sample index")
        self.ax_conc.set_ylabel("Score (arb.)")
        self.ax_conc.set_title("PCA scores")
        self.ax_conc.legend(fontsize=8)
        self.ax_conc.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_conc.draw()

        self.draw_errors(rmse_ps, resid_mean, "PCA", k)


        self.results = dict(
            k_auto=k_auto,
            k=k,
            scores=scores,
            loadings=loadings,
            mean=X_mean,
            evr=evr,
            energy=e,
            indices=idx_sel,
            rmse_ps=rmse_ps,
            resid_mean=resid_mean,
        )

    def export_results(self):
        """Export PCA plots to CSV: components, fractions (scores) or RMS errors."""
        if not self.results:
            QMessageBox.warning(self, "No PCA results", "Run PCA before exporting.")
            return
        option, ok = QInputDialog.getItem(
            self,
            "Export PCA results",
            "Select what to export:",
            ["Components", "Fractions", "RMS errors"],
            0,
            False,
        )
        if not ok:
            return

        k = int(self.results.get("k", 0) or 0)
        if k <= 0:
            QMessageBox.warning(self, "No PCA components", "There are no PCA components to export.")
            return

        if option == "Components":
            e = self.results.get("energy")
            loadings = self.results.get("loadings")
            X_mean = self.results.get("mean")
            if e is None or loadings is None or X_mean is None:
                QMessageBox.warning(self, "No PCA components", "Component data are not available for export.")
                return
            data = {"Energy_eV": e}
            for i in range(k):
                data[f"PC{i+1}"] = loadings[i] + X_mean
            df = pd.DataFrame(data)
            self._save_dataframe(df, "pca_components.csv", "Export PCA components")

        elif option == "Fractions":
            scores = self.results.get("scores")
            indices = self.results.get("indices")
            if scores is None or indices is None:
                QMessageBox.warning(self, "No PCA scores", "Score data are not available for export.")
                return
            labels = [self.model.sample_labels[i] for i in indices]
            data = {
                "SampleIndex": indices,
                "SampleLabel": labels,
            }
            for j in range(k):
                data[f"PC{j+1}"] = scores[:, j]
            df = pd.DataFrame(data)
            self._save_dataframe(df, "pca_fractions.csv", "Export PCA fractions")

        elif option == "RMS errors":
            rmse_ps = self.results.get("rmse_ps")
            indices = self.results.get("indices")
            if rmse_ps is None or indices is None:
                QMessageBox.warning(self, "No PCA RMSE", "RMSE data are not available for export.")
                return
            labels = [self.model.sample_labels[i] for i in indices]
            df = pd.DataFrame({
                "SampleIndex": indices,
                "SampleLabel": labels,
                "RMSE": rmse_ps,
            })
            self._save_dataframe(df, "pca_rmse.csv", "Export PCA RMSE")

class NMFTab(BaseAnalysisTab):
    def __init__(self, model: DataModel, pca_tab: PCATab):
        super().__init__(model, "NMF")
        self.pca_tab = pca_tab

        # Add NMF-specific controls AFTER k, BEFORE stretch
        self.ctrl.addWidget(QLabel("Init:"))
        self.cmb_init = QComboBox()
        self.cmb_init.addItems(["nndsvda", "random"])
        self.ctrl.addWidget(self.cmb_init)

        self.cmb_init.setToolTip(
            "Initialization for NMF. 'nndsvda' is a stable SVD-based guess (often recommended).\n"
            "'random' uses random starting values; use Seed to make runs reproducible."
        )

        self.ctrl.addWidget(QLabel("max_iter:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(100, 20000)
        self.spin_iter.setValue(3000)
        self.ctrl.addWidget(self.spin_iter)

        self.lbl_seed = QLabel("Seed:")
        self.lbl_seed.setToolTip(
            "Random seed for reproducibility.\n"
            "Only affects results when Init = 'random'.\n"
            "For Init = 'nndsvda', changing the seed will not change the result."
        )
        self.ctrl.addWidget(self.lbl_seed)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(0)
        self.spin_seed.setToolTip(
            "Random seed (integer) used only when Init = 'random'.\n"
            "Use different seeds to test stability; keep a fixed seed for reproducibility."
        )
        self.ctrl.addWidget(self.spin_seed)

        # Seed only matters for Init="random"
        self._update_seed_enabled()
        self.cmb_init.currentTextChanged.connect(lambda _t: self._update_seed_enabled())

        self.finalize_controls()
        self.btn_export.clicked.connect(self.export_results)
        self.btn_run.clicked.connect(self.run_nmf)

    
    def _update_seed_enabled(self):
        """Enable the Seed control only when Init is set to 'random'."""
        use = (self.cmb_init.currentText() == "random")
        if hasattr(self, "spin_seed"):
            self.spin_seed.setEnabled(use)
        if hasattr(self, "lbl_seed"):
            self.lbl_seed.setEnabled(use)

    def get_help_text(self) -> str:
        return (
            "<b>NMF – Non-Negative Matrix Factorization</b><br><br>"
            "NMF approximates the data matrix as X ≈ C · S with all elements constrained to be non-negative.<br>"
            "• <b>S</b> – component spectra (rows).<br>"
            "• <b>C</b> – contributions / concentrations (one row per sample).<br><br>"
            "Because intensities and concentrations cannot be negative in XAS, NMF often produces components that are easier to interpret "
            "chemically than PCA loadings. Still, the solution can depend on initialization and is not guaranteed to be unique.<br><br>"
            "<b>Controls</b><br>"
            "• <b>Auto k (PCA ≥99% EVR)</b> – uses PCA to suggest k (number of components) so that ≥99% of variance is captured.<br>"
            "• <b>k</b> – manual number of components when Auto k is off.<br>"
            "• <b>Init</b> – initialization method:<br>"
            "&nbsp;&nbsp;• <b>nndsvda</b> – Nonnegative Double Singular Value Decomposition (variant A). Stable SVD-based starting guess (recommended).<br>"
            "&nbsp;&nbsp;• <b>random</b> – random non-negative start (useful to test stability).<br>"
            "• <b>Seed</b> – random seed used <b>only</b> when Init = random. It is disabled for nndsvda because it would not change the result.<br>"
            "• <b>max_iter</b> – maximum iterations. Increase if convergence is slow.<br>"
            "• <b>Run</b> – performs NMF on the currently selected spectra.<br><br>"
            "<b>Status line</b><br>"
            "After Run, the status line reports <b>n</b> (how many spectra were used) and the <b>number of iterations actually performed</b>.<br><br>"
            "<b>Plots</b><br>"
            "• <b>Components</b> – extracted non-negative component spectra S.<br>"
            "• <b>Concentrations</b> – normalized rows of C (fractions per sample, summing to 1).<br>"
            "• <b>RMSE</b> – reconstruction error per sample (lower is better)."
        )


    def run_nmf(self):
        if not self.model.ready:
            QMessageBox.warning(self, "No data", "Load a CSV on the Data tab first.")
            return
        # Use only checked samples from Data tab
        idx_sel = (
            self.model.active_indices
            if self.model.active_indices is not None
            else list(range(len(self.model.sample_labels)))
        )
        if len(idx_sel) == 0:
            QMessageBox.warning(
                self,
                "No samples selected",
                "Check at least one sample on the Data tab before running the analysis.",
            )
            return

        # Build matrix only from samples that share a common energy axis
        try:
            e, X = self.model.get_matrix_for_indices(idx_sel)
        except ValueError as err:
            QMessageBox.warning(
                self,
                "Incompatible energy axes",
                str(err),
            )
            return

        # Handle missing values (NaNs) consistently (overlap trimming + optional interpolation)
        try:
            from flexpes_nexafs.utils.nan_policy import prepare_matrix_with_nan_policy
        except Exception:
            prepare_matrix_with_nan_policy = None

        if prepare_matrix_with_nan_policy is not None:
            labels = [self.model.sample_labels[i] for i in idx_sel]
            cleaned = prepare_matrix_with_nan_policy(
                self,
                e,
                X,
                labels,
                action_label="NMF",
            )
            if cleaned is None:
                return
            e, X = cleaned


        n, m = X.shape

        # k default from PCA

        k_auto = 2
        if self.pca_tab.results.get("k_auto") is not None:
            k_auto = int(self.pca_tab.results["k_auto"])
        k = self.choose_k(k_auto)
        self.set_k_default(k_auto)

        init = self.cmb_init.currentText()
        max_iter = int(self.spin_iter.value())
        seed = int(self.spin_seed.value())

        X_pos = np.maximum(X, 0.0)

        nmf = NMF(n_components=k, init=init, max_iter=max_iter, random_state=seed)
        C = nmf.fit_transform(X_pos)   # (n,k)
        n_iter = getattr(nmf, 'n_iter_', None)
        self.lbl_status.setText(f"NMF: n={n} spectra, iterations={n_iter}")
        seed_note = f"seed={seed}" if init == 'random' else "seed (ignored for nndsvda)"
        S = nmf.components_           # (k,m)
        Xhat = C @ S

        rmse_ps = rmse_per_sample(X, Xhat)
        resid_mean = mean_residual_vs_energy(X, Xhat)

        # Components
        self.ax_comp.clear()
        for i in range(k):
            self.ax_comp.plot(e, S[i], label=f"Comp {i+1}")
        self.ax_comp.set_xlabel("Photon energy (eV)")
        self.ax_comp.set_ylabel("Component intensity")
        self.ax_comp.set_title("NMF components")
        self.ax_comp.legend(fontsize=8)
        self.ax_comp.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_comp.draw()

        # Concentrations (fractions)
        C_frac = C / (C.sum(axis=1, keepdims=True) + 1e-12)
        self.ax_conc.clear()
        for i in range(k):
            self.ax_conc.plot(
                np.arange(n), C_frac[:, i],
                marker='o', linestyle='', label=f"Comp {i+1}"
            )
        self.ax_conc.set_xlabel("Sample index")
        self.ax_conc.set_ylabel("Estimated fraction")
        self.ax_conc.set_title("NMF concentrations (fraction)")
        self.ax_conc.legend(fontsize=8)
        self.ax_conc.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_conc.draw()

        self.draw_errors(rmse_ps, resid_mean, "NMF", k)


        self.results = dict(
            k=k,
            C=C_frac,
            S=S,
            Xhat=Xhat,
            rmse_ps=rmse_ps,
            resid_mean=resid_mean,
            energy=e,
            indices=idx_sel,
        )
        try:
            self.results_changed.emit()
        except Exception:
            pass

        try:
            self.results_changed.emit()
        except Exception:
            pass

    def export_results(self):
        """Export NMF plots to CSV: components, fractions or RMS errors."""
        if not self.results:
            QMessageBox.warning(self, "No NMF results", "Run NMF before exporting.")
            return
        option, ok = QInputDialog.getItem(
            self,
            "Export NMF results",
            "Select what to export:",
            ["Components", "Fractions", "RMS errors"],
            0,
            False,
        )
        if not ok:
            return

        k = int(self.results.get("k", 0) or 0)
        if k <= 0:
            QMessageBox.warning(self, "No NMF components", "There are no NMF components to export.")
            return

        if option == "Components":
            e = self.results.get("energy")
            S = self.results.get("S")
            if e is None or S is None:
                QMessageBox.warning(self, "No NMF components", "Component data are not available for export.")
                return
            data = {"Energy_eV": e}
            for i in range(k):
                data[f"Comp{i+1}"] = S[i]
            df = pd.DataFrame(data)
            self._save_dataframe(df, "nmf_components.csv", "Export NMF components")

        elif option == "Fractions":
            C = self.results.get("C")
            indices = self.results.get("indices")
            if C is None or indices is None:
                QMessageBox.warning(self, "No NMF fractions", "Fraction data are not available for export.")
                return
            labels = [self.model.sample_labels[i] for i in indices]
            data = {
                "SampleIndex": indices,
                "SampleLabel": labels,
            }
            for j in range(C.shape[1]):
                data[f"Comp{j+1}"] = C[:, j]
            df = pd.DataFrame(data)
            self._save_dataframe(df, "nmf_fractions.csv", "Export NMF fractions")

        elif option == "RMS errors":
            rmse_ps = self.results.get("rmse_ps")
            indices = self.results.get("indices")
            if rmse_ps is None or indices is None:
                QMessageBox.warning(self, "No NMF RMSE", "RMSE data are not available for export.")
                return
            labels = [self.model.sample_labels[i] for i in indices]
            df = pd.DataFrame({
                "SampleIndex": indices,
                "SampleLabel": labels,
                "RMSE": rmse_ps,
            })
            self._save_dataframe(df, "nmf_rmse.csv", "Export NMF RMSE")



class MCRTab(BaseAnalysisTab):
    results_changed = pyqtSignal()
    def __init__(self, model: DataModel, pca_tab: PCATab, nmf_tab: NMFTab):
        super().__init__(model, "MCR-ALS")
        self.pca_tab = pca_tab
        self.nmf_tab = nmf_tab

        # MCR-specific controls AFTER k, BEFORE stretch
        self.ctrl.addWidget(QLabel("Init:"))
        self.cmb_init = QComboBox()
        self.cmb_init.addItems(["NMF components", "random"])
        self.ctrl.addWidget(self.cmb_init)

        self.ctrl.addWidget(QLabel("max_iter:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(100, 50000)
        self.spin_iter.setValue(1000)
        self.ctrl.addWidget(self.spin_iter)

        self.lbl_tol = QLabel("ΔRMSE tol:")
        self.lbl_tol.setToolTip(
            "Convergence threshold: stop when the improvement in global RMSE between iterations\n"
            "is smaller than this value. Larger tol stops earlier (faster), smaller tol runs longer."
        )
        self.ctrl.addWidget(self.lbl_tol)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1e-12, 1e-2)
        self.spin_tol.setDecimals(10)
        self.spin_tol.setSingleStep(1e-8)
        self.spin_tol.setValue(1e-8)
        self.spin_tol.setToolTip(
            "Tolerance on change in global RMSE per iteration (ΔRMSE).\n"
            "If you see no effect, the algorithm may already be converging well before this limit,\n"
            "or it may be stopping at max_iter." 
        )
        self.ctrl.addWidget(self.spin_tol)

        self.chk_closure = QCheckBox("Closure (rows of C sum to 1)")
        self.chk_closure.setChecked(True)
        self.ctrl.addWidget(self.chk_closure)

        self.chk_smooth = QCheckBox("Smooth spectra")
        self.chk_smooth.setChecked(False)
        self.chk_smooth.setToolTip(
            "Apply gentle smoothing (regularization) to the component spectra S during MCR-ALS.\n"
            "This can suppress noise in extracted components, but too much smoothing can wash out real features."
        )
        self.ctrl.addWidget(self.chk_smooth)

        self.lbl_lambda = QLabel("λ:")
        self.lbl_lambda.setToolTip(
            "Smoothing strength (lambda). Larger values produce smoother component spectra.\n"
            "Typical mild values: 0.01–0.10. Set to 0 for no smoothing."
        )
        self.ctrl.addWidget(self.lbl_lambda)
        self.spin_lambda = QDoubleSpinBox()
        self.spin_lambda.setRange(0.0, 1.0)
        self.spin_lambda.setDecimals(3)
        self.spin_lambda.setSingleStep(0.01)
        self.spin_lambda.setValue(0.05)
        self.spin_lambda.setToolTip(
            "Smoothing strength (lambda). Only used when 'Smooth spectra' is enabled.\n"
            "Start with 0.05; reduce if peaks look too broadened; increase if components look noisy." 
        )
        self.spin_lambda.setEnabled(False)
        self.ctrl.addWidget(self.spin_lambda)

        self.chk_smooth.toggled.connect(self.spin_lambda.setEnabled)

        self.finalize_controls()
        self.btn_export.clicked.connect(self.export_results)
        self.btn_run.clicked.connect(self.run_mcr)

    def get_help_text(self) -> str:
        return (
            "<b>MCR-ALS – Multivariate Curve Resolution by Alternating Least Squares</b><br><br>"
            "MCR-ALS refines a factorization X ≈ C · S under chemical constraints. Like NMF it keeps C and S non-negative, "
            "and it can additionally enforce closure and smoothing of the component spectra.<br><br>"
            "• <b>S</b> – component spectra (rows).<br>"
            "• <b>C</b> – concentration profiles (one row per sample).<br><br>"
            "<b>Controls</b><br>"
            "• <b>Auto k (PCA ≥99% EVR)</b> – uses PCA to suggest the number of components k.<br>"
            "• <b>k</b> – manual number of components when Auto k is off.<br>"
            "• <b>Init</b> – initial guess for S (e.g. NMF components is often a good starting point).<br>"
            "• <b>max_iter</b> – maximum ALS iterations.<br>"
            "• <b>ΔRMSE tol</b> – convergence threshold: stop when the improvement in global RMSE per iteration becomes smaller than this value.<br>"
            "• <b>Closure</b> – if enabled, each row of C is normalized to sum to 1 (interpretable as fractions).<br>"
            "• <b>Smooth spectra</b> – applies gentle regularization to S during iterations to suppress noise in extracted components.<br>"
            "• <b>λ</b> – smoothing strength (lambda). Start small (≈0.01–0.1). Too large values can wash out real spectral features.<br>"
            "• <b>Run</b> – performs MCR-ALS on the currently selected spectra.<br><br>"
            "<b>Status line</b><br>"
            "After Run, the status line reports <b>n</b> (how many spectra were used) and the <b>number of iterations actually performed</b>.<br><br>"
            "<b>Plots</b><br>"
            "• <b>Components</b> – refined component spectra S (often most chemically interpretable).<br>"
            "• <b>Concentrations</b> – concentration profiles C vs sample index (with closure: fractions that sum to 1).<br>"
            "• <b>RMSE</b> – reconstruction error per sample (lower is better)."
        )


    def run_mcr(self):
        if not self.model.ready:
            QMessageBox.warning(self, "No data", "Load a CSV on the Data tab first.")
            return
        # Use only checked samples from Data tab
        idx_sel = (
            self.model.active_indices
            if self.model.active_indices is not None
            else list(range(len(self.model.sample_labels)))
        )
        if len(idx_sel) == 0:
            QMessageBox.warning(
                self,
                "No samples selected",
                "Check at least one sample on the Data tab before running the analysis.",
            )
            return

        # Build matrix only from samples that share a common energy axis
        try:
            e, X = self.model.get_matrix_for_indices(idx_sel)
        except ValueError as err:
            QMessageBox.warning(
                self,
                "Incompatible energy axes",
                str(err),
            )
            return

        # Handle missing values (NaNs) consistently (overlap trimming + optional interpolation)
        try:
            from flexpes_nexafs.utils.nan_policy import prepare_matrix_with_nan_policy
        except Exception:
            prepare_matrix_with_nan_policy = None

        if prepare_matrix_with_nan_policy is not None:
            labels = [self.model.sample_labels[i] for i in idx_sel]
            cleaned = prepare_matrix_with_nan_policy(
                self,
                e,
                X,
                labels,
                action_label="MCR",
            )
            if cleaned is None:
                return
            e, X = cleaned


        n, m = X.shape

        # k default from PCA

        k_auto = 2
        if self.pca_tab.results.get("k_auto") is not None:
            k_auto = int(self.pca_tab.results["k_auto"])
        k = self.choose_k(k_auto)
        self.set_k_default(k_auto)

        init_choice = self.cmb_init.currentText()
        max_iter = int(self.spin_iter.value())
        tol = float(self.spin_tol.value())
        closure = self.chk_closure.isChecked()
        smooth = self.chk_smooth.isChecked()
        lam = float(self.spin_lambda.value()) if smooth else 0.0

        # Init S from NMF if requested
        S_init = None
        if init_choice == "NMF components" and self.nmf_tab.results.get("S") is not None:
            S_init = self.nmf_tab.results["S"]
            if S_init.shape[0] != k:
                S_init = None  # rank changed

        # Ensure any initial guess has the correct energy grid length
        if S_init is not None and S_init.shape[1] != m:
            S_init = None

        X_pos = np.maximum(X, 0.0)
        C, S, err, n_iter, converged = mcr_als(
            X_pos, k=k, S_init=S_init, max_iter=max_iter,
            tol=tol, closure=closure, smooth=smooth,
            smooth_lambda=lam
        )
        self.lbl_status.setText(f"MCR-ALS: n={n} spectra, iterations={n_iter}")
        Xhat = C @ S


        rmse_ps = rmse_per_sample(X, Xhat)
        resid_mean = mean_residual_vs_energy(X, Xhat)

        # Components
        self.ax_comp.clear()
        for i in range(k):
            self.ax_comp.plot(e, S[i], label=f"Comp {i+1}")
        self.ax_comp.set_xlabel("Photon energy (eV)")
        self.ax_comp.set_ylabel("Component intensity")
        self.ax_comp.set_title("MCR-ALS components")
        self.ax_comp.legend(fontsize=8)
        self.ax_comp.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_comp.draw()

        # Concentrations
        self.ax_conc.clear()
        for i in range(k):
            self.ax_conc.plot(
                np.arange(n), C[:, i],
                marker='o', linestyle='', label=f"Comp {i+1}"
            )
        self.ax_conc.set_xlabel("Sample index")
        self.ax_conc.set_ylabel("Estimated fraction")
        self.ax_conc.set_title("MCR-ALS concentrations")
        self.ax_conc.legend(fontsize=8)
        self.ax_conc.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_conc.draw()

        self.draw_errors(rmse_ps, resid_mean, "MCR-ALS", k)


        self.results = dict(
            k=k,
            C=C,
            S=S,
            Xhat=Xhat,
            rmse_ps=rmse_ps,
            resid_mean=resid_mean,
            energy=e,
            indices=idx_sel,
        )

    def export_results(self):
        """Export MCR-ALS plots to CSV: components, fractions or RMS errors."""
        if not self.results:
            QMessageBox.warning(self, "No MCR-ALS results", "Run MCR-ALS before exporting.")
            return
        option, ok = QInputDialog.getItem(
            self,
            "Export MCR-ALS results",
            "Select what to export:",
            ["Components", "Fractions", "RMS errors"],
            0,
            False,
        )
        if not ok:
            return

        k = int(self.results.get("k", 0) or 0)
        if k <= 0:
            QMessageBox.warning(self, "No MCR-ALS components", "There are no MCR-ALS components to export.")
            return

        if option == "Components":
            e = self.results.get("energy")
            S = self.results.get("S")
            if e is None or S is None:
                QMessageBox.warning(self, "No MCR-ALS components", "Component data are not available for export.")
                return
            data = {"Energy_eV": e}
            for i in range(k):
                data[f"Comp{i+1}"] = S[i]
            df = pd.DataFrame(data)
            self._save_dataframe(df, "mcr_components.csv", "Export MCR-ALS components")

        elif option == "Fractions":
            C = self.results.get("C")
            indices = self.results.get("indices")
            if C is None or indices is None:
                QMessageBox.warning(self, "No MCR-ALS fractions", "Fraction data are not available for export.")
                return
            labels = [self.model.sample_labels[i] for i in indices]
            data = {
                "SampleIndex": indices,
                "SampleLabel": labels,
            }
            for j in range(C.shape[1]):
                data[f"Comp{j+1}"] = C[:, j]
            df = pd.DataFrame(data)
            self._save_dataframe(df, "mcr_fractions.csv", "Export MCR-ALS fractions")

        elif option == "RMS errors":
            rmse_ps = self.results.get("rmse_ps")
            indices = self.results.get("indices")
            if rmse_ps is None or indices is None:
                QMessageBox.warning(self, "No MCR-ALS RMSE", "RMSE data are not available for export.")
                return
            labels = [self.model.sample_labels[i] for i in indices]
            df = pd.DataFrame({
                "SampleIndex": indices,
                "SampleLabel": labels,
                "RMSE": rmse_ps,
            })
            self._save_dataframe(df, "mcr_rmse.csv", "Export MCR-ALS RMSE")



# -----------------------------
# Anchors tabs (Calibrate / Apply)
# -----------------------------

class AnchorsModel:
    """Holds raw anchors and calibrated ('good') anchors."""
    def __init__(self):
        self.energies = []        # list of 1D arrays
        self.spectra = []         # list of 1D arrays
        self.labels = []          # list of strings

        # Per-anchor parameters (same set as in this tool)
        self.params = []          # list of dicts with keys: dx,factor,broadening,background and constraint flags/within

        # Calibrated anchors sampled on a specific grid
        self.good_energy = None   # 1D array
        self.good_spectra = None  # 2D array shape (n_anchors, n_energy)
        self.good_params = None   # list of dicts (final parameters)

    def clear(self):
        self.energies = []
        self.spectra = []
        self.labels = []
        self.params = []
        self.good_energy = None
        self.good_spectra = None
        self.good_params = None

    @property
    def ready(self):
        return len(self.spectra) > 0 and len(self.labels) == len(self.spectra)

    def load_csv(self, path):
        """Load anchors from a single CSV, replacing any existing anchors."""
        self.clear()
        self.add_csv(path)

    def load_csvs(self, paths):
        """Load anchors from multiple CSV files, replacing any existing anchors."""
        self.clear()
        for p in paths:
            self.add_csv(p)

    def add_csv(self, path):
        """Append anchors from a CSV file.

        CSV format: first column = energy; remaining columns = one or more anchor spectra.
        If a file contains a single anchor (2 columns), the default label is the file stem.
        """
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("Anchors CSV must have at least two columns: energy + one anchor.")

        e = df.iloc[:, 0].to_numpy(dtype=float)
        file_stem = os.path.splitext(os.path.basename(path))[0]

        for j in range(1, df.shape[1]):
            y = df.iloc[:, j].to_numpy(dtype=float)
            col_name = str(df.columns[j]) if df.columns is not None else ""
            col_name = col_name.strip()

            if df.shape[1] == 2:
                label = file_stem if (col_name == "" or col_name.lower().startswith("unnamed")) else col_name
            else:
                base = col_name if (col_name != "" and not col_name.lower().startswith("unnamed")) else f"col{j}"
                label = f"{file_stem}:{base}"

            self.energies.append(e)
            self.spectra.append(y)
            self.labels.append(label)

            self.params.append(dict(
                dx=0.0,
                factor=1.0,
                broadening=0.0,
                background=0.0,
                c_dx_enabled=False,
                c_dx_within=0.0,
                c_factor_enabled=False,
                c_factor_within=0.0,
                c_broad_enabled=False,
                c_broad_within=0.0,
                c_bg_enabled=False,
                c_bg_within=0.0,
            ))


class AnchorsCalibrateTab(QWidget):
    anchors_changed = pyqtSignal()
    """Calibrate anchors by fitting MCR components as mixtures of transformed anchors."""
    def __init__(self, model: DataModel, mcr_tab: 'MCRTab'):
        super().__init__()
        self.model = model
        self.mcr_tab = mcr_tab
        try:
            self.mcr_tab.results_changed.connect(self.on_mcr_results_updated)
        except Exception:
            pass
        self.anchors = AnchorsModel()

        self._last_weights = None        # (n_comps, n_anchors_used)
        self._last_comp_indices = None   # list of comp indices
        self._last_comp_rmse = None
        self._fit_curves = {}      # vector of RMSE per comp

        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        left_w = QWidget()
        left = QVBoxLayout(left_w)
        splitter.addWidget(left_w)

        top = QHBoxLayout()
        left.addLayout(top, 2)

        self.fig_comp, self.canvas_comp = new_canvas()
        self.ax_comp = self.fig_comp.add_subplot(111)
        self.toolbar_comp = NavigationToolbar(self.canvas_comp, self)
        box_comp = QVBoxLayout()
        box_comp.addWidget(self.toolbar_comp)
        box_comp.addWidget(self.canvas_comp, 1)
        top.addLayout(box_comp, 1)

        self.fig_anch, self.canvas_anch = new_canvas()
        self.ax_anch = self.fig_anch.add_subplot(111)
        self.toolbar_anch = NavigationToolbar(self.canvas_anch, self)
        box_anch = QVBoxLayout()
        box_anch.addWidget(self.toolbar_anch)
        box_anch.addWidget(self.canvas_anch, 1)
        top.addLayout(box_anch, 1)

        self.fig_err, self.canvas_err = new_canvas()
        self.ax_err = self.fig_err.add_subplot(111)
        left.addWidget(self.canvas_err, 1)

        # Right: controls
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #444;")
        right.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV anchors…")
        self.btn_open.clicked.connect(self.open_anchors)
        btn_row.addWidget(self.btn_open)


        self.btn_run = QPushButton("Run calibration")
        self.btn_run.clicked.connect(self.run_calibration)
        btn_row.addWidget(self.btn_run)
        right.addLayout(btn_row)

        right.addWidget(QLabel("Anchors (used in calibration)"))
        self.list_anchors = QListWidget()
        self.list_anchors.itemChanged.connect(lambda _: self._plot_results_from_state())
        self.list_anchors.currentRowChanged.connect(self._on_anchor_selection_changed)
        right.addWidget(self.list_anchors, 1)

        right.addWidget(QLabel("MCR components to use (from last run)"))
        self.list_comps = QListWidget()
        self.list_comps.itemChanged.connect(lambda _: self._plot_results_from_state())
        right.addWidget(self.list_comps, 1)

        right.addWidget(QLabel("Fitted components (from calibration)"))
        self.list_fit = QListWidget()
        self.list_fit.itemChanged.connect(lambda _: self._plot_results_from_state())
        right.addWidget(self.list_fit, 1)


        grp_opt = QGroupBox("Optimize parameters")
        form_opt = QFormLayout(grp_opt)
        self.chk_opt_dx = QCheckBox("Optimize dX (shift)")
        self.chk_opt_dx.setChecked(True)
        self.chk_opt_bro = QCheckBox("Optimize broadening")
        self.chk_opt_bro.setChecked(False)
        self.chk_opt_fac = QCheckBox("Optimize factor (Y×)")
        self.chk_opt_fac.setChecked(False)
        self.chk_opt_bg = QCheckBox("Optimize background (integral)")
        self.chk_opt_bg.setChecked(True)
        form_opt.addRow(self.chk_opt_dx)
        form_opt.addRow(self.chk_opt_bro)
        form_opt.addRow(self.chk_opt_fac)
        form_opt.addRow(self.chk_opt_bg)
        right.addWidget(grp_opt)

        grp = QGroupBox("Selected anchor (start values + optional constraints)")
        form = QFormLayout(grp)

        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(-1000.0, 1000.0)
        self.spin_dx.setDecimals(4)
        self.spin_dx.setSingleStep(0.1)

        self.spin_factor = QDoubleSpinBox()
        self.spin_factor.setRange(0.0, 1000.0)
        self.spin_factor.setDecimals(4)
        self.spin_factor.setSingleStep(0.05)
        self.spin_factor.setValue(1.0)

        self.spin_bro = QDoubleSpinBox()
        self.spin_bro.setRange(0.0, 1000.0)
        self.spin_bro.setDecimals(4)
        self.spin_bro.setSingleStep(0.05)

        self.spin_bg = QDoubleSpinBox()
        self.spin_bg.setRange(0.0, 1000.0)
        self.spin_bg.setDecimals(4)
        self.spin_bg.setSingleStep(0.01)

        form.addRow("dX (eV):", self.spin_dx)
        form.addRow("Factor (Y×):", self.spin_factor)
        form.addRow("Broadening σ (eV):", self.spin_bro)
        form.addRow("Background (integral):", self.spin_bg)

        self.chk_c_dx = QCheckBox("Constrain dX within ±")
        self.spin_c_dx = QDoubleSpinBox()
        self.spin_c_dx.setRange(0.0, 1000.0)
        self.spin_c_dx.setDecimals(4)
        self.spin_c_dx.setSingleStep(0.05)
        form.addRow(self.chk_c_dx, self.spin_c_dx)

        self.chk_c_fac = QCheckBox("Constrain factor within ±")
        self.spin_c_fac = QDoubleSpinBox()
        self.spin_c_fac.setRange(0.0, 1000.0)
        self.spin_c_fac.setDecimals(4)
        self.spin_c_fac.setSingleStep(0.05)
        form.addRow(self.chk_c_fac, self.spin_c_fac)

        self.chk_c_bro = QCheckBox("Constrain broadening within ±")
        self.spin_c_bro = QDoubleSpinBox()
        self.spin_c_bro.setRange(0.0, 1000.0)
        self.spin_c_bro.setDecimals(4)
        self.spin_c_bro.setSingleStep(0.05)
        form.addRow(self.chk_c_bro, self.spin_c_bro)

        self.chk_c_bg = QCheckBox("Constrain background within ±")
        self.spin_c_bg = QDoubleSpinBox()
        self.spin_c_bg.setRange(0.0, 1000.0)
        self.spin_c_bg.setDecimals(4)
        self.spin_c_bg.setSingleStep(0.01)
        form.addRow(self.chk_c_bg, self.spin_c_bg)

        # connect editor
        for w in [self.spin_dx, self.spin_factor, self.spin_bro, self.spin_bg,
                  self.chk_c_dx, self.chk_c_fac, self.chk_c_bro, self.chk_c_bg,
                  self.spin_c_dx, self.spin_c_fac, self.spin_c_bro, self.spin_c_bg]:
            if isinstance(w, QCheckBox):
                w.toggled.connect(self._save_anchor_editor_to_model)
            else:
                w.valueChanged.connect(self._save_anchor_editor_to_model)

        right.addWidget(grp)

        grp_fit = QGroupBox("")
        form_fit = QFormLayout(grp_fit)
        self.spin_max_outer = QSpinBox()
        self.spin_max_outer.setRange(10, 2000)
        self.spin_max_outer.setValue(200)
        form_fit.addRow("Max iterations:", self.spin_max_outer)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_anchors)

        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_results)

        self.btn_help = QPushButton("Help")
        self.btn_help.clicked.connect(self.show_help)

        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        for b in (self.btn_clear, self.btn_export, self.btn_help):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btns.addWidget(self.btn_clear, 1)
        btns.addWidget(self.btn_export, 1)
        btns.addWidget(self.btn_help, 1)
        btns_w = QWidget()
        btns_w.setLayout(btns)
        form_fit.addRow(btns_w)
        right.addWidget(grp_fit)

        self._plot_empty()

    def get_help_text(self) -> str:
        return (
            "<b>Anchors – Calibrate</b><br><br>"
            "Goal: starting from raw anchors (e.g. reference or theoretical spectra for distinct chemical states), build <b>calibrated (“good”) anchors</b> "
            "that better match the experimental dataset’s energy scale and background, and then use them in ‘Anchors – Apply’.<br><br>"
            "<b>Typical workflow</b><br>"
            "1) Run <b>PCA</b> to estimate a suitable number of components.<br>"
            "2) Run <b>NMF</b> and/or <b>MCR-ALS</b> to obtain component spectra Comp1…CompK.<br>"
            "3) Load raw anchors (CSV).<br>"
            "4) Fit each selected component as a <b>non-negative mixture</b> of transformed anchors. Selected per-anchor parameters can be optimized to improve agreement.<br><br>"
            "<b>Adjustable parameters</b><br>"
            "• <b>dX</b> (eV): energy shift. Often the most important correction.<br>"
            "• <b>Background (integral)</b>: adds a smooth monotonic background proportional to the anchor’s cumulative integral. "
            "This is especially useful for theoretical anchors that may lack an experimental background contribution.<br>"
            "• <b>Broadening σ</b> (eV): Gaussian broadening. Can help in some cases, but it is <b>not</b> optimized by default.<br>"
            "• <b>Factor</b>: overall scaling of an anchor. Usually least relevant for area-normalized data and is OFF by default.<br><br>"
            "<b>Optimization toggles</b><br>"
            "For stability, only <b>dX</b> and <b>Background</b> are optimized by default. Enable broadening only when justified; enable Factor rarely.<br><br>"
            "<b>Constraints “within ±”</b><br>"
            "If enabled for a parameter, it is constrained to stay within ±(within) around the current value during the fit.<br><br>"
            "<b>Outputs</b><br>"
            "• Components plot: selected MCR components and their best mixtures of anchors.<br>"
            "• Anchors plot: calibrated (“good”) anchors on the MCR energy grid (used by ‘Anchors – Apply’).<br>"
            "• Error plot: RMSE per selected component."
        )

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Help – Anchors – Calibrate")
        dlg.setSizeGripEnabled(True)
        layout = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        f = txt.font()
        f.setPointSize(f.pointSize() + 5)
        txt.setFont(f)
        txt.setLineWrapMode(QTextEdit.WidgetWidth)
        txt.setAcceptRichText(True)
        txt.setHtml(self.get_help_text())
        layout.addWidget(txt, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        dlg.resize(900, 650)
        dlg.exec_()

    def _plot_empty(self):
        self.ax_comp.clear()
        self.ax_comp.set_title("Run MCR-ALS, then load anchors and run calibration")
        self.ax_comp.set_xlabel("Photon energy (eV)")
        self.ax_comp.set_ylabel("XAS (arb.)")
        self.ax_comp.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_comp.draw()

        self.ax_anch.clear()
        self.ax_anch.set_title("Anchors")
        self.ax_anch.set_xlabel("Photon energy (eV)")
        self.ax_anch.set_ylabel("XAS (arb.)")
        self.ax_anch.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_anch.draw()

        self.ax_err.clear()
        self.ax_err.set_title("RMS errors")
        self.ax_err.set_xlabel("Index")
        self.ax_err.set_ylabel("RMS error")
        self.ax_err.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_err.draw()

    def _refresh_components_list(self):
        self.list_comps.blockSignals(True)
        self.list_comps.clear()
        self.list_fit.clear()
        S = self.mcr_tab.results.get("S", None)
        if S is None:
            self.list_comps.blockSignals(False)
            return
        k = S.shape[0]
        for i in range(k):
            item = QListWidgetItem(f"Comp {i+1}", self.list_comps)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
        self.list_comps.blockSignals(False)


    def on_mcr_results_updated(self):
        """Refresh component list and plot when MCR-ALS results change."""
        self._refresh_components_list()
        self._plot_mcr_components()

    def _plot_mcr_components(self):
        """Plot using unified plotting logic."""
        self._plot_results_from_state()

    def open_anchors(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Open CSV anchors")
        dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.resize(900, 520)
        if dialog.exec_():
            paths = dialog.selectedFiles()
        else:
            paths = []
        if not paths:
            return
        try:
            self.anchors.load_csvs(paths)
        except Exception as exc:
            QMessageBox.critical(self, "Anchors load error", str(exc))
            return

        self.list_anchors.blockSignals(True)
        self.list_anchors.clear()
        for lab in self.anchors.labels:
            item = QListWidgetItem(str(lab), self.list_anchors)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
        self.list_anchors.blockSignals(False)

        if self.list_anchors.count() > 0:
            self.list_anchors.setCurrentRow(0)
        self.lbl_status.setText(f"Loaded {len(self.anchors.labels)} anchor(s).")

        self._refresh_components_list()
        # Plot using unified plotting (unique anchor colors)
        self._plot_results_from_state()

    def _comp_item_by_index(self, comp_index):
        """Return QListWidgetItem for 'Comp {i}' if present, else None."""
        for r in range(self.list_comps.count()):
            it = self.list_comps.item(r)
            if it and it.text().strip() == f"Comp {comp_index+1}":
                return it
        return None

    def _fit_item_by_index(self, comp_index):
        if not hasattr(self, "list_fit"):
            return None
        for r in range(self.list_fit.count()):
            it = self.list_fit.item(r)
            if it and it.data(Qt.UserRole) == comp_index:
                return it
        return None

    def _is_comp_visible(self, comp_index):
        it = self._comp_item_by_index(comp_index)
        return (it is None) or (it.checkState() == Qt.Checked)

    def _is_fit_visible(self, comp_index):
        it = self._fit_item_by_index(comp_index)
        return (it is not None) and (it.checkState() == Qt.Checked)

    def _anchor_color(self, j):
        """Anchor colors from Matplotlib 'tab10' colormap."""
        try:
            from matplotlib import cm
            cmap = cm.get_cmap("tab10")
            return cmap((j + 2) % 10)
        except Exception:
            # Fallback: return None to let Matplotlib choose default cycle
            return None



    def _comp_color(self, comp_index):
        from matplotlib import cm
        cmap = cm.get_cmap("tab10")
        return cmap(comp_index % cmap.N)

    def _plot_results_from_state(self):
        """Replot components + fitted components (if available) and anchors using checkbox state."""
        S = self.mcr_tab.results.get("S", None)
        e = self.mcr_tab.results.get("energy", None)
        if S is None or e is None:
            self._plot_empty()
            return

        # --- Components canvas ---
        self.ax_comp.clear()

        # Plot original components that are checked
        for ci in range(S.shape[0]):
            if not self._is_comp_visible(ci):
                continue
            self.ax_comp.plot(e, S[ci], color=self._comp_color(ci), label=f"Comp {ci+1}")

        # Plot fitted components (dashed) if available & checked
        if getattr(self, "_fit_curves", None):
            for ci, yhat in self._fit_curves.items():
                if self._is_fit_visible(ci):
                    self.ax_comp.plot(e, yhat, linestyle="--", color=self._comp_color(ci), label=f"Fit {ci+1}")

        self.ax_comp.set_title("MCR components and fitted components")
        self.ax_comp.set_xlabel("Photon energy (eV)")
        self.ax_comp.set_ylabel("XAS (arb.)")
        self.ax_comp.grid(True, which="both", alpha=0.5, linewidth=0.8)
        if self.ax_comp.lines:
            self.ax_comp.legend(fontsize=8)
        self.canvas_comp.draw()

        # --- Anchors canvas ---
        self.ax_anch.clear()
        if self.anchors.good_energy is not None and self.anchors.good_spectra is not None:
            x = self.anchors.good_energy
            labels_used = [self.anchors.labels[j] for j in self._selected_anchor_indices()]
            for j, lab in enumerate(labels_used):
                self.ax_anch.plot(x, self.anchors.good_spectra[j, :], color=self._anchor_color(j), label=str(lab))
            self.ax_anch.set_title("Calibrated ('good') anchors on MCR grid")
        else:
            for j in range(len(self.anchors.labels)):
                self.ax_anch.plot(self.anchors.energies[j], self.anchors.spectra[j], color=self._anchor_color(j), label=str(self.anchors.labels[j]))
            self.ax_anch.set_title("Raw anchors (before calibration)")

        self.ax_anch.set_xlabel("Photon energy (eV)")
        self.ax_anch.set_ylabel("XAS (arb.)")
        self.ax_anch.grid(True, which="both", alpha=0.5, linewidth=0.8)
        if self.ax_anch.lines:
            self.ax_anch.legend(fontsize=8)
        self.canvas_anch.draw()

        # --- Error canvas ---
        self.ax_err.clear()
        if getattr(self, "_last_comp_rmse", None) is not None:
            rmse = self._last_comp_rmse
            self.ax_err.scatter(np.arange(1, len(rmse)+1), rmse)
            self.ax_err.set_title("RMS errors per used component")
            self.ax_err.set_xlabel("Component (selected order)")
            self.ax_err.set_ylabel("RMS error")
        else:
            self.ax_err.set_title("RMS errors per component")
            self.ax_err.set_xlabel("Component")
            self.ax_err.set_ylabel("RMS error")
        self.ax_err.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_err.draw()

    def clear_anchors(self):
        self.anchors.clear()
        self.list_anchors.clear()
        self.list_comps.clear()
        self.list_fit.clear()
        self._last_weights = None
        self._last_comp_indices = None
        self._last_comp_rmse = None
        self._fit_curves = {}
        self.lbl_status.setText("Cleared.")
        self._plot_empty()
        try:
            self.anchors_changed.emit()
        except Exception:
            pass

    def _on_anchor_selection_changed(self, row):
        self._load_anchor_editor_from_model(row)

    def _load_anchor_editor_from_model(self, row):
        if row is None or row < 0 or row >= len(self.anchors.params):
            return
        p = self.anchors.params[row]
        widgets = [self.spin_dx, self.spin_factor, self.spin_bro, self.spin_bg,
                   self.chk_c_dx, self.chk_c_fac, self.chk_c_bro, self.chk_c_bg,
                   self.spin_c_dx, self.spin_c_fac, self.spin_c_bro, self.spin_c_bg]
        for w in widgets:
            w.blockSignals(True)
        self.spin_dx.setValue(float(p.get("dx", 0.0)))
        self.spin_factor.setValue(float(p.get("factor", 1.0)))
        self.spin_bro.setValue(float(p.get("broadening", 0.0)))
        self.spin_bg.setValue(float(p.get("background", 0.0)))

        self.chk_c_dx.setChecked(bool(p.get("c_dx_enabled", False)))
        self.spin_c_dx.setValue(float(p.get("c_dx_within", 0.0)))

        self.chk_c_fac.setChecked(bool(p.get("c_factor_enabled", False)))
        self.spin_c_fac.setValue(float(p.get("c_factor_within", 0.0)))

        self.chk_c_bro.setChecked(bool(p.get("c_broad_enabled", False)))
        self.spin_c_bro.setValue(float(p.get("c_broad_within", 0.0)))

        self.chk_c_bg.setChecked(bool(p.get("c_bg_enabled", False)))
        self.spin_c_bg.setValue(float(p.get("c_bg_within", 0.0)))
        for w in widgets:
            w.blockSignals(False)

    def _save_anchor_editor_to_model(self):
        row = self.list_anchors.currentRow()
        if row is None or row < 0 or row >= len(self.anchors.params):
            return
        p = self.anchors.params[row]
        p["dx"] = float(self.spin_dx.value())
        p["factor"] = float(self.spin_factor.value())
        p["broadening"] = float(self.spin_bro.value())
        p["background"] = float(self.spin_bg.value())

        p["c_dx_enabled"] = bool(self.chk_c_dx.isChecked())
        p["c_dx_within"] = float(self.spin_c_dx.value())

        p["c_factor_enabled"] = bool(self.chk_c_fac.isChecked())
        p["c_factor_within"] = float(self.spin_c_fac.value())

        p["c_broad_enabled"] = bool(self.chk_c_bro.isChecked())
        p["c_broad_within"] = float(self.spin_c_bro.value())

        p["c_bg_enabled"] = bool(self.chk_c_bg.isChecked())
        p["c_bg_within"] = float(self.spin_c_bg.value())

    def _selected_anchor_indices(self):
        idx = []
        for i in range(self.list_anchors.count()):
            if self.list_anchors.item(i).checkState() == Qt.Checked:
                idx.append(i)
        return idx

    def _selected_comp_indices(self):
        idx = []
        for i in range(self.list_comps.count()):
            if self.list_comps.item(i).checkState() == Qt.Checked:
                idx.append(i)
        return idx

    def _build_bounds(self, anchor_indices, p0):
        low = np.full_like(p0, -np.inf, dtype=float)
        high = np.full_like(p0, np.inf, dtype=float)

        opt_dx = self.chk_opt_dx.isChecked()
        opt_fac = self.chk_opt_fac.isChecked()
        opt_bro = self.chk_opt_bro.isChecked()
        opt_bg = self.chk_opt_bg.isChecked()

        for i, aidx in enumerate(anchor_indices):
            c = self.anchors.params[aidx]

            # dX
            if not opt_dx:
                low[4*i+0] = p0[4*i+0]
                high[4*i+0] = p0[4*i+0]
            elif bool(c.get("c_dx_enabled", False)):
                w = float(c.get("c_dx_within", 0.0))
                low[4*i+0] = p0[4*i+0] - w
                high[4*i+0] = p0[4*i+0] + w

            # factor (>=0)
            if not opt_fac:
                low[4*i+1] = p0[4*i+1]
                high[4*i+1] = p0[4*i+1]
            else:
                low[4*i+1] = 0.0
                if bool(c.get("c_factor_enabled", False)):
                    w = float(c.get("c_factor_within", 0.0))
                    low[4*i+1] = max(0.0, p0[4*i+1] - w)
                    high[4*i+1] = max(0.0, p0[4*i+1] + w)

            # broadening (>=0)
            if not opt_bro:
                low[4*i+2] = p0[4*i+2]
                high[4*i+2] = p0[4*i+2]
            else:
                low[4*i+2] = 0.0
                if bool(c.get("c_broad_enabled", False)):
                    w = float(c.get("c_broad_within", 0.0))
                    low[4*i+2] = max(0.0, p0[4*i+2] - w)
                    high[4*i+2] = max(0.0, p0[4*i+2] + w)

            # background (>=0)
            if not opt_bg:
                low[4*i+3] = p0[4*i+3]
                high[4*i+3] = p0[4*i+3]
            else:
                low[4*i+3] = 0.0
                if bool(c.get("c_bg_enabled", False)):
                    w = float(c.get("c_bg_within", 0.0))
                    low[4*i+3] = max(0.0, p0[4*i+3] - w)
                    high[4*i+3] = max(0.0, p0[4*i+3] + w)

        return low, high

    def run_calibration(self):
        S = self.mcr_tab.results.get("S", None)
        x_target = self.mcr_tab.results.get("energy", None)
        if S is None or x_target is None:
            QMessageBox.warning(self, "No MCR results", "Run MCR-ALS first.")
            return
        if not self.anchors.ready:
            QMessageBox.warning(self, "No anchors", "Load anchors CSV first.")
            return

        anchor_idx = self._selected_anchor_indices()
        if len(anchor_idx) < 1:
            QMessageBox.warning(self, "No anchors selected", "Select at least one anchor.")
            return

        comp_idx = self._selected_comp_indices()
        if len(comp_idx) < 1:
            QMessageBox.warning(self, "No components selected", "Select at least one MCR component.")
            return

        x_target = np.asarray(x_target, dtype=float)
        S_sel = np.asarray(S[comp_idx, :], dtype=float)
        kc, m = S_sel.shape
        p = len(anchor_idx)

        # p0
        p0 = np.zeros(4 * p, dtype=float)
        for i, aidx in enumerate(anchor_idx):
            par = self.anchors.params[aidx]
            p0[4*i+0] = float(par.get("dx", 0.0))
            p0[4*i+1] = float(par.get("factor", 1.0))
            p0[4*i+2] = float(par.get("broadening", 0.0))
            p0[4*i+3] = float(par.get("background", 0.0))

        low, high = self._build_bounds(anchor_idx, p0)

        # Base grid step
        dxt = np.diff(x_target)
        base_dx = float(np.median(dxt)) if dxt.size > 0 and np.isfinite(np.median(dxt)) else 1.0
        if base_dx == 0:
            base_dx = 1.0

        step = np.zeros_like(p0)
        for i in range(p):
            step[4*i+0] = 2.0 * abs(base_dx)
            step[4*i+1] = 0.1
            step[4*i+2] = 2.0 * abs(base_dx)
            step[4*i+3] = 0.01

        fixed_mask = np.isfinite(low) & np.isfinite(high) & (low == high)
        step[fixed_mask] = 0.0

        # limit step by bounds
        for j in range(len(step)):
            if step[j] <= 0:
                continue
            if np.isfinite(low[j]) and np.isfinite(high[j]):
                rng = high[j] - low[j]
                if rng >= 0:
                    step[j] = min(step[j], 0.5 * rng if rng > 0 else 0.0)

        tol = np.zeros_like(step)
        for i in range(p):
            tol[4*i+0] = abs(base_dx) * 1e-4
            tol[4*i+1] = 1e-4
            tol[4*i+2] = abs(base_dx) * 1e-4
            tol[4*i+3] = 1e-5
        tol[fixed_mask] = 0.0

        def clamp(pp):
            out = pp.copy()
            for j in range(len(out)):
                if np.isfinite(low[j]):
                    out[j] = max(out[j], low[j])
                if np.isfinite(high[j]):
                    out[j] = min(out[j], high[j])
            return out

        def build_A(pp):
            A = np.zeros((m, p), dtype=float)
            for i, aidx in enumerate(anchor_idx):
                x_a = self.anchors.energies[aidx]
                y_a = self.anchors.spectra[aidx]
                dx = float(pp[4*i+0])
                fac = float(pp[4*i+1])
                bro = max(0.0, float(pp[4*i+2]))
                bgp = max(0.0, float(pp[4*i+3]))
                A[:, i] = transform_curve_to_grid(x_a, y_a, x_target, dx=dx, factor=fac, broadening=bro, background=bgp)
            return A

        def sse(pp):
            A = build_A(pp)
            s = 0.0
            for r in range(kc):
                y = S_sel[r, :]
                w = nnls_solve(A, y)
                yhat = A @ w
                d = y - yhat
                s += float(np.sum(d * d))
            return s

        p_best = clamp(p0)
        best = sse(p_best)
        max_outer = int(self.spin_max_outer.value())
        outer = 0

        self.lbl_status.setText(f"Calibrating… SSE={best:.4g}")
        QApplication.processEvents()

        while outer < max_outer:
            improved = False
            for j in range(len(p_best)):
                if step[j] <= 0:
                    continue
                for sign in (-1.0, 1.0):
                    p_try = p_best.copy()
                    p_try[j] += sign * step[j]
                    p_try = clamp(p_try)
                    if p_try[j] == p_best[j]:
                        continue
                    val = sse(p_try)
                    if val < best:
                        best = val
                        p_best = p_try
                        improved = True
            if outer % 10 == 0:
                self.lbl_status.setText(f"Calibrating… iter {outer+1}/{max_outer}, SSE={best:.4g}")
                QApplication.processEvents()

            if not improved:
                step *= 0.5
                if np.all(step <= tol):
                    break
            outer += 1

        # Build final A and compute per-component weights/RMSE
        A_best = build_A(p_best)
        weights = np.zeros((kc, p), dtype=float)
        rmse = np.zeros(kc, dtype=float)
        for r in range(kc):
            y = S_sel[r, :]
            w = nnls_solve(A_best, y)
            weights[r, :] = w
            yhat = A_best @ w
            rmse[r] = float(np.sqrt(np.mean((y - yhat) ** 2)))

        # Store good anchors on x_target
        good_params = []
        good_spectra = np.zeros((p, m), dtype=float)
        for i, aidx in enumerate(anchor_idx):
            par = self.anchors.params[aidx].copy()
            par["dx"] = float(p_best[4*i+0])
            par["factor"] = float(p_best[4*i+1])
            par["broadening"] = float(max(0.0, p_best[4*i+2]))
            par["background"] = float(max(0.0, p_best[4*i+3]))
            good_params.append(par)
            # Update the editable anchor parameters so the GUI shows the optimized values
            try:
                self.anchors.params[aidx].update({
                    'dx': par['dx'],
                    'factor': par['factor'],
                    'broadening': par['broadening'],
                    'background': par['background'],
                })
            except Exception:
                pass
            good_spectra[i, :] = transform_curve_to_grid(
                self.anchors.energies[aidx], self.anchors.spectra[aidx], x_target,
                dx=par["dx"], factor=par["factor"], broadening=par["broadening"], background=par["background"]
            )

        self.anchors.good_energy = x_target.copy()
        self.anchors.good_spectra = good_spectra
        self.anchors.good_params = good_params

        # Refresh parameter editor for the currently selected anchor (shows optimized values)
        try:
            self._load_anchor_editor_from_model(self.list_anchors.currentRow())
        except Exception:
            pass

        self._last_weights = weights
        self._last_comp_indices = comp_idx
        self._last_comp_rmse = rmse

        self.lbl_status.setText(
            f"Calibration done. Anchors used={p}, components used={kc}, iterations={outer+1}."
        )
        try:
            self.anchors_changed.emit()
        except Exception:
            pass

        self._plot_results(comp_idx, S_sel, A_best, weights, rmse)

    def _plot_results(self, comp_idx, S_sel, A, W, rmse):
        # Store fitted curves per original component index
        self._fit_curves = {}
        for r in range(S_sel.shape[0]):
            ci = int(comp_idx[r])
            self._fit_curves[ci] = (A @ W[r, :]).copy()

        # Populate fitted-components list with checkboxes
        self.list_fit.blockSignals(True)
        self.list_fit.clear()
        for r in range(S_sel.shape[0]):
            ci = int(comp_idx[r])
            item = QListWidgetItem(f"Fit {ci+1}", self.list_fit)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, ci)
        self.list_fit.blockSignals(False)

        # Store RMSE and replot everything based on current checkbox state
        self._last_comp_rmse = rmse
        self._plot_results_from_state()

    def get_good_anchors(self):
        """Return (energy, spectra, labels) for good anchors, or (None, None, None)."""
        if self.anchors.good_energy is None or self.anchors.good_spectra is None:
            return None, None, None
        labels_used = [self.anchors.labels[j] for j in self._selected_anchor_indices()]
        return self.anchors.good_energy, self.anchors.good_spectra, labels_used



    def export_results(self):
        """Export calibration outputs to CSV."""
        if self.anchors.good_energy is None or self.anchors.good_spectra is None:
            QMessageBox.warning(self, "No calibrated anchors", "Run calibration before exporting.")
            return

        option, ok = QInputDialog.getItem(
            self,
            "Export anchors calibration results",
            "Select what to export:",
            ["Good anchors", "Mixture weights (components → anchors)", "RMSE per component", "Anchor parameters (dX/σ/factor/bg)"],
            0,
            False,
        )
        if not ok:
            return

        if option == "Good anchors":
            e = self.anchors.good_energy
            S = self.anchors.good_spectra  # shape (n_anchors, nE)
            data = {"PhotonEnergy_eV": e}
            for j, lab in enumerate(self.anchors.good_labels):
                data[str(lab)] = S[j, :]
            df = pd.DataFrame(data)
            save_dataframe(self, df, "good_anchors.csv", "Export good anchors")

        elif option == "Mixture weights (components → anchors)":
            if self._last_weights is None or self._last_comp_indices is None:
                QMessageBox.warning(self, "No weights", "Run calibration to compute mixture weights.")
                return
            w = np.asarray(self._last_weights, dtype=float)  # (n_comps, n_anchors_used)
            comp_ids = [f"Comp{int(i)+1}" for i in self._last_comp_indices]
            cols = [str(lab) for lab in self.anchors.good_labels]
            df = pd.DataFrame(w, columns=cols)
            df.insert(0, "Component", comp_ids)
            save_dataframe(self, df, "calibration_weights.csv", "Export calibration mixture weights")

        elif option == "RMSE per component":
            if self._last_comp_rmse is None or self._last_comp_indices is None:
                QMessageBox.warning(self, "No RMSE", "Run calibration to compute RMSE per component.")
                return
            df = pd.DataFrame({
                "Component": [f"Comp{int(i)+1}" for i in self._last_comp_indices],
                "RMSE": np.asarray(self._last_comp_rmse, dtype=float),
            })
            save_dataframe(self, df, "calibration_rmse.csv", "Export calibration RMSE")

        elif option == "Anchor parameters (dX/σ/factor/bg)":
            rows = []
            for lab, par in zip(self.anchors.labels, self.anchors.params):
                rows.append({
                    "Anchor": str(lab),
                    "dX_eV": par.get("dx", 0.0),
                    "Sigma_eV": par.get("broadening", 0.0),
                    "Factor": par.get("factor", 1.0),
                    "Background": par.get("background", 0.0),
                })
            df = pd.DataFrame(rows)
            save_dataframe(self, df, "anchor_parameters.csv", "Export anchor parameters")


class AnchorsApplyTab(QWidget):
    """Apply calibrated anchors to the original data cloud (non-negative fit per sample)."""
    def __init__(self, model: DataModel, calibrate_tab: AnchorsCalibrateTab):
        super().__init__()
        self.model = model
        self.calibrate_tab = calibrate_tab
        try:
            self.calibrate_tab.anchors_changed.connect(self.refresh_preview)
        except Exception:
            pass

        self.results = dict(C=None, Xhat=None, rmse=None, scales=None, energy=None, labels=None, sample_labels=None)

        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        left_w = QWidget()
        left = QVBoxLayout(left_w)
        splitter.addWidget(left_w)

        top = QHBoxLayout()
        left.addLayout(top, 2)

        self.fig_fit, self.canvas_fit = new_canvas()
        self.ax_fit = self.fig_fit.add_subplot(111)
        self.toolbar_fit = NavigationToolbar(self.canvas_fit, self)
        box_fit = QVBoxLayout()
        box_fit.addWidget(self.toolbar_fit)
        box_fit.addWidget(self.canvas_fit, 1)
        top.addLayout(box_fit, 1)

        self.fig_C, self.canvas_C = new_canvas()
        self.ax_C = self.fig_C.add_subplot(111)
        self.toolbar_C = NavigationToolbar(self.canvas_C, self)
        box_C = QVBoxLayout()
        box_C.addWidget(self.toolbar_C)
        box_C.addWidget(self.canvas_C, 1)
        top.addLayout(box_C, 1)

        self.fig_err, self.canvas_err = new_canvas()
        self.ax_err = self.fig_err.add_subplot(111)
        left.addWidget(self.canvas_err, 1)

        right_w = QWidget()
        right = QVBoxLayout(right_w)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #444;")
        right.addWidget(self.lbl_status)

        self.chk_closure = QCheckBox("Closure: normalize concentrations to sum=1")
        self.chk_closure.setChecked(True)
        right.addWidget(self.chk_closure)

        self.chk_scale_overlay = QCheckBox("Closure + best scale for overlay")
        self.chk_scale_overlay.setChecked(True)
        self.chk_scale_overlay.setToolTip("Keep coefficients as fractions (sum=1), but scale the reconstructed curve to best match the raw spectrum amplitude.")
        right.addWidget(self.chk_scale_overlay)

        # Keep the "best scale" option logically tied to Closure (fractions)
        self.chk_scale_overlay.setEnabled(self.chk_closure.isChecked())

        def _on_closure_toggled(state):
            enabled = bool(state)
            self.chk_scale_overlay.setEnabled(enabled)
            if not enabled:
                self.chk_scale_overlay.setChecked(False)

        def _on_scale_toggled(state):
            if bool(state) and (not self.chk_closure.isChecked()):
                # If user requests "Closure + best scale", automatically enable Closure
                self.chk_closure.setChecked(True)

        self.chk_closure.toggled.connect(_on_closure_toggled)
        self.chk_scale_overlay.toggled.connect(_on_scale_toggled)

        self.btn_run = QPushButton("Run anchor fit on selected spectra")
        self.btn_run.clicked.connect(self.run_apply)
        right.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_results)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_apply)

        self.btn_help = QPushButton("Help")
        self.btn_help.clicked.connect(self.show_help)

        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        for b in (self.btn_clear, self.btn_export, self.btn_help):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btns.addWidget(self.btn_clear, 1)
        btns.addWidget(self.btn_export, 1)
        btns.addWidget(self.btn_help, 1)
        btns_w = QWidget()
        btns_w.setLayout(btns)
        right.addWidget(btns_w)

        right.addWidget(QLabel("Show fit for sample:"))
        self.cmb_sample = QComboBox()
        self.cmb_sample.currentIndexChanged.connect(self._update_fit_plot)
        right.addWidget(self.cmb_sample)
        right.addStretch(1)

        self._plot_empty()
        self.refresh_preview()

    def get_help_text(self) -> str:
        return (
            "<b>Anchors – Apply</b><br><br>"
            "This tab uses calibrated (“good”) anchors from <b>Anchors – Calibrate</b> and fits each selected spectrum "
            "as a <b>non-negative mixture</b> of those anchors.<br><br>"
            "<b>Typical use</b><br>"
            "• If you have two anchors (e.g. two reference states), the resulting concentrations can be interpreted as the best "
            "non-negative mixing coefficients that reproduce each spectrum.<br><br>"
            "<b>Controls</b><br>"
            "• <b>Closure</b>: after NNLS, concentrations are normalized to sum to 1 for each spectrum. "
            "This is often convenient when interpreting mixtures as fractions.<br><br>"
            "<b>Outputs</b><br>"
            "• Fit plot: one chosen spectrum and its reconstruction from anchors.<br>"
            "• Concentrations plot: anchor coefficients across the selected spectra.<br>"
            "• Error plot: RMSE per spectrum."
        )

    def clear_apply(self):
        """Clear results on this tab (keeps calibrated anchors preview)."""
        self.results = dict(C=None, Xhat=None, rmse=None, scales=None, energy=None, labels=None, sample_labels=None)
        try:
            self.cmb_sample.blockSignals(True)
            self.cmb_sample.clear()
            self.cmb_sample.blockSignals(False)
        except Exception:
            pass
        try:
            self.lbl_status.setText("Cleared. (Calibrated anchors are kept.)")
        except Exception:
            pass
        # Clear plots, then restore anchor preview
        self._plot_empty()
        self.refresh_preview()

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Help – Anchors – Apply")
        dlg.setSizeGripEnabled(True)
        layout = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        f = txt.font()
        f.setPointSize(f.pointSize() + 5)
        txt.setFont(f)
        txt.setLineWrapMode(QTextEdit.WidgetWidth)
        txt.setAcceptRichText(True)
        txt.setHtml(self.get_help_text())
        layout.addWidget(txt, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        dlg.resize(900, 650)
        dlg.exec_()

    def _plot_empty(self):
        self.ax_fit.clear()
        self.ax_fit.set_title("Run calibration first, then run anchor fit")
        self.ax_fit.set_xlabel("Photon energy (eV)")
        self.ax_fit.set_ylabel("XAS (arb.)")
        self.ax_fit.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_fit.draw()

        self.ax_C.clear()
        self.ax_C.set_title("Concentrations")
        self.ax_C.set_xlabel("Sample index")
        self.ax_C.set_ylabel("Coefficient")
        self.ax_C.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_C.draw()

        self.ax_err.clear()
        self.ax_err.set_title("RMS errors per spectrum")
        self.ax_err.set_xlabel("Sample index")
        self.ax_err.set_ylabel("RMS error")
        self.ax_err.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_err.draw()

    def refresh_preview(self):
        """Plot calibrated ('good') anchors (if available) even before running the per-spectrum fit."""
        eA, SA, labelsA = self.calibrate_tab.get_good_anchors()
        self.ax_fit.clear()

        has_good = (eA is not None) and (SA is not None) and getattr(SA, "size", 0) > 0 and (labelsA is not None)
        if has_good:
            for j, lab in enumerate(labelsA):
                try:
                    col = self.calibrate_tab._anchor_color(j)
                except Exception:
                    col = None
                if col is None:
                    self.ax_fit.plot(eA, SA[j, :], label=str(lab))
                else:
                    self.ax_fit.plot(eA, SA[j, :], color=col, label=str(lab))
            self.ax_fit.set_title("Calibrated ('good') anchors")
            self.ax_fit.legend(fontsize=8)
        else:
            self.ax_fit.set_title("No calibrated anchors yet (run 'Anchors – Calibrate')")

        self.ax_fit.set_xlabel("Photon energy (eV)")
        self.ax_fit.set_ylabel("XAS (arb.)")
        self.ax_fit.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_fit.draw()


    def run_apply(self):
        eA, SA, labelsA = self.calibrate_tab.get_good_anchors()
        if eA is None or SA is None:
            QMessageBox.warning(self, "No good anchors", "Run 'Anchors – Calibrate' first (and make sure calibration completed).")
            return

        # Data selection from Data tab logic: DataModel provides selected indices
        idx = getattr(self.model, 'active_indices', None)
        if idx is None:
            idx = getattr(self.model, 'selected_indices', None)
        if idx is None or len(idx) < 1:
            QMessageBox.warning(self, "No spectra selected", "Select at least one spectrum on the Data tab.")
            return

        # Build data matrix (n,m) on a common grid
        energy, X = self.model.get_matrix_for_indices(idx)
        sample_labels = [self.model.sample_labels[i] for i in idx] if getattr(self.model, 'sample_labels', None) is not None else [str(i) for i in idx]
        if X is None:
            QMessageBox.warning(self, "Data error", "Could not build a data matrix (check energy grid consistency).")
            return

        energy = np.asarray(energy, dtype=float)
        X = np.asarray(X, dtype=float)
        n, m = X.shape

        # Anchor matrix on this grid: (m,p)
        p = SA.shape[0]
        A = np.zeros((m, p), dtype=float)
        for j in range(p):
            A[:, j] = np.interp(energy, eA, SA[j, :])

        # NNLS for each spectrum
        C = np.zeros((n, p), dtype=float)
        rmse = np.zeros(n, dtype=float)
        scales = np.ones(n, dtype=float)
        for i in range(n):
            y = X[i, :]
            c = nnls_solve(A, y)
            # Closure turns NNLS coefficients into fractions (sum=1)
            if self.chk_closure.isChecked():
                s = float(np.sum(c))
                if s > 0:
                    c = c / s
                # Optional: compute best global scale for overlay (does NOT change coefficients)
                if getattr(self, 'chk_scale_overlay', None) is not None and self.chk_scale_overlay.isChecked():
                    yhat0 = A @ c
                    denom = float(np.dot(yhat0, yhat0))
                    if denom > 0:
                        scales[i] = float(np.dot(y, yhat0) / denom)
                    else:
                        scales[i] = 1.0
            C[i, :] = c
            yhat = (scales[i] * (A @ c))
            rmse[i] = float(np.sqrt(np.mean((y - yhat) ** 2)))

        Xhat = (C @ A.T) * scales[:, None]  # (n,m) scaled overlay fit when enabled

        self.results = dict(C=C, Xhat=Xhat, rmse=rmse, scales=scales, energy=energy, labels=labelsA, sample_labels=sample_labels)

        scale_note = " (overlay scaled)" if (self.chk_closure.isChecked() and getattr(self, "chk_scale_overlay", None) is not None and self.chk_scale_overlay.isChecked()) else ""
        self.lbl_status.setText(f"Anchor fit done. n={n} spectra, anchors={p}.{scale_note}")
        self._populate_sample_selector(sample_labels)
        self._plot_concentrations(C, labelsA)
        self._plot_errors(rmse)
        self._update_fit_plot()

    def _populate_sample_selector(self, sample_labels):
        self.cmb_sample.blockSignals(True)
        self.cmb_sample.clear()
        for s in sample_labels:
            self.cmb_sample.addItem(str(s))
        self.cmb_sample.blockSignals(False)
        if self.cmb_sample.count() > 0:
            self.cmb_sample.setCurrentIndex(0)

    def _update_fit_plot(self):
        if self.results.get("C") is None:
            return
        i = self.cmb_sample.currentIndex()
        if i < 0:
            return
        energy = self.results["energy"]
        idx = getattr(self.model, 'active_indices', None)
        if idx is None:
            idx = getattr(self.model, 'selected_indices', None)
        energy0, X = self.model.get_matrix_for_indices(idx)
        X = np.asarray(X, dtype=float)
        y = X[i, :]
        yhat = self.results["Xhat"][i, :]

        self.ax_fit.clear()
        self.ax_fit.plot(energy, y, label="Data")
        self.ax_fit.plot(energy, yhat, linestyle="--", label="Fit (anchors)")
        self.ax_fit.set_title(f"Fit for sample: {self.results['sample_labels'][i]}")
        self.ax_fit.set_xlabel("Photon energy (eV)")
        self.ax_fit.set_ylabel("XAS (arb.)")
        self.ax_fit.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.ax_fit.legend(fontsize=8)
        self.canvas_fit.draw()

    def _plot_concentrations(self, C, labels):
        self.ax_C.clear()
        x = np.arange(C.shape[0])
        for j, lab in enumerate(labels):
            try:
                col = self.calibrate_tab._anchor_color(j)
            except Exception:
                col = None
            if col is None:
                self.ax_C.plot(x, C[:, j], marker="o", label=str(lab))
            else:
                self.ax_C.plot(x, C[:, j], marker="o", color=col, label=str(lab))
        self.ax_C.set_title("Anchor concentrations (NNLS coefficients)")
        self.ax_C.set_xlabel("Selected spectra (index in selection)")
        self.ax_C.set_ylabel("Coefficient")
        self.ax_C.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.ax_C.legend(fontsize=8)
        self.canvas_C.draw()

    def _plot_errors(self, rmse):
        self.ax_err.clear()
        self.ax_err.scatter(np.arange(len(rmse)), rmse)
        self.ax_err.set_title("RMS errors per selected spectrum")
        self.ax_err.set_xlabel("Selected spectra (index in selection)")
        self.ax_err.set_ylabel("RMS error")
        self.ax_err.grid(True, which="both", alpha=0.5, linewidth=0.8)
        self.canvas_err.draw()


# -----------------------------
# Main window
# -----------------------------


    def export_results(self):
        """Export anchor-fit outputs to CSV (concentrations, RMSE, reconstructed spectra)."""
        if self.results.get("C") is None or self.results.get("rmse") is None:
            QMessageBox.warning(self, "No anchor-fit results", "Run anchor fit before exporting.")
            return

        option, ok = QInputDialog.getItem(
            self,
            "Export anchors apply results",
            "Select what to export:",
            ["Concentrations", "RMSE per spectrum", "Reconstructed spectra (Xhat)"],
            0,
            False,
        )
        if not ok:
            return

        idx = getattr(self.model, "active_indices", None)
        if idx is None:
            idx = getattr(self.model, "selected_indices", None)
        if idx is None:
            idx = list(range(len(self.results.get("rmse", []))))

        sample_labels = self.results.get("sample_labels")
        if sample_labels is None:
            sample_labels = [self.model.sample_labels[i] for i in idx] if getattr(self.model, "sample_labels", None) is not None else [str(i) for i in idx]

        if option == "Concentrations":
            C = np.asarray(self.results["C"], dtype=float)  # (n_samples, n_anchors)
            labelsA = self.results.get("labels")
            if labelsA is None:
                labelsA = [f"Anchor{j+1}" for j in range(C.shape[1])]
            data = {"SampleIndex": idx, "SampleLabel": sample_labels}
            if self.results.get("scales") is not None:
                data["Scale"] = np.asarray(self.results["scales"], dtype=float)
            for j, lab in enumerate(labelsA):
                data[str(lab)] = C[:, j]
            df = pd.DataFrame(data)
            save_dataframe(self, df, "anchor_concentrations.csv", "Export concentrations")

        elif option == "RMSE per spectrum":
            rmse = np.asarray(self.results["rmse"], dtype=float)
            df = pd.DataFrame({"SampleIndex": idx, "SampleLabel": sample_labels, "RMSE": rmse})
            save_dataframe(self, df, "anchor_rmse.csv", "Export RMSE per spectrum")

        elif option == "Reconstructed spectra (Xhat)":
            e = self.results.get("energy")
            Xhat = self.results.get("Xhat")
            if e is None or Xhat is None:
                QMessageBox.warning(self, "No Xhat", "Reconstructed spectra are not available.")
                return
            data = {"PhotonEnergy_eV": e}
            for i_s, lab in enumerate(sample_labels):
                data[str(lab)] = Xhat[i_s, :]
            df = pd.DataFrame(data)
            save_dataframe(self, df, "anchor_reconstruction.csv", "Export reconstructed spectra")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XAS data decomposition")
        # Default placement/size when launched from the main app
        self.setGeometry(100, 30, 1550, 640)

        self.model = DataModel()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_data = DataTab(self.model)
        self.tab_pca = PCATab(self.model)
        self.tab_nmf = NMFTab(self.model, self.tab_pca)
        self.tab_mcr = MCRTab(self.model, self.tab_pca, self.tab_nmf)
        self.tab_anchors_cal = AnchorsCalibrateTab(self.model, self.tab_mcr)
        self.tab_anchors_apply = AnchorsApplyTab(self.model, self.tab_anchors_cal)

        self.tabs.addTab(self.tab_data, "Data")
        self.tabs.addTab(self.tab_pca, "PCA")
        self.tabs.addTab(self.tab_nmf, "NMF")
        self.tabs.addTab(self.tab_mcr, "MCR-ALS")
        self.tabs.addTab(self.tab_anchors_cal, "Anchors – Calibrate")
        self.tabs.addTab(self.tab_anchors_apply, "Anchors – Apply")

        # Refresh the component list in the calibration tab when switching to it
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def set_dataset(self, x, ys, labels):
        """Inject a prepared dataset from the main FlexPES app.

        Parameters
        ----------
        x : array-like
            Common energy axis (1D).
        ys : sequence of array-like
            Spectra arrays (each 1D).
        labels : sequence of str
            Labels for each spectrum.
        """
        import numpy as np
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QListWidgetItem

        if x is None or ys is None or len(ys) == 0:
            raise ValueError("Empty dataset received.")
        if labels is None or len(labels) != len(ys):
            raise ValueError("labels must have the same length as ys.")

        x = np.asarray(x).ravel()
        ys_np = [np.asarray(y).ravel() for y in ys]

        # Store into model (one energy axis per spectrum, as in CSV loading)
        self.model.energies = [x.copy() for _ in ys_np]
        self.model.spectra = [y.copy() for y in ys_np]
        self.model.sample_labels = [str(l) for l in labels]
        self.model.active_indices = list(range(len(ys_np)))

        # Populate the Data (CSV) tab list as open_csv() would do
        if hasattr(self, "tab_data") and hasattr(self.tab_data, "list_samples"):
            try:
                self.tab_data.list_samples.blockSignals(True)
            except Exception:
                pass
            self.tab_data.list_samples.clear()
            for lab in self.model.sample_labels:
                item = QListWidgetItem(str(lab), self.tab_data.list_samples)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            try:
                self.tab_data.list_samples.blockSignals(False)
            except Exception:
                pass

        # Disable CSV open button for now (kept for future use)
        if hasattr(self, "tab_data") and hasattr(self.tab_data, "btn_open"):
            try:
                self.tab_data.btn_open.setEnabled(False)
            except Exception:
                pass

        # Update selection and plot
        if hasattr(self, "tab_data") and hasattr(self.tab_data, "update_active_indices"):
            try:
                self.tab_data.update_active_indices()
            except Exception:
                pass
        if hasattr(self, "tab_data") and hasattr(self.tab_data, "plot_selected"):
            try:
                self.tab_data.plot_selected()
            except Exception:
                pass

        # Show the Data (CSV) tab first
        try:
            self.tabs.setCurrentIndex(0)
        except Exception:
            pass


    def _on_tab_changed(self, idx):
        """Refresh dependent tabs when the user switches tabs."""
        try:
            # Ensure anchor calibration sees the latest MCR components
            if self.tabs.tabText(idx).startswith('Anchors'):
                self.tab_anchors_cal.on_mcr_results_updated()
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("XAS data decomposition")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
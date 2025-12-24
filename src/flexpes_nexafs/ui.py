from . import processing
from . import data
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # allow concurrent readers on Windows
import csv
import h5py
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.ioff() 

from PyQt5.QtWidgets import (

    QApplication, QFileDialog, QMainWindow, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout, QTabWidget,
    QCheckBox, QComboBox, QSpinBox, QMessageBox, QSizePolicy, QSplitter,
    QInputDialog, QColorDialog, QListWidget, QListWidgetItem, QDialog,
    QMenu, QTextBrowser, QSlider, QDoubleSpinBox
, QAbstractItemView)

# ---- Compatibility shims (Phase 2b) ----
def _proc_robust_polyfit_on_normalized(self, xs, ys, deg, x_eval):
    try:
        from . import processing as _p
        fn = getattr(_p, "robust_polyfit_on_normalized", None)
        if fn is not None:
            return fn(xs, ys, deg, x_eval)
    except Exception:
        pass
    return self._robust_polyfit_on_normalized(xs, ys, deg, x_eval)

def _proc_safe_post_normalize(self, x, y, mode):
    try:
        from . import processing as _p
        fn = getattr(_p, "safe_post_normalize", None)
        if fn is not None:
            return fn(x, y, mode)
    except Exception:
        pass
    return self._safe_post_normalize(x, y, mode)
# ----------------------------------------
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from datetime import datetime
from .utils.sorting import parse_entry_number
from .widgets.curve_item import CurveListItemWidget

###############################################################################
# Helper function to parse integer from something like "entry1001"
# for sorting the Raw Data and Processed Data items in ascending order
###############################################################################


from .data import DataMixin
from .processing import ProcessingMixin
from .plotting import PlottingMixin
from .export import ExportMixin
from .library import LibraryMixin
class HDF5Viewer(DataMixin, ProcessingMixin, PlottingMixin, ExportMixin, LibraryMixin, QMainWindow):

    def on_plotted_list_reordered(self):
        """After drag-drop: recompute Waterfall using list order, rebuild legend, redraw."""
        # 1) Restore original data (if helper exists) so Waterfall applies cleanly
        try:
            if hasattr(self, "restore_original_line_data"):
                self.restore_original_line_data()
        except Exception:
            pass

        # 2) Re-apply Waterfall (reuse existing implementation)
        try:
            if hasattr(self, "recompute_waterfall_layout"):
                self.recompute_waterfall_layout()
            elif hasattr(self, "apply_waterfall_shift"):
                self.apply_waterfall_shift()
            else:
                if hasattr(self, "rescale_plotted_axes"):
                    self.rescale_plotted_axes()
        except Exception:
            pass

        # 3) Rebuild the legend to follow the Plotted list order
        try:
            if hasattr(self, "update_legend"):
                self.update_legend()
        except Exception:
            pass
    def _on_plotted_legend_mode_changed(self, _idx=0):
        """Qt slot: change legend behavior on Plotted Data panel."""
        try:
            mode = str(getattr(self, "legend_mode_combo").currentText())
        except Exception:
            mode = "User-defined"
        try:
            if hasattr(self, "set_plotted_legend_mode"):
                self.set_plotted_legend_mode(mode)
        except Exception:
            # Fallback: at least hide/show legend
            try:
                if mode.strip().lower() == "none":
                    self.toggle_plotted_legend(False)
                else:
                    self.toggle_plotted_legend(True)
            except Exception:
                pass

    def _on_plotted_annotation_checkbox_toggled(self, state):
        """Qt slot: show/hide annotation box on Plotted Data panel."""
        try:
            show = bool(state)
        except Exception:
            show = bool(state)
        try:
            if hasattr(self, "toggle_plotted_annotation"):
                self.toggle_plotted_annotation(show)
        except Exception:
            pass
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlexPES NEXAFS Plotter")
        self.setGeometry(100, 100, 1500, 600)

        self.VERSION_NUMBER = "2.1.0"
        self.CREATION_DATETIME = "2025-12-24"

        self.hdf5_files = {}
        self.plot_data = {}      # Keys: "abs_path##hdf5_path"
        self.energy_cache = {}

        self.manual_mode = False
        self.manual_points = []
        self.manual_bg_line = None
        self.manual_poly = None
        self.manual_poly_degree = 2

        self.last_sum_state = False
        self.last_normalize_state = False
        self.last_plot_len = 0

        self.plotted_curves = set()
        self.plotted_lines = {}
        self.custom_labels = {}

        self._sum_serial = 0  # unique id for summed curves in Plotted
        # For Raw Data panel: each dataset's visibility
        self.raw_visibility = {}

        # Region states for tree grouping
        self.region_states = {}
        self.proc_region_states = {}

        self.raw_tree_reset = False
        self.active_point = None

        #######################################################################
        # MAIN GUI LAYOUT
        #######################################################################
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left panel: file tree and controls
        self.left_panel_widget = QWidget()
        self.left_panel = QVBoxLayout(self.left_panel_widget)
        self.splitter.addWidget(self.left_panel_widget)

        self.open_close_layout = QHBoxLayout()
        self.open_button = QPushButton("Open HDF5 File")
        self.open_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.open_button.clicked.connect(self.open_file)
        self.open_close_layout.addWidget(self.open_button)

        self.close_button = QPushButton("Close all")
        self.close_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.close_button.clicked.connect(self.close_file)
        self.open_close_layout.addWidget(self.close_button)

        self.clear_button = QPushButton("Clear all")
        self.clear_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_button.clicked.connect(self.clear_all)
        self.open_close_layout.addWidget(self.clear_button)

        self.help_button = QPushButton("Help")
        self.help_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.help_menu = QMenu(self)
        self.usage_action = self.help_menu.addAction("Usage")
        self.about_action = self.help_menu.addAction("About")
        self.help_button.setMenu(self.help_menu)
        self.open_close_layout.addWidget(self.help_button)
        self.usage_action.triggered.connect(self.show_usage_info)
        self.about_action.triggered.connect(self.show_about_info)

        self.left_panel.addLayout(self.open_close_layout)

        self.file_label = QLabel("No file open")
        self.file_label.setWordWrap(True)
        self.left_panel.addWidget(self.file_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        try:
            from PyQt5.QtWidgets import QHeaderView
            self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        except Exception:
            pass
        self.tree.setHeaderLabels(["HDF5 Structure"])
        self.tree.header().resizeSection(0, 250)
        self.tree.itemExpanded.connect(self.load_subtree)
        self.tree.itemChanged.connect(self.toggle_plot)
        self.tree.itemClicked.connect(self.display_data)
        # Enable context menu on the HDF5 tree for per-file closing
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.left_panel.addWidget(self.tree)

        # Right panel: tab widget
        self.right_panel_widget = QWidget()
        self.right_panel = QVBoxLayout(self.right_panel_widget)
        self.splitter.addWidget(self.right_panel_widget)

        self.data_tabs = QTabWidget()
        self.data_tabs.currentChanged.connect(self.on_tab_changed)
        self.right_panel.addWidget(self.data_tabs)

        #######################################################################
        # RAW DATA TAB
        #######################################################################
        self.raw_tab = QWidget()
        self.raw_splitter = QSplitter(Qt.Horizontal)
        self.raw_tab_layout = QVBoxLayout(self.raw_tab)
        self.raw_tab_layout.addWidget(self.raw_splitter)

        self.raw_left_widget = QWidget()
        self.raw_left_layout = QVBoxLayout(self.raw_left_widget)
        self.raw_group_layout = QHBoxLayout()

        self.cb_all_tey = QCheckBox("All TEY data")
        self.cb_all_tey.stateChanged.connect(
            lambda state: self.set_group_visibility("_ch1", state == Qt.Checked)
        )
        self.raw_group_layout.addWidget(self.cb_all_tey)

        self.cb_all_pey = QCheckBox("All PEY data")
        self.cb_all_pey.stateChanged.connect(
            lambda state: self.set_group_visibility("_ch3", state == Qt.Checked)
        )
        self.raw_group_layout.addWidget(self.cb_all_pey)

        self.cb_all_tfy = QCheckBox("All TFY data")
        self.cb_all_tfy.stateChanged.connect(
            lambda state: self.set_group_visibility("roi2_dtc", state == Qt.Checked)
        )
        self.raw_group_layout.addWidget(self.cb_all_tfy)

        self.cb_all_pfy = QCheckBox("All PFY data")
        self.cb_all_pfy.stateChanged.connect(
            lambda state: self.set_group_visibility("roi1_dtc", state == Qt.Checked)
        )
        self.raw_group_layout.addWidget(self.cb_all_pfy)

        # --- Example-inspired: 'All in channel' checkbox + combo ---
        self.cb_all_in_channel = QCheckBox("All in channel:")
        self.combo_all_channel = QComboBox()
        try: self.combo_all_channel.setMinimumWidth(240)
        except Exception: pass
        self.raw_group_layout.addWidget(self.cb_all_in_channel)
        self.raw_group_layout.addWidget(self.combo_all_channel)
        # Wire behaviors
        self.cb_all_in_channel.stateChanged.connect(self._apply_all_in_channel_filter)
        self.combo_all_channel.currentTextChanged.connect(self._on_all_channel_selection_changed)
        self.raw_left_layout.addLayout(self.raw_group_layout)

        self.canvas_raw_fig, self.raw_ax = plt.subplots()
        self.canvas_raw = FigureCanvas(self.canvas_raw_fig)
        self.toolbar_raw = NavigationToolbar(self.canvas_raw, self.raw_left_widget)
        self.raw_left_layout.addWidget(self.toolbar_raw)
        self.raw_left_layout.addWidget(self.canvas_raw)

        self.scalar_display_raw = QLabel()
        self.scalar_display_raw.setWordWrap(True)
        self.scalar_display_raw.setFixedHeight(30)
        self.scalar_display_raw.setIndent(37)
        self.scalar_display_raw.setFont(QFont("Arial", 10))
        self.raw_left_layout.addWidget(self.scalar_display_raw)

        self.raw_splitter.addWidget(self.raw_left_widget)
        self.raw_tree = QTreeWidget()
        self.raw_tree.setHeaderHidden(True)
        (self.raw_tree.itemChanged.connect( self.raw_tree_item_changed )) if hasattr(self, 'raw_tree_item_changed') else None
        self.raw_splitter.addWidget(self.raw_tree)
        self.raw_splitter.setStretchFactor(0, 40)
        self.raw_splitter.setStretchFactor(1, 60)
        self.data_tabs.addTab(self.raw_tab, "Raw Data")

        #######################################################################
        # PROCESSED DATA TAB
        #######################################################################
        self.proc_tab = QWidget()
        self.proc_left_widget = QWidget()
        self.proc_left_layout = QVBoxLayout(self.proc_left_widget)

        # Top controls
        self.proc_controls_top = QWidget()
        self.proc_controls_top.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.proc_controls_top_layout = QHBoxLayout(self.proc_controls_top)
        self.proc_controls_top_layout.setContentsMargins(0, 0, 0, 0)
        self.proc_controls_top_layout.setSpacing(5)

        self.chk_normalize = QCheckBox("Normalize by I₀?")
        self.proc_controls_top_layout.addWidget(self.chk_normalize)
        self.proc_controls_top_layout.addWidget(QLabel("Set I₀ channel:"))
        self.combo_norm = QComboBox()
        self.proc_controls_top_layout.addWidget(self.combo_norm)
        self.combo_norm.setEnabled(False)

        self.chk_sum = QCheckBox("Sum up?")
        self.proc_controls_top_layout.addWidget(self.chk_sum)

        # Group background (Automatic BG + multiple selection)
        self.chk_group_bg = QCheckBox("Group BG")
        self.chk_group_bg.setChecked(False)
        self.chk_group_bg.setEnabled(False)
        self.chk_group_bg.setToolTip(
            "When two or more spectra are selected and BG mode is 'Automatic', "
            "enables group processing. With Area post-normalization, backgrounds are adjusted "
            "so that the pre-edge baseline is at zero and the absorption jump after Area normalization "
            "is consistent across the selected group."
        )
        self.proc_controls_top_layout.addWidget(self.chk_group_bg)

        # Optional: match pre-edge slope across the group (only meaningful in group Auto BG)
        self.chk_group_bg_slope = QCheckBox("Match pre-edge slope")
        self.chk_group_bg_slope.setChecked(False)
        self.chk_group_bg_slope.setEnabled(False)
        self.chk_group_bg_slope.setToolTip(
            "When Group BG is active (Automatic BG + multiple selected spectra), "
            "adjusts the backgrounds so that the pre-edge slope after BG subtraction "
            "is consistent across the selected group."
        )
        self.proc_controls_top_layout.addWidget(self.chk_group_bg_slope)
        self.proc_controls_top_layout.addStretch()

        self.pass_button = QPushButton("Pass")
        self.proc_controls_top_layout.addWidget(self.pass_button)
        self.export_ascii_button = QPushButton("Export")
        self.proc_controls_top_layout.addWidget(self.export_ascii_button)
        self.export_ascii_button.clicked.connect(self.export_ascii)
        self.pass_button.clicked.connect(self.pass_to_plotted_no_clear)

        self.chk_sum.stateChanged.connect(self.update_plot_processed)
        self.chk_normalize.stateChanged.connect(self._on_normalize_toggled)
        self.combo_norm.currentIndexChanged.connect(self.update_plot_processed)

        self.proc_left_layout.addWidget(self.proc_controls_top, 0)

        self.canvas_proc_fig, self.proc_ax = plt.subplots()
        self.canvas_proc = FigureCanvas(self.canvas_proc_fig)
        self.toolbar_proc = NavigationToolbar(self.canvas_proc, self.proc_left_widget)
        # Pre-edge marker dragging is handled via ProcessingMixin.on_press/on_motion/on_release
        # (same event pipeline as Manual BG anchors) for maximum robustness.
        self.proc_left_layout.addWidget(self.toolbar_proc)
        self.proc_left_layout.addWidget(self.canvas_proc, 1)

        # Bottom controls
        self.proc_controls_bottom = QWidget()
        self.proc_controls_bottom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.proc_controls_bottom_layout = QHBoxLayout(self.proc_controls_bottom)
        self.proc_controls_bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.proc_controls_bottom_layout.setSpacing(5)

        self.proc_controls_bottom_layout.addWidget(QLabel("Choose BG:"))
        self.combo_bg = QComboBox()
        self.combo_bg.addItems(["None", "Automatic", "Manual"])
        self.proc_controls_bottom_layout.addWidget(self.combo_bg)

        self.proc_controls_bottom_layout.addWidget(QLabel("Poly degree:"))
        self.combo_poly = QComboBox()
        self.combo_poly.addItems(["0", "1", "2", "3"])
        self.combo_poly.setCurrentIndex(2)  # default polynomial degree => 2
        self.proc_controls_bottom_layout.addWidget(self.combo_poly)

        self.proc_controls_bottom_layout.addWidget(QLabel("Pre-edge (%):"))
        self.spin_preedge = QSpinBox()
        self.spin_preedge.setRange(0, 100)
        self.spin_preedge.setValue(12)
        self.proc_controls_bottom_layout.addWidget(self.spin_preedge)

        self.chk_show_without_bg = QCheckBox("Subtract BG?")
        self.proc_controls_bottom_layout.addWidget(QLabel("Normalize:"))
        self.combo_post_norm = QComboBox()
        self.combo_post_norm.addItems(["None", "Max", "Jump", "Area"])
        self.combo_post_norm.setEnabled(False)
        self.proc_controls_bottom_layout.addWidget(self.combo_post_norm)
        self.proc_controls_bottom_layout.addWidget(self.chk_show_without_bg)


        self.proc_left_layout.addWidget(self.proc_controls_bottom, 0)

        self.proc_tree = QTreeWidget()
        self.proc_tree.setHeaderHidden(True)
        (self.proc_tree.itemChanged.connect( self.proc_tree_item_changed )) if hasattr(self, 'proc_tree_item_changed') else None

        self.proc_splitter = QSplitter(Qt.Horizontal)
        self.proc_splitter.addWidget(self.proc_left_widget)
        self.proc_splitter.addWidget(self.proc_tree)
        self.proc_splitter.setStretchFactor(0, 35)
        self.proc_splitter.setStretchFactor(1, 65)

        self.proc_tab_layout = QVBoxLayout(self.proc_tab)
        self.proc_tab_layout.addWidget(self.proc_splitter)
        self.data_tabs.addTab(self.proc_tab, "Processed Data")

        #######################################################################
        # PLOTTED DATA TAB
        #######################################################################
        self.plot_left_widget = QWidget()
        self.plot_left_layout = QVBoxLayout(self.plot_left_widget)
        self.plot_left_layout.setContentsMargins(10, 10, 10, 20)
        self.plot_left_layout.setSpacing(5)

        self.canvas_plotted_fig, self.plotted_ax = plt.subplots()
        self.plotted_ax.set_xlabel("Photon energy (eV)")
        self.plotted_ax.set_ylabel("XAS intensity (arb. units)")
        self.canvas_plotted = FigureCanvas(self.canvas_plotted_fig)
        self.canvas_plotted.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_plotted_fig.tight_layout()

        # Top controls for the Plotted Data panel
        self.plotted_controls_top = QWidget()
        self.plotted_controls_top.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.plotted_controls_top_layout = QHBoxLayout(self.plotted_controls_top)
        self.plotted_controls_top_layout.setContentsMargins(0, 0, 0, 0)
        self.plotted_controls_top_layout.setSpacing(5)

        self.plotted_controls_top_layout.addWidget(QLabel("Legend:"))
        self.legend_mode_combo = QComboBox()
        self.legend_mode_combo.addItems(["None", "User-defined", "Entry number"])
        self.legend_mode_combo.setCurrentText("User-defined")
        self.legend_mode_combo.setToolTip(
            "Legend labeling mode. 'User-defined' lets you click legend entries to rename. "
            "'Entry number' labels curves by entryXXXX -> XXXX. 'None' hides the legend."
        )
        self.plotted_controls_top_layout.addWidget(self.legend_mode_combo)

        self.chk_show_annotation = QCheckBox("Annotation")
        self.chk_show_annotation.setChecked(False)
        self.plotted_controls_top_layout.addWidget(self.chk_show_annotation)

        self.plotted_controls_top_layout.addStretch()

        self.legend_mode_combo.currentIndexChanged.connect(self._on_plotted_legend_mode_changed)
        self.chk_show_annotation.stateChanged.connect(self._on_plotted_annotation_checkbox_toggled)

        self.plot_left_layout.addWidget(self.plotted_controls_top, 0)
        self.toolbar_plotted = NavigationToolbar(self.canvas_plotted, self)
        self.plot_left_layout.addWidget(self.toolbar_plotted)
        self.plot_left_layout.addWidget(self.canvas_plotted, 1)

        # ----------------------------------------------------------------------
        # BOTTOM ROW (RE-ORDERED): Waterfall controls and Grid on the left; then stretch; then Export and Clear buttons
        # ----------------------------------------------------------------------
        self.plot_buttons_layout = QHBoxLayout()

        # Waterfall controls
        self.waterfall_mode_combo = QComboBox()
        self.waterfall_mode_combo.addItems(["None", "Adaptive step", "Uniform step"])
        self.waterfall_mode_combo.setCurrentIndex(0)
        self.plot_buttons_layout.addWidget(QLabel("Waterfall:"))
        self.plot_buttons_layout.addWidget(self.waterfall_mode_combo)
        self.waterfall_mode_combo.currentIndexChanged.connect(self.recompute_waterfall_layout)
        self.waterfall_slider = QSlider(Qt.Horizontal)
        self.waterfall_slider.setRange(0, 100)
        self.waterfall_slider.setValue(0)
        self.waterfall_slider.setEnabled(False)
        self.waterfall_slider.valueChanged.connect(self.on_waterfall_slider_changed)
        self.plot_buttons_layout.addWidget(self.waterfall_slider)

        self.waterfall_spin = QDoubleSpinBox()
        self.waterfall_spin.setRange(0.0, 1.0)
        self.waterfall_spin.setDecimals(2)
        self.waterfall_spin.setSingleStep(0.01)
        self.waterfall_spin.setValue(0.00)
        self.waterfall_spin.setEnabled(False)
        self.waterfall_spin.valueChanged.connect(self.on_waterfall_spin_changed)
        self.plot_buttons_layout.addWidget(self.waterfall_spin)

        # Grid density selector next to the waterfall controls
        self.grid_label = QLabel("Grid:")
        self.plot_buttons_layout.addWidget(self.grid_label)
        self.grid_mode_combo = QComboBox()
        self.grid_mode_combo.addItems(["None", "Coarse", "Fine", "Finest"])
        self.grid_mode_combo.currentIndexChanged.connect(self.on_grid_toggled)
        self.grid_mode_combo.setCurrentText("Finest")
        # Apply default grid setting
        try:
            self.on_grid_toggled()
        except Exception:
            pass
        self.plot_buttons_layout.addWidget(self.grid_mode_combo)
        # Button to load reference spectra from library
        self.load_reference_button = QPushButton("Load reference")
        self.load_reference_button.setToolTip("Load reference spectra from the library")
        try:
            self.load_reference_button.clicked.connect(self.on_load_reference_clicked)
        except Exception:
            pass
        self.plot_buttons_layout.addWidget(self.load_reference_button)

        # Stretch pushes the Export/ Clear buttons to the right
        self.plot_buttons_layout.addStretch()

        self.export_import_plotted_button = QPushButton("Export/Import")
        self.plot_buttons_layout.addWidget(self.export_import_plotted_button)

        # Export/Import popup menu for Plotted Data
        self.export_import_plotted_menu = QMenu(self)
        self.action_export_csv_plotted = self.export_import_plotted_menu.addAction("Export CSV")
        self.action_import_csv_plotted = self.export_import_plotted_menu.addAction("Import CSV")
        self.action_export_csv_plotted.triggered.connect(self.export_ascii_plotted)
        self.action_import_csv_plotted.triggered.connect(self.import_csv_plotted)
        self.export_import_plotted_button.clicked.connect(
            lambda: self.export_import_plotted_menu.exec_(
                self.export_import_plotted_button.mapToGlobal(
                    QPoint(0, self.export_import_plotted_button.height())
                )
            )
        )

        self.clear_plotted_data_button = QPushButton("Clear Plotted Data")
        self.clear_plotted_data_button.clicked.connect(self.clear_plotted_data)
        self.clear_plotted_data_button.setMaximumWidth(150)
        self.plot_buttons_layout.addWidget(self.clear_plotted_data_button)
        # ----------------------------------------------------------------------

        self.plot_left_layout.addLayout(self.plot_buttons_layout)

        self.plotted_splitter = QSplitter(Qt.Horizontal)
        self.plotted_splitter.addWidget(self.plot_left_widget)
        self.plotted_list = QListWidget()
        self.plotted_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.plotted_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.plotted_list.setDragEnabled(True)
        self.plotted_list.setDropIndicatorShown(True)
        self.plotted_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.plotted_splitter.addWidget(self.plotted_list)
        # Enable drag & drop reordering; replot on drop
        self.plotted_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.plotted_list.setDefaultDropAction(Qt.MoveAction)
        try:
            self.plotted_list.model().rowsMoved.connect(lambda *args: self.on_plotted_list_reordered())
        except Exception:
            pass
        self.plotted_splitter.setStretchFactor(0, 65)
        self.plotted_splitter.setStretchFactor(1, 35)
        self.data_tabs.addTab(self.plotted_splitter, "Plotted Data")

        #######################################################################
        # SIGNAL CONNECTIONS
        #######################################################################
        self.canvas_plotted.mpl_connect('pick_event', self.on_legend_pick)
        self.cid_press = self.canvas_proc.mpl_connect("button_press_event", self.on_press)
        self.cid_motion = self.canvas_proc.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_release = self.canvas_proc.mpl_connect("button_release_event", self.on_release)

        self.combo_bg.currentIndexChanged.connect(self.update_plot_processed)
        self.combo_poly.currentIndexChanged.connect(self.update_plot_processed)
        self.spin_preedge.valueChanged.connect(self.update_plot_processed)
        try:
            self.chk_group_bg.stateChanged.connect(self._on_group_bg_checkbox_toggled)
        except Exception:
            try:
                self.chk_group_bg.stateChanged.connect(self.update_plot_processed)
            except Exception:
                pass
        try:
            self.chk_group_bg_slope.stateChanged.connect(self._on_group_bg_slope_checkbox_toggled)
        except Exception:
            try:
                self.chk_group_bg_slope.stateChanged.connect(self.update_plot_processed)
            except Exception:
                pass
        self.chk_show_without_bg.stateChanged.connect(self._on_bg_subtract_toggled)
        self.combo_post_norm.currentIndexChanged.connect(self.update_plot_processed)

        # Dictionary for storing the unshifted data for each curve (for waterfall restoration)
        self.original_line_data = {}

        self.update_file_label()
        self.update_pass_button_state()
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

    def showEvent(self, event):
        """Set initial splitter proportions once the window is shown."""
        try:
            super().showEvent(event)
        except Exception:
            pass
        try:
            if not getattr(self, "_initial_splitters_set", False):
                self._initial_splitters_set = True
                QTimer.singleShot(0, self._apply_initial_splitter_sizes)
        except Exception:
            pass

    def _apply_initial_splitter_sizes(self):
        """Apply initial splitter sizes for a more balanced default layout."""
        try:
            # Main splitter: make left tree panel ~20% narrower than the old 1:2 default (~33% -> ~26%)
            total = int(self.splitter.width())
            if total > 0:
                left = max(180, int(total * 0.264))
                self.splitter.setSizes([left, max(200, total - left)])
        except Exception:
            pass

        # Tab splitters: widen the right-hand curve-tree widget vs previous defaults
        specs = [
            ("raw_splitter", 0.60),     # wider curve tree on Raw tab
            ("proc_splitter", 0.68),    # wider curve tree on Processed tab
            ("plotted_splitter", 0.35)  # narrower list; wider plot on Plotted tab
        ]
        for name, right_frac in specs:
            try:
                sp = getattr(self, name, None)
                if sp is None:
                    continue
                total = int(sp.width())
                if total <= 0:
                    continue
                right = max(220, int(total * float(right_frac)))
                left = max(220, total - right)
                sp.setSizes([left, right])
            except Exception:
                continue



    def _refresh_all_in_channel_combo(self):
        """
        Populate the 'All in channel' combobox with unique channel names (last path component)
        collected from all currently opened HDF5 files. Keeps selection stable and avoids
        re-entrant side effects by blocking signals during the update.
        """
        try:
            combo = getattr(self, "combo_all_channel", None)
            if combo is None:
                return
    
            # --- gather all 1D dataset relpaths from currently opened files ---
            rels = set()
            files = list(getattr(self, "hdf5_files", {}).keys()) if hasattr(self, "hdf5_files") else []
            for abs_path in files:
                try:
                    with self._open_h5_read(abs_path) as f:
                        def _visit(name, obj):
                            try:
                                import h5py
                                if isinstance(obj, h5py.Dataset):
                                    shp = tuple(getattr(obj, "shape", ()) or ())
                                    if len(shp) == 1 and getattr(obj, "size", 0) > 0:
                                        rels.add(name.lstrip("/"))
                            except Exception:
                                pass
                        f.visititems(_visit)
                except Exception:
                    # ignore unreadable files; continue with others
                    pass
    
            # --- turn relpaths into unique channel names (last path component) ---
            channels, seen = [], set()
            for s in sorted(rels, key=lambda x: x.lower() if isinstance(x, str) else str(x)):
                if not isinstance(s, str):
                    continue
                ch = s.split("/")[-1]
                if ch and ch not in seen:
                    seen.add(ch)
                    channels.append(ch)
    
            # --- no-op if items haven't changed (prevents snap-back) ---
            current = [combo.itemText(i) for i in range(combo.count())]
            if current == channels:
                return
    
            # Preserve user's intended selection if available
            desired = getattr(self, "_desired_all_channel_selection", None)
            prev_text = combo.currentText() if combo.count() else None
    
            combo.blockSignals(True)
            combo.clear()
            if channels:
                combo.addItems(channels)
                # Prefer desired, then previous text; otherwise leave the default (index 0)
                if isinstance(desired, str) and desired in channels:
                    combo.setCurrentText(desired)
                elif isinstance(prev_text, str) and prev_text in channels:
                    combo.setCurrentText(prev_text)
            combo.blockSignals(False)
    
        except Exception:
            # Best-effort cleanup
            try:
                self.combo_all_channel.blockSignals(False)
            except Exception:
                pass



    
    def _on_all_channel_selection_changed(self, *_):
        """Record user's desired channel selection; if checkbox is checked, apply it.
        Does NOT toggle anything when the checkbox is unchecked."""
        try:
            sel = (self.combo_all_channel.currentText() or "").strip()
            self._desired_all_channel_selection = sel
            if hasattr(self, "cb_all_in_channel") and self.cb_all_in_channel.isChecked():
                # Defer actual apply slightly to let UI settle and avoid re-entrancy
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._apply_all_in_channel_filter())
        except Exception:
            pass
        
    def _apply_all_in_channel_filter(self, *args):
            """Apply or clear the group of curves for the selected channel across all entries.
            - If checkbox is CHECKED: only the selected channel is loaded; switching clears previous channel.
            - If checkbox is UNCHECKED: clear curves for the selected channel (and previously active if different).
            Uses a re-entrancy guard and records desired selection to prevent combobox snap-back."""
            try:
                if not hasattr(self, "combo_all_channel") or not hasattr(self, "cb_all_in_channel"):
                    return
    
                selected = (self.combo_all_channel.currentText() or "").strip()
                checked = bool(self.cb_all_in_channel.isChecked())
                self._desired_all_channel_selection = selected  # remember user's intent
    
                prev_active = getattr(self, "_last_all_channel_filter", None)
    
                # Guard against re-entrant refreshes
                if getattr(self, "_in_all_channel_apply", False):
                    return
                self._in_all_channel_apply = True
    
                if checked:
                    # Load only the selected channel; clear previous active if different
                    if prev_active and prev_active != selected:
                        try:
                            self.set_group_visibility(prev_active, False)
                        except Exception:
                            pass
                    if selected:
                        self.set_group_visibility(selected, True)
                        self._last_all_channel_filter = selected
                else:
                    # Unchecked: clear selected, and also clear previous active if different
                    if selected:
                        try:
                            self.set_group_visibility(selected, False)
                        except Exception:
                            pass
                    if prev_active and prev_active != selected:
                        try:
                            self.set_group_visibility(prev_active, False)
                        except Exception:
                            pass
                    self._last_all_channel_filter = None
    
            except Exception:
                pass
            finally:
                # Allow pending timers to refresh safely now that we're done applying
                self._in_all_channel_apply = False
    def _on_tree_context_menu(self, pos):
        """Context menu handler for closing an individual HDF5 file from the main tree."""
        try:
            tree = getattr(self, "tree", None)
            if tree is None:
                return
            item = tree.itemAt(pos)
            # Only top-level items (files) should show the Close action
            if item is None or item.parent() is not None:
                return
            data = item.data(0, Qt.UserRole)
            if not isinstance(data, tuple) or len(data) != 2:
                return
            abs_path, hdf5_path = data
            # File-level items have an abs_path and empty internal HDF5 path
            if not abs_path or hdf5_path:
                return

            menu = QMenu(tree)
            close_action = menu.addAction("Close")
            chosen = menu.exec_(tree.viewport().mapToGlobal(pos))
            if chosen == close_action:
                self._confirm_and_close_file(abs_path, item)
        except Exception:
            # Fail silently: context menu is convenience only
            pass

    def _confirm_and_close_file(self, abs_path, item=None):
        """Ask the user to confirm closing a specific HDF5 file and delegate cleanup."""
        try:
            base = os.path.basename(abs_path) or abs_path
        except Exception:
            base = str(abs_path)

        try:
            reply = QMessageBox.question(
                self,
                "Close HDF5 file",
                f"Do you really want to close this file?\n{base}",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
        except Exception:
            return

        if reply != QMessageBox.Ok:
            return

        # Remove the top-level item from the tree, if provided
        try:
            tree = getattr(self, "tree", None)
            if tree is not None and item is not None:
                idx = tree.indexOfTopLevelItem(item)
                if idx >= 0:
                    tree.takeTopLevelItem(idx)
        except Exception:
            pass

        # Delegate the actual state cleanup to the data-layer helper
        try:
            if hasattr(self, "close_single_hdf5_file"):
                self.close_single_hdf5_file(abs_path)
        except Exception:
            pass

    def keyPressEvent(self, event):
        """Handle Delete on the main HDF5 tree to close a single file."""
        handled = False
        try:
            if event.key() == Qt.Key_Delete:
                tree = getattr(self, "tree", None)
                if tree is not None and tree.hasFocus():
                    item = tree.currentItem()
                    if item is not None and item.parent() is None:
                        data = item.data(0, Qt.UserRole)
                        if isinstance(data, tuple) and len(data) == 2:
                            abs_path, hdf5_path = data
                            if abs_path and not hdf5_path:
                                self._confirm_and_close_file(abs_path, item)
                                handled = True
        except Exception:
            handled = False

        if not handled:
            try:
                super().keyPressEvent(event)
            except Exception:
                pass

def launch():
    import sys
    # Use Qt from PyQt5 (add this import if not already present at the top)
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    # If your main window class has a different name than HDF5Viewer, change it here:
    w = HDF5Viewer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec_())


# ---- Split-structure compatibility layer ----
# Keep alias so the new entry point can import MainWindow uniformly.
try:
    MainWindow = HDF5Viewer  # alias to preserve class name externally
except NameError:
    # If class was renamed, leave MainWindow undefined to raise at import time
    pass

def launch():
    # Backwards compatibility helper if older scripts import launch()
    from PyQt5 import QtWidgets
    import sys
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

    def _enforce_all_channel_selection(self):
        """Ensure the combobox shows the user's desired selection, if available, without causing re-entrant applies."""
        try:
            desired = getattr(self, "_desired_all_channel_selection", None)
            if not isinstance(desired, str) or not desired:
                return
            # If already selected, nothing to do
            if self.combo_all_channel.currentText() == desired:
                return
            # If desired exists in items, set it with signals blocked
            count = self.combo_all_channel.count()
            if count <= 0:
                return
            idx = -1
            for i in range(count):
                if self.combo_all_channel.itemText(i) == desired:
                    idx = i; break
            if idx >= 0:
                self.combo_all_channel.blockSignals(True)
                try:
                    self.combo_all_channel.setCurrentIndex(idx)
                finally:
                    self.combo_all_channel.blockSignals(False)
        except Exception:
            pass
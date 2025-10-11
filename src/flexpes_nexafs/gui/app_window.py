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
)
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from datetime import datetime
from flexpes_nexafs.utils.sorting import parse_entry_number
from flexpes_nexafs.gui.widgets.curve_item import CurveListItemWidget

###############################################################################
# Helper function to parse integer from something like "entry1001"
# for sorting the Raw Data and Processed Data items in ascending order
###############################################################################


class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlexPES NEXAFS Plotter")
        self.setGeometry(100, 100, 1250, 600)

        self.VERSION_NUMBER = "1.9.2"
        self.CREATION_DATETIME = "2025-10-11"

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
        self.tree.setHeaderLabels(["HDF5 Structure", "Plot"])
        self.tree.header().resizeSection(0, 250)
        self.tree.header().resizeSection(1, 30)
        self.tree.itemExpanded.connect(self.load_subtree)
        self.tree.itemChanged.connect(self.toggle_plot)
        self.tree.itemClicked.connect(self.display_data)
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
        self.raw_tree.itemChanged.connect(self.raw_tree_item_changed)
        self.raw_splitter.addWidget(self.raw_tree)
        self.raw_splitter.setStretchFactor(0, 65)
        self.raw_splitter.setStretchFactor(1, 35)
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

        self.chk_sum = QCheckBox("Sum them up?")
        self.proc_controls_top_layout.addWidget(self.chk_sum)
        self.proc_controls_top_layout.addStretch()

        self.pass_button = QPushButton("Pass")
        self.proc_controls_top_layout.addWidget(self.pass_button)
        self.export_ascii_button = QPushButton("Export ASCII")
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
        self.proc_left_layout.addWidget(self.toolbar_proc)
        self.proc_left_layout.addWidget(self.canvas_proc, 1)

        # Bottom controls
        self.proc_controls_bottom = QWidget()
        self.proc_controls_bottom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.proc_controls_bottom_layout = QHBoxLayout(self.proc_controls_bottom)
        self.proc_controls_bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.proc_controls_bottom_layout.setSpacing(5)

        self.proc_controls_bottom_layout.addWidget(QLabel("Choose background:"))
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
        self.spin_preedge.setValue(5)
        self.proc_controls_bottom_layout.addWidget(self.spin_preedge)

        self.chk_show_without_bg = QCheckBox("Subtract background?")
        self.proc_controls_bottom_layout.addWidget(QLabel("Normalize:"))
        self.combo_post_norm = QComboBox()
        self.combo_post_norm.addItems(["None", "Max", "Jump", "Area"])
        self.combo_post_norm.setEnabled(False)
        self.proc_controls_bottom_layout.addWidget(self.combo_post_norm)
        self.proc_controls_bottom_layout.addWidget(self.chk_show_without_bg)


        self.proc_left_layout.addWidget(self.proc_controls_bottom, 0)

        self.proc_tree = QTreeWidget()
        self.proc_tree.setHeaderHidden(True)
        self.proc_tree.itemChanged.connect(self.proc_tree_item_changed)

        self.proc_splitter = QSplitter(Qt.Horizontal)
        self.proc_splitter.addWidget(self.proc_left_widget)
        self.proc_splitter.addWidget(self.proc_tree)
        self.proc_splitter.setStretchFactor(0, 57)
        self.proc_splitter.setStretchFactor(1, 43)

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

        # New Grid checkbox added next to the waterfall controls
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(False)
        self.grid_checkbox.stateChanged.connect(self.on_grid_toggled)
        self.plot_buttons_layout.addWidget(self.grid_checkbox)

        # Stretch pushes the Export/ Clear buttons to the right
        self.plot_buttons_layout.addStretch()

        self.export_ascii_plotted_button = QPushButton("Export ASCII")
        self.export_ascii_plotted_button.clicked.connect(self.export_ascii_plotted)
        self.plot_buttons_layout.addWidget(self.export_ascii_plotted_button)

        self.clear_plotted_data_button = QPushButton("Clear Plotted Data")
        self.clear_plotted_data_button.clicked.connect(self.clear_plotted_data)
        self.clear_plotted_data_button.setMaximumWidth(150)
        self.plot_buttons_layout.addWidget(self.clear_plotted_data_button)
        # ----------------------------------------------------------------------

        self.plot_left_layout.addLayout(self.plot_buttons_layout)

        self.plotted_splitter = QSplitter(Qt.Horizontal)
        self.plotted_splitter.addWidget(self.plot_left_widget)
        self.plotted_list = QListWidget()
        self.plotted_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.plotted_splitter.addWidget(self.plotted_list)
        self.plotted_splitter.setStretchFactor(0, 64)
        self.plotted_splitter.setStretchFactor(1, 36)
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
        self.chk_show_without_bg.stateChanged.connect(self._on_bg_subtract_toggled)
        self.combo_post_norm.currentIndexChanged.connect(self.update_plot_processed)

        # Dictionary for storing the unshifted data for each curve (for waterfall restoration)
        self.original_line_data = {}

        self.update_file_label()
        self.update_pass_button_state()
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)


    ###############################################################################
    # WATERFALL LOGIC
    ###############################################################################
    # ---- HDF5 concurrent read helpers ---------------------------------
    def _open_h5_read(self, path, retries: int = 3):
        """
        Open HDF5 file for reading with SWMR and retry a few times if busy.
        This helps tolerate concurrent writes by another process.
        """
          
        last_error = None
        for i in range(max(1, int(retries))):
            try:
                # Try modern SWMR-compatible open
                return h5py.File(path, "r", swmr=True, libver="latest", locking=False)
            except TypeError:
                # Older h5py: swmr/libver not supported; fall back safely
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
            return h5py.File(path, "r", locking=False)
        except Exception as e:
            # Give up after retries
            raise last_error or e


    def _filter_empty_plot_data(self):
        """Drop any entries with zero-length arrays to avoid reduction errors."""
        try:
            self.plot_data = {k: v for k, v in self.plot_data.items() if getattr(v, "size", 0) > 0}
        except Exception:
            pass
    # --------------------------------------------------------------------

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
    def apply_waterfall_shift(self):
        """Apply/update Waterfall offsets according to selected mode."""
        if not self.plotted_lines:
            return
        # Determine mode
        try:
            mode = self.waterfall_mode_combo.currentText()
        except Exception:
            # Backward compatibility: use checkbox if combo not present
            mode = "Adaptive step" if getattr(self, "waterfall_checkbox", None) and self.waterfall_checkbox.isChecked() else "None"

        # Restore originals first for consistent behavior
        try:
            self.restore_original_line_data()
        except Exception:
            pass

        import numpy as np

        # Build ordered visible keys
        plotted_keys_in_order = []
        for i in range(self.plotted_list.count()):
            item = self.plotted_list.item(i)
            widget = self.plotted_list.itemWidget(item)
            if widget:
                key = widget.key
                line = self.plotted_lines.get(key)
                if line is not None and line.get_visible():
                    plotted_keys_in_order.append(key)

        if not plotted_keys_in_order:
            self.rescale_plotted_axes()
            return

        if mode == "None":
            self.rescale_plotted_axes()
            return

        if mode == "Adaptive step":
            alpha = float(self.waterfall_spin.value())
            prev_max = None
            for i, key in enumerate(plotted_keys_in_order):
                line = self.plotted_lines[key]
                xdata = np.asarray(line.get_xdata())
                ydata = np.asarray(line.get_ydata(), dtype=float)
                if xdata.size < 1 or ydata.size < 1:
                    continue
                if prev_max is None:
                    mfin0 = np.isfinite(ydata)
                    prev_max = float(np.max(ydata[mfin0])) if np.any(mfin0) else 0.0
                    continue
                if np.isfinite(ydata[0]):
                    y0 = float(ydata[0])
                else:
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
        for key, line in self.plotted_lines.items():
            if key in self.original_line_data:
                x_orig, y_orig = self.original_line_data[key]
                line.set_xdata(x_orig)
                line.set_ydata(y_orig)
        self.canvas_plotted.draw()

    def get_line_color_for_key(self, key, ax):
        for line in ax.get_lines():
            if hasattr(line, 'dataset_key') and line.dataset_key == key:
                return line.get_color()
        return None

    ###############################################################################
    # pass_to_plotted_no_clear
    ###############################################################################
    def pass_to_plotted_no_clear(self):
        if not self.plot_data:
            return

        if self.chk_sum.isChecked():
            visible_keys = [key for key in self.plot_data if self.raw_visibility.get(key, False)]
            if not visible_keys:
                return
            key = visible_keys[0]
        else:
            visible_keys = [key for key in self.plot_data if self.raw_visibility.get(key, False)]
            if len(visible_keys) != 1:
                QMessageBox.warning(
                    self, "Warning", "Please select exactly one dataset, or enable 'Sum'."
                )
                return
            key = visible_keys[0]

                # Use synthetic key for summed curve so it doesn't collide with first component
        storage_key = f"SUMMED#{self._sum_serial + 1}" if self.chk_sum.isChecked() else key

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
            subtracted = self._safe_post_normalize(main_x, subtracted, norm_mode)
            main_y = subtracted
            bg_subtracted = True

        if self.chk_sum.isChecked():
            origin_label = "Summed Curve"
        else:
            parts = key.split("##", 1)
            origin_label = parts[1] if len(parts) == 2 else key

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
        self.original_line_data[storage_key] = (main_x.copy(), main_y.copy())

        item = QListWidgetItem()
        widget = CurveListItemWidget(origin_label, line.get_color(), storage_key)
        widget.colorChanged.connect(self.change_curve_color)
        widget.visibilityChanged.connect(self.change_curve_visibility)
        widget.styleChanged.connect(self.change_curve_style)
        item.setSizeHint(widget.sizeHint())
        self.plotted_list.addItem(item)
        self.plotted_list.setItemWidget(item, widget)

        self.data_tabs.setCurrentIndex(2)
        self.update_legend()
        if self.chk_sum.isChecked():
            self._sum_serial += 1

    def on_grid_toggled(self, state):
        if state == Qt.Checked:
            self.plotted_ax.grid(True)
        else:
            self.plotted_ax.grid(False)
        self.canvas_plotted.draw()

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
        self.original_line_data.clear()  # Also clear waterfall references

        # Reset Waterfall controls
        self.waterfall_mode_combo.setCurrentIndex(0)
        self.waterfall_slider.setValue(0)
        self.waterfall_spin.setValue(0.00)

        # Reset Grid option: uncheck grid and turn off grid on the axes
        self.grid_checkbox.setChecked(False)
        self.plotted_ax.grid(False)
        self.canvas_plotted.draw()

    def recursive_uncheck(self, item, col):
        if not item:
            return
        if item.data(col, Qt.UserRole):
            item.setCheckState(col, Qt.Unchecked)
        for i in range(item.childCount()):
            self.recursive_uncheck(item.child(i), col)

    def on_tab_changed(self, index):
        if index == 0:
            self.update_plot_raw()
            self.raw_tree.update()
        elif index == 1:
            self.update_plot_processed()
            self.proc_tree.update()

    def group_datasets(self):
        groups = []
        for key in self.plot_data.keys():
            parts = key.split("##", 1)
            if len(parts) != 2:
                continue
    
            abs_path, hdf5_path = parts
            parent = hdf5_path.rsplit("/", 1)[0] if "/" in hdf5_path else ""
            y_data = self.plot_data[key]
            x_data = self._lookup_energy(abs_path, parent, len(y_data))
    
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

    def update_raw_tree(self):
        self.raw_tree.blockSignals(True)
        self.raw_tree.clear()
        groups = self.group_datasets()

        for idx, group in enumerate(groups):
            region_id = f"region_{idx}"
            region_state = self.region_states.get(region_id, Qt.Checked)
            region_item = QTreeWidgetItem([f"Region {idx+1}"])
            region_item.setFlags(region_item.flags() | Qt.ItemIsUserCheckable)
            region_item.setCheckState(0, region_state)
            region_item.setData(0, Qt.UserRole+1, region_id)

            sorted_keys = sorted(
                group['keys'],
                key=lambda x: parse_entry_number(x.split("##",1)[1] if "##" in x else "")
            )

            for key in sorted_keys:
                parts = key.split("##", 1)
                label = self.shorten_label(parts[1]) if len(parts) == 2 else key
                child = QTreeWidgetItem([label])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child_state = Qt.Checked if self.raw_visibility.get(key, True) else Qt.Unchecked
                child.setCheckState(0, child_state)
                child.setData(0, Qt.UserRole, key)
                color = self.get_line_color_for_key(key, self.raw_ax)
                if color:
                    pixmap = QPixmap(16, 16)
                    pixmap.fill(QColor(color))
                    child.setIcon(0, QIcon(pixmap))
                region_item.addChild(child)
            self.raw_tree.addTopLevelItem(region_item)
            region_item.setExpanded(True)
            child_count = region_item.childCount()
            checked_count = sum(
                region_item.child(i).checkState(0) == Qt.Checked
                for i in range(child_count)
            )
            if checked_count == 0:
                region_item.setCheckState(0, Qt.Unchecked)
            elif checked_count == child_count:
                region_item.setCheckState(0, Qt.Checked)
            else:
                region_item.setCheckState(0, Qt.PartiallyChecked)

        self.raw_tree.blockSignals(False)
        self.raw_tree.update()
        self.raw_tree_reset = False
        self.raw_tree.repaint()

    def raw_tree_item_changed(self, item, column):
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
                checked_count = sum(
                    parent_item.child(i).checkState(0) == Qt.Checked
                    for i in range(child_count)
                )
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
        self.proc_tree.blockSignals(True)
        self.proc_tree.clear()
        groups = self.group_datasets()

        for idx, group in enumerate(groups):
            region_id = f"proc_region_{idx}"
            region_item = QTreeWidgetItem([f"Region {idx+1}"])
            region_item.setFlags(region_item.flags() | Qt.ItemIsUserCheckable)
            region_item.setCheckState(0, self.proc_region_states.get(region_id, Qt.Checked))
            region_item.setData(0, Qt.UserRole+1, region_id)

            sorted_keys = sorted(
                group['keys'],
                key=lambda x: parse_entry_number(x.split("##",1)[1] if "##" in x else "")
            )

            for key in sorted_keys:
                parts = key.split("##", 1)
                label = self.shorten_label(parts[1]) if len(parts) == 2 else key
                child = QTreeWidgetItem([label])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child_state = Qt.Checked if self.raw_visibility.get(key, True) else Qt.Unchecked
                child.setCheckState(0, child_state)
                child.setData(0, Qt.UserRole, key)
                color = self.get_line_color_for_key(key, self.proc_ax)
                if color:
                    pixmap = QPixmap(16, 16)
                    pixmap.fill(QColor(color))
                    child.setIcon(0, QIcon(pixmap))
                region_item.addChild(child)

            self.proc_tree.addTopLevelItem(region_item)
            region_item.setExpanded(True)
            child_count = region_item.childCount()
            checked_count = sum(
                region_item.child(i).checkState(0) == Qt.Checked
                for i in range(child_count)
            )
            if checked_count == 0:
                region_item.setCheckState(0, Qt.Unchecked)
            elif checked_count == child_count:
                region_item.setCheckState(0, Qt.Checked)
            else:
                region_item.setCheckState(0, Qt.PartiallyChecked)

        self.proc_tree.blockSignals(False)
        self.proc_tree.update()

    def proc_tree_item_changed(self, item, column):
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
                checked_count = sum(
                    parent_item.child(i).checkState(0) == Qt.Checked
                    for i in range(child_count)
                )
                if checked_count == 0:
                    parent_item.setCheckState(0, Qt.Unchecked)
                elif checked_count == child_count:
                    parent_item.setCheckState(0, Qt.Checked)
                else:
                    parent_item.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self.proc_tree.blockSignals(False)
        QTimer.singleShot(0, self.update_plot_processed)

    def shorten_label(self, hdf5_path):
        if "ch1" in hdf5_path:
            channel = "TEY"
        elif "ch3" in hdf5_path:
            channel = "PEY"
        elif "roi2_dtc" in hdf5_path:
            channel = "TFY"
        elif "roi1_dtc" in hdf5_path:
            channel = "PFY"
        else:
            channel = hdf5_path
        tokens = hdf5_path.split("/")
        entry = tokens[0] if tokens and tokens[0].startswith("entry") else ""
        return f"{channel} in {entry}" if entry else channel

    def set_group_visibility(self, filter_str: str, visible: bool):
        """
        Show or hide all 1D datasets whose names contain `filter_str`.
        Opens each HDF5 file briefly (non-locking) and updates plot visibility.
        """
        
        for abs_path in list(self.hdf5_files.keys()):
            try:
                with self._open_h5_read(abs_path) as f:
    
                    # ---- inner function defined INSIDE the with-block ----
                    def check_item(name, obj):
                        # This function can now see abs_path, filter_str, visible, f, etc.
                        if isinstance(obj, h5py.Dataset) and getattr(obj, "ndim", 0) == 1 and filter_str in name:
                            if getattr(obj, "size", 0) == 0:
                                return
                            combined_label = f"{abs_path}##{name}"
                            if visible:
                                # Read dataset immediately and close file after
                                self.plot_data[combined_label] = obj[()]
                                self.raw_visibility[combined_label] = True
                            else:
                                self.plot_data.pop(combined_label, None)
                                self.raw_visibility.pop(combined_label, None)
    
                    # ---- visit all items once ----
                    f.visititems(check_item)
    
            except Exception as e:
                print(f"Warning: could not open {abs_path}: {e}")
                continue
    
        # ---- refresh plots once after processing all files ----
        self.update_plot_raw()
        self.update_pass_button_state()



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

    def update_file_label(self):
        self.file_label.setText("\n".join(self.hdf5_files.keys()) if self.hdf5_files else "No file open")

    def visible_curves_count(self):
        return sum(1 for key, visible in self.raw_visibility.items() if visible)

    def update_pass_button_state(self):
        self.pass_button.setEnabled(self.chk_sum.isChecked() or self.visible_curves_count() == 1)

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
        usage_text = """
        <h2>FlexPES NEXAFS Plotter</h2>
        <h3><b>Overview:</b></h3>
        This application enables you to open HDF5 files containing NEXAFS spectra recorded at the FlexPES beamline (MAX IV Laboratory)
        and perform pre-processing, visualization, and export of raw and processed data.</p>

        <h3>File Controls (Top Left Panel):</h3>
        <ul>
            <li><b>Open HDF5 File:</b> Load one or more HDF5 files containing NEXAFS data.</li>
            <li><b>Close all:</b> Close all currently opened files.</li>
            <li><b>Clear all:</b> Remove all loaded data and reset the interface.</li>
            <li><b>Help:</b> Opens a menu with “Usage” and “About” options.</li>
        </ul>

        <h3>File Tree Panel:</h3>
        <p>Displays the hierarchical structure of the loaded HDF5 files. Expand groups to view individual datasets. Toggle checkboxes to select which 1D datasets (inside the "measurement" folder) to include in the plots.</p>

        <h3>Tabs (Right Panel):</h3>
        <ol>
            <li><b>Raw Data Tab:</b>
                <ul>
                    <li><i>Data Plot:</i> Shows the raw spectra, single or multiple.</li>
                    <li><i>Channel Checkboxes (TEY, PEY, TFY, PFY):</i> Quickly enable/disable groups of datasets.</li>
                    <li><i>Data Tree to the right:</i> Lists raw data items for detailed inspection. Datasets are grouped into energy-dependent "regions". Group checkboxes allow selection and deselection of specific absorption edges.</li>
                    <li><i>Scalar Display:</i> Presents data values for selected scalar or textual metadata items (under the plot canvas).</li>
                </ul>
            </li>
            <li><b>Processed Data Tab:</b>
                <ul>
                    <li><i>Normalization Controls:</i> Option to normalize spectra by a selected I₀ channel ("b107a_em_03_ch2" by default).</li>
                    <li><i>Summing Option:</i> Choose to sum multiple datasets. Mainly used to sum multiple sweeps for improved statistics.</li>
                    <li><i>Background Settings:</i> Select background subtraction mode (None, Automatic, Manual) with options to set the polynomial degree and pre-edge percentage used for the fit. In Automatic mode, an additional constraint ensures that the derivative at the end of the background curve matches the slope of the data curve. Manual mode can be used if the Automatic mode fails. In the Manual mode, adjust points interactively with the mouse.</li>
                    <li><i>Subtraction checkbox:</i> Visualizes the data with background subtracted. Can be applied only if one dataset is selected (or after summing up several curves).</li>
                    <li><i>Extra normalization combo box:</i> Allows to select extra normalization after background subtraction. "None" does nothing, "Max", "Jump" and "Area" normalize the BG-subtracted curve to its maximum, absorption jump (at last point) and to the area under the curve, rspectively.</li>
                    <li><i>PASS button:</i> Once the curve is pre-processed to its "final" shape (i.e., normalized, possibly summed up, and a suitable background subtracted) it can be passed to the "Plotted Data" tab for further visualization. In addition, it can also be saved as a CSV file with the "Export ASCII" button.</li>
                </ul>
            </li>
            <li><b>Plotted Data Tab:</b>
                <ul>
                    <li><i>Interactive Plot Canvas:</i> An embedded matplotlib canvas with built-in zooming, panning, and saving features.</li>
                    <li><i>Waterfall slider:</i> Enables waterfall representation if more then 1 curve are present in the plot.</li>
                    <li><i>Curve List:</i> A side panel that lists plotted curves and lets you adjust each curve’s color, style, and visibility.</li>
                    <li><i>Interactive Legends:</i> Click on legend items to rename curves for clarity. Drag the legend with the mouse to reposition it.</li>
                    <li><i>Export/Clear Buttons:</i> Export plotted data to CSV or clear the plot entirely.</li>
                </ul>
            </li>
        </ol>
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("About FlexPES NEXAFS Plotter")
        dlg.resize(600, 500)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setStyleSheet("font-size: 16px;")
        browser.setHtml(usage_text)
        layout.addWidget(browser)
        dlg.exec_()

    def reset_manual_mode(self):
        self.manual_mode = False
        self.manual_points = []
        self.manual_bg_line = None
        self.manual_poly = None
        self.manual_poly_degree = None
        self.last_sum_state = False
        self.last_normalize_state = False
        self.last_plot_len = 0

    def clear_all_except_plotted(self):
        for i in range(self.tree.topLevelItemCount()):
            self.recursive_uncheck(self.tree.topLevelItem(i), 1)
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
        # self.combo_poly.setCurrentIndex(0)
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
        # self.combo_poly.setCurrentIndex(0)
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
        self.original_line_data.clear()  # also clear for Waterfall
        self.update_pass_button_state()

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
            self.hdf5_files[abs_path] = True
    
            file_item = QTreeWidgetItem([os.path.basename(abs_path)])
            file_item.setData(0, 1, (abs_path, ""))
    
            has_children = False
            try:
                with self._open_h5_read(abs_path) as f:
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
    
        except Exception as e:
            self.file_label.setText(f"Error opening file: {e}")


    def populate_norm_channels(self, abs_path):
        """Populate normalization channels using a short-lived file open (non-locking)."""
        self.combo_norm.clear()
        default_channel = "b107a_em_03_ch2"
        default_index = -1
        try:
            with self._open_h5_read(abs_path) as f:
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
        data = item.data(0, 1)
        if not data:
            return
        abs_path, hdf5_path = data
        if abs_path not in self.hdf5_files:
            return
    
        try:
            with self._open_h5_read(abs_path) as f:
                if hdf5_path == "":
                    if item.childCount() == 1 and item.child(0).text(0) == "(click to expand)":
                        item.removeChild(item.child(0))
                        for key in f.keys():
                            child_item = QTreeWidgetItem([key])
                            child_item.setData(0, 1, (abs_path, key))
                            sub_obj = f[key]
                            if isinstance(sub_obj, h5py.Group) and sub_obj.keys():
                                child_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                                child_item.addChild(QTreeWidgetItem(["(click to expand)"]))
                            elif isinstance(sub_obj, h5py.Dataset) and sub_obj.ndim == 1:
                                child_item.setCheckState(1, Qt.Unchecked)
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
                                child_item.setData(0, 1, (abs_path, f"{hdf5_path}/{key}"))
                                sub_obj = obj[key]
                                if isinstance(sub_obj, h5py.Group) and sub_obj.keys():
                                    child_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                                    child_item.addChild(QTreeWidgetItem(["(click to expand)"]))
                                elif isinstance(sub_obj, h5py.Dataset) and sub_obj.ndim == 1:
                                    child_item.setCheckState(1, Qt.Unchecked)
                                    child_item.setData(1, Qt.UserRole, True)
                                item.addChild(child_item)
        except Exception:
            pass


    def toggle_plot(self, item, column):
        if column != 1:
            return
        data = item.data(0, 1)
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
                        # robust empty check (handles weird dsets while writer updates)
                        if getattr(ds_obj, "size", 0) == 0:
                            QMessageBox.warning(
                                self, "Empty dataset",
                                f'The dataset “{hdf5_path}” contains no data and will be ignored.'
                            )
                            item.setCheckState(1, Qt.Unchecked)
                            return
    
                        combined_label = f"{abs_path}##{hdf5_path}"
                        if item.checkState(1) == Qt.Checked:
                            self.plot_data[combined_label] = ds_obj[()]
                            self.raw_visibility[combined_label] = True
                        else:
                            if combined_label in self.plot_data:
                                del self.plot_data[combined_label]
                            self.raw_visibility[combined_label] = False
    
            # refresh once after the with-block (file now closed)
            self._filter_empty_plot_data()
            self.update_plot_raw()
            self.update_plot_processed()
            self.update_pass_button_state()
    
        except Exception:
            pass


    def display_data(self, item, column):
        if self.data_tabs.currentIndex() != 0:
            return
        data = item.data(0, 1)
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
            if isinstance(arr, np.ndarray) and arr.ndim in (0, 1):
                if arr.ndim == 0:
                    arr = arr.item()
                return
            if isinstance(arr, bytes):
                arr = arr.decode('utf-8')
            if isinstance(arr, str) and 'T' in arr and '-' in arr:
                try:
                    dt_obj = datetime.strptime(arr, "%Y-%m-%dT%H:%M:%S.%f")
                    arr = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
            self.scalar_display_raw.setText(str(arr) if not isinstance(arr, float) else f"{arr:.2f}")
        except Exception as e:
            self.scalar_display_raw.setText(f"Error displaying data: {e}")


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
            x_data = self._lookup_energy(abs_path, parent, len(y_data))
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
            x_data = self._lookup_energy(abs_path, parent, len(y_data))
            if getattr(x_data, "size", 0) == 0 or len(y_data) == 0:
                continue
            processed_y = self._apply_normalization(abs_path, parent, y_data)
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
            self._apply_automatic_bg_new(main_x, main_y, deg=int(self.combo_poly.currentText()), pre_edge_percent=float(self.spin_preedge.value())/100.0, ax=self.proc_ax, do_plot=True)
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
        self.proc_ax.set_xlabel("Photon energy (eV)")
        self.proc_ax.figure.tight_layout()
        self.canvas_proc.draw()
        self.update_pass_button_state()
        self.update_proc_tree()

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


    def _compute_background(self, main_x, main_y):
        mode = self.combo_bg.currentText()
        deg = int(self.combo_poly.currentText())
        if mode == "None":
            return np.zeros_like(main_y)
        elif mode == "Manual" and self.manual_points:
            try:
                xs = [pt["x"] for pt in self.manual_points]
                ys = [pt["y"] for pt in self.manual_points]
                background, _ = self._robust_polyfit_on_normalized(xs, ys, deg, main_x)
                background[0] = main_y[0]
            except Exception as ex:
                print("Error in manual background computation:", ex)
                background = np.zeros_like(main_y)
            return background
        elif mode == "Automatic":
            return self._apply_automatic_bg_new(main_x, main_y, do_plot=False)
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
                x_data = self._lookup_energy(abs_path, parent, len(y_data))
                processed_y = self._apply_normalization(abs_path, parent, y_data)
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
                        self._lookup_energy(abs_path, parent, len(y_data)),
                        self._apply_normalization(abs_path, parent, y_data),
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



    def _on_normalize_toggled(self, state):
        self.combo_norm.setEnabled(state == Qt.Checked)
        self.update_plot_processed()

    def _on_bg_subtract_toggled(self, state):
        """Enable/disable post‑normalisation combo and update plot."""
        self.combo_post_norm.setEnabled(state == Qt.Checked)
        self.update_plot_processed()

    def _apply_automatic_bg_new(self, main_x, main_y, deg=None, pre_edge_percent=None, ax=None, do_plot=True):
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
                    try: self._auto_bg_vline.remove()
                    except Exception: pass
                self._auto_bg_vline = None
                if hasattr(self, "_auto_bg_line") and getattr(self._auto_bg_line, "axes", None) is ax:
                    try: self._auto_bg_line.remove()
                    except Exception: pass
                self._auto_bg_line = None
            return background
        idx_end_finite = max(1, int(pre_edge_percent * Nf))
        idx_end_finite = min(idx_end_finite, Nf - 1)
        M = idx_end_finite
        deg_eff = int(max(0, deg))
        deg_eff = min(deg_eff, max(0, min(M, Nf) - 1))
        x_min = float(x_all[0]); x_max = float(x_all[-1])
        span = x_max - x_min
        if span <= 0: span = 1.0
        x_prime = 2.0 * (x_all - x_min) / span - 1.0
        x_pre = x_prime[:M]; y_pre = y_all[:M]
        x_end_prime = x_prime[-1]
        i_start = max(M + 1, int(0.95 * Nf))
        if i_start >= Nf - 1: i_start = max(Nf - 2, 0)
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
        if not np.isfinite(w) or w <= 0: w = 1.0
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
            try: ax.figure.canvas.draw_idle()
            except Exception: pass
        return background
    
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
        background, coeffs = self._robust_polyfit_on_normalized(xs, ys, d, main_x)
        if np.isfinite(main_y[0]):
            background[0] = main_y[0]
        self.manual_poly = coeffs
    
        # Draw anchors always
        for pt in self.manual_points:
            if pt.get("artist") is not None:
                try: pt["artist"].remove()
                except Exception: pass
                pt["artist"] = None
            marker, = self.proc_ax.plot(pt["x"], pt["y"], 'bo', markersize=8, picker=5)
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
            background = self._apply_automatic_bg_new(main_x, main_y, do_plot=False)
        elif mode == "Manual" and self.manual_points:
            try:
                d = int(self.combo_poly.currentText())
                xs = [pt["x"] for pt in self.manual_points]
                ys = [pt["y"] for pt in self.manual_points]
                background, _ = self._robust_polyfit_on_normalized(xs, ys, d, main_x)
                background[0] = main_y[0]
            except Exception:
                background = np.zeros_like(main_y)
        else:
            background = np.zeros_like(main_y)

        sub = main_y - background

        # post-normalisation (robust)
        if self.combo_post_norm.isEnabled():
            mode_norm = self.combo_post_norm.currentText()
            sub = self._safe_post_normalize(main_x, sub, mode_norm)
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
            bg, coeffs = self._robust_polyfit_on_normalized(xs, ys, deg, x_grid)
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

    def rescale_plotted_axes(self):
        x_all, y_all = [], []
        for key, line in self.plotted_lines.items():
            if line.get_visible():
                x = line.get_xdata()
                y = line.get_ydata()
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
            self.canvas_plotted_fig.tight_layout()
            self.canvas_plotted.draw()
    
    def export_ascii_plotted(self):
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
                x_data = self._lookup_energy(abs_path, parent, len(y_data))
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
                        self._lookup_energy(abs_path, parent, len(y_data)),
                        y_data
                    )
        return None, None

# --- GUI launcher used by app.py ---
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


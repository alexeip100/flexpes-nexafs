from . import processing
from . import data
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # allow concurrent readers on Windows
import h5py
import sys
import matplotlib.pyplot as plt
plt.ioff() 

from PyQt5.QtWidgets import (

    QApplication, QMainWindow, QTreeWidget, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout, QTabWidget,
    QCheckBox, QComboBox, QSpinBox, QMessageBox, QSizePolicy, QSplitter,
    QListWidget, QMenu, QSlider, QDoubleSpinBox
, QAbstractItemView)

# ---- Compatibility shims (Phase 2b) ----
# ----------------------------------------
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

###############################################################################
# Helper function to parse integer from something like "entry1001"
# for sorting the Raw Data and Processed Data items in ascending order
###############################################################################


from .data import DataMixin
from .processing import ProcessingMixin
from .plotting import PlottingMixin
from .export import ExportMixin
from .library import LibraryMixin
from .channel_setup import ChannelConfigManager, ChannelSetupDialog
import re
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

    def on_setup_channels_clicked(self):
        """Open the channel mapping dialog (beamline profiles)."""
        try:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Setup channels")
            msg.setText(
                "The default channel selection corresponds to the setup used at the MAX IV "
                "beamline FlexPES (branch A).\n\n"
                "Here you can assign channel names for other beamline/branchline setups and "
                "save them for future use."
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            ret = msg.exec_()
            if ret != QMessageBox.Ok:
                return
        except Exception:
            # If the message box fails for any reason, still try opening the dialog.
            pass

        dlg = ChannelSetupDialog(self.channel_config, parent=self)
        if dlg.exec_() == dlg.Accepted:
            try:
                # Persist updated mapping and refresh UI defaults.
                self.channel_config.save_to_user()
            except Exception:
                pass
            try:
                self._apply_channel_config_to_ui()
            except Exception:
                pass

    def _apply_channel_config_to_ui(self):
        """Refresh widgets that depend on channel mapping (I0 default, 'All TEY' filters, etc.)."""
        try:
            if hasattr(self, 'channel_profile_label'):
                self.channel_profile_label.setText(f"Active beamline: <b>{self.channel_config.active_profile()}</b>")
        except Exception:
            pass

        # Refresh normalization channels (I0 default selection)
        try:
            files = list(getattr(self, "hdf5_files", {}).keys())
            if files:
                self.populate_norm_channels(files[0])
        except Exception:
            pass

        # Re-apply group checkboxes if they are checked (patterns may have changed)
        try:
            for role, cb in [("TEY", getattr(self, "cb_all_tey", None)),
                             ("PEY", getattr(self, "cb_all_pey", None)),
                             ("TFY", getattr(self, "cb_all_tfy", None)),
                             ("PFY", getattr(self, "cb_all_pfy", None))]:
                if cb is not None and cb.isChecked():
                    self._on_all_role_checkbox(role, Qt.Checked)
        except Exception:
            pass

        # Replot to reflect any changes
        try:
            self.update_plot_raw()
            self.update_plot_processed()
        except Exception:
            pass

    def _on_all_role_checkbox(self, role: str, state):
        """Toggle selection of all datasets matching the configured pattern for a role."""
        try:
            checked = (state == Qt.Checked)
        except Exception:
            checked = bool(state)

        pattern = ""
        try:
            pattern = str(self.channel_config.get_pattern(role) or "")
        except Exception:
            pattern = ""
        if not pattern:
            return
        try:
            self.set_group_visibility(pattern, checked)
        except Exception:
            pass

    
    def _apply_basic_tooltips(self):
        """Assign helpful tooltips to buttons/checkboxes/comboboxes that lack one.

        - Only touches QPushButton, QCheckBox, QComboBox (per your request).
        - Does NOT touch trees/lists and does not add any new signal connections.
        - Uses curated, descriptive tooltips for the main controls, then falls back to a light
          generic tooltip only when the control is self-explanatory.
        """
        try:
            from PyQt6.QtWidgets import QPushButton, QCheckBox, QComboBox  # type: ignore
        except Exception:
            from PyQt5.QtWidgets import QPushButton, QCheckBox, QComboBox  # type: ignore

        # Curated tooltips keyed by attribute name on self (stable and unambiguous).
        by_attr = {
            # Main controls
            "open_button": "Open one or more HDF5 (.h5) files with XAS/NEXAFS spectra",
            "close_button": "Close the current HDF5 file and clear loaded data",
            "clear_button": "Clear loaded data and selections in all panels",
            "help_button": "Open the Usage help page",
            "setup_channels_button": "Configure channel mappings (TEY/PEY/TFY/PFY etc.) for the open files",
            "pass_button": "Pass selected processed curves to the Plotted Data panel",
            "export_ascii_button": "Export selected curves to an ASCII text file",
            "load_reference_button": "Load a reference spectrum for comparison",
            "export_import_plotted_button": "Export or import plotted curves as CSV",
            "pca_plotted_button": "Send eligible plotted curves to the decomposition app (PCA/NMF/MCR). Requires Waterfall OFF and post-normalization = Area",
            "clear_plotted_data_button": "Remove all curves from the Plotted Data panel",

            # Channel selection helpers
            "cb_all_tey": "Select all TEY channels in the HDF5 files",
            "cb_all_pey": "Select all PEY channels in the HDF5 files",
            "cb_all_tfy": "Select all TFY channels in the HDF5 files",
            "cb_all_pfy": "Select all PFY channels in the HDF5 files",
            "cb_all_in_channel": "Apply the current channel selection to all entries within the chosen channel",

            # Processing toggles
            "chk_normalize": "Enable curve normalization by an incident photon flux curve",
            "chk_sum": "Sum selected curves in Processed Data",
            "chk_group_bg": "Use a shared Automatic background model for selected spectra (Group BG)",
            "chk_group_bg_slope": "Match pre-edge slope across the group in Group BG mode",
            "chk_show_without_bg": "Show the spectra without background",
            "chk_show_annotation": "Show curve annotations/labels on the plot",
            "waterfall_checkbox": "Apply vertical offsets to curves to create a waterfall plot (must be OFF for PCA transfer)",

            # Comboboxes (choices)
            "combo_all_channel": "Choose detector/channel group to operate on (TEY/PEY/TFY/PFY)",
            "combo_norm": "Select the incident photon flux curve I₀ used for normalization",
            "combo_bg": "Choose background subtraction method",
            "combo_poly": "Choose polynomial order for polynomial background subtraction",
            "combo_post_norm": "Choose post-normalization method after BG subtraction",
            "legend_mode_combo": "Choose legend labels for plotted curves: None, User-defined, or Entry number",
            "grid_mode_combo": "Choose plot grid style",
        }

        # Apply curated tooltips first (do not clobber a tooltip that is already descriptive).
        generic_placeholders = {"Button", "Toggle option", "Choose an option"}
        for attr, tip in by_attr.items():
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                current = (w.toolTip() or "").strip()
                if current and current not in generic_placeholders and current != (getattr(w, "text", lambda: "")() or "").strip():
                    continue
                w.setToolTip(tip.rstrip('.'))
            except Exception:
                pass

        # Fallback for any remaining buttons/checkboxes/comboboxes without a tooltip.
        for w in self.findChildren((QPushButton, QCheckBox, QComboBox)):
            try:
                if (w.toolTip() or "").strip():
                    continue

                if isinstance(w, QPushButton):
                    t = (w.text() or "").strip()
                    # Slightly descriptive for most buttons; generic only for truly obvious cases.
                    if t.lower() in {"ok", "cancel", "close"}:
                        w.setToolTip(t)
                    elif t:
                        w.setToolTip(f"Action: {t}")
                    else:
                        w.setToolTip("Action button")

                elif isinstance(w, QCheckBox):
                    t = (w.text() or "").strip()
                    if t:
                        w.setToolTip(f"Toggle: {t}")
                    else:
                        w.setToolTip("Toggle option")

                else:  # QComboBox
                    # If we cannot infer the meaning, keep it short.
                    w.setToolTip("Select an option")
            except Exception:
                pass
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlexPES NEXAFS Plotter")
        # Default window position/size (tuned for smaller laptop screens)
        self.setGeometry(50, 50, 1650, 800)

        # Place the main window near the top-left of the available screen area
        # (helps on small laptop screens).
        try:
            from PyQt6 import QtGui  # type: ignore
        except Exception:
            try:
                from PyQt5 import QtGui  # type: ignore
            except Exception:
                QtGui = None  # type: ignore
        try:
            if QtGui is not None:
                scr = QtGui.QGuiApplication.primaryScreen()
                if scr is not None:
                    g = scr.availableGeometry()
                    self.move(g.left() + 50, g.top() + 50)
        except Exception:
            pass


        # Keep version/date in sync with the package metadata.
        try:
            from . import __version__ as _PKG_VERSION
            from . import __date__ as _PKG_DATE
        except Exception:
            _PKG_VERSION = "2.3.1"
            _PKG_DATE = "2026-01-07"
        self.VERSION_NUMBER = str(_PKG_VERSION)
        self.CREATION_DATETIME = str(_PKG_DATE)

        self.hdf5_files = {}
        self.plot_data = {}      # Keys: "abs_path##hdf5_path"
        self.energy_cache = {}

        # Channel mapping (beamline profiles)
        # Default: FlexPES-A (can be edited via the "Setup channels" dialog).
        self.channel_config = ChannelConfigManager()

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
        self.open_button = QPushButton("Open HDF5")
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

        # Channel mapping setup
        self.setup_channels_row = QHBoxLayout()
        self.setup_channels_button = QPushButton("Setup channels")
        # Keep the button as narrow as possible (do not stretch to full width).
        self.setup_channels_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        try:
            self.setup_channels_button.setFixedWidth(self.setup_channels_button.sizeHint().width())
        except Exception:
            pass
        self.setup_channels_button.clicked.connect(self.on_setup_channels_clicked)
        self.setup_channels_row.addWidget(self.setup_channels_button)

        active_profile = self.channel_config.active_profile()
        self.channel_profile_label = QLabel(f"Active beamline: <b>{active_profile}</b>")
        try:
            self.channel_profile_label.setTextFormat(Qt.RichText)
        except Exception:
            pass
        self.channel_profile_label.setWordWrap(False)
        self.channel_profile_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Keep label styling minimal so it follows the application's global font size.
        self.channel_profile_label.setStyleSheet("color: #555;")
        self.setup_channels_row.addWidget(self.channel_profile_label)

        self.left_panel.addLayout(self.setup_channels_row)

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
        self.cb_all_tey.stateChanged.connect(lambda state, r="TEY": self._on_all_role_checkbox(r, state))
        self.raw_group_layout.addWidget(self.cb_all_tey)

        self.cb_all_pey = QCheckBox("All PEY data")
        self.cb_all_pey.stateChanged.connect(lambda state, r="PEY": self._on_all_role_checkbox(r, state))
        self.raw_group_layout.addWidget(self.cb_all_pey)

        self.cb_all_tfy = QCheckBox("All TFY data")
        self.cb_all_tfy.stateChanged.connect(lambda state, r="TFY": self._on_all_role_checkbox(r, state))
        self.raw_group_layout.addWidget(self.cb_all_tfy)

        self.cb_all_pfy = QCheckBox("All PFY data")
        self.cb_all_pfy.stateChanged.connect(lambda state, r="PFY": self._on_all_role_checkbox(r, state))
        self.raw_group_layout.addWidget(self.cb_all_pfy)

        # --- Example-inspired: 'All in channel' checkbox + combo ---
        self.cb_all_in_channel = QCheckBox("All in channel:")
        self.combo_all_channel = QComboBox()
        try: self.combo_all_channel.setMinimumWidth(220)
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
        try: self.raw_tree.setMinimumWidth(300)
        except Exception: pass
        self.raw_tree.setHeaderHidden(True)
        (self.raw_tree.itemChanged.connect( self.raw_tree_item_changed )) if hasattr(self, 'raw_tree_item_changed') else None
        self.raw_splitter.addWidget(self.raw_tree)
        self.raw_splitter.setStretchFactor(0, 32)
        self.raw_splitter.setStretchFactor(1, 68)
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
        self.proc_controls_top_layout.addWidget(QLabel("Choose I₀:"))
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
            "When Automatic BG and multiple spectra are selected, fit a shared background. "
            "With Area normalization, it aligns pre-edge baseline and jump across the group"
        )
        self.proc_controls_top_layout.addWidget(self.chk_group_bg)

        # Optional: match pre-edge slope across the group (only meaningful in group Auto BG)
        self.chk_group_bg_slope = QCheckBox("Match pre-edge")
        self.chk_group_bg_slope.setChecked(False)
        self.chk_group_bg_slope.setEnabled(False)
        self.chk_group_bg_slope.setToolTip(
            "When Group BG is active (Automatic BG + multiple selected spectra), "
            "adjusts the backgrounds so that the pre-edge slope after BG subtraction "
            "is consistent across the selected group"
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
        self.combo_bg.addItems(["None", "Auto", "Manual"])
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
        self.spin_preedge.setToolTip("Set pre-edge window length (percent of energy span) used for baseline in Automatic/Group BG")
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
        try: self.proc_tree.setMinimumWidth(320)
        except Exception: pass
        self.proc_tree.setHeaderHidden(True)
        (self.proc_tree.itemChanged.connect( self.proc_tree_item_changed )) if hasattr(self, 'proc_tree_item_changed') else None

        self.proc_splitter = QSplitter(Qt.Horizontal)
        self.proc_splitter.addWidget(self.proc_left_widget)
        self.proc_splitter.addWidget(self.proc_tree)
        self.proc_splitter.setStretchFactor(0, 28)
        self.proc_splitter.setStretchFactor(1, 72)

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
            "'Entry number' labels curves by entryXXXX -> XXXX. 'None' hides the legend"
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
        # Waterfall (Uniform step only)
        self.waterfall_checkbox = QCheckBox("Waterfall")
        self.waterfall_checkbox.setChecked(False)
        self.plot_buttons_layout.addWidget(self.waterfall_checkbox)
        self.waterfall_checkbox.stateChanged.connect(self.on_waterfall_toggled)
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

        # PCA / decomposition handoff (opens legacy decomposition window)
        self.pca_plotted_button = QPushButton("PCA")
        self.pca_plotted_button.setToolTip("Send selected plotted curves to the PCA/NMF/MCR decomposition window")
        self.pca_plotted_button.clicked.connect(self.send_plotted_to_pca)
        self.pca_plotted_button.setMaximumWidth(90)
        self.plot_buttons_layout.addWidget(self.pca_plotted_button)

        self.clear_plotted_data_button = QPushButton("Clear Plotted")
        self.clear_plotted_data_button.clicked.connect(self.clear_plotted_data)
        self.clear_plotted_data_button.setMaximumWidth(150)
        self.plot_buttons_layout.addWidget(self.clear_plotted_data_button)
        # ----------------------------------------------------------------------

        self.plot_left_layout.addLayout(self.plot_buttons_layout)

        self.plotted_splitter = QSplitter(Qt.Horizontal)
        self.plotted_splitter.addWidget(self.plot_left_widget)
        self.plotted_list = QListWidget()
        try: self.plotted_list.setMinimumWidth(300)
        except Exception: pass
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
        self.plotted_splitter.setStretchFactor(0, 55)
        self.plotted_splitter.setStretchFactor(1, 45)
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

        # Tooltips: populate for buttons/checkboxes/comboboxes that lack one.
        self._apply_basic_tooltips()

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
            # Main splitter: make left tree panel ~20% narrower (~33% -> ~22%)
            total = int(self.splitter.width())
            if total > 0:
                left = max(200, int(total * 0.25))
                self.splitter.setSizes([left, max(200, total - left)])
        except Exception:
            pass

        # Tab splitters: widen the right-hand curve-tree widget vs previous defaults
        specs = [
            ("raw_splitter", 0.62),     # wider curve tree on Raw tab
            ("proc_splitter", 0.66),    # wider curve tree on Processed tab
            ("plotted_splitter", 0.42)  # wider plotted list
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


    def send_plotted_to_pca(self):
        """Send visible (checked) plotted curves to the decomposition window.

        This intentionally mirrors the Export CSV (Plotted Data) selection/naming logic,
        and adds strict scientific gating:
          - Waterfall must be OFF
          - post_normalization must be 'Area' for all selected curves
          - Legend must not be 'None' and names must be resolvable
        """
        try:
            from PyQt5.QtWidgets import QMessageBox
            import numpy as np

            # 1) Waterfall must be off
            try:
                if hasattr(self, "waterfall_checkbox") and self.waterfall_checkbox.isChecked():
                    QMessageBox.warning(self, "PCA Transfer Blocked",
                                        "Cannot send curves to PCA when Waterfall is enabled.\n"
                                        "Please uncheck 'Waterfall' (no offsets) and try again.")
                    return
            except Exception:
                pass

            # 2) Determine visible curves in current plotted-list order (same as Export CSV)
            visible_keys = []
            try:
                if hasattr(self, "plotted_list") and self.plotted_list is not None:
                    for row in range(self.plotted_list.count()):
                        item = self.plotted_list.item(row)
                        if item is None:
                            continue
                        key = item.data(Qt.UserRole)
                        if not key:
                            continue
                        ln = self.plotted_lines.get(key)
                        if ln is not None and ln.get_visible():
                            visible_keys.append(key)
            except Exception:
                pass

            if not visible_keys:
                QMessageBox.warning(self, "PCA Transfer Blocked",
                                    "No curves are selected/visible in Plotted Data.\n"
                                    "Please check (show) one or more curves and try again.")
                return

            # 3) Legend mode + naming rules (mirror Export CSV)
            legend_mode = None
            try:
                legend_mode = self.legend_mode_combo.currentText().strip()
            except Exception:
                legend_mode = "User-defined"

            if legend_mode == "None":
                QMessageBox.warning(self, "PCA Transfer Blocked",
                                    "Cannot send curves to PCA when Legend is set to 'None'.\n"
                                    "Please set Legend to 'User-defined' or 'Entry number'.")
                return
            # Helper: get entry number digits from metadata (mirrors Export CSV)
            def _entry_digits_from_meta(key: str):
                import re
                try:
                    meta = getattr(self, "plotted_metadata", {}) or {}
                    src_entry = (meta.get(key, {}) or {}).get("source_entry", "") or ""
                    m = re.search(r"entry(\d+)", str(src_entry))
                    if m:
                        return m.group(1)
                except Exception:
                    pass
                try:
                    m = re.search(r"entry(\d+)", str(key))
                    if m:
                        return m.group(1)
                except Exception:
                    pass
                return None

            headers = []
            y_columns = []

            # Use X from first visible line; min_len truncation mirrors Export CSV
            first_line = self.plotted_lines.get(visible_keys[0])
            if first_line is None:
                QMessageBox.warning(self, "PCA Transfer Blocked", "Internal error: first plotted curve line not found.")
                return
            x_data = np.asarray(first_line.get_xdata()).ravel()
            min_len = len(x_data)

            # 4) Scientific gating: post_normalization must be Area for all curves
            bad_area = []
            for key in visible_keys:
                md = getattr(self, "plotted_metadata", {}).get(key, {}) if hasattr(self, "plotted_metadata") else {}
                post_norm = str(md.get("post_normalization", "")).strip()
                if post_norm != "Area":
                    bad_area.append(key)

            if bad_area:
                QMessageBox.warning(self, "PCA Transfer Blocked",
                                    "Cannot send curves to PCA: all selected curves must be BG-subtracted and Area-normalized.\n"
                                    "Please ensure post-normalization is set to 'Area' before plotting/passing curves.")
                return

            # Build columns and headers (min_len truncation, no X-grid equality check — same as Export CSV)
            if legend_mode == "Entry number":
                for key in visible_keys:
                    ln = self.plotted_lines.get(key)
                    if ln is None:
                        continue
                    y = np.asarray(ln.get_ydata()).ravel()
                    min_len = min(min_len, len(y))
                    entry_num = _entry_digits_from_meta(key)
                    if not entry_num:
                        QMessageBox.warning(self, "PCA Transfer Blocked",
                                            "Cannot label curves by entry number: one or more selected curves has no entry id.")
                        return
                    headers.append(entry_num)
                    y_columns.append(y)
            else:  # User-defined
                for key in visible_keys:
                    ln = self.plotted_lines.get(key)
                    if ln is None:
                        continue
                    y = np.asarray(ln.get_ydata()).ravel()
                    min_len = min(min_len, len(y))
                    lbl = None
                    try:
                        lbl = self.custom_labels.get(key)
                    except Exception:
                        lbl = None
                    if not lbl or str(lbl).strip() == "" or str(lbl).strip() == "<select curve name>":
                        QMessageBox.warning(self, "PCA Transfer Blocked",
                                            "Cannot send curves to PCA: some selected curves have no user-defined name.\n"
                                            "Please name all selected curves (or switch Legend to 'Entry number').")
                        return
                    headers.append(str(lbl).strip())
                    y_columns.append(y)

            x_out = x_data[:min_len]
            y_out = [np.asarray(y)[:min_len] for y in y_columns]

            # 5) Open (or reuse) decomposition window and inject dataset
            try:
                from flexpes_nexafs.decomposition.legacy import MainWindow as DecompWindow
            except Exception as ex:
                QMessageBox.critical(self, "PCA Transfer Error",
                                     f"Failed to import decomposition module:\n{ex}")
                return

            try:
                if getattr(self, "_decomp_window", None) is None:
                    self._decomp_window = DecompWindow()
                self._decomp_window.set_dataset(x_out, y_out, headers)
                self._decomp_window.show()
                try:
                    self._decomp_window.raise_()
                    self._decomp_window.activateWindow()
                except Exception:
                    pass
            except Exception as ex:
                QMessageBox.critical(self, "PCA Transfer Error",
                                     f"Failed to send data to decomposition window:\n{ex}")
                return


        except Exception as _ex:
            try:
                from PyQt5.QtWidgets import QMessageBox
            except Exception:
                QMessageBox = None
            msg = f"Unexpected error in PCA transfer:\n{_ex}"
            if QMessageBox is not None:
                QMessageBox.critical(self, "PCA Transfer Error", msg)
            else:
                print(msg)

MainWindow = HDF5Viewer  # backward-compatible alias

def launch():
    """Launch the GUI (backwards-compatible helper)."""
    import sys
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------
# CurveListItemWidget — individual curve control widget for plot list
# --------------------------------------------------------------------

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QCheckBox, QComboBox, QSpinBox,
    QLabel, QHBoxLayout, QSizePolicy, QColorDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor


class CurveListItemWidget(QWidget):
    """Small row widget for controlling visibility, color, and style of one plotted curve."""

    colorChanged = pyqtSignal(str, str)        # key, new_color
    visibilityChanged = pyqtSignal(str, bool)  # key, visible
    styleChanged = pyqtSignal(str, str, float) # key, lineStyle, lineSize
    removeRequested = pyqtSignal(str)          # key

    def __init__(self, label, initial_color, key, parent=None):
        super().__init__(parent)
        self.key = key

        # --- Create controls ---
        self.color_button = QPushButton()
        self.color_button.setFixedSize(20, 20)
        self.color_button.setStyleSheet(f"background-color: {initial_color}")
        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedSize(20, 20)
        self.remove_button.setToolTip("Remove this curve from the plot")

        self.check_box = QCheckBox()
        self.check_box.setChecked(True)

        self.style_combo = QComboBox()
        self.style_combo.addItems(["Solid", "Dashed", "Scatter"])

        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 20)
        self.size_spin.setValue(2)

        self.label = QLabel(label)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.label.setToolTip(label)

        # --- Layout ---
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.remove_button)
        layout.addWidget(self.color_button)
        layout.addWidget(self.check_box)
        layout.addWidget(self.style_combo)
        layout.addWidget(self.size_spin)
        layout.addWidget(self.label)

        # --- Connections ---
        self.remove_button.clicked.connect(self.on_remove_clicked)
        self.color_button.clicked.connect(self.on_color_button_clicked)
        self.check_box.stateChanged.connect(self.on_visibility_changed)
        self.style_combo.currentTextChanged.connect(self.on_style_changed)
        self.size_spin.valueChanged.connect(self.on_style_changed)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def on_color_button_clicked(self):
        """Open color dialog and emit new color."""
        color = QColorDialog.getColor()
        if color.isValid():
            new_color = color.name()
            self.color_button.setStyleSheet(f"background-color: {new_color}")
            self.colorChanged.emit(self.key, new_color)

    def on_remove_clicked(self):
        """Request removal of this curve via signal."""
        self.removeRequested.emit(self.key)

    def on_visibility_changed(self, state):
        """Emit signal when visibility toggled."""
        self.visibilityChanged.emit(self.key, state == Qt.Checked)

    def on_style_changed(self):
        """Emit signal when style or size changed."""
        self.styleChanged.emit(
            self.key,
            self.style_combo.currentText(),
            float(self.size_spin.value())
        )
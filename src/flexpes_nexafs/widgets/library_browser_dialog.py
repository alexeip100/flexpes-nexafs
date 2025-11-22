from PyQt5.QtWidgets import (
    QDialog,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QVBoxLayout,
    QLabel,
)
from PyQt5.QtCore import Qt


class LibraryBrowserDialog(QDialog):
    """
    Dialog that lists reference spectra stored in the library and lets the user
    choose one or more to load into the plot.
    """

    def __init__(self, entries, parent=None):
        """
        entries: list of dicts with at least:
            - "label": display label
            - "group_name": HDF5 group name under /spectra
            - "meta": dict of attributes (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Load reference spectra")
        self.setModal(True)
        self._entries = list(entries or [])

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)

        if not self._entries:
            info = QListWidgetItem("No reference spectra found in the library.")
            info.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(info)
        else:
            for entry in self._entries:
                label = entry.get("label") or entry.get("group_name") or "<unnamed>"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, entry)
                tooltip_parts = []
                meta = entry.get("meta") or {}
                if meta.get("element") and meta.get("edge") and meta.get("compound"):
                    tooltip_parts.append(f"{meta.get('element')} {meta.get('edge')} in {meta.get('compound')}")
                if meta.get("detector"):
                    tooltip_parts.append(f"Detector: {meta.get('detector')}")
                if meta.get("resolution_meV") not in (None, "", 0):
                    tooltip_parts.append(f"Resolution: {meta.get('resolution_meV')} meV")
                if meta.get("comment"):
                    tooltip_parts.append(str(meta.get("comment")))
                if tooltip_parts:
                    item.setToolTip("\n".join(str(x) for x in tooltip_parts))
                self.list_widget.addItem(item)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            orientation=Qt.Horizontal,
            parent=self,
        )
        self.button_box.button(QDialogButtonBox.Ok).setText("Load selected")
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select one or more reference spectra to load:"))
        layout.addWidget(self.list_widget)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.resize(520, 400)

    def _on_accept(self):
        # If there are no selectable entries, do nothing
        if not self._entries:
            self.reject()
            return
        selected = self.list_widget.selectedItems()
        if not selected:
            # Allow OK with nothing selected, but treat as cancel.
            self.reject()
            return
        self.accept()

    def get_selected_entries(self):
        """Return list of entry dicts for selected items."""
        result = []
        for item in self.list_widget.selectedItems():
            entry = item.data(Qt.UserRole)
            if entry:
                result.append(entry)
        return result

from pathlib import Path


def test_version_metadata_is_241():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    init = Path("src/flexpes_nexafs/__init__.py").read_text(encoding="utf-8")
    assert 'version = "2.4.1"' in pyproject
    assert '__version__ = "2.4.1"' in init
    assert '__date__ = "2026-06-30"' in init


def test_hdf5_locking_is_set_before_h5py_import():
    data = Path("src/flexpes_nexafs/data.py").read_text(encoding="utf-8")
    assert data.index('os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")') < data.index("import h5py")


def test_group_bg_pass_to_plotted_no_stale_name_defined_ignore():
    raw_mixin = Path("src/flexpes_nexafs/plotting/mixin_raw_plot.py").read_text(encoding="utf-8")
    assert "type: ignore[name-defined]" not in raw_mixin
    assert "np.asarray(x_arr[:mlen]" not in raw_mixin


def test_hdf5_drag_drop_and_shared_loader_are_wired():
    ui = Path("src/flexpes_nexafs/ui.py").read_text(encoding="utf-8")
    data = Path("src/flexpes_nexafs/data.py").read_text(encoding="utf-8")
    assert "self.tree.setAcceptDrops(True)" in ui
    assert "installEventFilter(self)" in ui
    assert "def eventFilter" in ui
    assert "self.load_hdf5_paths(paths, source=\"drop\")" in ui
    assert "def load_hdf5_paths" in data
    assert "def refresh_hdf5_file" in data
    assert "Refresh already loaded HDF5 file(s)" in data

def test_normalization_refresh_guards_group_and_nested_payloads():
    helper = Path("src/flexpes_nexafs/hdf5_loading.py").read_text(encoding="utf-8")
    data = Path("src/flexpes_nexafs/data.py").read_text(encoding="utf-8")
    processed = Path("src/flexpes_nexafs/plotting/mixin_processed_plot.py").read_text(encoding="utf-8")
    assert "def normalize_hdf5_path" in helper
    assert "_looks_like_file_path" in helper
    assert "h5load.split_tree_payload" in data
    assert "h5load.normalize_hdf5_path" in processed
    assert "obj = f.get(norm_path, None)" in processed
    assert "isinstance(obj, _h5py.Dataset)" in processed

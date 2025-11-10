
"""State dataclasses (preparation for Session save/load).
These are *not* wired into the UI yet, so there is no behavior change.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class FileRef:
    id: str                   # stable id like 'f1'
    path: str                 # absolute or relative disk path
    size: Optional[int] = None
    mtime: Optional[float] = None
    sha256: Optional[str] = None

@dataclass
class PlottedCurve:
    label: str
    file_id: Optional[str]
    dataset_path: Optional[str]
    visible: bool = True
    color: Optional[str] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None
    marker: Optional[str] = None
    x: Optional[list] = None     # stored for exact-visual-fidelity restores (optional)
    y: Optional[list] = None

@dataclass
class Session:
    version: str = "1.0"
    app_version: Optional[str] = None
    files: List[FileRef] = field(default_factory=list)

    # UI selections/state (subset)
    selected_datasets: Dict[str, bool] = field(default_factory=dict)  # key = "abs##h5path"
    normalize_i0: bool = False
    i0_channel: Optional[str] = None
    background_mode: str = "None"          # "None" | "Automatic" | "Manual"
    background_degree: int = 2
    preedge_fraction: float = 0.20         # 0..1
    manual_anchors: List[Tuple[float, float]] = field(default_factory=list)

    post_norm_mode: str = "None"           # "None" | "Max" | "Jump" | "Area"
    sum_selected: bool = False

    # Plotted tab
    plotted: List[PlottedCurve] = field(default_factory=list)

    # Transient caches (not serialized typically)
    energy_cache: Dict[str, tuple] = field(default_factory=dict)

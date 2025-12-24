# FlexPES NEXAFS Plotter


## Overview
This application opens HDF5 files with NEXAFS spectra from the FlexPES
beamline (MAX IV Laboratory). It supports pre-processing, visualization,
and export of raw and processed data.

## File Controls (Top Left Panel)
- **Open HDF5 File:** Load one or more HDF5 files containing NEXAFS data.
- **Close all:** Close all currently opened files (but keep the application running). To close individual h5 files: select a file and either "Delete" or right-click and "Close".
- **Clear all:** Remove all loaded data and reset the interface.
- **Help:** Open the **Usage** and **About** dialogs.

## File Tree Panel
Shows the hierarchical structure of loaded HDF5 files.
Expand groups to view datasets. Tick checkboxes on **1-D datasets**
(typically under `entryXXXX/measurement/...`) to include or exclude curves
from the plots.

---

## Tabs (Right Panel):

## Raw Data Tab
**Purpose:** choose which datasets (curves) are visible.

- **Data Plot:** Shows raw spectra. Multiple curves can be shown at once.

- **Channel Families:**  
  Quick toggles by detector type:
  - **All TEY data**
  - **All PEY data**
  - **All TFY data**
  - **All PFY data**

  Each checkbox shows or hides *all* 1-D datasets whose HDF5 path contains
  the corresponding channel substring (e.g. `_ch1`, `_ch3`, etc.), across
  all opened files. You can combine these with the tree item checkboxes
  and with **All in channel** (see below).

- **All in channel:** *(checkbox + combobox of channel names)*  
  The combobox lists unique channel names found across all opened files
  (based on the **last path component** of each 1-D dataset).  
  - **Checked:**  
    Load the selected channel across **all entries**. If another channel
    was previously active, it is cleared first. Changing the selection
    while checked switches to the new channel and removes the previous one.
  - **Unchecked:**  
    Clear that channel group from the plot (and from the Raw Data tree’s
    visibility map). Changing the combobox while unchecked does not load
    anything.

- **Raw Data Tree (right of plot):**  
  A dedicated tree for raw data inspection. Datasets are grouped into
  energy-dependent “regions” (e.g. absorption edges). Checkboxes on group
  nodes allow quick selection or deselection of an entire edge, while
  individual items allow fine-grained control of which curves are visible.

- **Scalar Display (Raw):**  
  A small text area beneath the raw plot canvas that shows scalar or
  textual metadata for the currently selected item(s) (for example,
  experimental parameters or scan comments).

---

## Processed Data Tab
**Purpose:** apply non-destructive processing to the visible raw curves.

### Top Controls (normalization, summing, export)

- **Normalize by I₀?** *(checkbox + “Set I₀ channel” combobox)*  
  When checked, each raw spectrum is divided by the chosen I₀ / monitor
  channel from the same entry. This corrects for intensity fluctuations
  between scans before any background subtraction or post-normalization.

- **Set I₀ channel:**  
  Choose which dataset within the entry should be used as I₀. Only used
  when **Normalize by I₀?** is checked.

- **Sum them up?**  
  When checked, multiple visible curves are summed into one “Summed Curve”.
  This is useful for combining repeated sweeps to improve statistics.
  The summed result then behaves like any other main curve for background
  subtraction and post-normalization.


- **Group BG:** *(checkbox)*
  Enabled when **two or more** spectra are selected **and** **Choose BG** is set to **Automatic**.
  When checked, Automatic BG is applied in a *group-consistent* way so you can background-subtract and **Pass** the
  full selection to **Plotted Data** in one step.

  **Background model used in group mode (Automatic BG):**
  - Each spectrum starts from its *own* **Automatic BG** (a low-degree polynomial fitted to the pre-edge, with the same
    end-slope constraint as in single-spectrum mode).
  - If post-normalization is **Area**, the background is then adjusted per spectrum by adding a small **affine term**
    *(constant + global linear term)* so that (i) the **pre-edge baseline after subtraction is 0**, and (ii) the
    **absorption jump after Area normalization** is consistent across the selected spectra.

- **Match pre-edge slope:** *(checkbox)*
  Available only when **Group BG** is active. When enabled, the group mode additionally adjusts the
  backgrounds so that the **pre-edge slope after BG subtraction** is consistent across the selected spectra
  (median target).

  **Important:** With **Match pre-edge slope** enabled, the final background is **no longer a single pure polynomial**.
  Internally, the polynomial background is supplemented by a *localized pre-edge correction term* that is active in the
  pre-edge region and smoothly tapers to zero before the edge (so it does not distort the post-edge window). This extra
  degree of freedom makes it possible to align pre-edge slopes across a group while keeping the Area-normalized jump
  consistent.

- **Pass:**  
  Send the current processed main curve (possibly normalized, summed and
  background-subtracted) to the **Plotted Data** tab.  
  The curve is added to the Plotted list without clearing existing curves.

- **Export:**  
  Export the current **Processed Data** main curve to a CSV file.
  (This button is separate from the **Export/Import** menu in the *Plotted Data* tab.)

### Bottom Controls (BG + post-normalization)

- **Choose BG:** *(combobox: None / Automatic / Manual)*  
  - **None:** no background is fitted or subtracted.
  - **Automatic:** polynomial background fit with an additional constraint
    so that the derivative at the end of the background matches the slope
    of the data (intended to mimic a smooth pre-edge baseline).
  - **Manual:** manual background mode. The application starts from an
    automatic estimate, and you can adjust anchor points with the mouse
    if the automatic fit fails or needs refinement.

- **Poly degree:**  
  Polynomial degree for the background fit (typically 0–3).


- **Pre-edge (%):**  
  Fraction of the spectrum (in %) treated as “pre-edge” region for
  background estimation.

- **Subtract BG?**  
  When checked, the plotted curve on the Processed Data tab is shown as
  **(signal − background)**. This also enables *post-normalization*.

- **Normalize (post-processing):** *(None / Max / Jump / Area)*  
  This is a **post-normalization** applied after background subtraction
  (if enabled). Modes:
  - **None:** no further scaling.
  - **Max:** scale to the maximum value of the (subtracted) curve.
  - **Jump:** scale to the absorption jump at the last point of the curve.
  - **Area:** scale to the integrated area under the curve.

> **Note:** Background subtraction and post-normalization are only
> available when there is a single effective main curve (one dataset visible, or multiple datasets combined via **Sum them up?**).
> 
> If several raw curves are visible and summing is disabled, BG/post-normalization are normally disabled — **except** when
> BG mode is **Automatic** and **Group BG** is enabled.
> 
> In that group mode, when **Subtract BG?** is **OFF**, BG curves are shown for all selected curves.
> When **Subtract BG?** is **ON**, the view switches to the BG-subtracted curves, and the **Pass** button can pass
> **all selected curves** to the Plotted Data tab at once.

---

## Plotted Data Tab
**Purpose:** control plot appearance, curve styles, reference spectra, and export.

### Plot and axes

- **Interactive Plot Canvas:**  
  Embedded Matplotlib canvas with zoom, pan, and save tools (via the
  toolbar). Energy (x-axis) is auto-detected from typical energy arrays
  such as `pcap_energy_av` or `mono_traj_energy`, with a fallback to a
  simple index axis if none are found. Curve labels initially include
  information like *entry + channel* (or “Summed Curve”).

### Waterfall controls

- **Waterfall mode:** *(combobox: None / Adaptive step / Uniform step)*  
  - **None:** all curves are plotted on top of each other (no vertical shift).
  - **Adaptive step:** curves are offset vertically based on their
    amplitudes (adaptive spacing).
  - **Uniform step:** curves are offset by a constant step size.

- **Waterfall slider / numeric box:**  
  When a Waterfall mode (other than “None”) is active and more than one
  curve is plotted, the slider (and its paired numeric spin box) control
  the magnitude of the vertical offset between curves.

### Curve List (right side)

Side panel that lists all curves currently in the **Plotted Data** tab.

For each curve you can:

- **Show/hide:** toggle visibility without removing the curve.
- **Change color:** open a color picker to change the line color.
- **Change style:** choose between **Solid**, **Dashed**, and **Scatter**
  (marker mode).
- **Change thickness / marker size:** controlled by the style row’s size
  field.
- **Remove curve:** click the small **×** (cross) to remove the curve
  from the Plotted list and the Plotted axes.
- **Add to reference library:** click the **bookmark** button to add the
  curve to the reference spectra library (`library.h5`) together with
  metadata (element, edge, compound, resolution, comments, etc.).

The curve list is fully interactive:

- **Drag-and-drop reordering:**  
  Drag curves up or down with the mouse. When you release, the plot and
  legend are updated to follow the new order (particularly visible in
  the Waterfall representation).

### Legend

Legend behavior is controlled via the **Legend:** drop-down on the Plotted Data panel:

- **User-defined (default):** shows the legend and lets you set custom curve names.
  - In the legend, entries initially appear as `<select curve name>`.
  - Right-click a legend label to rename that curve. The new name is stored and reused when the legend is rebuilt.
- **Entry number:** shows the legend and automatically labels curves by their entry ID.
  - For example, `entry6567` becomes `6567` in the legend.
- **None:** removes the legend from the plot.

You can always reposition the legend (when shown) by dragging the legend box inside the axes.

### Annotation

- **Annotation checkbox:** show/hide a single annotation text on the
  Plotted Data plot.
- When visible:
  - Click the annotation text to edit it (via a dialog).
  - Drag the annotation with the left mouse button to reposition it.

### Grid

- **Grid:** *(combobox: None / Coarse / Fine / Finest)*  
  Applies major and minor grid lines to the Plotted Data axes:
  - **None:** no grid.
  - **Coarse:** major grid only (similar to the original single checkbox).
  - **Fine / Finest:** enable minor grid with successively denser spacing.
    Minor grid lines are drawn less prominently than major ones.

### Reference Spectra Library

- **Load reference:**  
  Button that opens a dialog listing spectra from the reference library
  file `library.h5`. Selected reference spectra are loaded into the
  Plotted Data tab as additional curves and marked as “reference” in
  their metadata (they cannot be re-added to the library from the curve list).

> **Important:** The reference-data library functionality is currently in
> an **early, experimental state**. The shipped `library.h5` content
> should be regarded as **demonstration only** and may include placeholder,
> test or otherwise unverified spectra. Treat any reference spectra as
> illustrative examples, not authoritative reference data.

### Export / Clear

- **Export/Import (Plotted):**  
  Click to open a menu:
  - **Export CSV:** export the curves currently shown in the Plotted Data tab as CSV.
  - **Import CSV:** load one or several previously exported CSV files into the Plotted Data plot.

- **Clear Plotted Data:**  
  Remove all curves from the Plotted Data tab (but keep raw and processed
  data untouched).
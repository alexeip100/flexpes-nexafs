# FlexPES NEXAFS Plotter

## Overview
This application opens HDF5 files with NEXAFS spectra from the FlexPES
beamline (MAX IV Laboratory). It supports pre‑processing, visualization,
and export of raw and processed data.

## File Controls (Top Left Panel)
- **Open HDF5 File:** Load one or more HDF5 files containing NEXAFS data.
- **Close all:** Close all currently opened files.
- **Clear all:** Remove all loaded data and reset the interface.
- **Help:** Open **Usage** and **About** dialogs.

## File Tree Panel
Shows the hierarchical structure of loaded HDF5 files.
Expand groups to view datasets. Tick checkboxes on **1‑D datasets**
(typically under `measurement/`) to include or exclude curves from plots.

## Tabs (Right Panel)

### Raw Data Tab
**Purpose:** choose which datasets (curves) are visible.

- **Data Plot:** Shows raw spectra. Multiple curves can be shown at once.
- **All in channel:** *(checkbox + combobox of channel names)*  
  The combobox lists unique channel names found across opened files
  (the last path component only).  
  - **Checked:** load the selected channel across **all entries**.
    Switching the selection clears the previous channel group and loads
    the new one.  
  - **Unchecked:** clear the selected channel group from the plot and
    the tree. Changing selection while unchecked does not load data.
- **Channel Families:** **All TEY**, **All PEY**, **All TFY**, **All PFY**.  
  Family toggles by detector type. Each checkbox shows or hides its
  family across entries. You can combine these with **All in channel**
  and with item checkboxes in the File Tree.
- **Data Tree (right of plot):** Lists raw items for inspection. Datasets
  are grouped into energy‑dependent “regions” (e.g., absorption edges).
  Group checkboxes allow quick selection or deselection of an entire edge.
- **Scalar Display:** Shows scalar or textual metadata for the selected
  items beneath the plot canvas.

### Processed Data Tab
**Purpose:** apply non‑destructive processing to all visible curves.

- **Background:** *(mode + polynomial degree + pre‑edge %)*  
  Modes: **None**, **Automatic**, **Manual**.  
  - **Automatic:** adds a constraint so the derivative at the end of the
    background matches the slope of the data.  
  - **Manual:** adjust anchor points with the mouse if Automatic fails.
- **Normalize:** *(None / Max / Jump / Area)*  
  Per‑curve normalization applied to the visible set.  
  - **Max:** scale to the maximum value.  
  - **Jump:** scale to the absorption jump at the last point.  
  - **Area:** scale to the integrated area under the curve.
- **Summing:** Sum multiple datasets (e.g., repeated sweeps) to improve
  statistics. The sum behaves like any other curve.
- **Subtraction:** Show the background‑subtracted curve. Requires a single
  dataset (or a sum) to be selected.
- **PASS:** When a curve is in its “final” processed state (normalized,
  optionally summed, with suitable background), send it to **Plotted Data**
  for presentation and export.  
  Use **Export ASCII** to save the processed curve as CSV.

### Plotted Data Tab
**Purpose:** control plot display, axes, style, and export.

- **Interactive Plot Canvas:** Embedded Matplotlib with zoom, pan, and save.
  Energy (x‑axis) is auto‑detected from common fields
  (`pcap_energy_av`, `mono_traj_energy`). Curve labels include *entry + channel*.
- **Waterfall slider:** If more than one curve is plotted, enable waterfall
  view and adjust vertical spacing.
- **Curve List:** Side panel that lists plotted curves. Adjust color, line
  style, and visibility per curve. The curve list is interactive:
  dragging curves up or down with the mouse instantly updates the plot and reorders the legend
  accordingly upon release (most noticeable in the Waterfall representation).
  Each curve can be removed from the "Plotted" list by pressing small cross in front of the curve.
- **Legend:** Show/hide; drag to reposition. If enabled in your build,
  you can rename legend entries.
- **Grid:** Grid with different line density can be applied to the plot.
- **Export / Clear:** Export plotted data (e.g., CSV of x/y) and save
  figures (e.g., PNG). **Clear** removes all plotted curves in this tab.


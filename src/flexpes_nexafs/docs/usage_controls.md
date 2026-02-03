# Overview
FlexPES-NEXAFS is a GUI tool for inspecting and processing **1D spectra** stored in HDF5 files (typical FlexPES/MAX IV style data).

You can use it to:
- browse HDF5 content safely (tree items are **lazy-loaded** as you expand them),
- quickly visualize groups of spectra corresponding to a specific detection mode (TEY/PEY/TFY/PFY), or groups of other recorded 1d data,
- apply common processing steps (optional **I₀ normalization**, **background subtraction**, and **post-normalization**),
- create **summed** spectra from multiple scans,
- prepare publication-style plots (with legend, annotation, grid, waterfall representation, curve style selection, etc),
- export/import curves as CSV, load reference spectra, and send curves to decomposition tools (PCA/NMF/MCR).

## Mental model (how parts of the GUI relate)
Think in three layers:

1) **Select** what you want to see (HDF5 Structure + Raw Data tab)  
2) **Process** selected spectra (Processed Data tab)  
3) **Present/export** selected curves (Plotted Data tab)

## Key terms used in the UI
- **Curve**: one 1D dataset shown as a line.
- **Entry number**: IDs like `entry6567`. Used by the “Entry number” legend mode.
- **Region**: a grouping used in the Raw/Processed trees to organize curves on the photon E scale.
- **I₀**: a monitor channel used for normalization when enabled.

## Abbreviations used in the UI
- **TEY**: Total Electron Yield (usually sample drain current)
- **PEY**: Partial Electron Yield (electron current with a low-E cutoff, usually from a channeltron multiplier or a multichannel plate (MCP) detector)
- **TFY**: Total Fluorescence Yield (either a photodiode, or an MCP counting detector, or sum of all channels in an energy-dispersive detector)
- **PFY**: Partial Fluorescence Yield (photon-counting signal with energy filtering, typically a region of interest in an energy-dispersive detector)
- **I₀ (I0)**: the incident-beam monitor; in the app this usually means the **I₀ signal/curve** used for normalization.
- **BG**: background (baseline) model used for subtraction from the raw XAS spectra.
- **PCA**: Principal Component Analysis (variance-based decomposition of the multiple spectra arrays)
- **NMF**: Non-negative Matrix Factorization (non-negative components)
- **MCR-ALS**: Multivariate Curve Resolution – Alternating Least Squares (mixture model with constraints)

---

## File Controls (top-left)
These buttons control file loading and global reset actions.

### **Open HDF5**
Opens one or more HDF5 files. Loaded files appear in the **HDF5 Structure** tree.

**Tip:** you can open multiple files at once; curves from all files can be selected and plotted together.

### **Close all**
Closes *all* opened files and clears all loaded data, trees, and plots.

**Also available:** close a *single* file:
- In **HDF5 Structure**, right-click the top-level file item → **Close** (confirmation shown).

### **Clear all**
Resets the UI state without exiting the application. Typical effects:
- clears plotted curves (Plotted Data),
- clears current raw/processed selections and curves shown in plots,
- resets processing controls.

Opened files remain visible in the HDF5 tree, so you can reselect curves quickly.

### **Help**
Opens:
- **What is what?** (this document, describes all UI elements)
- **How to?** (describes typical workflows/recipes)
- **About**

### **Setup channels** and **Active beamline**
Opens the channel mapping dialog and shows which profile (beamline or branch line) is currently active.

**See also (How to):** *Fix wrong TEY/PEY selection (Setup channels)*

---

## Setup channels (beamline profiles)
This dialog defines how the application recognizes roles such as **TEY / PEY / TFY / PFY**, **Energy**, and **I₀** based on dataset names/paths in the HDF5 file. It is used normally to define and use profiles specific for non-default beamlines (default is “FlexPES-A”)

### When to use it
- “All TEY data / All PEY data / …” selects the wrong curves (or nothing).
- The x-axis energy is not detected as expected.
- The default I₀ selection is not suitable for your files.

### Where mappings are stored
- A default mapping ships with the package: `docs/channel_mappings.json`
- Edits are saved to a per-user config file:
  `AppDataLocation/channel_mappings.json` (fallback: `~/.flexpes_nexafs/channel_mappings.json`)

### What it affects in the GUI
- Raw tab bulk selection:
  **All TEY data / All PEY data / All TFY data / All PFY data**
- Candidates shown in **Choose I₀**
- Energy lookup for plotting (x-axis)

### Practical guidance
- Treat the channel mapping as “how the software finds things” rather than “how the HDF5 is structured”.
- If your data uses different naming conventions, add the relevant substrings/candidates here.

---

## HDF5 Structure (tree in left panel)
This tree is for browsing the file contents and selecting specific datasets.

### Lazy loading
Tree nodes are populated only when expanded. This keeps large files responsive.

### Browsing metadata
Clicking on 0d data (like metadata strings or scalars) shows their values in a field under the plot on the Raw Data tab.

### Checking items
Checking a 1d dataset (toggling specific check boxes) affects what is considered “visible” and can contribute curves to Raw/Processed plots.

### Closing a single file
Right-click a top-level file item → **Close** (with confirmation). Alternatively, select the HDF5 file to close and press “Delete”. 

---

# Tabs (Right panel)
The right panel contains three tabs:
- **Raw Data**
- **Processed Data**
- **Plotted Data**

Each tab answers a different question:
- *Raw:* “What do I want to look at?”
- *Processed:* “How do I transform it (normalization/background/sum)?”
- *Plotted:* “How do I present/export it?”

---

# Raw Data tab
Purpose: select and inspect raw curves.

### Channel family selectors
Checkboxes:
- **All TEY data**
- **All PEY data**
- **All TFY data**
- **All PFY data**

These toggle visibility of all datasets matching the role substrings from the **active beamline profile**.

**Tip:** If these select nothing or the wrong curves, use **Setup channels** to adjust mappings.

### “All in channel”
- **All in channel:** checkbox + dropdown

The dropdown lists unique channel names (last component of dataset paths).  
When enabled, changing the dropdown switches the active channel selection across entries.

**When to use:** you want something else (I₀, energy, an encoder value, etc) across many entries.

### Raw plot
- Matplotlib toolbar is shown above the raw plot (zoom, pan, save, etc).

### Raw info line (below the plot)
A small text line under the plot displays a scalar value or short information for the currently displayed/selected 0d dataset (typically when browsing through HDF5 file metadata).

**See also (How to):** *Inspect many curves quickly (Raw Data selection)*

---

# Processed Data tab
Purpose: process visible curves (I₀ normalization, background handling, post-normalization), optionally create summed curves, then export or pass results to Plotted Data.

## Top row controls

### **Normalize by I₀?**
When checked, each curve is divided by the corresponding **I₀ signal/curve** from the same entry.

**When to use:** Almost always: corrects for the background and temporary flux variations (like storage ring injection events).  
**Tips:** If you choose an unsuitable I₀ signal (candidate), the result can be noisy or distorted.

### **Choose I₀**
Dropdown enabled only when **Normalize by I₀?** is checked.

Candidates come from the channel mapping profile.

### **Sum up?**
Opens the curve summation dialog to create new summed curves from selected curves. Mainly used when you have collected multiple scans from each sample for better statistics, and want to sum them up in groups corresponding to samples before proceeding with further treatment.

**See also (How to):** *Sum curves (“Sum up?” dialog)*

### **Group BG**
Enabled when you have two or more curves that you want to process *as a comparable group*.

**What it is for**  
Use **Group BG** when you want to compare a series of spectra (e.g. repeated scans, a time/temperature series, or multiple samples) and you don’t want the background handling and scaling to drift independently for each curve.

**What happens when you enable it**  
Group BG switches the app into a constrained group-processing mode and locks the relevant settings to keep the result comparable across the whole group:

- **Choose BG** is forced to **Auto**  
- **Subtract BG?** is forced **ON**  
- post **Normalize** is forced to **Area**

So the Processed Data plot shows a set of **background-subtracted, area-normalized** spectra processed in a consistent way.

**How the background is determined (conceptually)**  
Instead of fitting each curve completely independently, Group BG adjusts the automatic polynomial backgrounds with the goal that, across the group:

- the **post-subtraction scaling is comparable** (area-normalized output), and  
- the **pre-edge baseline behavior is consistent** enough to make curves directly comparable.

**When to use / when not to use**
- Use it for comparative plots and trend analysis across many similar spectra.
- Avoid it if each curve truly requires a different background model (very different baseline shapes or strong artefacts), because forcing consistency can hide genuine differences or introduce bias.

---

### **Match pre-edge**
Enabled only when **Group BG** is active.

**What it does**  
Adds an extra group constraint that makes the **pre-edge region line up more consistently** across the selected curves (in practice: the pre-edge slope/offset after subtraction is aligned across the group).

**Why you might want it**  
Sometimes Auto polynomial backgrounds are “good enough” curve-by-curve, but the group comparison still looks messy in the pre-edge (small slope differences dominate the visual comparison). Match pre-edge is meant to reduce that distraction and make the group comparison clearer.

**Important caution**  
Match pre-edge can improve consistency, but it does so by allowing a more flexible adjustment of the background specifically to make the pre-edge align. Sanity-check that you are not introducing artificial structures or distorting real low-energy features. A good practice is to toggle it on/off and confirm that the edge-region features you care about remain stable.

**See also (How to):** *Compare multiple curves consistently (Group BG)*

### **Pass**
Sends the current processed curve(s) to **Plotted Data** without clearing existing plotted curves.

**When to use:** you want to style/export from Plotted Data and keep multiple curves together.

### **Export**
Exports processed data to CSV. Works for individual curves, not a group. Group CSV export is possible from the Plotted Data tab.

**See also (How to):** *Make a single clean processed curve* and *Sum curves (“Sum up?” dialog)*

## Background and post-normalization controls (bottom row)

### **Choose BG** — None / Auto / Manual
- **None**: no background model
- **Auto**: polynomial pre-edge background
- **Manual**: anchor-based adjustment (built on top of an automatic estimate)

### **Poly degree**
Polynomial degree for Auto/Manual background.

### **Pre-edge (%)**
Pre-edge region length (percent of energy span) used for baseline fitting.

### **Subtract BG?**
Shows BG-subtracted curves when enabled.

### **Normalize** — None / Max / Jump / Area
Post-normalization scaling (enabled/disabled depending on mode).


---

# Plotted Data tab
Purpose: figure-like plotting and exporting.

## Plotted curve list (tree on the right)
Here you can manage which curves are shown and how they look. Each row corresponds to one curve currently loaded into Plotted Data.

### What you can do there

- Show / hide curves (visibility toggle per curve).
- Reorder curves by drag-and-drop (this affects plot order and legend order).
- Change appearance per curve:
-- color,
-- line style (solid / dashed / markers, depending on the control),
-- line width / marker size (depending on the control).
- Remove a curve from the plotted set (delete/remove control on the row).
- Add a curve to the reference library using the bookmark/add-to-library control (available on each curve row).

### Selection behavior

Selecting an item in the list makes that curve the “active” curve for certain actions (e.g. operations that act on one curve at a time, like naming in user-defined legend mode).

**Tip:** If you want a consistent export order and legend order, set it here by reordering the curves before exporting.


## Top row controls

### Legend
Dropdown: **Legend**
- **None**
- **User-defined**
- **Entry number**

Behavior:
- **User-defined**: rename labels by clicking legend entries.
- **Entry number**: labels by `entry#### → ####`.

**Tips:** If Legend is **None**, CSV export from Plotted Data is blocked (there is no naming scheme).

### Annotation
Checkbox: **Annotation**
- shows/hides one annotation box,
- right-click to edit text/style,
- drag to reposition.

## Bottom row controls

### Waterfall
Checkbox + slider/spinbox (enabled only when Waterfall is enabled).  
Applies a uniform vertical offset to separate many curves visually.

### Grid
Dropdown: `None / Coarse / Fine / Finest`

### Load reference
Button: **Load reference**
Opens the reference library dialog where you can load references into the plot.
The same dialog contains **Delete reference** (removes from the library).

### Export/Import
Button opens a menu:
- **Export CSV**
- **Import CSV**

Export naming depends on Legend mode:
- **Entry number** → headers are entry numbers (requires entry ids)
- **User-defined** → headers are custom names; placeholder `<select curve name>` is not allowed
- **None** → export is blocked

**See also (How to):** *Export plotted curves to CSV (Export/Import → Export CSV)*

### PCA (decomposition window)
Button: **PCA**

This opens the **decomposition window** and (optionally) sends curves from Plotted Data to it. The decomposition tools are meant to help you **separate mixed spectral contributions** and **compare multiple spectra systematically**, for example:
- identify common components across a series,
- find dominant variations (PCA),
- try non-negative component models (NMF),
- fit linear mixtures with constraints (MCR-ALS),
- explore anchor-based / guided decompositions (when available in the window).

#### What you typically use PCA for
- You have many spectra (e.g. a spatial scan, temperature series, time series) and want to understand:
  - “How many independent spectral shapes are present?”
  - “Which spectra vary together?”
  - “Can I represent the series as a mixture of a few components?”

#### What happens when you click PCA
- If eligible curves are available in Plotted Data, they are transferred to the decomposition window.
- If no visible curves are selected/eligible, the window can still open (useful if you plan to load data from within that window).

#### Requirements enforced by the app (transfer from Plotted Data)
The transfer is blocked unless:
- **Waterfall** is OFF (waterfall changes y-values and is for visualization only),
- post **Normalize** is **Area** for all selected curves (ensures comparability),
- **Legend** is not **None** and curve names are resolvable (needed to label spectra in the decomposition window).

**See also (How to):** *Send curves to PCA / decomposition window*


### Clear Plotted
Removes all curves from Plotted Data without changing Raw/Processed selections.


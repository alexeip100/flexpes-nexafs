# Quickstart
You’ve just opened FlexPES‑NEXAFS and you want to get from “a file on disk” to “a curve I can trust”.

Start simple:

1) Click **Open HDF5** and select your file(s). The file(s) appear in the **HDF5 Structure** tree.  
2) Go to **Raw Data** and make your first broad choice: *which kind of signal do I want to look at?*  
   - If you’re after an electron-yield view, try **All TEY data** or **All PEY data**.  
   - If you’re working with fluorescence, try **All TFY data** or **All PFY data**.  
   - If you already know the exact channel name you need across many entries, enable **All in channel** and pick it in the dropdown.  
3) Switch to **Processed Data** and decide what “clean” means for your case:
   - Do you need normalization by the incident-beam monitor (**Normalize by I₀?** + **Choose I₀**)?
   - Do you need a baseline/background correction (**Choose BG**, **Pre-edge (%)**, **Poly degree**)?
   - Do you want to *see* the BG‑subtracted result (**Subtract BG?**)?
   - Do you want a defined scaling for comparisons (**Normalize**, often **Area**)?
4) When the processed curve looks right, either:
   - click **Export** (best when you are exporting one curve), or
   - click **Pass** to move curves into **Plotted Data**, where you can style, reorder, annotate, and export multiple curves together.

**Controls used (What is what?):** *Raw Data tab*, *Processed Data tab*, *Export*, *Pass*

## Abbreviations used below (quick reference)
- **TEY**: Total Electron Yield
- **PEY**: Partial Electron Yield
- **TFY**: Total Fluorescence Yield
- **PFY**: Partial Fluorescence Yield
- **I₀ (I0)**: incident-beam monitor; in practice you select the **I₀ signal/curve** to normalize your spectra
- **BG**: background (baseline) model/subtraction
- **PCA**: Principal Component Analysis
- **NMF**: Non-negative Matrix Factorization
- **MCR-ALS**: Multivariate Curve Resolution – Alternating Least Squares


---

# Fix wrong TEY/PEY selection (Setup channels)
Sometimes the app does exactly what you asked—just not what you *meant*. The classic example: you click **All TEY data** and either nothing happens, or the “TEY” curves clearly aren’t the signal you expected.

This almost always means the file’s channel naming doesn’t match the current channel profile.

**How to fix it (in human terms):**
1) Click **Setup channels**. Think of this dialog as the app’s “dictionary” for your file naming conventions.  
2) Pick the profile that is closest to your data (for example, a FlexPES profile).  
3) If needed, adjust the substrings/candidates so that the app can correctly recognize TEY/PEY/TFY/PFY and suitable I₀ candidates.  
4) Save/apply, confirm the label **Active beamline:** shows what you expect, then go back and try **All TEY data** again.

If the selector now lights up the right curves, you’re done.

**Controls used (What is what?):** *Setup channels (beamline profiles)*, *Channel family selectors*

---

# Inspect many curves quickly (Raw Data selection)
Imagine you have a dataset with many entries: repeated scans, a temperature series, or a map. You *could* click individual datasets in the HDF5 tree—but you don’t want to spend your afternoon doing that.

The Raw Data tab is built for this situation.

## Option A: “show me all curves of a detector family”
If you want the “standard” signals:
- Click **All TEY data** (or PEY/TFY/PFY).

You’ll immediately see the family of curves, which makes it easy to spot outliers, drift, bad scans, or obvious trends before you process anything.

## Option B: “follow one channel across many entries”
If you know you always want the same channel name:
- Enable **All in channel**, then choose the channel in the dropdown.

This is great for non-standard channels, monitor signals, or anything that isn’t reliably captured by the TEY/PEY/TFY/PFY grouping.

**If nothing sensible shows up:** go back to **Setup channels** and fix the mapping first.

**Controls used (What is what?):** *Raw Data tab*, *Channel family selectors*, *All in channel*

---

# Make a single clean processed curve
This workflow is for the common situation: you’re not trying to compare 50 curves yet—you just want *one* spectrum that is corrected, readable, and exportable.

Think of it as polishing one curve until it’s “presentation ready”.

1) In **Processed Data**, decide whether you need incident-beam normalization.  
   If yes: enable **Normalize by I₀?** and pick the correct **I₀ signal/curve** in **Choose I₀**.  
2) Choose how to deal with baseline/background:
   - **Auto** is the normal starting point.
   - Adjust **Pre-edge (%)** and **Poly degree** if the baseline fit is clearly wrong.
   - Use **Manual** only if you really need anchor-style correction.  
3) Decide what you want to look at:
   - Enable **Subtract BG?** if you want the BG-subtracted spectrum.
   - Disable it if you want to keep the baseline in the displayed curve.  
4) If your next step is comparison (or PCA later), choose a stable scaling via post **Normalize** (often **Area**).  
5) When the curve looks right:
   - **Export** if you’re exporting one curve, or
   - **Pass** if you want to assemble a figure or export multiple curves from Plotted Data.

**A practical note:** Processed‑tab **Export** is intentionally strict—if multiple curves are visible, export from **Plotted Data** instead, or create a summed curve first.

**Controls used (What is what?):** *Processed Data tab*, *Choose BG*, *Subtract BG?*, *Normalize*, *Export*, *Pass*

---

# Sum curves (“Sum up?” dialog)
Summation is what you reach for when you have repeats: several scans that are “the same measurement”, just noisy or slightly shifted, and you want one cleaner curve.

When you click **Sum up?**, you’re essentially telling the app: *“Treat these curves as one group and give me a single representative curve.”*

**How it works in practice:**
1) In **Processed Data**, make sure the curves you want are visible.  
2) Click **Sum up?** and build one or more groups (each group becomes one summed curve).  
3) Name each group—those names become the new curve names.  
4) Confirm.

Behind the scenes, the app uses the overlapping energy range, interpolates onto a common grid, and then sums. If **Normalize by I₀?** is enabled, that normalization happens before the summation.

If you don’t get a summed curve, it usually means the curves don’t overlap in energy enough to build a shared grid.

**Controls used (What is what?):** *Sum up?*, *Normalize by I₀?*, *Export*

---

# Compare multiple curves consistently (Group BG)
Once you move from “one nice curve” to “a set of curves I want to compare”, the risk changes: tiny differences in baseline handling can dominate your interpretation.

That’s what **Group BG** is for: it puts the app into a consistent group mode so the set behaves like a set.

**A good mental model:** “I want these curves to be processed the same way, so I can focus on real spectral differences.”

1) In **Processed Data**, make two or more curves visible—the ones you want to compare.  
2) Enable **Group BG**. The app locks into a consistent mode:
   - **Choose BG = Auto**
   - **Subtract BG? = ON**
   - **Normalize = Area**  
3) If the pre-edge region still looks uneven and distracts from the comparison, enable **Match pre-edge** to align pre-edge baseline/slope more consistently.  
4) Click **Pass** to move the group into **Plotted Data** for styling, ordering, annotation, and export.

**Controls used (What is what?):** *Group BG mode*, *Match pre-edge*, *Pass*

---

# Prepare a publication-style plot (Plotted Data)
Processed Data is where you *compute*; Plotted Data is where you *compose*.

In Plotted Data you can treat curves like figure elements:
- choose naming via **Legend**,
- reorder curves for a sensible narrative,
- add annotation,
- choose grid and waterfall display,
- export the final dataset as CSV.

A typical “figure-making” rhythm:
1) Send curves in using **Pass** (from Processed Data).  
2) Pick a **Legend** mode:
   - **User-defined** if you want meaningful labels (“Sample A”, “Annealed 450°C”…),
   - **Entry number** if you want raw bookkeeping labels.  
3) Use the plotted curve list (tree on the right) to show/hide and reorder curves—this also controls legend order.  
4) Add **Annotation** when you want the figure to explain itself.  
5) Add a **Grid** if it helps reading values; use **Waterfall** when curves overlap too much.  
6) Export:
   - **Export/Import → Export CSV** for data,
   - Matplotlib toolbar Save for an image/PDF if you want a figure file.

**Controls used (What is what?):** *Plotted Data tab*, *Legend*, *Annotation*, *Grid*, *Waterfall*, *Export/Import*

---

# Export plotted curves to CSV (Export/Import → Export CSV)
Exporting from Plotted Data is for when you have *a set* of curves and you want them in one tidy CSV.

Before you export, decide how you want the columns to be named:
- **User-defined** legend → your custom names become headers.
- **Entry number** legend → entry numbers become headers.
- **None** → export is blocked (there is no naming scheme).

Then:
1) Click **Export/Import → Export CSV**.
2) Choose where to save.

**Controls used (What is what?):** *Export/Import*, *Legend*

---

# Import CSV into Plotted Data
Import is the mirror of export: it lets you bring curves back in, or overlay external curves for comparison.

1) Click **Export/Import → Import CSV**.
2) Select one or more CSV files.

The curves appear in the plotted list and on the plot, ready for styling and export.

**Controls used (What is what?):** *Export/Import*

---

# Use reference spectra (Load reference)
References are for the moments when you want a known spectrum on top of your data—an internal standard, a textbook reference, or a saved “good sample” spectrum.

1) Click **Load reference** in Plotted Data.  
2) Pick one or more references and confirm.  
3) If you’re cleaning up the library, **Delete reference** is available inside the same dialog.

**Controls used (What is what?):** *Load reference*

---

# Send curves to PCA / decomposition window
This workflow is for *series thinking*: you have many spectra and you suspect the set is a mixture of a few underlying components.

The decomposition window helps you answer questions like:
- “How many independent spectral shapes are present?”
- “Do these spectra move together (one dominant trend) or in several independent ways?”
- “Can I explain the series as mixtures of a few endmembers?”

**How to get there smoothly:**
1) In **Processed Data**, make curves comparable (often you’ll end up with post **Normalize = Area**).  
2) Click **Pass** to move them into **Plotted Data**.  
3) Make sure **Waterfall** is OFF (waterfall is for visualization, not analysis).  
4) Make sure **Legend** is not **None** so the spectra have names.  
5) Click **PCA**.

If requirements are met, curves are sent into the decomposition window where you can run PCA/NMF/MCR‑ALS.

**Controls used (What is what?):** *PCA (decomposition window)*, *Pass*, *Plotted Data tab*

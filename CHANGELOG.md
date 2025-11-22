# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Added
- New features or improvements that are not yet released.

### Changed
- Updates or behavioral changes that alter existing functionality.

### Fixed
- Bug fixes that improve stability or correctness.

### Removed
- Deprecated or obsolete functionality removed.

---

## [1.9.1] – 2025-10-09

### Changed

- H5 files are no longer open for the lifetime of the app -> other software can open them simultaneously for writing (e.g. Sardana) 
- Better "Open file" dialog

### Fixed

- Improved stability of background subtraction toggle (`Subtract background?`) for manual BG in case of summed curves.


## [1.9.2] – 2025-10-11

### Added
- New features of Waterfall representation in the Plotted Data panel

### Fixed

- Fixed bugs in the post-normalization of the summed curves (to Max, Jump and Area).
- Fixed bug of passing a summed curve and all curves constituting the sum: toggling Waterfall does not remove the very first curve any more.


## [1.9.4] – 2025-11-11

### Added
- Group loading of any channel (not only TEY, PEY, TFY and PFY) is now enabled by using a combo box "All in channel:"

### Changed
- Package structure has been completely refactored: the code is now split into several modules (ui.py, data.py, plotting.py, etc) to simplify development.
- Help updated
- Help is refactored from plotting.py into a dedicated markdown file (help.md in DOCS) for easier updating. 

## [1.9.5] – 2025-11-12

### Added
- The curve list in the “Plotted Data” panel is now interactive: dragging curves up or down with the mouse instantly updates the plot and reorders the legend accordingly upon release (most noticeable in the Waterfall representation).


## [1.9.6] – 2025-11-15

### Added
- It is possible now to remove curves from the list in the "Plotted Data" panel by clicking a "Cross" button in front of any specific curve, with corresponding adjustment of the plot.

### Changed
- "Grid" check box is replaced with the "Grid:" combo box, which allows to apply grids with different line density.
- "Help"->"Usage" is updated. 


## [1.9.7] – 2025-11-15

### Fixed
- A bug crashing the application is fixed.


## [1.9.8] – 2025-11-22

### Fixed
- A few bugs related to the appearance of curves and legends in the "Plotted Data" plot are fixed.

### Added
- Annotation option is added for the plot in the "Plotted Data" panel.
- Reference spectra library file (library.h5) is added; spectra in the list on the "Plotted Data" panel can be added to the library using the new "bookmark" button in each row.
- "Load reference" button allows to load a reference spectrum saved in the file library.h5 (for these spectra the "bookmark" buttons are disabled).

### Changed
- Help -> Usage text is updated.


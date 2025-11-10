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


## [1.9.3] – 2025-11-10

### Changed
- Package structure has been completely refactored: the code is now split into several modules (ui.py, data.py, plotting.py, etc) to simplify development.
- No changes should be noticeable to the end user compared to v1.9.2. 


## [1.9.4] – 2025-11-11

### Added
- Group loading of any channel (not only TEY, PEY, TFY and PFY) is now enabled by using a combo box "All in channel:"

### Changed
- Help updated
- Help is refactored from plotting.py into a dedicated markdown file (help.md in DOCS) for easier updating. 

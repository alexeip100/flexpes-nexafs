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

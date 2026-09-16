# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Release draft: move this file to the repository root in the `0.2.0`
> release PR, replace `Unreleased` with the release date, and review the
> entries against the final merged state.

## [0.2.0] - 2026-09-16

### Added

- Local adaptation of Corey relative-permeability parameters for production
  wells where calculated current oil saturation exceeds initial saturation.
- Severity-based influence multipliers for inconsistent well trajectory
  points, applied in both full-memory and batched calculation paths.
- Smooth local water-cut correction around adapted wells inside their
  effective drainage radii.
- Per-well adapted relative-permeability diagnostics exposed through
  `ResultMaps.adapted_relative_permeability`.
- Point-level `So_current`, `eps_so`, and weight-multiplier diagnostics in the
  processed well data.
- Configuration options for local relative-permeability adaptation,
  production-scaled saturation tolerance, physical penalties, and water-cut
  smoothing.
- Tests covering local adaptation, point-weight reduction, full and batched
  weighting paths, water-cut correction, and reserve constraints.

### Changed

- Current-saturation interpolation now downweights inconsistent well data and
  enforces `So_current <= So_initial` after interpolation.
- Material-balance optimization now uses normalized loss and includes a
  configurable penalty for residual recoverable reserves exceeding initial
  recoverable reserves.
- Batched optimization streams fresh batches instead of writing intermediate
  optimization matrices to disk.
- Full-mode memory selection now accounts for peak distance-matrix allocation,
  reducing the risk of out-of-memory execution.
- Residual recoverable reserves are clipped to the physical interval from zero
  to initial recoverable reserves.
- Water-cut calculation can apply locally adapted Corey parameters while
  retaining the original calculation when no adaptation is required.
- `calculate_current_saturation` now returns the saturation map together with
  the processed well diagnostics used by the result pipeline.
- Example files now resolve input and output paths relative to the example
  directory and display adapted relative-permeability diagnostics.
- Documentation now describes water cut in percent and lists the new options
  and result field.

### Fixed

- Fractional-flow endpoint handling now returns physically bounded values when
  either water or oil relative permeability is zero.
- Validation rejects invalid endpoint saturation combinations and zero Corey
  endpoint values.
- Validation correctly accepts optional fracture and memory parameters when
  they are not configured.
- Interpolation edge noise can no longer produce current saturation above the
  initial saturation map.


# GLIM Rover Core Milestones

## M0: Safe Mapping Core

Goal: make the core robust to empty and filtered-empty LiDAR scans.

Tasks:
- M0.1 ScanGuard: reject empty raw and filtered point clouds safely
- M0.2 ScanGuard config: min_raw_points, min_filtered_points, burst handling
- M0.3 ScanGuard diagnostics topic/logs

Acceptance:
- Empty raw cloud does not crash.
- Cloud that becomes empty after filtering does not create LiDAR factors/keyframes.
- Consecutive skipped scans are counted and logged.
- Recovery after valid scan is logged.

## M1: PointAttributes v1

PR290-style point attributes.

## M2: Colored PCD Export

XYZ/XYZI/XYZRGB map export.

## M3: MapCleaner Dataset Export

MapCleaner-compatible scans + poses + metadata export.

## M4: Diagnostics and QualitySnapshot

Unified diagnostics and QualitySnapshot.

#pragma once

#include <cstddef>
#include <string>

namespace glim {

/**
 * @brief Configuration for LiDAR scan safety checks.
 *
 * Stage M0 goal:
 * - reject empty raw scans
 * - reject filtered-empty scans
 * - prevent invalid LiDAR factors/keyframes
 * - keep diagnostics for debugging and QualitySnapshot
 */
struct ScanGuardConfig {
  bool enable = true;

  std::size_t min_raw_points = 100;
  std::size_t min_filtered_points = 80;
  std::size_t min_effective_points = 50;

  int max_empty_scan_burst = 10;

  bool drop_empty_frame = true;
  bool allow_imu_only_prediction = true;
  bool publish_diagnostics = true;
};

enum class ScanGuardRejectReason {
  NONE = 0,
  RAW_EMPTY,
  RAW_TOO_SMALL,
  FILTERED_EMPTY,
  FILTERED_TOO_SMALL,
  EFFECTIVE_TOO_SMALL,
  TOO_MANY_CONSECUTIVE_SKIPS
};

struct ScanGuardStatus {
  double stamp = 0.0;

  std::size_t raw_points = 0;
  std::size_t filtered_points = 0;
  std::size_t effective_points = 0;

  bool accepted = true;
  bool lidar_update_skipped = false;

  int consecutive_skipped_frames = 0;

  ScanGuardRejectReason reason = ScanGuardRejectReason::NONE;
  std::string reason_text = "accepted";
};

class ScanGuard {
public:
  ScanGuard() = default;
  explicit ScanGuard(const ScanGuardConfig& config);

  void set_config(const ScanGuardConfig& config);
  const ScanGuardConfig& config() const;

  ScanGuardStatus evaluate_raw(double stamp, std::size_t raw_points);
  ScanGuardStatus evaluate_filtered(
    double stamp,
    std::size_t raw_points,
    std::size_t filtered_points,
    std::size_t effective_points);

  void reset_burst();

  static const char* reason_to_string(ScanGuardRejectReason reason);

private:
  ScanGuardStatus accept(
    double stamp,
    std::size_t raw_points,
    std::size_t filtered_points,
    std::size_t effective_points);

  ScanGuardStatus reject(
    double stamp,
    std::size_t raw_points,
    std::size_t filtered_points,
    std::size_t effective_points,
    ScanGuardRejectReason reason);

private:
  ScanGuardConfig config_;
  int consecutive_skipped_frames_ = 0;
};

}  // namespace glim

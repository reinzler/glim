#include <glim/preprocess/scan_guard.hpp>

namespace glim {

ScanGuard::ScanGuard(const ScanGuardConfig& config) : config_(config) {}

void ScanGuard::set_config(const ScanGuardConfig& config) {
  config_ = config;
}

const ScanGuardConfig& ScanGuard::config() const {
  return config_;
}

void ScanGuard::reset_burst() {
  consecutive_skipped_frames_ = 0;
}

const char* ScanGuard::reason_to_string(ScanGuardRejectReason reason) {
  switch (reason) {
    case ScanGuardRejectReason::NONE:
      return "accepted";
    case ScanGuardRejectReason::RAW_EMPTY:
      return "raw_empty";
    case ScanGuardRejectReason::RAW_TOO_SMALL:
      return "raw_too_small";
    case ScanGuardRejectReason::FILTERED_EMPTY:
      return "filtered_empty";
    case ScanGuardRejectReason::FILTERED_TOO_SMALL:
      return "filtered_too_small";
    case ScanGuardRejectReason::EFFECTIVE_TOO_SMALL:
      return "effective_too_small";
    case ScanGuardRejectReason::TOO_MANY_CONSECUTIVE_SKIPS:
      return "too_many_consecutive_skips";
    default:
      return "unknown";
  }
}

ScanGuardStatus ScanGuard::evaluate_raw(double stamp, std::size_t raw_points) {
  if (!config_.enable) {
    return accept(stamp, raw_points, raw_points, raw_points);
  }

  if (raw_points == 0) {
    return reject(stamp, raw_points, 0, 0, ScanGuardRejectReason::RAW_EMPTY);
  }

  if (raw_points < config_.min_raw_points) {
    return reject(stamp, raw_points, raw_points, raw_points, ScanGuardRejectReason::RAW_TOO_SMALL);
  }

  return accept(stamp, raw_points, raw_points, raw_points);
}

ScanGuardStatus ScanGuard::evaluate_filtered(
  double stamp,
  std::size_t raw_points,
  std::size_t filtered_points,
  std::size_t effective_points) {
  if (!config_.enable) {
    return accept(stamp, raw_points, filtered_points, effective_points);
  }

  if (raw_points == 0) {
    return reject(stamp, raw_points, filtered_points, effective_points, ScanGuardRejectReason::RAW_EMPTY);
  }

  if (raw_points < config_.min_raw_points) {
    return reject(stamp, raw_points, filtered_points, effective_points, ScanGuardRejectReason::RAW_TOO_SMALL);
  }

  if (filtered_points == 0) {
    return reject(stamp, raw_points, filtered_points, effective_points, ScanGuardRejectReason::FILTERED_EMPTY);
  }

  if (filtered_points < config_.min_filtered_points) {
    return reject(stamp, raw_points, filtered_points, effective_points, ScanGuardRejectReason::FILTERED_TOO_SMALL);
  }

  if (effective_points < config_.min_effective_points) {
    return reject(stamp, raw_points, filtered_points, effective_points, ScanGuardRejectReason::EFFECTIVE_TOO_SMALL);
  }

  return accept(stamp, raw_points, filtered_points, effective_points);
}

ScanGuardStatus ScanGuard::accept(
  double stamp,
  std::size_t raw_points,
  std::size_t filtered_points,
  std::size_t effective_points) {
  consecutive_skipped_frames_ = 0;

  ScanGuardStatus status;
  status.stamp = stamp;
  status.raw_points = raw_points;
  status.filtered_points = filtered_points;
  status.effective_points = effective_points;
  status.accepted = true;
  status.lidar_update_skipped = false;
  status.consecutive_skipped_frames = consecutive_skipped_frames_;
  status.reason = ScanGuardRejectReason::NONE;
  status.reason_text = reason_to_string(status.reason);
  return status;
}

ScanGuardStatus ScanGuard::reject(
  double stamp,
  std::size_t raw_points,
  std::size_t filtered_points,
  std::size_t effective_points,
  ScanGuardRejectReason reason) {
  consecutive_skipped_frames_++;

  ScanGuardStatus status;
  status.stamp = stamp;
  status.raw_points = raw_points;
  status.filtered_points = filtered_points;
  status.effective_points = effective_points;
  status.accepted = false;
  status.lidar_update_skipped = true;
  status.consecutive_skipped_frames = consecutive_skipped_frames_;
  status.reason = reason;
  status.reason_text = reason_to_string(reason);

  if (config_.max_empty_scan_burst > 0 &&
      consecutive_skipped_frames_ > config_.max_empty_scan_burst) {
    status.reason = ScanGuardRejectReason::TOO_MANY_CONSECUTIVE_SKIPS;
    status.reason_text = reason_to_string(status.reason);
  }

  return status;
}

}  // namespace glim

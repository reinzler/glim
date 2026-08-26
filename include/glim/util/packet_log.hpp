#pragma once

#include <cstddef>
#include <string>

namespace glim {

/**
 * Thread-safe CSV logger for packaging / ISAM2 diagnostics.
 * Enabled when GLIM_LOG_PACKETS is set (path or "1").
 * Optional ISAM2 lines when GLIM_LOG_ISAM2 is set (reuses same file if path matches,
 * otherwise appends to GLIM_LOG_ISAM2 path / glim_isam2.csv).
 */
void log_packet_row(const char* site, std::size_t n_imu, std::size_t n_frames);

void log_isam2_update(
  const char* site,
  std::size_t n_factors,
  std::size_t n_values,
  int recovery_depth,
  bool skipped);

}  // namespace glim

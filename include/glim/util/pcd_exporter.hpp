#pragma once

#include <string>
#include <cstdint>

#include <gtsam_points/types/point_cloud.hpp>

namespace glim {

enum class PCDColorMode {
  HEIGHT,
  INTENSITY,
  CAMERA
};

struct PCDExportStats {
  std::size_t input_points = 0;
  std::size_t written_points = 0;
  bool has_intensity = false;
};

bool save_pcd_xyz(
  const std::string& path,
  const gtsam_points::PointCloud& cloud,
  PCDExportStats* stats = nullptr);

bool save_pcd_xyzi(
  const std::string& path,
  const gtsam_points::PointCloud& cloud,
  PCDExportStats* stats = nullptr);

bool save_pcd_xyzrgb(
  const std::string& path,
  const gtsam_points::PointCloud& cloud,
  PCDColorMode color_mode,
  PCDExportStats* stats = nullptr);

}  // namespace glim

#pragma once

#include <vector>

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <gtsam_points/types/point_cloud.hpp>

namespace glim {

inline constexpr const char* MAPCLEANER_KEYFRAMES_KEY = "mapcleaner_keyframes";

struct MapCleanerKeyframe {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp = 0.0;
  Eigen::Isometry3d T_origin_lidar = Eigen::Isometry3d::Identity();
  gtsam_points::PointCloud::ConstPtr cloud;
};

using MapCleanerKeyframes =
  std::vector<MapCleanerKeyframe, Eigen::aligned_allocator<MapCleanerKeyframe>>;

}  // namespace glim

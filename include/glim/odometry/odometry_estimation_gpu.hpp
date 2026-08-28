#pragma once

#include <glim/odometry/odometry_estimation_imu.hpp>

namespace gtsam_points {
class VoxelizedFrame;
class StreamTempBufferRoundRobin;
class CUDAStream;
class GaussianVoxelMapCPU;
}  // namespace gtsam_points

namespace glim {

/**
 * @brief Parameters for OdometryEstimationGPU
 */
struct OdometryEstimationGPUParams : public OdometryEstimationIMUParams {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationGPUParams();
  virtual ~OdometryEstimationGPUParams();

  enum class KeyframeUpdateStrategy { OVERLAP, DISPLACEMENT, ENTROPY, FIXED_INTERVAL_FIFO };
  enum class MatchingMode { FRAME_TO_MODEL, KEYFRAME };

public:
  MatchingMode matching_mode;
  bool enable_surface_validation;

  // Frame-to-model VGICP (same algorithm as CPU odometry)
  int max_iterations;
  int lru_thresh;
  double target_downsampling_rate;
  double vgicp_resolution;
  int vgicp_voxelmap_levels;
  double vgicp_voxelmap_scaling_factor;

  // Per-frame voxelmaps (KEYFRAME matching)
  double voxel_resolution;
  double voxel_resolution_max;
  double voxel_resolution_dmin;
  double voxel_resolution_dmax;
  int voxelmap_levels;
  double voxelmap_scaling_factor;

  int max_num_keyframes;
  int full_connection_window_size;

  // Keyframe management params
  KeyframeUpdateStrategy keyframe_strategy;
  double keyframe_min_overlap;
  double keyframe_max_overlap;
  double keyframe_delta_trans;
  double keyframe_delta_rot;
  double keyframe_entropy_thresh;
  int keyframe_fixed_interval;
};

/**
 * @brief GPU-based tightly coupled LiDAR-IMU odometry
 *
 */
class OdometryEstimationGPU : public OdometryEstimationIMU {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationGPU(const OdometryEstimationGPUParams& params = OdometryEstimationGPUParams());
  virtual ~OdometryEstimationGPU() override;

private:
  virtual void create_frame(EstimationFrame::Ptr& frame) override;
  virtual gtsam::NonlinearFactorGraph create_factors(const int current, const gtsam_points::shared_ptr<gtsam::ImuFactor>& imu_factor, gtsam::Values& new_values) override;
  virtual void update_frames(const int current, const gtsam::NonlinearFactorGraph& new_factors) override;

  gtsam::NonlinearFactorGraph create_factors_frame_to_model(const int current, gtsam::Values& new_values);
  gtsam::NonlinearFactorGraph create_factors_keyframe(const int current);
  void update_target(const int current, const Eigen::Isometry3d& T_target_imu);

  void update_keyframes_overlap(int current);
  void update_keyframes_displacement(int current);
  void update_keyframes_entropy(const gtsam::NonlinearFactorGraph& matching_cost_factors, int current);
  void update_keyframes_fixed_interval_fifo(int current);

private:
  // Frame-to-model target (incremental CPU voxelmap, matching on GPU)
  std::mt19937 mt;
  Eigen::Isometry3d last_T_target_imu;
  std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> target_voxelmaps;

  // Keyframe params
  int entropy_num_frames;
  double entropy_running_average;
  std::vector<EstimationFrame::ConstPtr> keyframes;

  // CUDA-related
  std::unique_ptr<gtsam_points::CUDAStream> stream;
  std::unique_ptr<gtsam_points::StreamTempBufferRoundRobin> stream_buffer_roundrobin;
};

}  // namespace glim

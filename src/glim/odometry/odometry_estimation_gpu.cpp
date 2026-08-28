#include <glim/odometry/odometry_estimation_gpu.hpp>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam_points/cuda/cuda_stream.hpp>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu.hpp>

#include <glim/util/config.hpp>
#include <glim/common/imu_integration.hpp>
#include <glim/common/cloud_deskewing.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>

#include <glim/odometry/callbacks.hpp>

#ifdef GTSAM_USE_TBB
#include <tbb/task_arena.h>
#endif

namespace glim {

namespace {

enum class GPUFrameFenceMode {
  DISABLED,
  AFTER_SMOOTHER,
  BEFORE_OVERLAP,
  BOTH,
};

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !*value) {
    return false;
  }

  return std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0 &&
         std::strcmp(value, "off") != 0 &&
         std::strcmp(value, "OFF") != 0;
}

GPUFrameFenceMode gpu_frame_fence_mode() {
  static const GPUFrameFenceMode mode = [] {
    const char* value = std::getenv("GLIM_GPU_FRAME_FENCE");
    if (!value || !*value || std::strcmp(value, "off") == 0 || std::strcmp(value, "0") == 0) {
      return GPUFrameFenceMode::DISABLED;
    }
    if (std::strcmp(value, "after_smoother") == 0) {
      return GPUFrameFenceMode::AFTER_SMOOTHER;
    }
    if (std::strcmp(value, "before_overlap") == 0) {
      return GPUFrameFenceMode::BEFORE_OVERLAP;
    }
    if (std::strcmp(value, "both") == 0) {
      return GPUFrameFenceMode::BOTH;
    }

    spdlog::warn(
      "[gpu-frame-probe] unknown GLIM_GPU_FRAME_FENCE='{}'; disabling frame fence",
      value);
    return GPUFrameFenceMode::DISABLED;
  }();

  return mode;
}

const char* gpu_frame_fence_mode_name() {
  switch (gpu_frame_fence_mode()) {
    case GPUFrameFenceMode::DISABLED:
      return "off";
    case GPUFrameFenceMode::AFTER_SMOOTHER:
      return "after_smoother";
    case GPUFrameFenceMode::BEFORE_OVERLAP:
      return "before_overlap";
    case GPUFrameFenceMode::BOTH:
      return "both";
  }

  return "off";
}

bool gpu_frame_trace_enabled() {
  static const bool enabled = env_flag_enabled("GLIM_GPU_FRAME_TRACE");
  return enabled;
}

void synchronize_gpu_frame_boundary(const char* boundary, int current) {
  const cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    spdlog::critical(
      "[gpu-frame-probe] cudaDeviceSynchronize failed boundary={} current={} error={}",
      boundary,
      current,
      cudaGetErrorString(status));
    throw std::runtime_error("GLIM GPU frame-boundary synchronization failed");
  }

  if (gpu_frame_trace_enabled()) {
    spdlog::info("[gpu-frame-trace] current={} phase={} fence=complete", current, boundary);
  }
}

}  // namespace

using Callbacks = OdometryEstimationCallbacks;

using gtsam::symbol_shorthand::B;  // IMU bias
using gtsam::symbol_shorthand::V;  // IMU velocity   (v_world_imu)
using gtsam::symbol_shorthand::X;  // IMU pose       (T_world_imu)

OdometryEstimationGPUParams::OdometryEstimationGPUParams() : OdometryEstimationIMUParams() {
  // odometry config
  Config config(GlobalConfig::get_config_path("config_odometry"));

  const std::string matching = config.param<std::string>("odometry_estimation", "gpu_matching_mode", "FRAME_TO_MODEL");
  if (matching == "KEYFRAME") {
    matching_mode = MatchingMode::KEYFRAME;
  } else {
    if (matching != "FRAME_TO_MODEL") {
      spdlog::warn("unknown gpu_matching_mode '{}'; using FRAME_TO_MODEL", matching);
    }
    matching_mode = MatchingMode::FRAME_TO_MODEL;
  }
  enable_surface_validation = config.param<bool>("odometry_estimation", "enable_surface_validation", false);

  max_iterations = config.param<int>("odometry_estimation", "max_iterations", 5);
  lru_thresh = config.param<int>("odometry_estimation", "lru_thresh", 100);
  target_downsampling_rate = config.param<double>("odometry_estimation", "target_downsampling_rate", 0.1);

  voxel_resolution = config.param<double>("odometry_estimation", "voxel_resolution", 0.5);
  voxel_resolution_max = config.param<double>("odometry_estimation", "voxel_resolution_max", voxel_resolution);
  voxel_resolution_dmin = config.param<double>("odometry_estimation", "voxel_resolution_dmin", 4.0);
  voxel_resolution_dmax = config.param<double>("odometry_estimation", "voxel_resolution_dmax", 12.0);

  voxelmap_levels = config.param<int>("odometry_estimation", "voxelmap_levels", 2);
  voxelmap_scaling_factor = config.param<double>("odometry_estimation", "voxelmap_scaling_factor", 2.0);

  vgicp_resolution = config.param<double>("odometry_estimation", "vgicp_resolution", voxel_resolution);
  vgicp_voxelmap_levels = config.param<int>("odometry_estimation", "vgicp_voxelmap_levels", voxelmap_levels);
  vgicp_voxelmap_scaling_factor = config.param<double>("odometry_estimation", "vgicp_voxelmap_scaling_factor", voxelmap_scaling_factor);

  max_num_keyframes = config.param<int>("odometry_estimation", "max_num_keyframes", 10);
  full_connection_window_size = config.param<int>("odometry_estimation", "full_connection_window_size", 3);

  const std::string strategy = config.param<std::string>("odometry_estimation", "keyframe_update_strategy", "OVERLAP");
  if (strategy == "OVERLAP") {
    keyframe_strategy = KeyframeUpdateStrategy::OVERLAP;
  } else if (strategy == "DISPLACEMENT") {
    keyframe_strategy = KeyframeUpdateStrategy::DISPLACEMENT;
  } else if (strategy == "ENTROPY") {
    keyframe_strategy = KeyframeUpdateStrategy::ENTROPY;
  } else if (strategy == "FIXED_INTERVAL_FIFO") {
    keyframe_strategy = KeyframeUpdateStrategy::FIXED_INTERVAL_FIFO;
  } else {
    spdlog::error("unknown keyframe update strategy {}", strategy);
    spdlog::warn("falling back to keyframe update strategy OVERLAP");
    keyframe_strategy = KeyframeUpdateStrategy::OVERLAP;
  }

  keyframe_min_overlap = config.param<double>("odometry_estimation", "keyframe_min_overlap", 0.1);
  keyframe_max_overlap = config.param<double>("odometry_estimation", "keyframe_max_overlap", 0.9);
  keyframe_delta_trans = config.param<double>("odometry_estimation", "keyframe_delta_trans", 1.0);
  keyframe_delta_rot = config.param<double>("odometry_estimation", "keyframe_delta_rot", 0.25);
  keyframe_entropy_thresh = config.param<double>("odometry_estimation", "keyframe_entropy_thresh", 0.99);
  keyframe_fixed_interval = std::max(1, config.param<int>("odometry_estimation", "keyframe_fixed_interval", 10));
}

OdometryEstimationGPUParams::~OdometryEstimationGPUParams() {}

OdometryEstimationGPU::OdometryEstimationGPU(const OdometryEstimationGPUParams& params) : OdometryEstimationIMU(std::make_unique<OdometryEstimationGPUParams>(params)) {
  entropy_num_frames = 0;
  entropy_running_average = 0.0;

  // Controls only the GPU odometry RoundRobin pool.
  // Sub-mapping and global-mapping use independent pools.
  int odometry_cuda_streams = 4;

  const char* streams_env = std::getenv("GTSAM_POINTS_CUDA_STREAMS");
  if (streams_env != nullptr) {
    char* parse_end = nullptr;
    const long requested = std::strtol(streams_env, &parse_end, 10);

    if (
      streams_env[0] != '\0' &&
      parse_end != streams_env &&
      *parse_end == '\0' &&
      requested >= 1 &&
      requested <= 64
    ) {
      odometry_cuda_streams = static_cast<int>(requested);
    } else {
      spdlog::warn(
        "[gpu-roundrobin] invalid GTSAM_POINTS_CUDA_STREAMS='{}'; "
        "using default=4 (valid range: 1..64)",
        streams_env);
    }
  }

  stream.reset(new gtsam_points::CUDAStream());
  stream_buffer_roundrobin.reset(
    new gtsam_points::StreamTempBufferRoundRobin(odometry_cuda_streams));

  spdlog::info(
    "[gpu-roundrobin] odometry_streams={}",
    odometry_cuda_streams);

  spdlog::info(
    "[gpu-frame-probe] frame_fence={} trace={}",
    gpu_frame_fence_mode_name(),
    gpu_frame_trace_enabled() ? "on" : "off");

  const auto gpu_params = static_cast<const OdometryEstimationGPUParams*>(this->params.get());
  last_T_target_imu.setIdentity();
  if (gpu_params->matching_mode == OdometryEstimationGPUParams::MatchingMode::FRAME_TO_MODEL) {
    target_voxelmaps.resize(gpu_params->vgicp_voxelmap_levels);
    for (int i = 0; i < gpu_params->vgicp_voxelmap_levels; i++) {
      const double resolution = gpu_params->vgicp_resolution * std::pow(gpu_params->vgicp_voxelmap_scaling_factor, i);
      target_voxelmaps[i] = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
      target_voxelmaps[i]->set_lru_horizon(gpu_params->lru_thresh);
    }
    spdlog::info(
      "[gpu-odom] matching=FRAME_TO_MODEL vgicp_resolution={} levels={} lru={} surface_validation={}",
      gpu_params->vgicp_resolution,
      gpu_params->vgicp_voxelmap_levels,
      gpu_params->lru_thresh,
      gpu_params->enable_surface_validation);
  } else if (gpu_params->keyframe_strategy == OdometryEstimationGPUParams::KeyframeUpdateStrategy::FIXED_INTERVAL_FIFO) {
    spdlog::info(
      "[keyframe-policy] strategy=FIXED_INTERVAL_FIFO interval={} max_keyframes={} overlap_gpu=off",
      gpu_params->keyframe_fixed_interval,
      gpu_params->max_num_keyframes);
  }
}

OdometryEstimationGPU::~OdometryEstimationGPU() {
  frames.clear();
  keyframes.clear();
  smoother.reset();
}

void OdometryEstimationGPU::create_frame(EstimationFrame::Ptr& new_frame) {
  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());

  new_frame->frame = gtsam_points::PointCloudGPU::clone(*new_frame->frame, *stream);
  stream->sync();

  if (params->matching_mode == OdometryEstimationGPUParams::MatchingMode::FRAME_TO_MODEL) {
    return;
  }

  // Adaptively determine the voxel resolution based on the median distance
  const int max_scan_count = 256;
  const double dist_median = gtsam_points::median_distance(new_frame->frame, max_scan_count);
  const double p = std::max(0.0, std::min(1.0, (dist_median - params->voxel_resolution_dmin) / (params->voxel_resolution_dmax - params->voxel_resolution_dmin)));
  const double base_resolution = params->voxel_resolution + p * (params->voxel_resolution_max - params->voxel_resolution);

  for (int i = 0; i < params->voxelmap_levels; i++) {
    if (!new_frame->frame->size()) {
      break;
    }

    const double resolution = base_resolution * std::pow(params->voxelmap_scaling_factor, i);
    auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(resolution, 8192 * 2, 10, 1e-3, *stream);
    voxelmap->insert(*new_frame->frame);
    new_frame->voxelmaps.push_back(voxelmap);
  }
}

void OdometryEstimationGPU::update_frames(const int current, const gtsam::NonlinearFactorGraph& new_factors) {
  const auto fence_mode = gpu_frame_fence_mode();
  if (fence_mode == GPUFrameFenceMode::AFTER_SMOOTHER || fence_mode == GPUFrameFenceMode::BOTH) {
    synchronize_gpu_frame_boundary("after_smoother", current);
  }

  if (gpu_frame_trace_enabled()) {
    logger->info(
      "[gpu-frame-trace] current={} stamp={:.9f} phase=before_estimates factors={} keyframes={}",
      current,
      frames[current]->stamp,
      new_factors.size(),
      keyframes.size());
  }

  OdometryEstimationIMU::update_frames(current, new_factors);

  if (gpu_frame_trace_enabled()) {
    const auto& translation = frames[current]->T_world_imu.translation();
    logger->info(
      "[gpu-frame-trace] current={} stamp={:.9f} phase=after_estimates xyz=[{:.6f},{:.6f},{:.6f}] keyframes={}",
      current,
      frames[current]->stamp,
      translation.x(),
      translation.y(),
      translation.z(),
      keyframes.size());
  }

  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());
  if (params->matching_mode == OdometryEstimationGPUParams::MatchingMode::FRAME_TO_MODEL) {
    return;
  }

  if (fence_mode == GPUFrameFenceMode::BEFORE_OVERLAP || fence_mode == GPUFrameFenceMode::BOTH) {
    synchronize_gpu_frame_boundary("before_overlap", current);
  }

  switch (params->keyframe_strategy) {
    case OdometryEstimationGPUParams::KeyframeUpdateStrategy::OVERLAP:
      update_keyframes_overlap(current);
      break;
    case OdometryEstimationGPUParams::KeyframeUpdateStrategy::DISPLACEMENT:
      update_keyframes_displacement(current);
      break;
    case OdometryEstimationGPUParams::KeyframeUpdateStrategy::ENTROPY:
      update_keyframes_entropy(new_factors, current);
      break;
    case OdometryEstimationGPUParams::KeyframeUpdateStrategy::FIXED_INTERVAL_FIFO:
      update_keyframes_fixed_interval_fifo(current);
      break;
  }

  Callbacks::on_update_keyframes(keyframes);
}

gtsam::NonlinearFactorGraph OdometryEstimationGPU::create_factors(const int current, const gtsam_points::shared_ptr<gtsam::ImuFactor>& imu_factor, gtsam::Values& new_values) {
  (void)imu_factor;
  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());
  if (params->matching_mode == OdometryEstimationGPUParams::MatchingMode::FRAME_TO_MODEL) {
    return create_factors_frame_to_model(current, new_values);
  }
  return create_factors_keyframe(current);
}

gtsam_points::GaussianVoxelMapGPU::Ptr gpu_voxelmap_from_cpu(
  const gtsam_points::GaussianVoxelMapCPU& cpu,
  gtsam_points::CUDAStream& stream) {
  auto gpu = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(static_cast<float>(cpu.voxel_resolution()), 8192 * 2, 10, 1e-3, stream);
  const size_t n = cpu.num_voxels();
  if (n == 0) {
    return gpu;
  }

  std::vector<Eigen::Vector3i> coords(n);
  std::vector<int> num_points(n);
  std::vector<Eigen::Vector3f> means(n);
  std::vector<Eigen::Matrix3f> covs(n);
  std::vector<float> intensities(n, 0.0f);
  for (size_t i = 0; i < n; i++) {
    const auto& voxel = cpu.lookup_voxel(static_cast<int>(i));
    coords[i] = cpu.voxel_coord(voxel.mean);
    num_points[i] = static_cast<int>(voxel.num_points);
    means[i] = voxel.mean.head<3>().cast<float>();
    covs[i] = voxel.cov.topLeftCorner<3, 3>().cast<float>();
    intensities[i] = static_cast<float>(voxel.intensity);
  }
  gpu->upload_voxels(coords, num_points, means, covs, intensities);
  return gpu;
}

gtsam::NonlinearFactorGraph OdometryEstimationGPU::create_factors_frame_to_model(const int current, gtsam::Values& new_values) {
  const auto params = static_cast<const OdometryEstimationGPUParams*>(this->params.get());
  const int last = current - 1;

  if (current == 0) {
    last_T_target_imu = frames[current]->T_world_imu;
    update_target(current, frames[current]->T_world_imu);
    return gtsam::NonlinearFactorGraph();
  }

  const Eigen::Isometry3d pred_T_last_current = frames[last]->T_world_imu.inverse() * frames[current]->T_world_imu;
  const Eigen::Isometry3d pred_T_target_imu = last_T_target_imu * pred_T_last_current;

  gtsam::Values values;
  values.insert(X(current), gtsam::Pose3(pred_T_target_imu.matrix()));

  auto stream_buffer = stream_buffer_roundrobin->get_stream_buffer();
  const auto& factor_stream = stream_buffer.first;
  const auto& buffer = stream_buffer.second;

  gtsam::NonlinearFactorGraph matching_cost_factors;
  std::vector<gtsam_points::GaussianVoxelMapGPU::Ptr> gpu_targets;
  gpu_targets.reserve(target_voxelmaps.size());
  for (const auto& voxelmap : target_voxelmaps) {
    if (!voxelmap || voxelmap->num_voxels() == 0) {
      continue;
    }
    auto gpu_map = gpu_voxelmap_from_cpu(*voxelmap, *stream);
    gpu_targets.push_back(gpu_map);
    auto vgicp_factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(gtsam::Pose3(), X(current), gpu_map, frames[current]->frame, factor_stream, buffer);
    vgicp_factor->set_enable_surface_validation(params->enable_surface_validation);
    matching_cost_factors.add(vgicp_factor);
  }

  gtsam::NonlinearFactorGraph graph;
  graph.add(matching_cost_factors);

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.setMaxIterations(params->max_iterations);
  lm_params.setAbsoluteErrorTol(0.1);

  gtsam::Pose3 last_estimate = values.at<gtsam::Pose3>(X(current));
  lm_params.termination_criteria = [&](const gtsam::Values& values) {
    const gtsam::Pose3 current_pose = values.at<gtsam::Pose3>(X(current));
    const gtsam::Pose3 delta = last_estimate.inverse() * current_pose;

    const double delta_t = delta.translation().norm();
    const double delta_r = Eigen::AngleAxisd(delta.rotation().matrix()).angle();
    last_estimate = current_pose;

    if (delta_t < 1e-10 && delta_r < 1e-10) {
      return false;
    }
    return delta_t < 1e-3 && delta_r < 1e-3 * M_PI / 180.0;
  };

  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
#ifdef GTSAM_USE_TBB
  auto arena = static_cast<tbb::task_arena*>(this->tbb_task_arena.get());
  arena->execute([&] {
#endif
    values = optimizer.optimize();
#ifdef GTSAM_USE_TBB
  });
#endif

  const Eigen::Isometry3d T_target_imu = Eigen::Isometry3d(values.at<gtsam::Pose3>(X(current)).matrix());
  Eigen::Isometry3d T_last_current = last_T_target_imu.inverse() * T_target_imu;
  T_last_current.linear() = Eigen::Quaterniond(T_last_current.linear()).normalized().toRotationMatrix();
  frames[current]->T_world_imu = frames[last]->T_world_imu * T_last_current;
  new_values.insert_or_assign(X(current), gtsam::Pose3(frames[current]->T_world_imu.matrix()));

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(last), X(current), gtsam::Pose3(T_last_current.matrix()), gtsam::noiseModel::Isotropic::Precision(6, 1e3));
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(current), gtsam::Pose3(T_target_imu.matrix()), gtsam::noiseModel::Isotropic::Precision(6, 1e3));

  update_target(current, T_target_imu);
  last_T_target_imu = T_target_imu;
  return factors;
}

void OdometryEstimationGPU::update_target(const int current, const Eigen::Isometry3d& T_target_imu) {
  const auto params = static_cast<const OdometryEstimationGPUParams*>(this->params.get());
  auto frame = frames[current]->frame;
  if (current >= 5) {
    frame = gtsam_points::random_sampling(frames[current]->frame, params->target_downsampling_rate, mt);
  }

  auto transformed = gtsam_points::transform(frame, T_target_imu);
  for (auto& target_voxelmap : target_voxelmaps) {
    target_voxelmap->insert(*transformed);
  }
}

gtsam::NonlinearFactorGraph OdometryEstimationGPU::create_factors_keyframe(const int current) {
  if (current == 0 || !frames[current]->frame->size()) {
    return gtsam::NonlinearFactorGraph();
  }

  const auto create_binary_factor = [this](
                                      gtsam::NonlinearFactorGraph& factors,
                                      gtsam::Key target_key,
                                      gtsam::Key source_key,
                                      const glim::EstimationFrame::ConstPtr& target,
                                      const glim::EstimationFrame::ConstPtr& source) {
    auto stream_buffer = stream_buffer_roundrobin->get_stream_buffer();
    const auto& stream = stream_buffer.first;
    const auto& buffer = stream_buffer.second;

    for (const auto& voxelmap : target->voxelmaps) {
      auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(target_key, source_key, voxelmap, source->frame, stream, buffer);
      factor->set_enable_surface_validation(static_cast<OdometryEstimationGPUParams*>(this->params.get())->enable_surface_validation);
      factors.add(factor);
    }
  };

  const auto create_unary_factor = [this](
                                     gtsam::NonlinearFactorGraph& factors,
                                     const gtsam::Pose3& fixed_target_pose,
                                     gtsam::Key source_key,
                                     const glim::EstimationFrame::ConstPtr& target,
                                     const glim::EstimationFrame::ConstPtr& source) {
    auto stream_buffer = stream_buffer_roundrobin->get_stream_buffer();
    const auto& stream = stream_buffer.first;
    const auto& buffer = stream_buffer.second;

    for (const auto& voxelmap : target->voxelmaps) {
      auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(fixed_target_pose, source_key, voxelmap, source->frame, stream, buffer);
      factor->set_enable_surface_validation(static_cast<OdometryEstimationGPUParams*>(this->params.get())->enable_surface_validation);
      factors.add(factor);
    }
  };

  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());

  gtsam::NonlinearFactorGraph factors;
  if (current == 0) {
    return factors;
  }

  // There must be at least one factor between consecutive frames
  for (int target = current - params->full_connection_window_size; target < current; target++) {
    if (target < 0) {
      continue;
    }

    create_binary_factor(factors, X(target), X(current), frames[target], frames[current]);
  }

  for (const auto& keyframe : keyframes) {
    if (keyframe->id >= current - params->full_connection_window_size) {
      // There already exists a factor
      continue;
    }

    double span = frames[current]->stamp - keyframe->stamp;
    if (span > params->smoother_lag - 0.1 || !frames[keyframe->id]) {
      // Create unary factor
      const gtsam::Pose3 key_T_world_imu(keyframe->T_world_imu.matrix());
      create_unary_factor(factors, key_T_world_imu, X(current), keyframe, frames[current]);
    } else {
      // Create binary factor
      const int target = keyframe->id;
      create_binary_factor(factors, X(target), X(current), frames[target], frames[current]);
    }
  }

  return factors;
}

/**
 * @brief Deterministic keyframe management based only on frame indices.
 *
 * A keyframe is inserted after keyframe_fixed_interval valid frames have
 * elapsed since the most recently inserted keyframe. When the active set is
 * full, the oldest keyframe is removed. No overlap or pose-dependent metric
 * is used for insertion or eviction decisions.
 */
void OdometryEstimationGPU::update_keyframes_fixed_interval_fifo(int current) {
  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());

  if (!frames[current]->frame->size()) {
    return;
  }

  if (keyframes.empty()) {
    keyframes.push_back(frames[current]);
    if (gpu_frame_trace_enabled()) {
      logger->info(
        "[gpu-frame-trace] current={} stamp={:.9f} phase=fixed_interval_fifo action=seed interval={} keyframes=1",
        current,
        frames[current]->stamp,
        params->keyframe_fixed_interval);
    }
    return;
  }

  const int frames_since_last_keyframe = current - keyframes.back()->id;
  if (frames_since_last_keyframe < params->keyframe_fixed_interval) {
    if (gpu_frame_trace_enabled()) {
      logger->info(
        "[gpu-frame-trace] current={} stamp={:.9f} phase=fixed_interval_fifo action=retain since_last={} interval={} keyframes={}",
        current,
        frames[current]->stamp,
        frames_since_last_keyframe,
        params->keyframe_fixed_interval,
        keyframes.size());
    }
    return;
  }

  keyframes.push_back(frames[current]);

  std::vector<EstimationFrame::ConstPtr> marginalized_keyframes;
  const int max_keyframes = std::max(1, params->max_num_keyframes);
  while (keyframes.size() > max_keyframes) {
    marginalized_keyframes.push_back(keyframes.front());
    keyframes.erase(keyframes.begin());
  }

  if (gpu_frame_trace_enabled()) {
    const int evicted_id = marginalized_keyframes.empty() ? -1 : marginalized_keyframes.back()->id;
    logger->info(
      "[gpu-frame-trace] current={} stamp={:.9f} phase=fixed_interval_fifo action=add interval={} evicted={} keyframes={}",
      current,
      frames[current]->stamp,
      params->keyframe_fixed_interval,
      evicted_id,
      keyframes.size());
  }

  if (!marginalized_keyframes.empty()) {
    Callbacks::on_marginalized_keyframes(marginalized_keyframes);
  }
}

/**
 * @brief Keyframe management based on an overlap metric
 * @ref   Koide et al., "Globally Consistent and Tightly Coupled 3D LiDAR Inertial Mapping", ICRA2022
 */
void OdometryEstimationGPU::update_keyframes_overlap(int current) {
  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());

  if (!frames[current]->frame->size()) {
    return;
  }

  if (keyframes.empty()) {
    keyframes.push_back(frames[current]);
    if (gpu_frame_trace_enabled()) {
      logger->info(
        "[gpu-frame-trace] current={} stamp={:.9f} phase=overlap action=seed keyframes=1",
        current,
        frames[current]->stamp);
    }
    return;
  }

  std::vector<gtsam_points::GaussianVoxelMap::ConstPtr> keyframes_(keyframes.size());
  std::vector<Eigen::Isometry3d> delta_from_keyframes(keyframes.size());
  for (int i = 0; i < keyframes.size(); i++) {
    keyframes_[i] = keyframes[i]->voxelmaps.back();
    delta_from_keyframes[i] = keyframes[i]->T_world_imu.inverse() * frames[current]->T_world_imu;
  }

  const double overlap = gtsam_points::overlap_gpu(keyframes_, frames[current]->frame, delta_from_keyframes, *stream);
  if (overlap > params->keyframe_max_overlap) {
    if (gpu_frame_trace_enabled()) {
      logger->info(
        "[gpu-frame-trace] current={} stamp={:.9f} phase=overlap value={:.9f} action=retain keyframes={}",
        current,
        frames[current]->stamp,
        overlap,
        keyframes.size());
    }
    return;
  }

  const auto& new_keyframe = frames[current];
  keyframes.push_back(new_keyframe);

  if (gpu_frame_trace_enabled()) {
    logger->info(
      "[gpu-frame-trace] current={} stamp={:.9f} phase=overlap value={:.9f} action=add keyframes={}",
      current,
      frames[current]->stamp,
      overlap,
      keyframes.size());
  }

  if (keyframes.size() <= params->max_num_keyframes) {
    return;
  }

  std::vector<EstimationFrame::ConstPtr> marginalized_keyframes;

  // Remove keyframes without overlap to the new keyframe
  for (int i = 0; i < keyframes.size(); i++) {
    const Eigen::Isometry3d delta = keyframes[i]->T_world_imu.inverse() * new_keyframe->T_world_imu;
    const double overlap = gtsam_points::overlap_gpu(keyframes[i]->voxelmaps.back(), new_keyframe->frame, delta, *stream);
    if (overlap < params->keyframe_min_overlap) {
      marginalized_keyframes.push_back(keyframes[i]);
      keyframes.erase(keyframes.begin() + i);
      i--;
    }
  }

  if (keyframes.size() <= params->max_num_keyframes) {
    Callbacks::on_marginalized_keyframes(marginalized_keyframes);
    return;
  }

  // Remove the keyframe with the minimum score
  std::vector<double> scores(keyframes.size() - 1, 0.0);
  for (int i = 0; i < keyframes.size() - 1; i++) {
    const auto& keyframe = keyframes[i];
    const double overlap_latest = gtsam_points::overlap_gpu(keyframe->voxelmaps.back(), new_keyframe->frame, keyframe->T_world_imu.inverse() * new_keyframe->T_world_imu, *stream);

    std::vector<gtsam_points::GaussianVoxelMap::ConstPtr> other_keyframes;
    std::vector<Eigen::Isometry3d> delta_from_others;
    for (int j = 0; j < keyframes.size() - 1; j++) {
      if (i == j) {
        continue;
      }

      const auto& other = keyframes[j];
      other_keyframes.push_back(other->voxelmaps.back());
      delta_from_others.push_back(other->T_world_imu.inverse() * keyframe->T_world_imu);
    }

    const double overlap_others = gtsam_points::overlap_gpu(other_keyframes, keyframe->frame, delta_from_others, *stream);
    scores[i] = overlap_latest * (1.0 - overlap_others);
  }

  double min_score = scores[0];
  int frame_to_eliminate = 0;
  for (int i = 1; i < scores.size(); i++) {
    if (scores[i] < min_score) {
      min_score = scores[i];
      frame_to_eliminate = i;
    }
  }

  marginalized_keyframes.push_back(keyframes[frame_to_eliminate]);
  keyframes.erase(keyframes.begin() + frame_to_eliminate);
  Callbacks::on_marginalized_keyframes(marginalized_keyframes);
}

/**
 * @brief Keyframe management based on displacement criteria
 * @ref   Engel et al., "Direct Sparse Odometry", IEEE Trans. PAMI, 2018
 */
void OdometryEstimationGPU::update_keyframes_displacement(int current) {
  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());

  if (keyframes.empty()) {
    keyframes.push_back(frames[current]);
    return;
  }

  const Eigen::Isometry3d delta_from_last = keyframes.back()->T_world_imu.inverse() * frames[current]->T_world_imu;
  const double delta_trans = delta_from_last.translation().norm();
  const double delta_rot = Eigen::AngleAxisd(delta_from_last.linear()).angle();

  if (delta_trans < params->keyframe_delta_trans && delta_rot < params->keyframe_delta_rot) {
    return;
  }

  const auto& new_keyframe = frames[current];
  keyframes.push_back(new_keyframe);

  if (keyframes.size() <= params->max_num_keyframes) {
    return;
  }

  for (int i = 0; i < keyframes.size() - 1; i++) {
    const Eigen::Isometry3d delta = keyframes[i]->T_world_imu.inverse() * new_keyframe->T_world_imu;
    const double overlap = gtsam_points::overlap_gpu(keyframes[i]->voxelmaps.back(), new_keyframe->frame, delta, *stream);

    if (overlap < 0.01) {
      std::vector<EstimationFrame::ConstPtr> marginalized_keyframes;
      marginalized_keyframes.push_back(keyframes[i]);
      keyframes.erase(keyframes.begin() + i);
      Callbacks::on_marginalized_keyframes(marginalized_keyframes);
      return;
    }
  }

  const int leave_window = 2;
  const double eps = 1e-3;
  std::vector<double> scores(keyframes.size() - 1, 0.0);
  for (int i = leave_window; i < keyframes.size() - 1; i++) {
    double sum_inv_dist = 0.0;
    for (int j = 0; j < keyframes.size() - 1; j++) {
      if (i == j) {
        continue;
      }

      const double dist = (keyframes[i]->T_world_imu.translation() - keyframes[j]->T_world_imu.translation()).norm();
      sum_inv_dist += 1.0 / (dist + eps);
    }

    const double d0 = (keyframes[i]->T_world_imu.translation() - new_keyframe->T_world_imu.translation()).norm();
    scores[i] = std::sqrt(d0) * sum_inv_dist;
  }

  const auto max_score_loc = std::max_element(scores.begin(), scores.end());
  const int max_score_index = std::distance(scores.begin(), max_score_loc);

  std::vector<EstimationFrame::ConstPtr> marginalized_keyframes;
  marginalized_keyframes.push_back(keyframes[max_score_index]);
  keyframes.erase(keyframes.begin() + max_score_index);
  Callbacks::on_marginalized_keyframes(marginalized_keyframes);
}

/**
 * @brief Keyframe management based on entropy measure
 * @ref   Kuo et al., "Redesigning SLAM for Arbitrary Multi-Camera Systems", ICRA2020
 */
void OdometryEstimationGPU::update_keyframes_entropy(const gtsam::NonlinearFactorGraph& matching_cost_factors, int current) {
  const auto params = static_cast<OdometryEstimationGPUParams*>(this->params.get());

  gtsam::Values values = smoother->calculateEstimate();

  gtsam::NonlinearFactorGraph valid_factors;
  for (const auto& factor : matching_cost_factors) {
    bool valid = std::all_of(factor->keys().begin(), factor->keys().end(), [&](const gtsam::Key key) { return values.exists(key); });
    if (!valid) {
      continue;
    }

    valid_factors.push_back(factor->clone());
  }

  gtsam_points::NonlinearFactorSetGPU factor_set;
  factor_set.add(valid_factors);
  factor_set.linearize(values);
  auto linearized = valid_factors.linearize(values);

  gtsam::Matrix6 H = linearized->hessianBlockDiagonal()[X(current)];
  double negative_entropy = std::log(H.determinant());

  entropy_num_frames++;
  entropy_running_average += (negative_entropy - entropy_running_average) / entropy_num_frames;

  if (!keyframes.empty() && negative_entropy > entropy_running_average * params->keyframe_entropy_thresh) {
    return;
  }

  entropy_num_frames = 0;
  entropy_running_average = 0.0;

  const auto& new_keyframe = frames[current];
  keyframes.push_back(new_keyframe);

  if (keyframes.size() <= params->max_num_keyframes) {
    return;
  }

  std::vector<EstimationFrame::ConstPtr> marginalized_keyframes;
  marginalized_keyframes.push_back(keyframes.front());
  keyframes.erase(keyframes.begin());
  Callbacks::on_marginalized_keyframes(marginalized_keyframes);
}

}  // namespace glim

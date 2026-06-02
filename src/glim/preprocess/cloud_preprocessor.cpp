#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/preprocess/callbacks.hpp>

#include <fstream>
#include <iostream>
#include <limits>
#include <spdlog/spdlog.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/parallelism.hpp>

#include <glim/util/config.hpp>
#include <glim/util/convert_to_string.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#endif

namespace glim {

namespace {

std::vector<std::size_t> find_nearest_raw_indices(
    const std::vector<Eigen::Vector4d>& raw_points,
    const std::vector<Eigen::Vector4d>& processed_points) {
  std::vector<std::size_t> indices(processed_points.size(), 0);

  if (raw_points.empty() || processed_points.empty()) {
    return indices;
  }

  gtsam_points::KdTree tree(raw_points.data(), static_cast<int>(raw_points.size()));

  for (std::size_t i = 0; i < processed_points.size(); i++) {
    std::size_t nearest_index = 0;
    double nearest_squared_distance = std::numeric_limits<double>::max();

    const auto found = tree.knn_search(
      processed_points[i].data(),
      1,
      &nearest_index,
      &nearest_squared_distance);

    if (found > 0) {
      indices[i] = nearest_index;
    }
  }

  return indices;
}

}  // namespace


CloudPreprocessorParams::CloudPreprocessorParams() {
  Config config(GlobalConfig::get_config_path("config_preprocess"));
  Config sensor_config(GlobalConfig::get_config_path("config_sensors"));

  global_shutter = sensor_config.param<bool>("sensors", "global_shutter_lidar", false);

  distance_near_thresh = config.param<double>("preprocess", "distance_near_thresh", 1.0);
  distance_far_thresh = config.param<double>("preprocess", "distance_far_thresh", 100.0);
  use_random_grid_downsampling = config.param<bool>("preprocess", "use_random_grid_downsampling", false);
  downsample_resolution = config.param<double>("preprocess", "downsample_resolution", 0.15);
  downsample_target = config.param<int>("preprocess", "random_downsample_target", 0);
  downsample_rate = config.param<double>("preprocess", "random_downsample_rate", 0.3);
  enable_outlier_removal = config.param<bool>("preprocess", "enable_outlier_removal", false);
  outlier_removal_k = config.param<int>("preprocess", "outlier_removal_k", 10);
  outlier_std_mul_factor = config.param<double>("preprocess", "outlier_std_mul_factor", 2.0);

  enable_cropbox_filter = config.param<bool>("preprocess", "enable_cropbox_filter", false);
  crop_bbox_frame = "lidar";
  crop_bbox_min.setZero();
  crop_bbox_max.setZero();

  if (enable_cropbox_filter) {
    Eigen::Isometry3d T_lidar_imu = sensor_config.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity());
    T_imu_lidar = T_lidar_imu.inverse();

    crop_bbox_frame = config.param<std::string>("preprocess", "crop_bbox_frame", "lidar");
    crop_bbox_min = config.param<Eigen::Vector3d>("preprocess", "crop_bbox_min", Eigen::Vector3d(0.0, 0.0, 0.0));
    crop_bbox_max = config.param<Eigen::Vector3d>("preprocess", "crop_bbox_max", Eigen::Vector3d(0.0, 0.0, 0.0));

    if (crop_bbox_frame != "lidar" && crop_bbox_frame != "imu") {
      throw std::runtime_error(fmt::format("Unsupported crop bbox frame: {}", crop_bbox_frame));
    } else if ((crop_bbox_min.array() > crop_bbox_max.array()).any()) {
      throw std::runtime_error(fmt::format("Misconfigured bbox: min={}, max={}", convert_to_string(crop_bbox_min), convert_to_string(crop_bbox_max)));
    }
  }

  k_correspondences = config.param<int>("preprocess", "k_correspondences", 8);

  num_threads = config.param<int>("preprocess", "num_threads", 2);

  scan_guard.enable = config.param<bool>("scan_guard", "enable", true);
  scan_guard.min_raw_points = config.param<int>("scan_guard", "min_raw_points", 100);
  scan_guard.min_filtered_points = config.param<int>("scan_guard", "min_filtered_points", 80);
  scan_guard.min_effective_points = config.param<int>("scan_guard", "min_effective_points", 50);
  scan_guard.max_empty_scan_burst = config.param<int>("scan_guard", "max_empty_scan_burst", 10);
  scan_guard.drop_empty_frame = config.param<bool>("scan_guard", "drop_empty_frame", true);
  scan_guard.allow_imu_only_prediction = config.param<bool>("scan_guard", "allow_imu_only_prediction", true);
  scan_guard.publish_diagnostics = config.param<bool>("scan_guard", "publish_diagnostics", true);
}

CloudPreprocessorParams::~CloudPreprocessorParams() {}

CloudPreprocessor::CloudPreprocessor(const CloudPreprocessorParams& params) : params(params) {
#ifdef GTSAM_POINTS_USE_TBB
  if (gtsam_points::is_tbb_default()) {
    tbb_task_arena.reset(new tbb::task_arena(params.num_threads));
  }
#endif

  spdlog::info(
    "[scan_guard] enable={} min_raw={} min_filtered={} min_effective={} max_burst={} drop_empty_frame={}",
    params.scan_guard.enable,
    params.scan_guard.min_raw_points,
    params.scan_guard.min_filtered_points,
    params.scan_guard.min_effective_points,
    params.scan_guard.max_empty_scan_burst,
    params.scan_guard.drop_empty_frame);
}

CloudPreprocessor::~CloudPreprocessor() {}

PreprocessedFrame::Ptr CloudPreprocessor::preprocess(const RawPoints::ConstPtr& raw_points) {
  if (!raw_points) {
    spdlog::warn("[scan_guard] raw_points is nullptr, skip preprocessing");
    return nullptr;
  }

  auto preprocessed = preprocess_impl(raw_points);

  if (!preprocessed) {
    spdlog::warn(
      "[scan_guard] preprocessor returned nullptr for stamp={} raw_points={}",
      raw_points->stamp,
      raw_points->size());
  }

  return preprocessed;
}

PreprocessedFrame::Ptr CloudPreprocessor::preprocess_impl(const RawPoints::ConstPtr& raw_points) {
  static thread_local ScanGuard scan_guard_local;
  scan_guard_local.set_config(params.scan_guard);

  const double raw_stamp = raw_points ? raw_points->stamp : 0.0;
  const std::size_t raw_size = raw_points ? raw_points->size() : 0;

  const auto raw_guard_status = scan_guard_local.evaluate_raw(raw_stamp, raw_size);
  if (!raw_guard_status.accepted && params.scan_guard.drop_empty_frame) {
    spdlog::warn(
      "[scan_guard] skip raw scan stamp={} raw={} reason={} burst={}",
      raw_guard_status.stamp,
      raw_guard_status.raw_points,
      raw_guard_status.reason_text,
      raw_guard_status.consecutive_skipped_frames);
    return nullptr;
  }

  spdlog::trace("preprocessing input: {} points", raw_size);

  gtsam_points::PointCloudCPU::Ptr frame = std::make_shared<gtsam_points::PointCloudCPU>();
  frame->add_times(raw_points->times);
  frame->add_points(raw_points->points);
  if (raw_points->intensities.size()) {
    frame->add_intensities(raw_points->intensities);
  }
  PreprocessCallbacks::on_preprocessing_begin(frame);

  // Downsampling
  if (params.use_random_grid_downsampling) {
    const double rate = params.downsample_target > 0 ? static_cast<double>(params.downsample_target) / frame->size() : params.downsample_rate;
    frame = gtsam_points::randomgrid_sampling(frame, params.downsample_resolution, rate, mt, params.num_threads);
  } else {
    frame = gtsam_points::voxelgrid_sampling(frame, params.downsample_resolution, params.num_threads);
  }
  PreprocessCallbacks::on_downsampling_finished(frame);

  if (frame->size() < 100) {
    spdlog::warn("too few points in the downsampled cloud ({} points)", frame->size());
  }

  // Distance filter
  std::vector<int> indices;
  indices.reserve(frame->size());
  double squared_distance_near_thresh = params.distance_near_thresh * params.distance_near_thresh;
  double squared_distance_far_thresh = params.distance_far_thresh * params.distance_far_thresh;

  for (int i = 0; i < frame->size(); i++) {
    const bool is_finite = frame->points[i].allFinite();
    const double squared_dist = (Eigen::Vector4d() << frame->points[i].head<3>(), 0.0).finished().squaredNorm();
    if (squared_dist > squared_distance_near_thresh && squared_dist < squared_distance_far_thresh && is_finite) {
      indices.push_back(i);
    }
  }

  if (indices.size() < 100) {
    spdlog::warn("too few points in the filtered cloud ({} points)", indices.size());
  }

  // Sort by time
  std::sort(indices.begin(), indices.end(), [&](const int lhs, const int rhs) { return frame->times[lhs] < frame->times[rhs]; });
  frame = gtsam_points::sample(frame, indices);

  if (params.global_shutter) {
    std::fill(frame->times, frame->times + frame->size(), 0.0);
  }

  // Cropbox filter
  if (params.enable_cropbox_filter) {
    if (params.crop_bbox_frame == "lidar") {
      auto is_inside_bbox = [&](const Eigen::Vector3d& p_lidar) {
        return (p_lidar.array() >= params.crop_bbox_min.array()).all() && (p_lidar.array() <= params.crop_bbox_max.array()).all();
      };

      frame = gtsam_points::filter(frame, [&](const auto& pt) { return !is_inside_bbox(pt.template head<3>()); });

    } else if (params.crop_bbox_frame == "imu") {
      auto is_inside_bbox = [&](const Eigen::Vector3d& p_lidar) {
        const auto p_imu = params.T_imu_lidar * p_lidar;
        return (p_imu.array() >= params.crop_bbox_min.array()).all() && (p_imu.array() <= params.crop_bbox_max.array()).all();
      };

      frame = gtsam_points::filter(frame, [&](const auto& pt) { return !is_inside_bbox(pt.template head<3>()); });

    } else {
      throw std::runtime_error(fmt::format("Unsupported crop bbox frame: {}", params.crop_bbox_frame));
    }
  }

  // Outlier removal
  if (params.enable_outlier_removal) {
    frame = gtsam_points::remove_outliers(frame, params.outlier_removal_k, params.outlier_std_mul_factor, params.num_threads);
  }

  PreprocessCallbacks::on_filtering_finished(frame);

  const auto filtered_guard_status = scan_guard_local.evaluate_filtered(
    raw_points->stamp,
    raw_points->size(),
    frame ? frame->size() : 0,
    frame ? frame->size() : 0);

  if (!filtered_guard_status.accepted && params.scan_guard.drop_empty_frame) {
    spdlog::warn(
      "[scan_guard] skip filtered scan stamp={} raw={} filtered={} effective={} reason={} burst={}",
      filtered_guard_status.stamp,
      filtered_guard_status.raw_points,
      filtered_guard_status.filtered_points,
      filtered_guard_status.effective_points,
      filtered_guard_status.reason_text,
      filtered_guard_status.consecutive_skipped_frames);
    return nullptr;
  }

  // Create a preprocessed frame
  PreprocessedFrame::Ptr preprocessed(new PreprocessedFrame);
  preprocessed->stamp = raw_points->stamp;
  preprocessed->scan_end_time = frame->size() ? raw_points->stamp + frame->times[frame->size() - 1] : raw_points->stamp;

  preprocessed->times.assign(frame->times, frame->times + frame->size());
  preprocessed->points.assign(frame->points, frame->points + frame->size());
  if (frame->intensities) {
    preprocessed->intensities.assign(frame->intensities, frame->intensities + frame->size());
  }

  // PR290-style attributes.
  //
  // gtsam_points preserves standard times internally, but intensity can be lost
  // or zeroed by some sampling/filtering paths.  Arbitrary fields such as
  // line/tag/scanner_id/rgb are also not preserved by gtsam_points.
  //
  // Recover attributes by mapping every final preprocessed point to the nearest
  // source raw point.  This keeps legacy PreprocessedFrame::intensities valid for
  // downstream SubMap / GlobalMapping export.
  std::vector<std::size_t> nearest_raw_indices;

  if (!raw_points->attrs.empty()) {
    std::string attrs_error;
    if (raw_points->attrs.validate(raw_points->points.size(), &attrs_error)) {
      nearest_raw_indices =
        find_nearest_raw_indices(raw_points->points, preprocessed->points);

      preprocessed->attrs = raw_points->attrs.sample(nearest_raw_indices);

      if (raw_points->attrs.intensity) {
        preprocessed->intensities.resize(preprocessed->size());

        for (std::size_t i = 0; i < nearest_raw_indices.size(); i++) {
          preprocessed->intensities[i] =
            static_cast<double>((*raw_points->attrs.intensity)[nearest_raw_indices[i]]);
        }
      } else if (!raw_points->intensities.empty()) {
        preprocessed->intensities.resize(preprocessed->size());

        for (std::size_t i = 0; i < nearest_raw_indices.size(); i++) {
          preprocessed->intensities[i] =
            raw_points->intensities[nearest_raw_indices[i]];
        }
      }
    } else {
      spdlog::warn("skip raw PointAttributes propagation: {}", attrs_error);
    }
  }

  // Timestamp from final frame is authoritative after TimeKeeper/preprocessing.
  preprocessed->attrs.timestamp = preprocessed->times;

  // Intensity must mirror final legacy PreprocessedFrame::intensities because
  // SubMap / GlobalMapping still consume this legacy field.
  if (!preprocessed->intensities.empty()) {
    auto& dst = preprocessed->attrs.intensity.emplace();
    dst.resize(preprocessed->intensities.size());

    for (std::size_t i = 0; i < preprocessed->intensities.size(); i++) {
      dst[i] = static_cast<float>(preprocessed->intensities[i]);
    }
  }

  preprocessed->attrs.throw_if_invalid(preprocessed->size());

  if (!preprocessed->intensities.empty()) {
    const auto [min_it, max_it] = std::minmax_element(
      preprocessed->intensities.begin(),
      preprocessed->intensities.end());

    spdlog::debug(
      "[PointAttributes] preprocessed intensity range: min={} max={} count={}",
      *min_it,
      *max_it,
      preprocessed->intensities.size());
  }

  preprocessed->k_neighbors = params.k_correspondences;
  preprocessed->neighbors = find_neighbors(frame->points, frame->size(), params.k_correspondences);

  spdlog::trace("preprocessed: {} -> {} points", raw_points->size(), preprocessed->size());

  return preprocessed;
}

std::vector<int> CloudPreprocessor::find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const {
  gtsam_points::KdTree tree(points, num_points);

  std::vector<int> neighbors(num_points * k);

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_indices(k, i);
    std::vector<double> k_sq_dists(k);
    size_t num_found = tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.begin() + num_found, neighbors.begin() + i * k);
  };

  if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 8)
    for (int i = 0; i < num_points; i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error : TBB is not enabled" << std::endl;
    abort();
#endif
  }

  return neighbors;
}

}  // namespace glim

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <glim/odometry/estimation_frame.hpp>

namespace spdlog {
class logger;
}

namespace glim {

/** What to do with a lidar frame while initial state is not ready (bag-time gate). */
enum class PreInitFrameAction {
  Wait,  // hold frame; need more IMU
  Drop,  // discard; frame stamp is before init window (deterministic)
  Ready  // init complete; insert normally
};

/**
 * @brief Initial sensor state estimator
 *
 */
class InitialStateEstimation {
public:
  InitialStateEstimation();
  virtual ~InitialStateEstimation() {}

  /**
   * @brief Insert an IMU data
   * @param stamp         Timestamp
   * @param linear_acc    Linear acceleration
   * @param angular_vel   Angular velocity
   */
  virtual void insert_imu(double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) {}

  /**
   * @brief Insert a point cloud
   * @param raw_frame Preprocessed point cloud
   */
  virtual void insert_frame(const PreprocessedFrame::ConstPtr& raw_frame) {}

  /**
   * @brief Get the initial estimation result
   * @return EstimationFrame::ConstPtr  Estimated initial state
   * @return nullptr                    If it's not ready
   */
  virtual EstimationFrame::ConstPtr initial_pose() = 0;

  /**
   * @brief Bag-time gate for frames arriving before / during IMU init.
   */
  virtual PreInitFrameAction preinit_action(double frame_stamp) const { return PreInitFrameAction::Ready; }

protected:
  // Logging
  std::shared_ptr<spdlog::logger> logger;
};

/**
 * @brief Naive initial state estimator that simply calculates a pose that aligns linear acc with the gravity direction
 *        Would not work well when the sensor is moving
 */
class NaiveInitialStateEstimation : public InitialStateEstimation {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  NaiveInitialStateEstimation(const Eigen::Isometry3d& T_lidar_imu, const Eigen::Matrix<double, 6, 1>& imu_bias);
  virtual ~NaiveInitialStateEstimation() override;

  virtual void insert_imu(double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) override;
  virtual EstimationFrame::ConstPtr initial_pose() override;
  virtual PreInitFrameAction preinit_action(double frame_stamp) const override;

  void set_init_state(const Eigen::Isometry3d& init_T_world_imu, const Eigen::Vector3d& init_v_world_imu);

private:
  double window_size;

  bool ready;
  double init_stamp;
  double stamp;
  Eigen::Vector3d sum_acc;

  Eigen::Matrix<double, 6, 1> imu_bias;
  Eigen::Isometry3d T_lidar_imu;

  bool force_init;
  Eigen::Vector3d init_v_world_imu;
  Eigen::Isometry3d init_T_world_imu;
};

// TODO: Implement Loose-coupling-based initial state estimator

}  // namespace glim

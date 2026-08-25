#include <glim/mapping/async_sub_mapping.hpp>

#include <cstdlib>
#include <fstream>
#include <mutex>

namespace glim {

namespace {
std::once_flag sub_packet_log_once;
std::ofstream sub_packet_log_ofs;

bool sub_packet_log_enabled() {
  const char* v = std::getenv("GLIM_LOG_PACKETS");
  return v && v[0] != '\0' && v[0] != '0';
}

void log_sub_packet(size_t n_imu, size_t n_frames) {
  if (!sub_packet_log_enabled()) {
    return;
  }
  std::call_once(sub_packet_log_once, [] {
    const char* path = std::getenv("GLIM_LOG_PACKETS");
    const std::string p = (path && std::string(path) != "1" && std::string(path) != "true") ? path : "glim_packets.csv";
    sub_packet_log_ofs.open(p, std::ios::out | std::ios::app);
    if (sub_packet_log_ofs.tellp() == 0) {
      sub_packet_log_ofs << "site,n_imu,n_lidar_frames\n";
    }
  });
  if (sub_packet_log_ofs) {
    sub_packet_log_ofs << "sub_drain," << n_imu << ',' << n_frames << '\n';
  }
}
}  // namespace

AsyncSubMapping::AsyncSubMapping(const std::shared_ptr<glim::SubMappingBase>& sub_mapping) : sub_mapping(sub_mapping) {
  kill_switch = false;
  end_of_sequence = false;
  force_close_requested = false;
  thread = std::thread([this] { run(); });
}

AsyncSubMapping::~AsyncSubMapping() {
  kill_switch = true;
  join();
}

#ifdef GLIM_USE_OPENCV
void AsyncSubMapping::insert_image(const double stamp, const cv::Mat& image) {
  input_image_queue.push_back(std::make_pair(stamp, image));
}
#endif

void AsyncSubMapping::insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) {
  Eigen::Matrix<double, 7, 1> imu_data;
  imu_data << stamp, linear_acc, angular_vel;
  input_imu_queue.push_back(imu_data);
}

void AsyncSubMapping::insert_frame(const EstimationFrame::ConstPtr& odom_frame) {
  input_frame_queue.push_back(odom_frame);
}

void AsyncSubMapping::join() {
  end_of_sequence = true;
  if (thread.joinable()) {
    thread.join();
  }
}

int AsyncSubMapping::workload() const {
  return input_frame_queue.size();
}

std::vector<SubMap::Ptr> AsyncSubMapping::get_results() {
  return output_submap_queue.get_all_and_clear();
}

void AsyncSubMapping::request_force_close_submap() {
  force_close_requested.store(true);
}

void AsyncSubMapping::run() {
  while (!kill_switch) {
    auto submaps = sub_mapping->get_submaps();
    output_submap_queue.insert(submaps);

#ifdef GLIM_USE_OPENCV
    auto images = input_image_queue.get_all_and_clear();
#endif
    auto imu_frames = input_imu_queue.get_all_and_clear();
    auto odom_frames = input_frame_queue.get_all_and_clear();
    log_sub_packet(imu_frames.size(), odom_frames.size());

    if (force_close_requested.exchange(false)) {
      if (sub_mapping->force_close_submap()) {
        auto forced = sub_mapping->get_submaps();
        output_submap_queue.insert(forced);
      }
    }

    if (
#ifdef GLIM_USE_OPENCV
      images.empty() &&
#endif
      imu_frames.empty() && odom_frames.empty()) {
      if (end_of_sequence) {
        break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    for (const auto& imu : imu_frames) {
      const double stamp = imu[0];
      const Eigen::Vector3d linear_acc = imu.block<3, 1>(1, 0);
      const Eigen::Vector3d angular_vel = imu.block<3, 1>(4, 0);
      sub_mapping->insert_imu(stamp, linear_acc, angular_vel);
    }

#ifdef GLIM_USE_OPENCV
    for (const auto& image : images) {
      sub_mapping->insert_image(image.first, image.second);
    }
#endif

    for (const auto& frame : odom_frames) {
      sub_mapping->insert_frame(frame);
    }
  }

  output_submap_queue.insert(sub_mapping->submit_end_of_sequence());
}
}  // namespace glim

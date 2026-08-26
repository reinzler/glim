#include <glim/util/packet_log.hpp>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

namespace glim {

namespace {

std::mutex& log_mutex() {
  static std::mutex m;
  return m;
}

bool env_on(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] != '\0' && v[0] != '0';
}

std::string env_path(const char* name, const char* fallback_file) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0' || v[0] == '0') {
    return {};
  }
  if (std::string(v) == "1" || std::string(v) == "true") {
    return fallback_file;
  }
  return v;
}

std::ofstream& packet_stream() {
  static std::ofstream ofs;
  static bool opened = false;
  if (!opened) {
    opened = true;
    const std::string p = env_path("GLIM_LOG_PACKETS", "glim_packets.csv");
    if (!p.empty()) {
      ofs.open(p, std::ios::out | std::ios::trunc);
      if (ofs) {
        ofs << "site,n_imu,n_lidar_frames\n";
      }
    }
  }
  return ofs;
}

std::ofstream& isam2_stream() {
  static std::ofstream ofs;
  static bool opened = false;
  if (!opened) {
    opened = true;
    std::string p = env_path("GLIM_LOG_ISAM2", "glim_isam2.csv");
    if (p.empty() && env_on("GLIM_LOG_PACKETS")) {
      // Sibling of packets file.
      p = env_path("GLIM_LOG_PACKETS", "glim_packets.csv");
      const auto slash = p.find_last_of("/\\");
      const std::string dir = (slash == std::string::npos) ? std::string() : p.substr(0, slash + 1);
      p = dir + "glim_isam2.csv";
    }
    if (!p.empty()) {
      ofs.open(p, std::ios::out | std::ios::trunc);
      if (ofs) {
        ofs << "site,n_factors,n_values,recovery_depth,skipped\n";
      }
    }
  }
  return ofs;
}

}  // namespace

void log_packet_row(const char* site, std::size_t n_imu, std::size_t n_frames) {
  if (!env_on("GLIM_LOG_PACKETS")) {
    return;
  }
  std::lock_guard<std::mutex> lock(log_mutex());
  auto& ofs = packet_stream();
  if (ofs) {
    ofs << site << ',' << n_imu << ',' << n_frames << '\n';
    ofs.flush();
  }
}

void log_isam2_update(
  const char* site,
  std::size_t n_factors,
  std::size_t n_values,
  int recovery_depth,
  bool skipped) {
  if (!env_on("GLIM_LOG_ISAM2") && !env_on("GLIM_LOG_PACKETS")) {
    return;
  }
  std::lock_guard<std::mutex> lock(log_mutex());
  auto& ofs = isam2_stream();
  if (ofs) {
    ofs << site << ',' << n_factors << ',' << n_values << ',' << recovery_depth << ',' << (skipped ? 1 : 0) << '\n';
    ofs.flush();
  }
}

}  // namespace glim

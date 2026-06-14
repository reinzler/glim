#include <glim/util/pcd_exporter.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <tuple>
#include <vector>

namespace glim {
namespace {

bool is_finite_point(const Eigen::Vector4d& p) {
  return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
}

std::uint32_t pack_rgb_u32(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  return (static_cast<std::uint32_t>(r) << 16) |
         (static_cast<std::uint32_t>(g) << 8) |
         static_cast<std::uint32_t>(b);
}

float pack_rgb_float(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  const std::uint32_t packed = pack_rgb_u32(r, g, b);
  float out;
  static_assert(sizeof(out) == sizeof(packed), "float and uint32_t must have the same size");
  std::memcpy(&out, &packed, sizeof(float));
  return out;
}

std::uint8_t clamp_u8(double v) {
  v = std::max(0.0, std::min(255.0, v));
  return static_cast<std::uint8_t>(std::lround(v));
}

std::tuple<std::uint8_t, std::uint8_t, std::uint8_t> grayscale(double t) {
  const auto v = clamp_u8(255.0 * std::max(0.0, std::min(1.0, t)));
  return {v, v, v};
}

// Simple blue -> cyan -> green -> yellow -> red ramp.
// Good enough for CloudCompare/RViz visual inspection without adding dependencies.
std::tuple<std::uint8_t, std::uint8_t, std::uint8_t> ramp_color(double t) {
  t = std::max(0.0, std::min(1.0, t));

  const double x = 4.0 * t;
  const double r = std::min(std::max(x - 1.5, 0.0), 1.0);
  const double g = std::min(std::max(1.0 - std::abs(x - 2.0), 0.0), 1.0);
  const double b = std::min(std::max(1.5 - x, 0.0), 1.0);

  return {
    clamp_u8(255.0 * r),
    clamp_u8(255.0 * g),
    clamp_u8(255.0 * b)
  };
}

std::pair<double, double> minmax_z(const gtsam_points::PointCloud& cloud) {
  double mn = std::numeric_limits<double>::max();
  double mx = -std::numeric_limits<double>::max();

  for (int i = 0; i < cloud.size(); i++) {
    const auto& p = cloud.points[i];
    if (!is_finite_point(p)) {
      continue;
    }

    mn = std::min(mn, p.z());
    mx = std::max(mx, p.z());
  }

  if (!std::isfinite(mn) || !std::isfinite(mx) || mn > mx) {
    return {0.0, 1.0};
  }

  if (std::abs(mx - mn) < 1e-9) {
    mx = mn + 1.0;
  }

  return {mn, mx};
}

std::pair<double, double> minmax_intensity(const gtsam_points::PointCloud& cloud) {
  if (!cloud.has_intensities()) {
    return {0.0, 1.0};
  }

  double mn = std::numeric_limits<double>::max();
  double mx = -std::numeric_limits<double>::max();

  for (int i = 0; i < cloud.size(); i++) {
    const double v = cloud.intensities[i];
    if (!std::isfinite(v)) {
      continue;
    }

    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }

  if (!std::isfinite(mn) || !std::isfinite(mx) || mn > mx) {
    return {0.0, 1.0};
  }

  if (std::abs(mx - mn) < 1e-9) {
    mx = mn + 1.0;
  }

  return {mn, mx};
}


std::pair<double, double> robust_minmax_intensity(const gtsam_points::PointCloud& cloud) {
  if (!cloud.has_intensities()) {
    return {0.0, 1.0};
  }

  std::vector<double> values;
  values.reserve(cloud.size());

  for (int i = 0; i < cloud.size(); i++) {
    const double v = cloud.intensities[i];
    if (std::isfinite(v)) {
      values.push_back(v);
    }
  }

  if (values.empty()) {
    return {0.0, 1.0};
  }

  std::sort(values.begin(), values.end());

  const auto q = [&](double p) {
    const std::size_t idx = static_cast<std::size_t>(
      std::max(0.0, std::min(1.0, p)) * static_cast<double>(values.size() - 1));
    return values[idx];
  };

  double mn = q(0.02);
  double mx = q(0.98);

  if (!std::isfinite(mn) || !std::isfinite(mx) || std::abs(mx - mn) < 1e-9) {
    mn = values.front();
    mx = values.back();
  }

  if (std::abs(mx - mn) < 1e-9) {
    mx = mn + 1.0;
  }

  return {mn, mx};
}

std::vector<int> finite_indices(const gtsam_points::PointCloud& cloud) {
  std::vector<int> indices;
  indices.reserve(cloud.size());

  for (int i = 0; i < cloud.size(); i++) {
    if (is_finite_point(cloud.points[i])) {
      indices.push_back(i);
    }
  }

  return indices;
}

void fill_stats(
    const gtsam_points::PointCloud& cloud,
    const std::size_t written_points,
    PCDExportStats* stats) {
  if (!stats) {
    return;
  }

  stats->input_points = static_cast<std::size_t>(cloud.size());
  stats->written_points = written_points;
  stats->has_intensity = cloud.has_intensities();
}

}  // namespace

bool save_pcd_xyz(
    const std::string& path,
    const gtsam_points::PointCloud& cloud,
    PCDExportStats* stats) {
  const auto indices = finite_indices(cloud);

  std::ofstream ofs(path);
  if (!ofs) {
    return false;
  }

  ofs << "# .PCD v0.7 - GLIM rover core export\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z\n";
  ofs << "SIZE 4 4 4\n";
  ofs << "TYPE F F F\n";
  ofs << "COUNT 1 1 1\n";
  ofs << "WIDTH " << indices.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << indices.size() << "\n";
  ofs << "DATA ascii\n";

  ofs.setf(std::ios::fixed);
  ofs.precision(6);

  for (const int idx : indices) {
    const auto& p = cloud.points[idx];
    ofs << static_cast<float>(p.x()) << ' '
        << static_cast<float>(p.y()) << ' '
        << static_cast<float>(p.z()) << '\n';
  }

  fill_stats(cloud, indices.size(), stats);
  return true;
}

bool save_pcd_xyzi(
    const std::string& path,
    const gtsam_points::PointCloud& cloud,
    PCDExportStats* stats) {
  if (!cloud.has_intensities()) {
    fill_stats(cloud, 0, stats);
    return false;
  }

  const auto indices = finite_indices(cloud);

  std::ofstream ofs(path);
  if (!ofs) {
    return false;
  }

  ofs << "# .PCD v0.7 - GLIM rover core export\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z intensity\n";
  ofs << "SIZE 4 4 4 4\n";
  ofs << "TYPE F F F F\n";
  ofs << "COUNT 1 1 1 1\n";
  ofs << "WIDTH " << indices.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << indices.size() << "\n";
  ofs << "DATA ascii\n";

  ofs.setf(std::ios::fixed);
  ofs.precision(6);

  for (const int idx : indices) {
    const auto& p = cloud.points[idx];
    ofs << static_cast<float>(p.x()) << ' '
        << static_cast<float>(p.y()) << ' '
        << static_cast<float>(p.z()) << ' '
        << static_cast<float>(cloud.intensities[idx]) << '\n';
  }

  fill_stats(cloud, indices.size(), stats);
  return true;
}

bool save_pcd_xyzrgb(
    const std::string& path,
    const gtsam_points::PointCloud& cloud,
    const PCDColorMode color_mode,
    PCDExportStats* stats) {
  const auto indices = finite_indices(cloud);

  if (color_mode == PCDColorMode::INTENSITY && !cloud.has_intensities()) {
    fill_stats(cloud, 0, stats);
    return false;
  }

  const Eigen::Vector4f* camera_colors = nullptr;
  if (color_mode == PCDColorMode::CAMERA) {
    if (!cloud.aux_attributes.count("colors")) {
      fill_stats(cloud, 0, stats);
      return false;
    }
    camera_colors = cloud.aux_attribute<Eigen::Vector4f>("colors");
    if (!camera_colors) {
      fill_stats(cloud, 0, stats);
      return false;
    }
  }

  const auto [z_min, z_max] = minmax_z(cloud);
  const auto [i_min, i_max] = robust_minmax_intensity(cloud);

  //
  // Important:
  // PCL/PCD RGB is traditionally stored as a packed RGB value reinterpreted
  // as float.  If we write that float in ASCII with fixed precision, many
  // valid RGB bit patterns are printed as "0.000000", so viewers show a
  // monochrome cloud.  Use binary PCD to preserve the exact RGB bytes.
  //
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    return false;
  }

  ofs << "# .PCD v0.7 - GLIM rover core export\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z rgb\n";
  ofs << "SIZE 4 4 4 4\n";
  ofs << "TYPE F F F F\n";
  ofs << "COUNT 1 1 1 1\n";
  ofs << "WIDTH " << indices.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << indices.size() << "\n";
  ofs << "DATA binary\n";

  for (const int idx : indices) {
    const auto& p = cloud.points[idx];

    std::uint8_t r = 255;
    std::uint8_t g = 255;
    std::uint8_t b = 255;

    if (color_mode == PCDColorMode::HEIGHT) {
      const double t = (p.z() - z_min) / (z_max - z_min);
      std::tie(r, g, b) = ramp_color(t);
    } else if (color_mode == PCDColorMode::INTENSITY) {
      double t = (cloud.intensities[idx] - i_min) / (i_max - i_min);
      t = std::max(0.0, std::min(1.0, t));

      // Gamma < 1 brightens low-reflectivity Livox/Velodyne maps.
      t = std::sqrt(t);

      std::tie(r, g, b) = ramp_color(t);
    } else if (color_mode == PCDColorMode::CAMERA) {
      const Eigen::Vector4f& c = camera_colors[idx];
      r = static_cast<std::uint8_t>(std::clamp(c.x(), 0.0f, 1.0f) * 255.0f);
      g = static_cast<std::uint8_t>(std::clamp(c.y(), 0.0f, 1.0f) * 255.0f);
      b = static_cast<std::uint8_t>(std::clamp(c.z(), 0.0f, 1.0f) * 255.0f);
    }

    const std::array<float, 4> row = {
      static_cast<float>(p.x()),
      static_cast<float>(p.y()),
      static_cast<float>(p.z()),
      pack_rgb_float(r, g, b)
    };

    ofs.write(reinterpret_cast<const char*>(row.data()), sizeof(float) * row.size());
  }

  fill_stats(cloud, indices.size(), stats);
  return static_cast<bool>(ofs);
}


}  // namespace glim

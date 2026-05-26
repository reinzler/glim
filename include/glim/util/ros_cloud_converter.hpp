#pragma once

#include <algorithm>
#include <memory>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <spdlog/spdlog.h>
#include <boost/format.hpp>
#include <cmath>
#include <cstdint>

#include <Eigen/Core>
#include <gtsam_points/types/point_cloud.hpp>
#include <glim/util/raw_points.hpp>

#ifdef GLIM_ROS2
#include <sensor_msgs/msg/point_cloud2.hpp>
namespace glim {
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointCloud2Ptr = sensor_msgs::msg::PointCloud2::SharedPtr;
using PointCloud2ConstPtr = sensor_msgs::msg::PointCloud2::ConstSharedPtr;
using PointField = sensor_msgs::msg::PointField;

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}

inline builtin_interfaces::msg::Time from_sec(const double time) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = std::floor(time);
  stamp.nanosec = (time - stamp.sec) * 1e9;
  return stamp;
}

}  // namespace glim
#else
#include <sensor_msgs/PointCloud2.h>
namespace glim {
using PointCloud2 = sensor_msgs::PointCloud2;
using PointCloud2Ptr = sensor_msgs::PointCloud2::Ptr;
using PointCloud2ConstPtr = sensor_msgs::PointCloud2::ConstPtr;
using PointField = sensor_msgs::PointField;

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.toSec();
}

inline ros::Time from_sec(const double time) {
  ros::Time stamp;
  stamp.sec = std::floor(time);
  stamp.nsec = (time - stamp.sec) * 1e9;
  return stamp;
}

}  // namespace glim
#endif

namespace glim {

template <typename T>
Eigen::Vector4d get_vec4(const void* x, const void* y, const void* z) {
  return Eigen::Vector4d(*reinterpret_cast<const T*>(x), *reinterpret_cast<const T*>(y), *reinterpret_cast<const T*>(z), 1.0);
}

static bool read_uint8_like_field(const PointCloud2& points_msg, int datatype, int offset, int point_index, std::uint8_t& out) {
  const auto* ptr = &points_msg.data[points_msg.point_step * point_index + offset];

  switch (datatype) {
    case PointField::INT8:
      out = static_cast<std::uint8_t>(std::clamp<int>(*reinterpret_cast<const std::int8_t*>(ptr), 0, 255));
      return true;
    case PointField::UINT8:
      out = *reinterpret_cast<const std::uint8_t*>(ptr);
      return true;
    case PointField::INT16:
      out = static_cast<std::uint8_t>(std::clamp<int>(*reinterpret_cast<const std::int16_t*>(ptr), 0, 255));
      return true;
    case PointField::UINT16:
      out = static_cast<std::uint8_t>(std::clamp<int>(*reinterpret_cast<const std::uint16_t*>(ptr), 0, 255));
      return true;
    case PointField::INT32:
      out = static_cast<std::uint8_t>(std::clamp<long>(*reinterpret_cast<const std::int32_t*>(ptr), 0L, 255L));
      return true;
    case PointField::UINT32:
      out = static_cast<std::uint8_t>(std::min<std::uint32_t>(*reinterpret_cast<const std::uint32_t*>(ptr), 255U));
      return true;
    case PointField::FLOAT32:
      out = static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(std::lround(*reinterpret_cast<const float*>(ptr))), 0, 255));
      return true;
    case PointField::FLOAT64:
      out = static_cast<std::uint8_t>(std::clamp<long>(static_cast<long>(std::llround(*reinterpret_cast<const double*>(ptr))), 0L, 255L));
      return true;
    default:
      return false;
  }
}

static RawPoints::Ptr extract_raw_points(const PointCloud2& points_msg, const std::string& intensity_channel, const std::string& ring_channel) {
  int num_points = points_msg.width * points_msg.height;

  int x_type = 0;
  int y_type = 0;
  int z_type = 0;
  int time_type = 0;  // ouster and livox
  int intensity_type = 0;
  int color_type = 0;
  int ring_type = 0;
  int tag_type = 0;
  int scanner_id_type = 0;

  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  int time_offset = -1;
  int intensity_offset = -1;
  int color_offset = -1;
  int ring_offset = -1;
  int tag_offset = -1;
  int scanner_id_offset = -1;

  std::unordered_map<std::string, std::pair<int*, int*>> fields;
  fields["x"] = std::make_pair(&x_type, &x_offset);
  fields["y"] = std::make_pair(&y_type, &y_offset);
  fields["z"] = std::make_pair(&z_type, &z_offset);
  fields["t"] = std::make_pair(&time_type, &time_offset);
  fields["time"] = std::make_pair(&time_type, &time_offset);
  fields["time_stamp"] = std::make_pair(&time_type, &time_offset);
  fields["timestamp"] = std::make_pair(&time_type, &time_offset);
  fields[intensity_channel] = std::make_pair(&intensity_type, &intensity_offset);
  fields["rgba"] = std::make_pair(&color_type, &color_offset);
  fields["rgb"] = std::make_pair(&color_type, &color_offset);
  fields[ring_channel] = std::make_pair(&ring_type, &ring_offset);
  fields["tag"] = std::make_pair(&tag_type, &tag_offset);
  fields["scanner_id"] = std::make_pair(&scanner_id_type, &scanner_id_offset);
  fields["scanner"] = std::make_pair(&scanner_id_type, &scanner_id_offset);
  fields["lidar_id"] = std::make_pair(&scanner_id_type, &scanner_id_offset);

  for (const auto& field : points_msg.fields) {
    auto found = fields.find(field.name);
    if (found == fields.end()) {
      continue;
    }

    *found->second.first = field.datatype;
    *found->second.second = field.offset;
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    spdlog::warn("missing point coordinate fields");
    return nullptr;
  }

  if ((x_type != PointField::FLOAT32 && x_type != PointField::FLOAT64) || x_type != y_type || x_type != y_type) {
    spdlog::warn("unsupported points type");
    return nullptr;
  }

  auto raw_points = std::make_shared<RawPoints>();

  raw_points->points.resize(num_points);

  if (x_type == PointField::FLOAT32 && y_offset == x_offset + sizeof(float) && z_offset == y_offset + sizeof(float)) {
    // Special case: contiguous 3 floats
    for (int i = 0; i < num_points; i++) {
      const auto* x_ptr = &points_msg.data[points_msg.point_step * i + x_offset];
      raw_points->points[i] << Eigen::Map<const Eigen::Vector3f>(reinterpret_cast<const float*>(x_ptr)).cast<double>(), 1.0;
    }
  } else if (x_type == PointField::FLOAT64 && y_offset == x_offset + sizeof(double) && z_offset == y_offset + sizeof(double)) {
    // Special case: contiguous 3 doubles
    for (int i = 0; i < num_points; i++) {
      const auto* x_ptr = &points_msg.data[points_msg.point_step * i + x_offset];
      raw_points->points[i] << Eigen::Map<const Eigen::Vector3d>(reinterpret_cast<const double*>(x_ptr)), 1.0;
    }
  } else {
    for (int i = 0; i < num_points; i++) {
      const auto* x_ptr = &points_msg.data[points_msg.point_step * i + x_offset];
      const auto* y_ptr = &points_msg.data[points_msg.point_step * i + y_offset];
      const auto* z_ptr = &points_msg.data[points_msg.point_step * i + z_offset];

      if (x_type == PointField::FLOAT32) {
        raw_points->points[i] = get_vec4<float>(x_ptr, y_ptr, z_ptr);
      } else {
        raw_points->points[i] = get_vec4<double>(x_ptr, y_ptr, z_ptr);
      }
    }
  }

  if (time_offset >= 0) {
    raw_points->times.resize(num_points);

    for (int i = 0; i < num_points; i++) {
      const auto* time_ptr = &points_msg.data[points_msg.point_step * i + time_offset];
      switch (time_type) {
        case PointField::UINT32:
          raw_points->times[i] = *reinterpret_cast<const uint32_t*>(time_ptr) / 1e9;
          break;
        case PointField::FLOAT32:
          raw_points->times[i] = *reinterpret_cast<const float*>(time_ptr);
          break;
        case PointField::FLOAT64:
          raw_points->times[i] = *reinterpret_cast<const double*>(time_ptr);
          break;
        default:
          spdlog::warn("unsupported time type {}", time_type);
          return nullptr;
      }
    }
  }

  if (intensity_offset >= 0) {
    raw_points->intensities.resize(num_points);

    for (int i = 0; i < num_points; i++) {
      const auto* intensity_ptr = &points_msg.data[points_msg.point_step * i + intensity_offset];
      switch (intensity_type) {
        case PointField::UINT8:
          raw_points->intensities[i] = *reinterpret_cast<const std::uint8_t*>(intensity_ptr);
          break;
        case PointField::UINT16:
          raw_points->intensities[i] = *reinterpret_cast<const std::uint16_t*>(intensity_ptr);
          break;
        case PointField::UINT32:
          raw_points->intensities[i] = *reinterpret_cast<const std::uint32_t*>(intensity_ptr);
          break;
        case PointField::FLOAT32:
          raw_points->intensities[i] = *reinterpret_cast<const float*>(intensity_ptr);
          break;
        case PointField::FLOAT64:
          raw_points->intensities[i] = *reinterpret_cast<const double*>(intensity_ptr);
          break;
        default:
          spdlog::warn("unsupported intensity type {}", intensity_type);
          return nullptr;
      }
    }
  }

  if (color_offset >= 0) {
    if (color_type != PointField::UINT32) {
      spdlog::warn("unsupported color type {}", color_type);
    } else {
      raw_points->colors.resize(num_points);

      for (int i = 0; i < num_points; i++) {
        const auto* color_ptr = &points_msg.data[points_msg.point_step * i + color_offset];
        raw_points->colors[i] = Eigen::Matrix<unsigned char, 4, 1>(reinterpret_cast<const std::uint8_t*>(color_ptr)).cast<double>() / 255.0;
      }
    }
  }

  if (ring_offset >= 0) {
    raw_points->rings.resize(num_points);

    for (int i = 0; i < num_points; i++) {
      const auto* ring_ptr = &points_msg.data[points_msg.point_step * i + ring_offset];
      switch (ring_type) {
        case PointField::UINT8:
          raw_points->rings[i] = *reinterpret_cast<const std::uint8_t*>(ring_ptr);
          break;
        case PointField::UINT16:
          raw_points->rings[i] = *reinterpret_cast<const std::uint16_t*>(ring_ptr);
          break;
        case PointField::UINT32:
          raw_points->rings[i] = *reinterpret_cast<const std::uint32_t*>(ring_ptr);
          break;
        default:
          spdlog::warn("unsupported ring type {}", ring_type);
          return nullptr;
      }
    }
  }


  // Fill PR290-style attributes from legacy fields and extra PointCloud2 channels.
  if (!raw_points->times.empty()) {
    raw_points->attrs.timestamp = raw_points->times;
  }

  if (!raw_points->intensities.empty()) {
    auto& dst = raw_points->attrs.intensity.emplace();
    dst.resize(raw_points->intensities.size());
    for (std::size_t i = 0; i < raw_points->intensities.size(); i++) {
      dst[i] = static_cast<float>(raw_points->intensities[i]);
    }
  }

  if (!raw_points->rings.empty()) {
    auto& dst = raw_points->attrs.line.emplace();
    dst.resize(raw_points->rings.size());
    for (std::size_t i = 0; i < raw_points->rings.size(); i++) {
      dst[i] = static_cast<std::uint8_t>(std::clamp<int>(raw_points->rings[i], 0, 255));
    }
  }

  if (!raw_points->colors.empty()) {
    auto& dst = raw_points->attrs.rgb.emplace();
    dst.resize(raw_points->colors.size());
    for (std::size_t i = 0; i < raw_points->colors.size(); i++) {
      dst[i] = raw_points->colors[i].template head<3>().cast<float>();
    }
  }

  if (tag_offset >= 0) {
    auto& dst = raw_points->attrs.tag.emplace();
    dst.resize(num_points);

    for (int i = 0; i < num_points; i++) {
      if (!read_uint8_like_field(points_msg, tag_type, tag_offset, i, dst[i])) {
        spdlog::warn("unsupported tag type {}", tag_type);
        return nullptr;
      }
    }
  }

  if (scanner_id_offset >= 0) {
    auto& dst = raw_points->attrs.scanner_id.emplace();
    dst.resize(num_points);

    for (int i = 0; i < num_points; i++) {
      if (!read_uint8_like_field(points_msg, scanner_id_type, scanner_id_offset, i, dst[i])) {
        spdlog::warn("unsupported scanner_id type {}", scanner_id_type);
        return nullptr;
      }
    }
  }

  std::string attrs_error;
  if (!raw_points->attrs.validate(raw_points->points.size(), &attrs_error)) {
    spdlog::warn("invalid point attributes: {}", attrs_error);
    raw_points->attrs.clear();
  }

  raw_points->stamp = to_sec(points_msg.header.stamp);
  return raw_points;
}

static RawPoints::Ptr extract_raw_points(const PointCloud2& points_msg, const std::string& intensity_channel = "intensity") {
  return extract_raw_points(points_msg, intensity_channel, "");
}

static RawPoints::Ptr extract_raw_points(const PointCloud2ConstPtr& points_msg, const std::string& intensity_channel = "intensity") {
  return extract_raw_points(*points_msg, intensity_channel, "");
}

static PointCloud2ConstPtr frame_to_pointcloud2(const std::string& frame_id, const double stamp, const gtsam_points::PointCloud& frame) {
  PointCloud2Ptr msg(new PointCloud2);
  msg->header.frame_id = frame_id;
  msg->header.stamp = from_sec(stamp);

  msg->width = frame.size();
  msg->height = 1;

  const auto create_field = [](const std::string& name, int offset, int datatype, int count) {
    PointField field;
    field.name = name;
    field.offset = offset;
    field.datatype = datatype;
    field.count = count;
    return field;
  };

  int point_step = 0;
  msg->fields.reserve(6);

  msg->fields.emplace_back(create_field("x", sizeof(float) * 0, PointField::FLOAT32, 1));
  msg->fields.emplace_back(create_field("y", sizeof(float) * 1, PointField::FLOAT32, 1));
  msg->fields.emplace_back(create_field("z", sizeof(float) * 2, PointField::FLOAT32, 1));
  point_step += sizeof(float) * 3;

  if (frame.times) {
    msg->fields.emplace_back(create_field("t", point_step, PointField::FLOAT32, 1));
    point_step += sizeof(float);
  }

  if (frame.intensities) {
    msg->fields.emplace_back(create_field("intensity", point_step, PointField::FLOAT32, 1));
    point_step += sizeof(float);
  }

  const Eigen::Vector4f* colors = frame.aux_attributes.count("colors") ? frame.aux_attribute<Eigen::Vector4f>("colors") : nullptr;
  if (colors) {
    msg->fields.emplace_back(create_field("rgba", point_step, PointField::UINT32, 1));
    point_step += sizeof(std::uint32_t);
  }

  msg->is_bigendian = false;
  msg->point_step = point_step;
  msg->row_step = point_step * frame.size();

  msg->data.resize(point_step * frame.size());
  for (int i = 0; i < frame.size(); i++) {
    unsigned char* point_bytes = msg->data.data() + msg->point_step * i;
    Eigen::Map<Eigen::Vector3f> xyz(reinterpret_cast<float*>(point_bytes));
    xyz = frame.points[i].head<3>().cast<float>();
    point_bytes += sizeof(float) * 3;

    if (frame.times) {
      *reinterpret_cast<float*>(point_bytes) = frame.times[i];
      point_bytes += sizeof(float);
    }

    if (frame.intensities) {
      *reinterpret_cast<float*>(point_bytes) = frame.intensities[i];
      point_bytes += sizeof(float);
    }

    if (colors) {
      const Eigen::Matrix<std::uint8_t, 4, 1> rgba = (colors[i].array() * 255.0f).min(255.0f).max(0.0f).cast<std::uint8_t>();
      point_bytes[0] = rgba[0];
      point_bytes[1] = rgba[1];
      point_bytes[2] = rgba[2];
      point_bytes[3] = rgba[3];
      point_bytes += sizeof(std::uint32_t);
    }
  }

  return msg;
}

}  // namespace glim
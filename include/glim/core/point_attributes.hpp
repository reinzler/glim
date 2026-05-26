#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace glim {

/**
 * @brief PR290-style per-point attribute container.
 *
 * Rule:
 *   if an attribute exists, attribute.size() must equal num_points.
 *
 * M1.1 is intentionally non-invasive:
 *   - define the standard schema
 *   - add validation helpers
 *   - do not yet rewrite RawPoints / PreprocessedFrame pipeline
 */
struct PointAttributes {
  std::optional<std::vector<float>> intensity;
  std::optional<std::vector<double>> timestamp;

  std::optional<std::vector<std::uint8_t>> line;
  std::optional<std::vector<std::uint8_t>> tag;
  std::optional<std::vector<std::uint8_t>> scanner_id;

  std::optional<std::vector<Eigen::Vector3f>> rgb;
  std::optional<std::vector<Eigen::Vector3f>> normals;
  std::optional<std::vector<Eigen::Matrix3f>> covariances;

  // Future MapCleaner / dynamic object pipeline.
  std::optional<std::vector<float>> static_score;
  std::optional<std::vector<std::uint8_t>> dynamic_label;

  void clear() {
    intensity.reset();
    timestamp.reset();
    line.reset();
    tag.reset();
    scanner_id.reset();
    rgb.reset();
    normals.reset();
    covariances.reset();
    static_score.reset();
    dynamic_label.reset();
  }

  bool empty() const {
    return !intensity &&
           !timestamp &&
           !line &&
           !tag &&
           !scanner_id &&
           !rgb &&
           !normals &&
           !covariances &&
           !static_score &&
           !dynamic_label;
  }

  bool validate(std::size_t num_points, std::string* error = nullptr) const {
    auto check = [&](const auto& attr, const char* name) {
      if (!attr) {
        return true;
      }

      if (attr->size() == num_points) {
        return true;
      }

      if (error) {
        *error = std::string("PointAttributes size mismatch: ") +
                 name +
                 ".size()=" +
                 std::to_string(attr->size()) +
                 " num_points=" +
                 std::to_string(num_points);
      }

      return false;
    };

    return check(intensity, "intensity") &&
           check(timestamp, "timestamp") &&
           check(line, "line") &&
           check(tag, "tag") &&
           check(scanner_id, "scanner_id") &&
           check(rgb, "rgb") &&
           check(normals, "normals") &&
           check(covariances, "covariances") &&
           check(static_score, "static_score") &&
           check(dynamic_label, "dynamic_label");
  }

  void throw_if_invalid(std::size_t num_points) const {
    std::string error;
    if (!validate(num_points, &error)) {
      throw std::runtime_error(error);
    }
  }
};

}  // namespace glim

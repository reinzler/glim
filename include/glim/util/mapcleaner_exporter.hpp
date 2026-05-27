#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <glim/mapping/sub_map.hpp>
#include <glim/util/mapcleaner_types.hpp>

namespace glim {

struct MapCleanerExportStats {
  std::size_t submaps = 0;
  std::size_t frames_seen = 0;
  std::size_t keyframes_seen = 0;
  std::size_t scans_written = 0;
  std::size_t poses_written = 0;
  std::size_t skipped_empty_frames = 0;
  std::size_t submaps_with_keyframes = 0;
};

bool export_mapcleaner_dataset(
  const std::string& output_dir,
  const std::vector<SubMap::Ptr>& submaps,
  MapCleanerExportStats* stats = nullptr);

}  // namespace glim

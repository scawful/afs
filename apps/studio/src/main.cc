/// AFS Training Data Visualization - Main Entry Point
///
/// Usage: afs_viz [data_path]
///   data_path: Path to training data directory (default: ~/.context/training)
///
/// Build:
///   cmake -B build -S src/cc -DAFS_BUILD_VIZ=ON
///   cmake --build build
///
/// Keys:
///   F5 - Refresh data
///   Ctrl+Q - Quit
///   Ctrl+/ - Shortcut editor

#include <iostream>
#include <string>

#include "app.h"
#include "core/logger.h"
#include "core/filesystem.h"

int main(int argc, char* argv[]) {
  using afs::studio::core::FileSystem;
  
  // Determine data path
  std::string data_path_str;
  if (argc > 1) {
    data_path_str = argv[1];
  } else {
    data_path_str = "~/.context/training";
  }

  std::filesystem::path data_path = FileSystem::ResolvePath(data_path_str);

  LOG_INFO("AFS Studio Starting...");
  LOG_INFO("Data path: " + data_path.string());
  LOG_INFO("Press F5 to refresh data");

  afs::viz::App app(data_path.string());
  return app.Run();
}

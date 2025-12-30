/// AFS Studio - Main Entry Point
///
/// Usage: afs_studio [--data PATH]
///   --data PATH: Path to training data directory (default: ~/src/training if present)
///   --version: Print version and exit
///
/// Build:
///   cmake -S apps/studio -B build/studio
///   cmake --build build/studio --target afs_studio
///
/// Keys:
///   F5 - Refresh data
///   Ctrl+Q - Quit
///   Ctrl+/ - Shortcut editor

#include <cstdlib>
#include <iostream>
#include <string>

#include "app.h"
#include "core/filesystem.h"
#include "core/logger.h"
#include "core/version.h"

namespace {

void PrintUsage() {
  std::cout << "afs_studio [--data PATH]\n"
            << "  --data PATH  Training data root (default: ~/src/training)\n"
            << "  --version    Print version and exit\n"
            << "  -h, --help   Show this help\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  using afs::studio::core::FileSystem;

  // Determine data path
  std::string data_path_str;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    }
    if (arg == "--version" || arg == "-v") {
      std::cout << "AFS Studio " << afs::studio::core::StudioVersion() << "\n";
      return 0;
    }
    if (arg == "--data" || arg == "--data-path") {
      if (i + 1 < argc) {
        data_path_str = argv[++i];
        continue;
      }
      std::cerr << "Missing value for --data\n";
      return 1;
    }
    if (data_path_str.empty() && !arg.empty() && arg[0] != '-') {
      data_path_str = arg;
    }
  }
  if (data_path_str.empty()) {
    const char* env_root = std::getenv("AFS_TRAINING_ROOT");
    if (env_root && env_root[0] != '\0') {
      data_path_str = env_root;
    } else {
      auto preferred = FileSystem::ResolvePath("~/src/training");
      data_path_str = FileSystem::Exists(preferred) ? preferred.string()
                                                    : "~/.context/training";
    }
  }

  std::filesystem::path data_path = FileSystem::ResolvePath(data_path_str);

  LOG_INFO(std::string("AFS Studio ") + afs::studio::core::StudioVersion());
  LOG_INFO("Data path: " + data_path.string());
  LOG_INFO("Press F5 to refresh data");

  afs::viz::App app(data_path.string());
  return app.Run();
}

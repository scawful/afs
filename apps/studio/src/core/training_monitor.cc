#include "training_monitor.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include "filesystem.h"
#include "logger.h"

namespace afs {
namespace studio {

namespace {

std::optional<std::filesystem::path> ResolveTrainingMonitorConfigPath() {
  const char* env_path = std::getenv("AFS_TRAINING_MONITOR_CONFIG");
  if (env_path && env_path[0] != '\0') {
    auto candidate = core::FileSystem::ResolvePath(env_path);
    if (core::FileSystem::Exists(candidate)) {
      return candidate;
    }
  }

  const char* env_mounts = std::getenv("AFS_SCAWFUL_MOUNTS");
  if (env_mounts && env_mounts[0] != '\0') {
    auto candidate = core::FileSystem::ResolvePath(env_mounts);
    if (core::FileSystem::Exists(candidate)) {
      return candidate;
    }
  }

  const std::filesystem::path candidates[] = {
      core::FileSystem::ResolvePath("~/.config/afs/afs_scawful/mounts.json"),
      core::FileSystem::ResolvePath("~/.config/afs/afs_scawful/training_monitor.json"),
      core::FileSystem::ResolvePath("~/.config/afs/plugins/afs_scawful/config/mounts.json"),
      core::FileSystem::ResolvePath(
          "~/.config/afs/plugins/afs_scawful/config/training_monitor.json"),
  };

  for (const auto& candidate : candidates) {
    if (core::FileSystem::Exists(candidate)) {
      return candidate;
    }
  }

  return std::nullopt;
}

std::filesystem::path NormalizeWindowsSubpath(const std::string& raw_path) {
  if (raw_path.empty()) {
    return {};
  }

  std::string cleaned = raw_path;
  std::replace(cleaned.begin(), cleaned.end(), '\\', '/');
  auto colon = cleaned.find(':');
  if (colon != std::string::npos) {
    cleaned = cleaned.substr(colon + 1);
  }
  while (!cleaned.empty() && (cleaned.front() == '/' || cleaned.front() == '\\')) {
    cleaned.erase(cleaned.begin());
  }
  return std::filesystem::path(cleaned);
}

bool LoadTrainingMonitorConfig(const std::filesystem::path& config_path,
                               TrainingMonitorConfig* config,
                               bool allow_mount_override,
                               bool* use_training_dir) {
  if (!config || !use_training_dir) {
    return false;
  }

  std::ifstream file(config_path);
  if (!file.is_open()) {
    return false;
  }

  try {
    nlohmann::json data = nlohmann::json::parse(file);
    if (data.contains("training_monitor") && data["training_monitor"].is_object()) {
      data = data["training_monitor"];
    }
    if (!data.is_object()) {
      return false;
    }

    if (allow_mount_override && data.contains("windows_mount_path") &&
        data["windows_mount_path"].is_string()) {
      config->windows_mount_path =
          core::FileSystem::ResolvePath(data["windows_mount_path"].get<std::string>());
    }

    if (data.contains("windows_training_dir") && data["windows_training_dir"].is_string()) {
      config->windows_training_dir = data["windows_training_dir"].get<std::string>();
      *use_training_dir = !config->windows_training_dir.empty();
    }

    if (data.contains("ssh_host") && data["ssh_host"].is_string()) {
      config->ssh_host = data["ssh_host"].get<std::string>();
    }
    if (data.contains("ssh_user") && data["ssh_user"].is_string()) {
      config->ssh_user = data["ssh_user"].get<std::string>();
    }

    if (data.contains("auto_refresh") && data["auto_refresh"].is_boolean()) {
      config->auto_refresh = data["auto_refresh"].get<bool>();
    }
    if (data.contains("refresh_interval_seconds") &&
        data["refresh_interval_seconds"].is_number_integer()) {
      config->refresh_interval_seconds =
          data["refresh_interval_seconds"].get<int>();
    }

    if (data.contains("watched_paths") && data["watched_paths"].is_array()) {
      config->watched_paths.clear();
      for (const auto& entry : data["watched_paths"]) {
        if (entry.is_string()) {
          config->watched_paths.push_back(entry.get<std::string>());
        }
      }
    }

    return true;
  } catch (const nlohmann::json::exception& e) {
    LOG_WARN(std::string("TrainingMonitor: Failed to parse config: ") + e.what());
  }

  return false;
}

}  // namespace

TrainingMonitor::TrainingMonitor() {
  bool allow_mount_override = true;
  // Default Windows mount (optional).
  const char* env_mount = std::getenv("AFS_WINDOWS_MOUNT");
  if (env_mount && env_mount[0] != '\0') {
    config_.windows_mount_path = core::FileSystem::ResolvePath(env_mount);
    allow_mount_override = false;
  }

  const char* home = std::getenv("HOME");
  if (allow_mount_override && home) {
    config_.windows_mount_path =
        std::filesystem::path(home) / "Mounts" / "windows-training";
  }

  const char* env_training_dir = std::getenv("AFS_WINDOWS_TRAINING_DIR");
  if (env_training_dir && env_training_dir[0] != '\0') {
    config_.windows_training_dir = env_training_dir;
    use_training_dir_ = true;
  }

  auto config_path = ResolveTrainingMonitorConfigPath();
  if (config_path) {
    LoadTrainingMonitorConfig(*config_path, &config_, allow_mount_override,
                              &use_training_dir_);
  }
}

TrainingMonitor::TrainingMonitor(const TrainingMonitorConfig& config)
    : config_(config),
      use_training_dir_(!config.windows_training_dir.empty()) {}

std::filesystem::path TrainingMonitor::ResolveWindowsMount() const {
  // Check if mount is accessible
  if (core::FileSystem::Exists(config_.windows_mount_path)) {
    std::error_code ec;
    if (std::filesystem::is_directory(config_.windows_mount_path, ec)) {
        return config_.windows_mount_path;
    }
  }
  return {};
}

std::filesystem::path TrainingMonitor::FindLatestCheckpoint(
    const std::filesystem::path& model_dir) {
  std::filesystem::path latest;
  std::filesystem::file_time_type latest_time;

  try {
    std::error_code ec;
    auto it = std::filesystem::directory_iterator(model_dir, ec);
    if (!ec) {
      for (const auto& entry : it) {
        std::error_code entry_ec;
        if (entry.is_directory(entry_ec)) {
          std::string name = entry.path().filename().string();
          // Look for checkpoint-* directories
          if (name.find("checkpoint-") == 0) {
            auto time = entry.last_write_time(entry_ec);
            if (!entry_ec && (latest.empty() || time > latest_time)) {
              latest = entry.path();
              latest_time = time;
            }
          }
        }
      }
    }
  } catch (...) {
    // Directory iteration failed
  }

  return latest;
}

bool TrainingMonitor::Poll(std::string* error) {
  last_error_.clear();
  last_poll_time_ = std::chrono::steady_clock::now();

  // Try mount first
  std::filesystem::path mount = ResolveWindowsMount();
  if (!mount.empty()) {
    // Look for active training dirs
    std::filesystem::path models_dir = mount;
    if (use_training_dir_ && !config_.windows_training_dir.empty()) {
      auto subdir = NormalizeWindowsSubpath(config_.windows_training_dir);
      if (!subdir.empty()) {
        models_dir /= subdir;
      }
    }
    models_dir /= "models";
    if (core::FileSystem::Exists(models_dir)) {
      // Find the most recently modified model directory
      std::filesystem::path latest_model;
      std::filesystem::file_time_type latest_time;

      try {
        std::error_code ec;
        auto it = std::filesystem::directory_iterator(models_dir, ec);
        if (!ec) {
          for (const auto& entry : it) {
            std::error_code entry_ec;
            if (entry.is_directory(entry_ec)) {
              auto time = entry.last_write_time(entry_ec);
              if (!entry_ec && (latest_model.empty() || time > latest_time)) {
                latest_model = entry.path();
                latest_time = time;
              }
            }
          }
        }
      } catch (...) {
        // Directory iteration failed
      }

      if (!latest_model.empty()) {
        // Look for trainer_state.json in latest checkpoint
        std::filesystem::path checkpoint = FindLatestCheckpoint(latest_model);
        if (!checkpoint.empty()) {
          std::filesystem::path state_file = checkpoint / "trainer_state.json";
          if (core::FileSystem::Exists(state_file)) {
            return LoadFromPath(state_file, error);
          }
        }

        // Try trainer_state.json in model root
        std::filesystem::path root_state = latest_model / "trainer_state.json";
        if (core::FileSystem::Exists(root_state)) {
          return LoadFromPath(root_state, error);
        }
      }
    }

    last_error_ = "No active training found in mount";
    if (error) *error = last_error_;
    state_.status = TrainingStatus::kIdle;
    return true;  // Not an error, just no active training
  }

  last_error_ = "Windows mount not accessible: " + config_.windows_mount_path.string();
  if (error) *error = last_error_;
  return false;
}

bool TrainingMonitor::LoadFromPath(const std::filesystem::path& path,
                                    std::string* error) {
  std::ifstream file(path);
  if (!file.is_open()) {
    last_error_ = "Failed to open: " + path.string();
    if (error) *error = last_error_;
    return false;
  }

  try {
    nlohmann::json json = nlohmann::json::parse(file);
    if (!ParseTrainerState(json)) {
      last_error_ = "Failed to parse trainer state";
      if (error) *error = last_error_;
      return false;
    }

    state_.source_path = path.string();
    state_.source_location = "windows";
    state_.is_remote = true;
    return true;

  } catch (const nlohmann::json::exception& e) {
    last_error_ = std::string("JSON parse error: ") + e.what();
    if (error) *error = last_error_;
    return false;
  }
}

bool TrainingMonitor::ParseTrainerState(const nlohmann::json& json) {
  // Reset state
  state_ = TrainingState{};

  // Helper to safely get values
  auto get_int = [&](const char* key) -> int {
    if (json.contains(key) && json[key].is_number()) {
      return json[key].get<int>();
    }
    return 0;
  };

  auto get_float = [&](const char* key) -> float {
    if (json.contains(key) && json[key].is_number()) {
      return json[key].get<float>();
    }
    return 0.0f;
  };

  auto get_string = [&](const char* key) -> std::string {
    if (json.contains(key) && json[key].is_string()) {
      return json[key].get<std::string>();
    }
    return "";
  };

  // Parse basic info
  state_.current_epoch = get_int("epoch");
  state_.total_epochs = get_int("num_train_epochs");
  state_.current_step = get_int("global_step");
  state_.total_steps = get_int("max_steps");

  // Calculate progress
  if (state_.total_steps > 0) {
    state_.progress_percent =
        static_cast<float>(state_.current_step) / state_.total_steps * 100.0f;
  }

  // Determine status
  if (state_.current_step > 0 && state_.current_step < state_.total_steps) {
    state_.status = TrainingStatus::kRunning;
  } else if (state_.current_step >= state_.total_steps && state_.total_steps > 0) {
    state_.status = TrainingStatus::kCompleted;
  } else {
    state_.status = TrainingStatus::kIdle;
  }

  // Parse loss history
  if (json.contains("log_history") && json["log_history"].is_array()) {
    for (const auto& entry : json["log_history"]) {
      if (entry.contains("loss") && entry["loss"].is_number()) {
        LossPoint point;
        if (entry.contains("step") && entry["step"].is_number()) {
          point.step = entry["step"].get<int>();
        }
        point.loss = entry["loss"].get<float>();
        if (entry.contains("eval_loss") && entry["eval_loss"].is_number()) {
          point.eval_loss = entry["eval_loss"].get<float>();
        }
        state_.loss_history.push_back(point);

        // Track current and best loss
        state_.current_loss = point.loss;
        if (state_.best_loss == 0.0f || point.loss < state_.best_loss) {
          state_.best_loss = point.loss;
          state_.best_step = point.step;
        }
      }
    }
  }

  // Parse timing info
  if (json.contains("total_flos") && json["total_flos"].is_number()) {
    // Estimate based on FLOPS if available
  }

  // Extract model name from path if available
  if (json.contains("best_model_checkpoint") &&
      json["best_model_checkpoint"].is_string()) {
    std::string path = json["best_model_checkpoint"].get<std::string>();
    // Extract model name from path
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) pos = path.rfind('\\');
    if (pos != std::string::npos && pos > 0) {
      size_t start = path.rfind('/', pos - 1);
      if (start == std::string::npos) start = path.rfind('\\', pos - 1);
      if (start != std::string::npos) {
        state_.model_name = path.substr(start + 1, pos - start - 1);
      }
    }
  }

  return true;
}

bool TrainingMonitor::ShouldRefresh() const {
  if (!config_.auto_refresh) return false;

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      now - last_poll_time_);

  return elapsed.count() >= config_.refresh_interval_seconds;
}

}  // namespace studio
}  // namespace afs

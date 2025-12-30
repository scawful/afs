#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace afs {
namespace studio {

// Training run status
enum class TrainingStatus {
  kUnknown,
  kIdle,
  kRunning,
  kCompleted,
  kFailed,
  kPaused
};

// Single epoch progress data
struct EpochProgress {
  int epoch = 0;
  int step = 0;
  int total_steps = 0;
  float loss = 0.0f;
  float learning_rate = 0.0f;
  std::string timestamp;
};

// Loss data point for charts
struct LossPoint {
  int step = 0;
  float loss = 0.0f;
  float eval_loss = 0.0f;
};

// Training state from trainer_state.json
struct TrainingState {
  // Identity
  std::string run_name;
  std::string model_name;
  std::string base_model;

  // Progress
  TrainingStatus status = TrainingStatus::kUnknown;
  int current_epoch = 0;
  int total_epochs = 0;
  int current_step = 0;
  int total_steps = 0;
  float progress_percent = 0.0f;

  // Timing
  std::string started_at;
  std::string updated_at;
  int elapsed_minutes = 0;
  int estimated_remaining_minutes = 0;

  // Metrics
  float current_loss = 0.0f;
  float best_loss = 0.0f;
  int best_step = 0;
  std::optional<float> eval_loss;
  std::optional<float> perplexity;

  // Loss history for plotting
  std::vector<LossPoint> loss_history;

  // Hardware
  std::string device;
  std::optional<float> gpu_memory_percent;
  std::optional<float> gpu_utilization_percent;

  // Source
  std::string source_path;
  std::string source_location;  // "windows", "mac", "cloud"
  bool is_remote = false;
};

// Configuration for training monitor
struct TrainingMonitorConfig {
  // Windows mount point
  std::filesystem::path windows_mount_path;
  std::string windows_training_dir = "D:/afs_training";

  // SSH config (fallback if mount not available)
  std::string ssh_host;
  std::string ssh_user;

  // Auto-refresh
  bool auto_refresh = true;
  int refresh_interval_seconds = 10;

  // What to monitor
  std::vector<std::string> watched_paths;
};

// Training monitor - reads training state from local or remote
class TrainingMonitor {
 public:
  TrainingMonitor();
  explicit TrainingMonitor(const TrainingMonitorConfig& config);

  // Poll for current training state
  bool Poll(std::string* error = nullptr);

  // Get current training state
  const TrainingState& GetState() const { return state_; }

  // Check if any training is active
  bool IsTrainingActive() const {
    return state_.status == TrainingStatus::kRunning;
  }

  // Get last poll time
  const std::chrono::steady_clock::time_point& GetLastPollTime() const {
    return last_poll_time_;
  }

  // Check if refresh is needed based on interval
  bool ShouldRefresh() const;

  // Configuration
  const TrainingMonitorConfig& GetConfig() const { return config_; }
  void SetConfig(const TrainingMonitorConfig& config) { config_ = config; }

  // Error handling
  const std::string& GetLastError() const { return last_error_; }

 private:
  bool LoadFromPath(const std::filesystem::path& path, std::string* error);
  bool ParseTrainerState(const nlohmann::json& json);
  std::filesystem::path FindLatestCheckpoint(const std::filesystem::path& model_dir);
  std::filesystem::path ResolveWindowsMount() const;

  TrainingMonitorConfig config_;
  TrainingState state_;
  std::chrono::steady_clock::time_point last_poll_time_;
  std::string last_error_;
};

}  // namespace studio
}  // namespace afs

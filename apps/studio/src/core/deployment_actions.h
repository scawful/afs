#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace afs {
namespace studio {

// Result of an async action
enum class ActionStatus {
  kPending,
  kRunning,
  kCompleted,
  kFailed
};

struct ActionResult {
  ActionStatus status = ActionStatus::kPending;
  std::string output;
  std::string error;
  int exit_code = 0;
  float progress = 0.0f;
};

// Deployment target backends
enum class DeploymentBackend {
  kOllama,
  kLlamaCpp,
  kHalextNode
};

// Quantization options for GGUF conversion
struct QuantizationOption {
  std::string name;
  std::string description;
  float size_ratio;  // Relative to F16
};

// Available quantization options
inline std::vector<QuantizationOption> GetQuantizationOptions() {
  return {
    {"Q3_K_S", "Smallest size, reduced quality", 0.35f},
    {"Q4_K_M", "Good balance (recommended)", 0.45f},
    {"Q5_K_M", "Better quality, larger size", 0.55f},
    {"Q8_0", "Best quality, largest size", 0.65f},
    {"F16", "Full precision (no quantization)", 1.0f},
  };
}

// Deployment actions - executes afs CLI commands
class DeploymentActions {
 public:
  using ProgressCallback = std::function<void(const std::string&, float)>;

  DeploymentActions();

  // Pull model from remote location
  ActionResult PullModel(const std::string& model_id,
                         const std::string& source = "auto",
                         ProgressCallback callback = nullptr);

  // Convert model to GGUF format
  ActionResult ConvertToGGUF(const std::string& model_id,
                              const std::string& quantization = "Q4_K_M",
                              ProgressCallback callback = nullptr);

  // Deploy to Ollama
  ActionResult DeployToOllama(const std::string& model_id,
                               const std::string& ollama_name = "",
                               const std::string& quantization = "Q4_K_M",
                               ProgressCallback callback = nullptr);

  // Test deployed model
  ActionResult TestModel(const std::string& model_id,
                          DeploymentBackend backend,
                          const std::string& test_prompt = "");

  // Check if Ollama is running
  bool IsOllamaRunning() const;

  // Check if llama.cpp is available
  bool IsLlamaCppAvailable() const;

  // Get last error
  const std::string& GetLastError() const { return last_error_; }

 private:
  // Execute command and capture output
  ActionResult ExecuteCommand(const std::vector<std::string>& args,
                               int timeout_seconds = 300);

  std::string afs_cli_path_;
  std::string llama_cpp_path_;
  std::string last_error_;
};

}  // namespace studio
}  // namespace afs

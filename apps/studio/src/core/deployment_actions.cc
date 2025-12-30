#include "deployment_actions.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>

namespace afs {
namespace studio {

DeploymentActions::DeploymentActions() {
  // Find afs CLI
  const char* home = std::getenv("HOME");
  if (home) {
    std::filesystem::path path(home);
    // Try common locations
    std::vector<std::filesystem::path> candidates = {
        path / "Code" / "afs" / ".venv" / "bin" / "afs",
        path / ".local" / "bin" / "afs",
        std::filesystem::path("/usr/local/bin/afs"),
    };
    for (const auto& p : candidates) {
      if (std::filesystem::exists(p)) {
        afs_cli_path_ = p.string();
        break;
      }
    }

    // Find llama.cpp
    llama_cpp_path_ = (path / "Code" / "llama.cpp").string();
  }
}

ActionResult DeploymentActions::ExecuteCommand(
    const std::vector<std::string>& args, int timeout_seconds) {
  ActionResult result;
  result.status = ActionStatus::kRunning;

  // Build command string
  std::string cmd;
  for (const auto& arg : args) {
    if (!cmd.empty()) cmd += " ";
    // Simple shell escaping
    if (arg.find(' ') != std::string::npos) {
      cmd += "\"" + arg + "\"";
    } else {
      cmd += arg;
    }
  }
  cmd += " 2>&1";

  // Execute command
  std::array<char, 4096> buffer;
  std::string output;

  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    result.status = ActionStatus::kFailed;
    result.error = "Failed to execute command: " + cmd;
    return result;
  }

  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output += buffer.data();
  }

  int ret = pclose(pipe);
  result.exit_code = WEXITSTATUS(ret);
  result.output = output;

  if (result.exit_code == 0) {
    result.status = ActionStatus::kCompleted;
    result.progress = 1.0f;
  } else {
    result.status = ActionStatus::kFailed;
    result.error = "Command failed with exit code " +
                   std::to_string(result.exit_code);
  }

  return result;
}

ActionResult DeploymentActions::PullModel(const std::string& model_id,
                                           const std::string& source,
                                           ProgressCallback callback) {
  if (afs_cli_path_.empty()) {
    ActionResult result;
    result.status = ActionStatus::kFailed;
    result.error = "afs CLI not found";
    return result;
  }

  std::vector<std::string> args = {afs_cli_path_, "models", "pull", model_id};
  if (source != "auto") {
    args.push_back("--source");
    args.push_back(source);
  }

  if (callback) callback("Pulling model...", 0.1f);
  auto result = ExecuteCommand(args, 600);  // 10 minute timeout
  if (callback) callback("Pull complete", 1.0f);

  return result;
}

ActionResult DeploymentActions::ConvertToGGUF(const std::string& model_id,
                                               const std::string& quantization,
                                               ProgressCallback callback) {
  if (afs_cli_path_.empty()) {
    ActionResult result;
    result.status = ActionStatus::kFailed;
    result.error = "afs CLI not found";
    return result;
  }

  std::vector<std::string> args = {afs_cli_path_, "models", "convert",
                                    model_id,      "gguf"};
  if (!quantization.empty()) {
    args.push_back("--quant");
    args.push_back(quantization);
  }

  if (callback) callback("Converting to GGUF...", 0.2f);
  auto result = ExecuteCommand(args, 1800);  // 30 minute timeout
  if (callback) callback("Conversion complete", 1.0f);

  return result;
}

ActionResult DeploymentActions::DeployToOllama(const std::string& model_id,
                                                const std::string& ollama_name,
                                                const std::string& quantization,
                                                ProgressCallback callback) {
  if (afs_cli_path_.empty()) {
    ActionResult result;
    result.status = ActionStatus::kFailed;
    result.error = "afs CLI not found";
    return result;
  }

  std::vector<std::string> args = {afs_cli_path_, "models", "deploy",
                                    model_id,      "ollama"};
  if (!ollama_name.empty()) {
    args.push_back("--name");
    args.push_back(ollama_name);
  }
  if (!quantization.empty()) {
    args.push_back("--quant");
    args.push_back(quantization);
  }

  if (callback) callback("Deploying to Ollama...", 0.3f);
  auto result = ExecuteCommand(args, 1800);
  if (callback) callback("Deployment complete", 1.0f);

  return result;
}

ActionResult DeploymentActions::TestModel(const std::string& model_id,
                                           DeploymentBackend backend,
                                           const std::string& test_prompt) {
  if (afs_cli_path_.empty()) {
    ActionResult result;
    result.status = ActionStatus::kFailed;
    result.error = "afs CLI not found";
    return result;
  }

  std::string backend_str;
  switch (backend) {
    case DeploymentBackend::kOllama:
      backend_str = "ollama";
      break;
    case DeploymentBackend::kLlamaCpp:
      backend_str = "llama.cpp";
      break;
    case DeploymentBackend::kHalextNode:
      backend_str = "halext-node";
      break;
  }

  std::vector<std::string> args = {afs_cli_path_, "models", "test", model_id,
                                    backend_str};

  return ExecuteCommand(args, 60);  // 1 minute timeout
}

bool DeploymentActions::IsOllamaRunning() const {
  // Check if Ollama is running
  FILE* pipe = popen("pgrep -x ollama 2>/dev/null", "r");
  if (!pipe) return false;

  char buffer[128];
  bool running = fgets(buffer, sizeof(buffer), pipe) != nullptr;
  pclose(pipe);

  return running;
}

bool DeploymentActions::IsLlamaCppAvailable() const {
  return !llama_cpp_path_.empty() &&
         std::filesystem::exists(llama_cpp_path_ + "/llama-quantize");
}

}  // namespace studio
}  // namespace afs

#pragma once

#include <string>
#include <vector>
#include <deque>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace afs::viz {

struct ChatMessage {
  std::string role;  // "user", "assistant", "system"
  std::string content;
};

struct LlamaConfig {
  std::string llama_cli_path = "~/llama.cpp/build/bin/llama-cli";
  std::string model_path;
  std::string rpc_servers;  // Comma-separated list of host:port
  int context_size = 4096;
  int n_predict = 256;
  float temperature = 0.7f;
  float top_p = 0.9f;
  bool use_rpc = true;
};

// Callback for streaming tokens
using TokenCallback = std::function<void(const std::string&)>;
using CompletionCallback = std::function<void(bool success, const std::string& error)>;

class LlamaClient {
 public:
  LlamaClient();
  ~LlamaClient();

  // Configuration
  void SetConfig(const LlamaConfig& config);
  const LlamaConfig& GetConfig() const { return config_; }

  // Connection management
  bool CheckHealth();
  bool IsReady() const { return is_ready_; }
  bool IsGenerating() const { return is_generating_; }
  std::string GetStatusMessage() const;

  // Chat interface
  void SendMessage(const std::string& message,
                   TokenCallback on_token,
                   CompletionCallback on_complete);
  void StopGeneration();
  void ClearHistory();

  // History access
  const std::vector<ChatMessage>& GetHistory() const { return history_; }

 private:
  void GenerationThread(const std::string& prompt,
                        TokenCallback on_token,
                        CompletionCallback on_complete);
  std::string BuildPrompt(const std::string& user_message);
  std::string ExpandPath(const std::string& path);

  LlamaConfig config_;
  std::vector<ChatMessage> history_;

  std::atomic<bool> is_ready_{false};
  std::atomic<bool> is_generating_{false};
  std::atomic<bool> stop_requested_{false};

  std::thread generation_thread_;
  std::mutex mutex_;
  std::string status_message_;

  // Process management
  pid_t current_pid_ = -1;
};

}  // namespace afs::viz

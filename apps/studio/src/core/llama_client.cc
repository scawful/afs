#include "llama_client.h"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <poll.h>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace afs::viz {

LlamaClient::LlamaClient() {
  status_message_ = "Not configured";
}

LlamaClient::~LlamaClient() {
  StopGeneration();
  if (generation_thread_.joinable()) {
    generation_thread_.join();
  }
}

void LlamaClient::SetConfig(const LlamaConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  status_message_ = "Configured, checking health...";
}

std::string LlamaClient::ExpandPath(const std::string& path) {
  if (path.empty()) return path;
  if (path[0] == '~') {
    const char* home = getenv("HOME");
    if (home) {
      return std::string(home) + path.substr(1);
    }
  }
  return path;
}

bool LlamaClient::CheckHealth() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string cli_path = ExpandPath(config_.llama_cli_path);

  // Check if llama-cli exists
  if (!std::filesystem::exists(cli_path)) {
    status_message_ = "llama-cli not found: " + cli_path;
    is_ready_ = false;
    return false;
  }

  // Check if model exists (if specified)
  if (!config_.model_path.empty()) {
    std::string model_path = ExpandPath(config_.model_path);
    if (!std::filesystem::exists(model_path)) {
      status_message_ = "Model not found: " + model_path;
      is_ready_ = false;
      return false;
    }
  }

  // Check RPC connectivity if enabled
  if (config_.use_rpc && !config_.rpc_servers.empty()) {
    // Parse first server
    size_t colon = config_.rpc_servers.find(':');
    if (colon != std::string::npos) {
      std::string host = config_.rpc_servers.substr(0, colon);
      std::string port_str = config_.rpc_servers.substr(colon + 1);
      // Remove any comma for multiple servers
      size_t comma = port_str.find(',');
      if (comma != std::string::npos) {
        port_str = port_str.substr(0, comma);
      }

      // Quick TCP check
      std::string cmd = "nc -z -w 2 " + host + " " + port_str + " 2>/dev/null";
      int result = system(cmd.c_str());
      if (result != 0) {
        status_message_ = "RPC server unreachable: " + host + ":" + port_str;
        is_ready_ = false;
        return false;
      }
    }
  }

  status_message_ = "Ready";
  is_ready_ = true;
  return true;
}

std::string LlamaClient::GetStatusMessage() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
  return status_message_;
}

std::string LlamaClient::BuildPrompt(const std::string& user_message) {
  std::ostringstream prompt;

  // Build chat history in ChatML format
  prompt << "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n";

  // History already contains the current user message (added before calling BuildPrompt)
  // So we just iterate through all history
  for (const auto& msg : history_) {
    prompt << "<|im_start|>" << msg.role << "\n" << msg.content << "<|im_end|>\n";
  }

  prompt << "<|im_start|>assistant\n";

  return prompt.str();
}

void LlamaClient::SendMessage(const std::string& message,
                               TokenCallback on_token,
                               CompletionCallback on_complete) {
  if (is_generating_) {
    if (on_complete) {
      on_complete(false, "Already generating");
    }
    return;
  }

  if (!is_ready_) {
    if (on_complete) {
      on_complete(false, "Client not ready: " + status_message_);
    }
    return;
  }

  // Add user message to history
  history_.push_back({"user", message});

  // Build prompt
  std::string prompt = BuildPrompt(message);

  // Start generation in background thread
  if (generation_thread_.joinable()) {
    generation_thread_.join();
  }

  is_generating_ = true;
  stop_requested_ = false;

  generation_thread_ = std::thread(&LlamaClient::GenerationThread, this,
                                    prompt, on_token, on_complete);
}

void LlamaClient::GenerationThread(const std::string& prompt,
                                    TokenCallback on_token,
                                    CompletionCallback on_complete) {
  std::string cli_path = ExpandPath(config_.llama_cli_path);
  std::string model_path = ExpandPath(config_.model_path);

  // Build command arguments
  std::vector<std::string> args;
  args.push_back(cli_path);

  if (config_.use_rpc && !config_.rpc_servers.empty()) {
    args.push_back("--rpc");
    args.push_back(config_.rpc_servers);
  }

  args.push_back("-m");
  args.push_back(model_path);
  args.push_back("-c");
  args.push_back(std::to_string(config_.context_size));
  args.push_back("-n");
  args.push_back(std::to_string(config_.n_predict));
  args.push_back("--temp");
  args.push_back(std::to_string(config_.temperature));
  args.push_back("--top-p");
  args.push_back(std::to_string(config_.top_p));
  args.push_back("-p");
  args.push_back(prompt);
  args.push_back("--no-display-prompt");
  args.push_back("--single-turn");  // Process prompt and exit, don't wait for input
  args.push_back("-e");  // Escape sequences

  // Create pipe for stdout
  int stdout_pipe[2];
  if (pipe(stdout_pipe) < 0) {
    is_generating_ = false;
    if (on_complete) on_complete(false, "Failed to create pipe");
    return;
  }

  // Fork and exec
  pid_t pid = fork();
  if (pid < 0) {
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    is_generating_ = false;
    if (on_complete) on_complete(false, "Failed to fork");
    return;
  }

  if (pid == 0) {
    // Child process
    close(stdout_pipe[0]);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stdout_pipe[1], STDERR_FILENO);
    close(stdout_pipe[1]);

    // Convert args to char**
    std::vector<char*> argv;
    for (auto& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    execvp(argv[0], argv.data());
    _exit(127);
  }

  // Parent process
  close(stdout_pipe[1]);
  current_pid_ = pid;

  // Set non-blocking
  int flags = fcntl(stdout_pipe[0], F_GETFL, 0);
  fcntl(stdout_pipe[0], F_SETFL, flags | O_NONBLOCK);

  std::string accumulated_response;
  std::string pending_buffer;  // Buffer to accumulate partial lines
  char buffer[256];
  bool in_preamble = true;  // Skip all preamble until we see actual generation

  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_message_ = "Loading model...";
  }

  while (!stop_requested_) {
    struct pollfd pfd;
    pfd.fd = stdout_pipe[0];
    pfd.events = POLLIN;

    int poll_result = poll(&pfd, 1, 100);  // 100ms timeout

    if (poll_result > 0 && (pfd.revents & POLLIN)) {
      ssize_t n = read(stdout_pipe[0], buffer, sizeof(buffer) - 1);
      if (n > 0) {
        buffer[n] = '\0';
        std::string chunk(buffer);

        // Skip preamble (ggml logs, loading messages, spinner, banner)
        if (in_preamble) {
          // Check for signs we're still in preamble
          bool is_preamble =
              chunk.find("ggml_") != std::string::npos ||
              chunk.find("Loading") != std::string::npos ||
              chunk.find("llama_") != std::string::npos ||
              chunk.find("GPU") != std::string::npos ||
              chunk.find("Metal") != std::string::npos ||
              chunk.find("tensor") != std::string::npos ||
              chunk.find("simd") != std::string::npos ||
              chunk.find("MTL") != std::string::npos ||
              chunk.find("build") != std::string::npos ||
              chunk.find("rpc") != std::string::npos ||
              chunk.find("CUDA") != std::string::npos ||
              chunk.find("Backend") != std::string::npos ||
              chunk.find("modalities") != std::string::npos ||
              chunk.find("available commands") != std::string::npos ||
              chunk.find("/exit") != std::string::npos ||
              chunk.find("/regen") != std::string::npos ||
              chunk.find("/clear") != std::string::npos ||
              chunk.find("/read") != std::string::npos ||
              // ASCII art banner
              chunk.find("▄") != std::string::npos ||
              chunk.find("█") != std::string::npos ||
              chunk.find("▀") != std::string::npos ||
              // Spinner characters (often appear alone)
              (chunk.length() <= 3 && (
                chunk.find("|") != std::string::npos ||
                chunk.find("-") != std::string::npos ||
                chunk.find("\\") != std::string::npos ||
                chunk.find("/") != std::string::npos));

          if (is_preamble) {
            // Update status with loading progress hints
            if (chunk.find("Loading") != std::string::npos) {
              std::lock_guard<std::mutex> lock(mutex_);
              status_message_ = "Loading model...";
            } else if (chunk.find("rpc") != std::string::npos) {
              std::lock_guard<std::mutex> lock(mutex_);
              status_message_ = "Connecting to RPC...";
            }
            continue;
          }

          // We've passed the preamble
          in_preamble = false;
          {
            std::lock_guard<std::mutex> lock(mutex_);
            status_message_ = "Generating...";
          }
        }

        accumulated_response += chunk;
        if (on_token) on_token(chunk);
      } else if (n == 0) {
        break;  // EOF
      }
    } else if (poll_result == 0) {
      // Timeout, check if process is still running
      int status;
      pid_t result = waitpid(pid, &status, WNOHANG);
      if (result == pid) {
        // Process finished, read any remaining output
        ssize_t remaining;
        while ((remaining = read(stdout_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
          buffer[remaining] = '\0';
          std::string chunk(buffer);
          accumulated_response += chunk;
          if (on_token) on_token(chunk);
        }
        break;
      }
    }
  }

  close(stdout_pipe[0]);

  // Stop if requested
  if (stop_requested_ && current_pid_ > 0) {
    kill(current_pid_, SIGTERM);
    usleep(100000);  // 100ms
    kill(current_pid_, SIGKILL);
  }

  // Wait for process
  int status;
  waitpid(pid, &status, 0);
  current_pid_ = -1;

  // Add assistant response to history
  if (!accumulated_response.empty()) {
    // Clean up response (remove ChatML end token if present)
    size_t end_pos = accumulated_response.find("<|im_end|>");
    if (end_pos != std::string::npos) {
      accumulated_response = accumulated_response.substr(0, end_pos);
    }
    history_.push_back({"assistant", accumulated_response});
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_message_ = "Ready";
  }

  is_generating_ = false;

  if (on_complete) {
    if (stop_requested_) {
      on_complete(false, "Stopped by user");
    } else {
      on_complete(true, "");
    }
  }
}

void LlamaClient::StopGeneration() {
  stop_requested_ = true;
  if (current_pid_ > 0) {
    kill(current_pid_, SIGTERM);
  }
}

void LlamaClient::ClearHistory() {
  std::lock_guard<std::mutex> lock(mutex_);
  history_.clear();
}

}  // namespace afs::viz

#include "chat_panel.h"
#include "../core.h"
#include "../../icons.h"
#include <imgui.h>
#include <vector>
#include <mutex>

namespace afs::viz::ui {

namespace {
  // Streaming response state (shared between callbacks and render)
  std::mutex streaming_mutex;
  std::string streaming_response;
  bool is_streaming = false;
}

void RenderChatPanel(AppState& state,
                     LlamaClient& llama_client,
                     std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  // Toolbar
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  // Status indicator
  bool is_ready = llama_client.IsReady();
  bool is_generating = llama_client.IsGenerating();

  if (is_generating) {
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), ICON_MD_PENDING " Generating...");
  } else if (is_ready) {
    ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), ICON_MD_CHECK_CIRCLE " Ready");
  } else {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), ICON_MD_ERROR " Not Connected");
  }

  ImGui::SameLine();
  if (ImGui::Button(ICON_MD_REFRESH " Check")) {
    llama_client.CheckHealth();
  }

  ImGui::SameLine();
  if (ImGui::Button(ICON_MD_DELETE " Clear")) {
    state.logs.clear();
    llama_client.ClearHistory();
    std::lock_guard<std::mutex> lock(streaming_mutex);
    streaming_response.clear();
    is_streaming = false;
  }

  ImGui::SameLine();
  if (is_generating && ImGui::Button(ICON_MD_STOP " Stop")) {
    llama_client.StopGeneration();
  }

  // Config section (collapsible)
  ImGui::SameLine();
  if (ImGui::CollapsingHeader(ICON_MD_SETTINGS " Config", ImGuiTreeNodeFlags_None)) {
    LlamaConfig config = llama_client.GetConfig();
    bool config_changed = false;

    ImGui::SetNextItemWidth(300);
    static char model_path[512] = "";
    if (model_path[0] == '\0' && !config.model_path.empty()) {
      strncpy(model_path, config.model_path.c_str(), sizeof(model_path) - 1);
    }
    if (ImGui::InputText("Model Path", model_path, sizeof(model_path))) {
      config.model_path = model_path;
      config_changed = true;
    }

    ImGui::SetNextItemWidth(200);
    static char rpc_servers[256] = "";
    if (rpc_servers[0] == '\0' && !config.rpc_servers.empty()) {
      strncpy(rpc_servers, config.rpc_servers.c_str(), sizeof(rpc_servers) - 1);
      rpc_servers[sizeof(rpc_servers) - 1] = '\0';
    }
    if (ImGui::InputText("RPC Servers", rpc_servers, sizeof(rpc_servers))) {
      config.rpc_servers = rpc_servers;
      config_changed = true;
    }

    if (ImGui::Checkbox("Use RPC", &config.use_rpc)) {
      config_changed = true;
    }

    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderFloat("Temperature", &config.temperature, 0.0f, 2.0f)) {
      config_changed = true;
    }

    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("Max Tokens", &config.n_predict, 32, 2048)) {
      config_changed = true;
    }

    if (config_changed) {
      llama_client.SetConfig(config);
    }
  }

  ImGui::PopStyleVar();
  ImGui::Separator();

  // Chat Area
  float footer_height = 50.0f;
  ImVec2 chat_size = ImVec2(0, ImGui::GetContentRegionAvail().y - footer_height);

  ImGui::BeginChild("ChatHistory", chat_size, true, ImGuiWindowFlags_AlwaysVerticalScrollbar);

  // Display history from LlamaClient
  const auto& history = llama_client.GetHistory();
  for (const auto& msg : history) {
    ImVec4 color;
    const char* icon;

    if (msg.role == "user") {
      color = ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
      icon = ICON_MD_PERSON;
    } else if (msg.role == "assistant") {
      color = ImVec4(0.4f, 1.0f, 0.6f, 1.0f);
      icon = ICON_MD_SMART_TOY;
    } else {
      color = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
      icon = ICON_MD_INFO;
    }

    ImGui::TextColored(color, "%s %s", icon, msg.role.c_str());
    ImGui::Indent();
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
    ImGui::TextUnformatted(msg.content.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Unindent();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
  }

  // Show streaming response
  {
    std::lock_guard<std::mutex> lock(streaming_mutex);
    if (is_streaming && !streaming_response.empty()) {
      ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), ICON_MD_SMART_TOY " assistant");
      ImGui::Indent();
      ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
      ImGui::TextUnformatted(streaming_response.c_str());
      ImGui::PopTextWrapPos();

      // Blinking cursor
      static float blink_timer = 0.0f;
      blink_timer += ImGui::GetIO().DeltaTime;
      if (fmod(blink_timer, 1.0f) < 0.5f) {
        ImGui::SameLine(0, 0);
        ImGui::Text("_");
      }

      ImGui::Unindent();
    }
  }

  // Auto-scroll
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 50) {
    ImGui::SetScrollHereY(1.0f);
  }

  ImGui::EndChild();

  // Input Area
  ImGui::Separator();
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 8));

  float send_button_width = 80.0f;
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - send_button_width - 10);

  bool send_triggered = false;
  if (ImGui::InputTextWithHint("##ChatInput", "Type a message...",
                                state.chat_input.data(), state.chat_input.size(),
                                ImGuiInputTextFlags_EnterReturnsTrue)) {
    send_triggered = true;
  }

  ImGui::SameLine();
  bool can_send = is_ready && !is_generating && state.chat_input[0] != '\0';

  if (!can_send) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button(ICON_MD_SEND " Send", ImVec2(send_button_width, 0)) || send_triggered) {
    if (can_send) {
      std::string message(state.chat_input.data());

      // Log user message
      if (log_callback) {
        log_callback("user", message, "user");
      }

      // Clear streaming state
      {
        std::lock_guard<std::mutex> lock(streaming_mutex);
        streaming_response.clear();
        is_streaming = true;
      }

      // Send to LlamaClient
      llama_client.SendMessage(
        message,
        // Token callback - accumulate streaming response
        [](const std::string& token) {
          std::lock_guard<std::mutex> lock(streaming_mutex);
          streaming_response += token;
        },
        // Completion callback
        [log_callback](bool success, const std::string& error) {
          std::lock_guard<std::mutex> lock(streaming_mutex);
          is_streaming = false;

          if (!success && log_callback) {
            log_callback("system", "Error: " + error, "system");
          }
        }
      );

      // Clear input
      state.chat_input[0] = '\0';
      ImGui::SetKeyboardFocusHere(-1);
    }
  }

  if (!can_send) {
    ImGui::EndDisabled();
  }

  ImGui::PopStyleVar();

  // Status bar
  ImGui::TextDisabled("%s", llama_client.GetStatusMessage().c_str());
}

} // namespace afs::viz::ui

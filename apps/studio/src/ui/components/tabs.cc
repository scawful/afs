#include "tabs.h"
#include "../core.h"
#include "../../icons.h"
#include <implot.h>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include "../../core/logger.h"

namespace afs {
namespace viz {
namespace ui {

void RenderKnobsTab(AppState& state, const DataLoader& loader, const std::string& data_path, std::function<void(const char*)> refresh_callback, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  ImGui::Text("Runtime Controls");
  if (ImGui::Button("Refresh Now")) {
    if (refresh_callback) refresh_callback("ui");
  }
  ImGui::SameLine();
  ImGui::Checkbox("Auto Refresh", &state.auto_refresh);
  ImGui::SliderFloat("Refresh Interval (s)", &state.refresh_interval_sec, 2.0f, 30.0f);

  ImGui::Separator();
  ImGui::Text("Simulation");
  ImGui::Checkbox("Simulate Activity", &state.simulate_activity);
  ImGui::SliderFloat("Agent Activity", &state.agent_activity_scale, 0.3f, 3.0f);
  ImGui::SliderFloat("Mission Bias", &state.mission_priority_bias, 0.5f, 2.0f);

  ImGui::Separator();
  ImGui::Text("Visualization");
  ImGui::SliderFloat("Chart Height (compact)", &state.chart_height, 130.0f, 240.0f);
  ImGui::Checkbox("Compact Charts", &state.compact_charts);
  ImGui::Checkbox("Show Status Strip", &state.show_status_strip);
  ImGui::Checkbox("Show Controls", &state.show_controls);
  ImGui::Checkbox("Show Systems Panel", &state.show_systems_panel);
  ImGui::Checkbox("Allow Workspace Scroll", &state.allow_workspace_scroll);
  ImGui::Checkbox("Plot Interaction", &state.enable_plot_interaction);
  if (state.enable_plot_interaction) {
    ImGui::Checkbox("Plot Zoom Requires Shift", &state.plot_interaction_requires_modifier);
    if (state.plot_interaction_requires_modifier) {
      ImGui::TextDisabled("Hold Shift + scroll/drag to pan/zoom plots.");
    }
  }
  ImGui::Checkbox("Reset Layout on Workspace Change", &state.reset_layout_on_workspace_change);
  ImGui::Checkbox("Auto Columns", &state.auto_chart_columns);
  if (!state.auto_chart_columns) {
    ImGui::SliderInt("Chart Columns", &state.chart_columns, 2, 4);
  }
  ImGui::SliderFloat("Embedding Sample Rate", &state.embedding_sample_rate, 0.1f, 1.0f);
  ImGui::SliderFloat("Quality Threshold", &state.quality_threshold, 0.4f, 0.95f);
  ImGui::SliderInt("Mission Concurrency", &state.mission_concurrency, 1, 12);
  ImGui::Checkbox("Verbose Logs", &state.verbose_logs);
  ImGui::Checkbox("Pulse Animations", &state.use_pulse_animations);
  ImGui::Checkbox("Plot Legends", &state.show_plot_legends);
  ImGui::Checkbox("Plot Markers", &state.show_plot_markers);
  ImGui::Checkbox("Data Scientist Mode", &state.data_scientist_mode);
  ImGui::Checkbox("Show All Chart Windows", &state.show_all_charts);

  ImGui::Separator();
  ImGui::Text("Advanced");
  if (ImGui::Button("Purge Mission Queue")) {
      state.missions.clear();
      if (log_callback) log_callback("system", "Mission queue purged.", "system");
  }
  ImGui::SameLine();
  if (ImGui::Button("Force Reconnect")) {
      if (log_callback) log_callback("system", "Forcing backend reconnect...", "system");
  }

  ImGui::Separator();
  ImGui::Text(ICON_MD_MODEL_TRAINING " Agent Training");
  ImGui::SliderFloat("Learning Rate", &state.trainer_lr, 0.00001f, 0.001f, "%.5f");
  ImGui::SliderInt("Batch Size", &state.trainer_batch_size, 8, 128);
  ImGui::SliderInt("Epochs", &state.trainer_epochs, 1, 100);
  ImGui::SliderFloat("Gen Temperature", &state.generator_temp, 0.1f, 2.0f);
  ImGui::SliderFloat("Rejection Floor", &state.rejection_threshold, 0.2f, 0.95f);

  ImGui::Separator();
  ImGui::TextDisabled("Data Path");
  ImGui::TextWrapped("%s", data_path.c_str());
}

void RenderAgentsTab(AppState& state, ImFont* font_ui, ImFont* font_header, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  ImGui::Text("Background Agents");
  ImGui::InputTextWithHint("##AgentName", "Agent name", state.new_agent_name.data(), state.new_agent_name.size());
  ImGui::InputTextWithHint("##AgentRole", "Role", state.new_agent_role.data(), state.new_agent_role.size());

  if (ImGui::Button("Spawn Agent")) {
    std::string name(state.new_agent_name.data());
    std::string role(state.new_agent_role.data());
    if (name.empty()) name = "Agent " + std::to_string(state.agents.size() + 1);
    if (role.empty()) role = "Generalist";

    AgentState agent;
    agent.name = name;
    agent.role = role;
    agent.status = "Active";
    agent.enabled = true;
    agent.activity_phase = 0.3f * static_cast<float>(state.agents.size() + 1);
    state.agents.push_back(std::move(agent));

    if (log_callback) log_callback(name, "Agent provisioned.", "agent");
  }

  ImGui::SameLine();
  ImGui::SetNextItemWidth(60.0f);
  ImGui::InputInt("##SpawnAgentCount", &state.spawn_agent_count);
  state.spawn_agent_count = std::max(1, std::min(state.spawn_agent_count, 12));
  ImGui::SameLine();
  if (ImGui::Button("Spawn Batch")) {
    std::string base_nameExtra(state.new_agent_name.data());
    if (base_nameExtra.empty()) base_nameExtra = "Agent";
    std::string role(state.new_agent_role.data());
    if (role.empty()) role = "Generalist";

    for (int i = 0; i < state.spawn_agent_count; ++i) {
      AgentState agent;
      agent.name = base_nameExtra + " " + std::to_string(state.agents.size() + 1);
      agent.role = role;
      agent.status = "Active";
      agent.enabled = true;
      agent.activity_phase = 0.3f * static_cast<float>(state.agents.size() + 1);
      state.agents.push_back(std::move(agent));
    }
    if (log_callback) log_callback("system", "Batch spawn complete.", "system");
  }

  if (ImGui::Button("Pause All")) {
    for (auto& agent : state.agents) {
      agent.enabled = false;
      agent.status = "Paused";
    }
    if (log_callback) log_callback("system", "All agents paused.", "system");
  }
  ImGui::SameLine();
  if (ImGui::Button("Resume All")) {
    for (auto& agent : state.agents) {
      agent.enabled = true;
      if (agent.status == "Paused") agent.status = "Idle";
    }
    if (log_callback) log_callback("system", "All agents resumed.", "system");
  }

  float table_height = ImGui::GetContentRegionAvail().y;
  if (ImGui::BeginTable("AgentsTable", 8, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY, ImVec2(0, table_height))) {
    ImGui::TableSetupColumn("Agent");
    ImGui::TableSetupColumn("Role");
    ImGui::TableSetupColumn("Status");
    ImGui::TableSetupColumn("Queue");
    ImGui::TableSetupColumn("Success");
    ImGui::TableSetupColumn("Latency");
    ImGui::TableSetupColumn("CPU/Mem");
    ImGui::TableSetupColumn("On");
    ImGui::TableHeadersRow();

    for (size_t i = 0; i < state.agents.size(); ++i) {
      auto& agent = state.agents[i];
      ImGui::PushID(static_cast<int>(i));
      const char* status = agent.enabled ? (agent.queue_depth > 0 ? "Busy" : "Idle") : "Paused";

      ImGui::TableNextRow();
      bool is_selected = (state.selected_agent_index == static_cast<int>(i));
      
      float pulse = 0.0f;
      if (state.use_pulse_animations && agent.enabled && agent.status != "Idle") {
          pulse = 0.5f + 0.5f * std::sin(state.pulse_timer * 6.0f);
      }
      
      if (pulse > 0.01f) {
          ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(GetStepColor(pulse * 0.2f, state)));
      }

      if (ImGui::Selectable(agent.name.c_str(), is_selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
        state.selected_agent_index = static_cast<int>(i);
      }

      ImGui::TableSetColumnIndex(1); ImGui::Text("%s", agent.role.c_str());
      ImGui::TableSetColumnIndex(2); ImGui::Text("%s", status);
      ImGui::TableSetColumnIndex(3); ImGui::Text("%d", agent.queue_depth);
      ImGui::TableSetColumnIndex(4); ImGui::Text("%.0f%%", agent.success_rate * 100.0f);
      ImGui::TableSetColumnIndex(5); ImGui::Text("%.1f ms", agent.avg_latency_ms);
      ImGui::TableSetColumnIndex(6); ImGui::Text("%.0f/%.0f", agent.cpu_pct, agent.mem_pct);
      ImGui::TableSetColumnIndex(7); ImGui::Checkbox("##enabled", &agent.enabled);
      ImGui::PopID();
    }
    ImGui::EndTable();
  }

  if (state.selected_agent_index >= 0 && state.selected_agent_index < static_cast<int>(state.agents.size())) {
    auto& agent = state.agents[state.selected_agent_index];
    ImGui::Separator();
    ImGui::Text("Agent Details: %s", agent.name.c_str());
    ImGui::Columns(2, "AgentDetailCols", false);
    ImGui::Text("Role: %s", agent.role.c_str());
    ImGui::Text("Status: %s", agent.status.c_str());
    ImGui::Text("Tasks: %d", agent.tasks_completed);
    ImGui::NextColumn();
    
    if (state.sparkline_data.size() < 40) {
       for(int i=0; i<40; ++i) state.sparkline_data.push_back(0.4f + 0.5f * (float)rand()/RAND_MAX);
    }
    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
    if (ImPlot::BeginPlot("##Sparkline", ImVec2(-1, 60), ImPlotFlags_CanvasOnly)) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::PlotLine("Activity", state.sparkline_data.data(), 40);
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleColor();
    ImGui::Columns(1);
    
    if (ImGui::Button("Reset Agent Stats")) {
        agent.tasks_completed = 0;
        if (log_callback) log_callback(agent.name, "Stats reset.", "agent");
    }
    ImGui::SameLine();
    if (ImGui::Button("Force Restart")) {
        if (log_callback) log_callback(agent.name, "Restarting...", "system");
    }
  }
}

void RenderMissionsTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  ImGui::Text("Mission Queue");
  ImGui::InputTextWithHint("##MissionName", "Mission name", state.new_mission_name.data(), state.new_mission_name.size());
  ImGui::InputTextWithHint("##MissionOwner", "Owner", state.new_mission_owner.data(), state.new_mission_owner.size());
  ImGui::SliderInt("Priority", &state.new_mission_priority, 1, 5);

  if (ImGui::Button("Create Mission")) {
    std::string name(state.new_mission_name.data());
    std::string owner(state.new_mission_owner.data());
    if (name.empty()) name = "Mission " + std::to_string(state.missions.size() + 1);
    if (owner.empty()) owner = "Ops";

    MissionState mission;
    mission.name = name;
    mission.owner = owner;
    mission.status = "Queued";
    mission.priority = state.new_mission_priority;
    state.missions.push_back(std::move(mission));
    if (log_callback) log_callback("system", "Mission queued: " + name, "system");
  }

  ImGui::SameLine();
  ImGui::SetNextItemWidth(60.0f);
  ImGui::InputInt("##SpawnMissionCount", &state.spawn_mission_count);
  state.spawn_mission_count = std::max(1, std::min(state.spawn_mission_count, 10));
  ImGui::SameLine();
  if (ImGui::Button("Spawn Batch")) {
    std::string base_nameExtra(state.new_mission_name.data());
    if (base_nameExtra.empty()) base_nameExtra = "Mission";
    std::string owner(state.new_mission_owner.data());
    if (owner.empty()) owner = "Ops";

    for (int i = 0; i < state.spawn_mission_count; ++i) {
      MissionState mission;
      mission.name = base_nameExtra + " " + std::to_string(state.missions.size() + 1);
      mission.owner = owner;
      mission.status = "Queued";
      mission.priority = state.new_mission_priority;
      state.missions.push_back(std::move(mission));
    }
    if (log_callback) log_callback("system", "Batch missions queued.", "system");
  }

  if (ImGui::Button("Clear Completed")) {
    state.missions.erase(std::remove_if(state.missions.begin(), state.missions.end(),
                                   [](const MissionState& mission) {
                                     return mission.status == "Complete" && !mission.data_backed;
                                   }), state.missions.end());
  }

  float table_height = ImGui::GetContentRegionAvail().y;
  if (ImGui::BeginTable("MissionsTable", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY, ImVec2(0, table_height))) {
    ImGui::TableSetupColumn("Mission");
    ImGui::TableSetupColumn("Owner");
    ImGui::TableSetupColumn("Status");
    ImGui::TableSetupColumn("Priority");
    ImGui::TableSetupColumn("Progress");
    ImGui::TableSetupColumn("Data");
    ImGui::TableHeadersRow();

    for (size_t i = 0; i < state.missions.size(); ++i) {
      auto& mission = state.missions[i];
      ImGui::PushID(static_cast<int>(i));
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0); ImGui::Text("%s", mission.name.c_str());
      ImGui::TableSetColumnIndex(1); ImGui::Text("%s", mission.owner.c_str());
      ImGui::TableSetColumnIndex(2); ImGui::Text("%s", mission.status.c_str());
      ImGui::TableSetColumnIndex(3); ImGui::Text("%d", mission.priority);
      ImGui::TableSetColumnIndex(4); ImGui::ProgressBar(mission.progress, ImVec2(-FLT_MIN, 0.0f));
      ImGui::TableSetColumnIndex(5); ImGui::Text("%s", mission.data_backed ? "data" : "live");
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
}

void RenderLogsTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  ImGui::Text(ICON_MD_CHAT " Agent Chat & Orchestration");
  
  // Provider Selection
  const char* providers[] = { "Local (Ollama)", "Remote (OpenWebUI)", "Direct (afs SVC)", "Mock" };
  static int current_provider = 0;
  ImGui::SetNextItemWidth(200);
  ImGui::Combo("Provider", &current_provider, providers, IM_ARRAYSIZE(providers));
  ImGui::SameLine();
  
  std::vector<const char*> agent_labels;
  agent_labels.push_back("All Agents");
  for (const auto& agent : state.agents) {
    agent_labels.push_back(agent.name.c_str());
  }
  if (state.log_agent_index < 0 || state.log_agent_index >= static_cast<int>(agent_labels.size())) {
    state.log_agent_index = 0;
  }
  ImGui::SetNextItemWidth(150);
  ImGui::Combo("Target", &state.log_agent_index, agent_labels.data(), static_cast<int>(agent_labels.size()));

  ImGui::Separator();

  float footer_height = 80.0f;
  float log_height = ImGui::GetContentRegionAvail().y - footer_height;
  if (log_height < 120.0f) log_height = 120.0f;

  ImGui::BeginChild("LogList", ImVec2(0, log_height), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);

  std::string filter(state.log_filter.data());
  for (const auto& entry : state.logs) {
    if (state.log_agent_index > 0 && entry.agent != agent_labels[state.log_agent_index]) continue;
    if (!filter.empty()) {
      if (entry.message.find(filter) == std::string::npos && entry.agent.find(filter) == std::string::npos) continue;
    }

    bool is_user = (entry.kind == "user");
    bool is_system = (entry.kind == "system");
    
    ImVec2 size = ImGui::GetContentRegionAvail();
    float bubble_width = std::min(size.x * 0.75f, 500.0f);
    
    if (is_user) ImGui::SetCursorPosX(size.x - bubble_width - 8.0f);
    else ImGui::SetCursorPosX(8.0f);

    ImGui::BeginGroup();
    
    // Bubble header
    ImGui::TextDisabled("%s", entry.agent.c_str());
    
    // Bubble body
    ImVec4 bg_color = is_user ? ImVec4(0.2f, 0.4f, 0.8f, 0.4f) : (is_system ? ImVec4(0.3f, 0.3f, 0.3f, 0.4f) : ImVec4(0.2f, 0.6f, 0.4f, 0.4f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_color);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 8));
    
    std::string child_id = "msg_" + std::to_string(&entry - &state.logs[0]);
    float text_h = ImGui::CalcTextSize(entry.message.c_str(), nullptr, false, bubble_width - 20).y + 20;
    
    ImGui::BeginChild(child_id.c_str(), ImVec2(bubble_width, text_h), true, ImGuiWindowFlags_NoScrollbar);
    ImGui::TextWrapped("%s", entry.message.c_str());
    ImGui::EndChild();
    
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    
    ImGui::EndGroup();
    ImGui::Spacing();
  }
  
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
      ImGui::SetScrollHereY(1.0f);

  ImGui::EndChild();

  ImGui::Separator();
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
  if (ImGui::InputTextWithHint("##ChatInput", "Message swarm...", state.chat_input.data(), state.chat_input.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
      goto send_msg;
  }
  ImGui::SameLine();
  if (ImGui::Button("Send", ImVec2(90, 0))) {
  send_msg:
    std::string message(state.chat_input.data());
    if (!message.empty()) {
      std::string target = agent_labels[state.log_agent_index];
      if (log_callback) log_callback("user", message, "user");
      
      // Simulate response
      if (state.log_agent_index > 0) {
        if (log_callback) log_callback(target, "Processing request via " + std::string(providers[current_provider]) + ": " + message, "agent");
      } else {
        if (log_callback) log_callback("Orchestrator", "Coordinating agents via " + std::string(providers[current_provider]) + "...", "system");
      }
      
      state.chat_input[0] = '\0';
    }
  }
}

void RenderContextTab(AppState& state, TextEditor& text_editor, MemoryEditorWidget& memory_editor, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  ImGui::Text("AFS Context Browser");
  ImGui::Separator();

  if (ImGui::Button("..") || ImGui::Button("Up")) {
    if (state.current_browser_path.has_parent_path()) {
      state.current_browser_path = state.current_browser_path.parent_path();
      state.browser_entries.clear();
    }
  }
  ImGui::SameLine();
  ImGui::TextDisabled("Path: %s", state.current_browser_path.string().c_str());

  if (state.browser_entries.empty()) RefreshBrowserEntries(state);

  float table_height = ImGui::GetContentRegionAvail().y * 0.6f;
  if (ImGui::BeginTable("FileBrowser", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, table_height))) {
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableHeadersRow();

    for (const auto& entry : state.browser_entries) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      if (entry.is_directory) {
        if (ImGui::Selectable((entry.name + "/").c_str(), false)) {
          state.current_browser_path = entry.path;
          state.browser_entries.clear();
          break;
        }
      } else {
        if (ImGui::Selectable(entry.name.c_str(), state.selected_file_path == entry.path)) {
            LoadFile(state, entry.path, text_editor);
        }
      }

      ImGui::TableNextColumn();
      if (!entry.is_directory) ImGui::TextDisabled("%.1f KB", entry.size / 1024.0f);

      ImGui::TableNextColumn();
      if (!entry.is_directory) {
        if (ImGui::Button(("Add##" + entry.name).c_str())) {
          ContextItem item;
          item.name = entry.name;
          item.path = entry.path;
          item.type = entry.path.extension().string();
          state.selected_context.push_back(item);
          if (log_callback) log_callback("system", "Added to context: " + entry.name, "system");
        }
      }
    }
    ImGui::EndTable();
  }

  ImGui::Separator();
  ImGui::Text("File View (%s)", state.selected_file_path.filename().string().c_str());
  if (state.is_binary_view) {
      memory_editor.DrawContents(state.binary_data.data(), state.binary_data.size());
  } else {
      text_editor.Render("TextEditor", ImVec2(0, 0), true);
  }

  ImGui::Separator();
  ImGui::Text("Selected Context (%d items)", (int)state.selected_context.size());
  ImGui::BeginChild("ContextList", ImVec2(0, 0), true);
  for (size_t i = 0; i < state.selected_context.size(); ++i) {
    auto& item = state.selected_context[i];
    ImGui::PushID((int)i);
    ImGui::Checkbox("##on", &item.enabled);
    ImGui::SameLine();
    ImGui::Text("%s", item.name.c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", item.path.string().c_str());
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 40);
    if (ImGui::Button("X")) {
      state.selected_context.erase(state.selected_context.begin() + i);
      ImGui::PopID();
      break;
    }
    ImGui::PopID();
  }
  ImGui::EndChild();
}

void RenderAgentPromptTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  ImGui::Text("Agent Orchestrator Prompt");
  ImGui::Separator();

  ImGui::Text("System Prompt");
  ImGui::InputTextMultiline("##SysPrompt", state.system_prompt.data(), state.system_prompt.size(), ImVec2(-1, 80));

  ImGui::Spacing();
  ImGui::Text("User Message");
  ImGui::InputTextMultiline("##UserPrompt", state.user_prompt.data(), state.user_prompt.size(), ImVec2(-1, 120));

  static int target_agent = 0;
  std::vector<const char*> agent_names = {"Orchestrator", "Coordinator"};
  for (const auto& a : state.agents) agent_names.push_back(a.name.c_str());
  
  ImGui::Combo("Target", &target_agent, agent_names.data(), (int)agent_names.size());

  if (ImGui::Button("Trigger Background Agent", ImVec2(-1, 40))) {
    std::string msg = "Sent prompt to " + std::string(agent_names[target_agent]);
    if (log_callback) log_callback("user", msg, "user");
    if (log_callback) log_callback(agent_names[target_agent], "Analyzing context with " + std::to_string(state.selected_context.size()) + " files...", "agent");
    state.user_prompt[0] = '\0';
  }

  ImGui::Separator();
  ImGui::TextDisabled("Quick Context Meta:");
  for (const auto& item : state.selected_context) {
    if (item.enabled) {
      ImGui::TextDisabled(" - %s (%s)", item.name.c_str(), item.type.c_str());
    }
  }
}

void RenderTablesTab(AppState& state, const DataLoader& loader) {
  if (ImGui::BeginTabBar("TableGroups")) {
    if (ImGui::BeginTabItem("Generator Detailed")) {
      const auto& stats = loader.GetGeneratorStats();
      if (ImGui::BeginTable("GenDetailed", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Generator");
        ImGui::TableSetupColumn("Accepted");
        ImGui::TableSetupColumn("Rejected");
        ImGui::TableSetupColumn("Rate %");
        ImGui::TableSetupColumn("Avg Q");
        ImGui::TableSetupColumn("Status");
        ImGui::TableHeadersRow();

        for (const auto& s : stats) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn(); ImGui::Text("%s", s.name.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%d", s.samples_accepted);
          ImGui::TableNextColumn(); ImGui::Text("%d", s.samples_rejected);
          ImGui::TableNextColumn(); 
          float rate = s.acceptance_rate * 100.0f;
          ImGui::Text("%.1f%%", rate);
          if (rate < 40.0f) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1,0,0,1), "[!] "); }
          
          ImGui::TableNextColumn(); ImGui::Text("%.3f", s.avg_quality);
          ImGui::TableNextColumn(); 
          if (s.samples_rejected > s.samples_accepted) ImGui::TextColored(ImVec4(1,0.5,0,1), "Struggling");
          else ImGui::TextColored(ImVec4(0,1,0.5,1), "Healthy");
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }
    
    if (ImGui::BeginTabItem("Quality Metrics")) {
      const auto& trends = loader.GetQualityTrends();
      if (ImGui::BeginTable("QualityDetailed", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Domain");
        ImGui::TableSetupColumn("Metric");
        ImGui::TableSetupColumn("Mean Score");
        ImGui::TableSetupColumn("Trend");
        ImGui::TableSetupColumn("Sparkline");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < trends.size(); ++i) {
          const auto& t = trends[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn(); ImGui::Text("%s", t.domain.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%s", t.metric.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%.4f", t.mean);
          ImGui::TableNextColumn(); 
          if (t.trend_direction == "improving") ImGui::TextColored(ImVec4(0,1,0,1), "INC");
          else if (t.trend_direction == "declining") ImGui::TextColored(ImVec4(1,0,0,1), "DEC");
          else ImGui::Text("---");
          
          ImGui::TableNextColumn();
          ImGui::PushID((int)i);
          if (ImPlot::BeginPlot("##Spark", ImVec2(-1, 24), ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs)) {
            ImPlot::SetupAxes(nullptr,nullptr,ImPlotAxisFlags_NoDecorations,ImPlotAxisFlags_NoDecorations);
            ImPlot::PlotLine("##v", t.values.data(), (int)t.values.size());
            ImPlot::EndPlot();
          }
          ImGui::PopID();
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Training Runs")) {
        const auto& runs = loader.GetTrainingRuns();
        if (ImGui::BeginTable("TrainingDetailed", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("Run ID");
            ImGui::TableSetupColumn("Model");
            ImGui::TableSetupColumn("Samples");
            ImGui::TableSetupColumn("Loss");
            ImGui::TableSetupColumn("Eval Metrics");
            ImGui::TableHeadersRow();

            for (const auto& r : runs) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("%s", r.run_id.substr(0, 16).c_str());
                ImGui::TableNextColumn(); ImGui::Text("%s", r.model_name.c_str());
                ImGui::TableNextColumn(); ImGui::Text("%d", r.samples_count);
                ImGui::TableNextColumn(); ImGui::Text("%.5f", r.final_loss);
                ImGui::TableNextColumn();
                for (const auto& [name, val] : r.eval_metrics) {
                    ImGui::TextDisabled("%s: %.3f", name.c_str(), val);
                }
            }
            ImGui::EndTable();
        }
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void RenderServicesTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  ImGui::Text("Core Services");
  
  auto render_service = [&log_callback](const char* name, const char* status, float health, const char* desc) {
    ImGui::PushID(name);
    ImGui::BeginGroup();
    ImGui::Text("%s", name);
    ImGui::TextDisabled("%s", desc);
    
    ImVec4 status_color = ImVec4(0.4f, 0.8f, 0.4f, 1.0f);
    if (strcmp(status, "Warning") == 0) status_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);
    else if (strcmp(status, "Error") == 0) status_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
    
    ImGui::TextColored(status_color, "Status: %s", status);
    ImGui::ProgressBar(health, ImVec2(-1.0f, 0.0f));
    ImGui::EndGroup();
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
  };

  render_service("Orchestrator", "Active", 0.98f, "Main mission control & dispatch");
  render_service("Knowledge Base", "Active", 0.92f, "Vector DB & Concept Index");
  render_service("Trainer", "Active", 0.85f, "Active learning loop coordination");
  render_service("Embedding SVC", "Warning", 0.65f, "Latency spike detected in region 42");
  
  if (ImGui::Button("Reset Services")) {
    if (log_callback) log_callback("system", "Services reset signal sent.", "system");
  }
}

void RefreshBrowserEntries(AppState& state) {
  state.browser_entries.clear();
  if (state.current_browser_path.has_parent_path()) {
      state.browser_entries.push_back({"..", state.current_browser_path.parent_path(), true, 0, false});
  }
  
  std::error_code ec;
  auto it = std::filesystem::directory_iterator(state.current_browser_path, ec);
  if (ec) {
      LOG_ERROR("Failed to open directory: " + state.current_browser_path.string() + " (" + ec.message() + ")");
      return;
  }

  try {
    for (const auto& entry : it) {
      std::error_code entry_ec;
      if (entry.is_directory(entry_ec)) {
          std::error_code context_ec;
          bool has_context = false;
          try {
              has_context = std::filesystem::exists(entry.path() / ".context", context_ec);
          } catch (...) {}
          state.browser_entries.push_back({entry.path().filename().string(), entry.path(), true, 0, has_context});
      } else if (entry.is_regular_file(entry_ec)) {
          state.browser_entries.push_back({entry.path().filename().string(), entry.path(), false, entry.file_size(entry_ec), false});
      }
    }
  } catch (...) {
      LOG_WARN("Error during directory iteration in " + state.current_browser_path.string());
  }
  
  std::sort(state.browser_entries.begin(), state.browser_entries.end(), [](const FileEntry& a, const FileEntry& b) {
      if (a.is_directory != b.is_directory) return a.is_directory > b.is_directory;
      return a.name < b.name;
  });
}

void LoadFile(AppState& state, const std::filesystem::path& path, TextEditor& text_editor) {
  state.selected_file_path = path;
  std::string ext = path.extension().string();
  static const std::vector<std::string> text_exts = {
      ".cpp", ".cc", ".c", ".h", ".hpp", ".py", ".md", ".json", ".txt", ".xml", ".org", ".asm", ".s", ".cmake", ".yml", ".yaml", ".sh"
  };
  
  bool is_text = false;
  for (const auto& e : text_exts) if (ext == e) { is_text = true; break; }
  
  if (is_text) {
      state.is_binary_view = false;
      std::ifstream t(path);
      if (t.is_open()) {
          std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
          text_editor.SetText(str);
          if (ext == ".cpp" || ext == ".cc" || ext == ".h" || ext == ".hpp") text_editor.SetLanguageDefinition(TextEditor::LanguageDefinition::CPlusPlus());
          else if (ext == ".sql") text_editor.SetLanguageDefinition(TextEditor::LanguageDefinition::SQL());
          else text_editor.SetLanguageDefinition(TextEditor::LanguageDefinition());
      }
  } else {
      state.is_binary_view = true;
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (file.is_open()) {
          std::streamsize size = file.tellg();
          file.seekg(0, std::ios::beg);
          if (size <= 10 * 1024 * 1024) {
              state.binary_data.resize(size);
              file.read((char*)state.binary_data.data(), size);
          } else state.binary_data.clear();
      }
  }
}

void RenderMarkdown(const std::string& content, ImFont* font_ui, ImFont* font_header, ThemeProfile current_theme) {
  if (font_ui) ImGui::PushFont(font_ui);
  const char* p = content.c_str();
  const char* end = p + content.size();
  while (p < end) {
      const char* line_end = strchr(p, '\n');
      if (!line_end) line_end = end;
      std::string line(p, line_end);
      if (line.substr(0, 2) == "# ") {
          if (font_header) ImGui::PushFont(font_header);
          ImGui::TextColored(GetThemeColor(ImGuiCol_PlotLines, current_theme), "%s", line.substr(2).c_str());
          if (font_header) ImGui::PopFont();
      } else if (line.substr(0, 3) == "## ") {
          if (font_header) ImGui::PushFont(font_header);
          ImGui::Text("%s", line.substr(3).c_str());
          if (font_header) ImGui::PopFont();
      } else if (line.substr(0, 4) == "### ") ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", line.substr(4).c_str());
      else if (line.substr(0, 2) == "- ") { ImGui::Bullet(); ImGui::SameLine(); ImGui::TextWrapped("%s", line.substr(2).c_str()); }
      else ImGui::TextWrapped("%s", line.c_str());
      p = line_end + 1;
  }
  if (font_ui) ImGui::PopFont();
}

} // namespace ui
} // namespace viz
} // namespace afs

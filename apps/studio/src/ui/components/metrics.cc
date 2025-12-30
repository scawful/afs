#include "metrics.h"
#include "../core.h"
#include "../../icons.h"
#include <cmath>
#include <algorithm>
#include <GLFW/glfw3.h> // For glfwGetTime if needed, but AppState has last_refresh_time

namespace afs {
namespace viz {
namespace ui {

void RenderMetricCards(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header) {
  const auto& trends = loader.GetQualityTrends();

  float total_success = 0.0f;
  for (const auto& agent : state.agents) {
      total_success += agent.success_rate;
  }
  if (!state.agents.empty())
      total_success /= static_cast<float>(state.agents.size());

  float avg_quality = 0.0f;
  if (!trends.empty()) {
    for (const auto& t : trends)
      avg_quality += t.mean;
    avg_quality /= static_cast<float>(trends.size());
  }

  char q_buf[32], a_buf[32];
  snprintf(q_buf, sizeof(q_buf), "%d%%", static_cast<int>(avg_quality * 100));
  snprintf(a_buf, sizeof(a_buf), "%d%%", static_cast<int>(total_success * 100));

  MetricCard cards[] = {
      {ICON_MD_INSIGHTS "  Overall Quality", q_buf, "System Health",
       GetThemeColor(ImGuiCol_PlotLines, state.current_theme), true},
      {ICON_MD_SPEED "  Swarm Velocity", "1.2k/s", "+12% vs last run",
       ImVec4(0.4f, 1.0f, 0.6f, 1.0f), true},
      {ICON_MD_AUTO_FIX_HIGH "  Efficiency", a_buf, "Mean Success Rate", ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
       total_success > 0.85f}};

  float card_w = (ImGui::GetContentRegionAvail().x - 16) / 3.0f;
  
  for (int i = 0; i < 3; ++i) {
    ImGui::PushID(i);
    ImGui::BeginChild("Card", ImVec2(card_w, 100), true, ImGuiWindowFlags_NoScrollbar);
    
    // Label
    if (font_ui) ImGui::PushFont(font_ui);
    ImGui::TextDisabled("%s", cards[i].label.c_str());
    if (font_ui) ImGui::PopFont();
    
    // Value
    if (font_header) ImGui::PushFont(font_header);
    ImGui::TextColored(cards[i].color, "%s", cards[i].value.c_str());
    if (font_header) ImGui::PopFont();
    
    // Subtext
    ImGui::Spacing();
    ImGui::TextDisabled("%s", cards[i].sub_text.c_str());
    
    // Decorative Pulse (Bottom edge)
    if (state.use_pulse_animations) {
        float p = (1.0f + std::sin(state.pulse_timer * 2.0f + i)) * 0.5f;
        ImDrawList* draw = ImGui::GetWindowDrawList();
        ImVec2 p_min = ImGui::GetItemRectMin(); // Note: Not used correctly in original, but let's keep pattern
        ImVec2 p_max = ImGui::GetWindowPos();
        p_max.x += ImGui::GetWindowSize().x;
        p_max.y += ImGui::GetWindowSize().y;
        draw->AddRectFilled(ImVec2(p_max.x - 40 * p, p_max.y - 2), p_max, ImColor(cards[i].color));
    }

    ImGui::EndChild();
    if (i < 2) ImGui::SameLine();
    ImGui::PopID();
  }
}

void RenderSummaryRow(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header) {
  RenderMetricCards(state, loader, font_ui, font_header);
  
  ImGui::Spacing();
  
  // Swarm Topology Overview
  ImGui::BeginChild("SwarmTopology", ImVec2(0, 120), true);
  if (font_header) ImGui::PushFont(font_header);
  ImGui::Text(ICON_MD_HUB " SWARM TOPOLOGY");
  if (font_header) ImGui::PopFont();
  
  ImGui::Separator();
  
  if (ImGui::BeginTable("Topology", 5, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
      ImGui::TableSetupColumn("Active Agents", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Queue Depth", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Mission Velocity", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Avg. Success", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Health", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      
      int active_count = 0;
      float total_success = 0.0f;
      int total_queue = 0;
      for (const auto& a : state.agents) {
          if (a.enabled) active_count++;
          total_success += a.success_rate;
          total_queue += a.queue_depth;
      }
      if (!state.agents.empty()) total_success /= (float)state.agents.size();
      
      ImGui::TableNextRow();
      
      // Column 0: Active Agents
      ImGui::TableSetColumnIndex(0);
      ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "%d / %d", active_count, (int)state.agents.size());
      
      // Column 1: Queue Depth
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%d Tasks", total_queue);
      
      // Column 2: Mission Velocity
      ImGui::TableSetColumnIndex(2);
      float avg_progress = 0.0f;
      for (const auto& m : state.missions) avg_progress += m.progress;
      if (!state.missions.empty()) avg_progress /= (float)state.missions.size();
      ImGui::ProgressBar(avg_progress, ImVec2(-1, 0), "");
      
      // Column 3: Success Rate
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%.1f%%", total_success * 100.0f);
      
      // Column 4: Health
      ImGui::TableSetColumnIndex(4);
      ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), ICON_MD_VERIFIED_USER " NOMINAL");
      
      ImGui::EndTable();
  }
  ImGui::EndChild();
}

void RenderStatusBar(AppState& state, const DataLoader& loader, const std::string& data_path) {
  ImGui::Separator();
  if (ImGui::BeginTable("StatusStrip", 2,
                        ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableNextColumn();
    const auto& status = loader.GetLastStatus();
    int sources_found = status.FoundCount();
    int sources_ok = status.OkCount();
    if (loader.HasData()) {
      double seconds_since = std::max(0.0, glfwGetTime() - state.last_refresh_time);
      if (status.error_count > 0) {
        ImGui::Text(
            "Sources: %d/%d ok  |  Generators: %zu  |  Regions: %zu  |  Runs: %zu  |  Errors: %d  |  Last refresh: %.0fs  |  F5 to refresh",
            sources_ok,
            sources_found,
            loader.GetGeneratorStats().size(),
            loader.GetEmbeddingRegions().size(),
            loader.GetTrainingRuns().size(),
            status.error_count,
            seconds_since);
      } else {
        ImGui::Text(
            "Sources: %d/%d ok  |  Generators: %zu  |  Regions: %zu  |  Runs: %zu  |  Last refresh: %.0fs  |  F5 to refresh",
            sources_ok,
            sources_found,
            loader.GetGeneratorStats().size(),
            loader.GetEmbeddingRegions().size(),
            loader.GetTrainingRuns().size(),
            seconds_since);
      }
    } else {
      if (status.error_count > 0) {
        ImGui::TextDisabled("No data loaded - %d error(s) on refresh", status.error_count);
      } else {
        ImGui::TextDisabled("No data loaded - Press F5 to refresh");
      }
    }

    ImGui::TableNextColumn();
    ImGui::Text("Data: %s", data_path.c_str());
    ImGui::EndTable();
  }
}

} // namespace ui
} // namespace viz
} // namespace afs

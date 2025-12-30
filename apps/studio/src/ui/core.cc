#include "core.h"
#include <implot.h>
#include <algorithm>
#include <vector>
#include "../icons.h"

namespace afs {
namespace viz {
namespace ui {

void HelpMarker(const char* desc) {
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

void ApplyPremiumPlotStyles(const char* plot_id, AppState& state) {
  // Custom theme-like styling for AFS
  ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.18f);
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, state.line_weight);
  ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, state.show_plot_markers ? 5.5f : 0.0f);
  ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, 1.4f);
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(14, 14));
  ImPlot::PushStyleVar(ImPlotStyleVar_LabelPadding, ImVec2(6, 4));
  
  // Standard grid line color (subtle)
  ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(1, 1, 1, 0.12f));
  ImPlot::PushStyleColor(ImPlotCol_Line, GetThemeColor(ImGuiCol_PlotLines, state.current_theme));
}

void RenderChartHeader(PlotKind kind, const char* title, const char* desc, AppState& state) {
  ImGui::PushStyleColor(ImGuiCol_Text, GetThemeColor(ImGuiCol_PlotLines, state.current_theme));
  ImGui::Text("%s", title);
  ImGui::PopStyleColor();
  if (desc && desc[0] != '\0') {
    ImGui::SameLine();
    HelpMarker(desc);
  }

  if (kind != PlotKind::None) {
    float button_size = ImGui::GetFrameHeight();
    ImGui::SameLine();
    float right_edge = ImGui::GetWindowContentRegionMax().x - button_size;
    ImGui::SetCursorPosX(right_edge);
    ImGui::PushID(static_cast<int>(kind));
    const char* icon = state.is_rendering_expanded_plot ? ICON_MD_CLOSE_FULLSCREEN
                                                   : ICON_MD_OPEN_IN_FULL;
    if (ImGui::Button(icon, ImVec2(button_size, button_size))) {
      if (state.is_rendering_expanded_plot) {
        state.expanded_plot = PlotKind::None;
      } else {
        state.expanded_plot = kind;
      }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip(state.is_rendering_expanded_plot ? "Exit full screen"
                                                     : "Expand to full screen");
    }

    // New: Pop-out button (GIMP style)
    ImGui::SameLine();
    ImGui::SetCursorPosX(right_edge - button_size - 4);
    if (ImGui::Button(ICON_MD_OPEN_IN_NEW, ImVec2(button_size, button_size))) {
      bool found = false;
      for (auto f : state.active_floaters) {
        if (f == kind) { found = true; break; }
      }
      if (!found) state.active_floaters.push_back(kind);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Pop out to standalone window");

    // Update inspector context if clicked
    if (ImGui::IsItemClicked()) {
        state.inspector_context = kind;
    }

    ImGui::PopID();
  }

  ImGui::Dummy(ImVec2(0.0f, 4.0f));
}

void HandlePlotContextMenu(PlotKind kind, AppState& state) {
  if (kind == PlotKind::None) return;
  ImGui::PushID(static_cast<int>(kind));
  if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    ImGui::OpenPopup("PlotContext");
  }
    if (ImGui::BeginPopup("PlotContext")) {
    if (state.expanded_plot == kind) {
      if (ImGui::MenuItem(ICON_MD_FULLSCREEN_EXIT " Exit Full Screen")) {
        state.expanded_plot = PlotKind::None;
      }
    } else if (kind != PlotKind::None) {
      if (ImGui::MenuItem(ICON_MD_FULLSCREEN " Expand to Full Screen")) {
        state.expanded_plot = kind;
      }
    }
    
    if (state.focus_chart == kind) {
        if (ImGui::MenuItem(ICON_MD_REMOVE_CIRCLE_OUTLINE " Remove from Focus")) {
            state.focus_chart = PlotKind::None;
        }
    } else if (kind != PlotKind::None) {
        if (ImGui::MenuItem(ICON_MD_CENTER_FOCUS_STRONG " Focus on Dashboard")) {
            state.focus_chart = kind;
            state.current_workspace = Workspace::Dashboard;
        }
    }

    if (ImGui::MenuItem(ICON_MD_OPEN_IN_NEW " Pop out to Window")) {
        bool already_open = false;
        for (auto k : state.active_floaters) { if (k == kind) { already_open = true; break; } }
        if (!already_open) state.active_floaters.push_back(kind);
    }

    if (ImGui::MenuItem("Reset View")) {
      ImPlot::SetNextAxesToFit();
    }

    ImGui::Separator();

    if (ImGui::BeginMenu("Settings")) {
      ImGui::Checkbox("Legends", &state.show_plot_legends);
      ImGui::Checkbox("Markers", &state.show_plot_markers);
      ImGui::Checkbox("Interaction", &state.enable_plot_interaction);
      ImGui::EndMenu();
    }

    ImGui::EndPopup();
  }
  ImGui::PopID();
}

ImVec4 GetThemeColor(ImGuiCol col, ThemeProfile theme) {
  switch (theme) {
    case ThemeProfile::Amber:
      if (col == ImGuiCol_Text) return ImVec4(1.0f, 0.7f, 0.0f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(1.0f, 0.6f, 0.0f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
      break;
    case ThemeProfile::Emerald:
      if (col == ImGuiCol_Text) return ImVec4(0.0f, 1.0f, 0.4f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(0.0f, 0.8f, 0.3f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(0.2f, 1.0f, 0.5f, 1.0f);
      break;
    case ThemeProfile::Cyberpunk:
      if (col == ImGuiCol_Text) return ImVec4(1.0f, 0.0f, 0.5f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(0.1f, 0.0f, 0.2f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(0.0f, 1.0f, 1.0f, 1.0f);
      break;
    case ThemeProfile::Monochrome:
      if (col == ImGuiCol_Text) return ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(0.2f, 0.2f, 0.2f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      break;
    case ThemeProfile::Solarized:
      if (col == ImGuiCol_Text) return ImVec4(0.52f, 0.60f, 0.00f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(0.03f, 0.21f, 0.26f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(0.15f, 0.45f, 0.55f, 1.0f);
      break;
    case ThemeProfile::Nord:
      if (col == ImGuiCol_Text) return ImVec4(0.56f, 0.80f, 0.71f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(0.18f, 0.20f, 0.25f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(0.53f, 0.75f, 0.82f, 1.0f);
      break;
    case ThemeProfile::Dracula:
      if (col == ImGuiCol_Text) return ImVec4(1.00f, 0.47f, 0.77f, 1.0f);
      if (col == ImGuiCol_Header) return ImVec4(0.16f, 0.17f, 0.24f, 0.4f);
      if (col == ImGuiCol_PlotLines) return ImVec4(0.74f, 0.57f, 0.97f, 1.0f);
      break;
    default: // Cobalt
      if (col == ImGuiCol_PlotLines) return ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
      break;
  }
  return ImGui::GetStyleColorVec4(col);
}

ImVec4 GetSeriesColor(int index) {
  static const ImVec4 palette[] = {
      ImVec4(0.20f, 0.65f, 0.96f, 1.0f),
      ImVec4(0.96f, 0.54f, 0.24f, 1.0f),
      ImVec4(0.90f, 0.28f, 0.46f, 1.0f),
      ImVec4(0.38f, 0.88f, 0.46f, 1.0f),
      ImVec4(0.86f, 0.80f, 0.26f, 1.0f),
      ImVec4(0.62f, 0.38f, 0.96f, 1.0f),
      ImVec4(0.20f, 0.86f, 0.82f, 1.0f),
      ImVec4(0.92f, 0.34f, 0.22f, 1.0f),
      ImVec4(0.64f, 0.72f, 0.98f, 1.0f),
      ImVec4(0.75f, 0.56f, 0.36f, 1.0f),
      ImVec4(0.95f, 0.78f, 0.36f, 1.0f),
      ImVec4(0.42f, 0.74f, 0.90f, 1.0f),
  };

  const int count = static_cast<int>(sizeof(palette) / sizeof(palette[0]));
  int safe_index = index % count;
  if (safe_index < 0) safe_index += count;
  return palette[safe_index];
}

ImVec4 GetStepColor(float step, AppState& state) {
  ImVec4 base = GetThemeColor(ImGuiCol_PlotLines, state.current_theme);
  float alpha = 0.2f + 0.8f * step;
  return ImVec4(base.x, base.y, base.z, alpha);
}

int GetPlotAxisFlags(const AppState& state) {
  int flags = ImPlotAxisFlags_None;
  if (!state.enable_plot_interaction) {
    flags |= ImPlotAxisFlags_Lock;
  }
  return flags;
}

ImPlotFlags BasePlotFlags(const AppState& state, bool allow_legend) {
  ImPlotFlags flags = ImPlotFlags_NoMenus;
  if (!allow_legend || !state.show_plot_legends) {
    flags |= ImPlotFlags_NoLegend;
  }
  return flags;
}

const std::vector<PlotOption>& PlotOptions() {
  static const std::vector<PlotOption> options = {
      {PlotKind::QualityTrends, "Quality Trends"},
      {PlotKind::GeneratorEfficiency, "Generator Efficiency"},
      {PlotKind::CoverageDensity, "Coverage Density"},
      {PlotKind::TrainingLoss, "Training Loss (Top Runs)"},
      {PlotKind::LossVsSamples, "Loss vs Samples"},
      {PlotKind::DomainCoverage, "Domain Coverage"},
      {PlotKind::EmbeddingQuality, "Embedding Quality"},
      {PlotKind::AgentThroughput, "Agent Workload"},
      {PlotKind::QualityDirection, "Quality Direction"},
      {PlotKind::GeneratorMix, "Generator Mix"},
      {PlotKind::EmbeddingDensity, "Embedding Density"},
      {PlotKind::EvalMetrics, "Eval Metrics"},
      {PlotKind::Rejections, "Rejection Reasons"},
      {PlotKind::KnowledgeGraph, "Context Graph"},
      {PlotKind::LatentSpace, "Embedding Map"},
      {PlotKind::Effectiveness, "Domain Effectiveness"},
      {PlotKind::Thresholds, "Threshold Sensitivity"},
      {PlotKind::MountsStatus, "Local Mounts Status"},
      {PlotKind::DatasetInventory, "Dataset Inventory"},
  };
  return options;
}

int PlotKindToIndex(PlotKind kind) {
  const auto& options = PlotOptions();
  for (size_t i = 0; i < options.size(); ++i) {
    if (options[i].kind == kind) return static_cast<int>(i);
  }
  return 0;
}

PlotKind IndexToPlotKind(int index) {
  const auto& options = PlotOptions();
  if (index < 0 || static_cast<size_t>(index) >= options.size()) {
    return options.front().kind;
  }
  return options[index].kind;
}

float Clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

void AppendLog(AppState& state, const std::string& agent, const std::string& message, const std::string& kind) {
  if (message.empty()) return;
  constexpr size_t kMaxLogs = 300;
  state.logs.push_back(LogEntry{agent, message, kind});
  if (state.logs.size() > kMaxLogs) state.logs.pop_front();
}

AgentState* FindAgentByName(std::vector<AgentState>& agents, const std::string& name) {
  for (auto& agent : agents) {
    if (agent.name == name) return &agent;
  }
  return nullptr;
}

} // namespace ui
} // namespace viz
} // namespace afs

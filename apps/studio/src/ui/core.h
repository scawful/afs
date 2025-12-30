#pragma once

#include <imgui.h>
#include <implot.h>
#include "../models/state.h"

namespace afs {
namespace viz {

namespace ui {

// Utility Functions
void HelpMarker(const char* desc);
void ApplyPremiumPlotStyles(const char* plot_id, AppState& state);
void RenderChartHeader(PlotKind kind, const char* title, const char* desc, AppState& state);
void HandlePlotContextMenu(PlotKind kind, AppState& state);
float Clamp01(float value);
void AppendLog(AppState& state, const std::string& agent, const std::string& message, const std::string& kind);
AgentState* FindAgentByName(std::vector<AgentState>& agents, const std::string& name);

// Theme Helpers
ImVec4 GetThemeColor(ImGuiCol col, ThemeProfile theme);
ImVec4 GetSeriesColor(int index);
ImVec4 GetStepColor(float step, AppState& state); 
ImVec4 GetStepColor(float step, AppState& state); 
int GetPlotAxisFlags(const AppState& state);
ImPlotFlags BasePlotFlags(const AppState& state, bool allow_legend);

// Grid Helpers
struct PlotOption {
  PlotKind kind;
  const char* label;
};

const std::vector<PlotOption>& PlotOptions();
int PlotKindToIndex(PlotKind kind);
PlotKind IndexToPlotKind(int index);

} // namespace ui
} // namespace viz
} // namespace afs

#include "generator_efficiency.h"
#include "../core.h"
#include <algorithm>
#include <vector>
#include <string>

namespace afs::viz::ui {

void GeneratorEfficiencyChart::Render(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::GeneratorEfficiency,
                    "GENERATOR EFFICIENCY",
                    "Acceptance rates for active data generators. Rates < 40% (Warning Zone) indicate generators struggling with current model constraints.",
                    state);

  const auto& stats = loader.GetGeneratorStats();
  if (stats.empty()) {
    ImGui::TextDisabled("No generator stats available");
    return;
  }
  
  struct GeneratorRow {
    std::string name;
    float rate = 0.0f;
  };
  std::vector<GeneratorRow> rows;
  rows.reserve(stats.size());
  for (const auto& s : stats) {
    std::string name = s.name;
    size_t pos = name.find("DataGenerator");
    if (pos != std::string::npos) name = name.substr(0, pos);
    rows.push_back({name, s.acceptance_rate * 100.0f});
  }
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.rate > b.rate; });

  std::vector<const char*> labels;
  std::vector<float> rates;
  std::vector<std::string> label_storage;
  labels.reserve(rows.size());
  rates.reserve(rows.size());
  label_storage.reserve(rows.size());
  for (const auto& row : rows) {
    label_storage.push_back(row.name);
    rates.push_back(row.rate);
  }
  for (const auto& s : label_storage) labels.push_back(s.c_str());

  if (!rows.empty()) {
    const auto& top = rows.front();
    const auto& bottom = rows.back();
    ImGui::TextDisabled("Top: %s (%.1f%%)  |  Bottom: %s (%.1f%%)",
                        top.name.c_str(), top.rate, bottom.name.c_str(), bottom.rate);
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##GeneratorStats", state);
  if (ImPlot::BeginPlot("##GeneratorStats", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Generator", "Acceptance %", axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Once);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::GeneratorEfficiency, state);
    
    // Warning Zone Overlay
    double wx[2] = {-1, 100};
    double wy1[2] = {0, 0};
    double wy2[2] = {40, 40};
    ImPlot::SetNextFillStyle(ImVec4(1, 0, 0, 0.1f));
    ImPlot::PlotShaded("Low Efficiency", wx, wy1, wy2, 2);

    ImPlot::SetNextFillStyle(GetSeriesColor(1), 0.8f);
    ImPlot::PlotBars("Rate", rates.data(), static_cast<int>(rates.size()), 0.67);

    // Custom Tooltip
    if (ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse = ImPlot::GetPlotMousePos();
      int idx = (int)std::round(mouse.x);
      if (idx >= 0 && idx < (int)rates.size()) {
        ImGui::BeginTooltip();
        ImGui::Text("Generator: %s", label_storage[idx].c_str());
        ImGui::Text("Acceptance Rate: %.1f%%", rates[idx]);
        ImGui::Separator();
        ImGui::TextDisabled("Status: %s", rates[idx] < 40.0f ? "WARNING (Low Efficiency)" : "Healthy");
        ImGui::EndTooltip();
      }
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

} // namespace afs::viz::ui

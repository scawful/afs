#include "quality_trends.h"
#include "../core.h"
#include <algorithm>
#include <vector>

namespace afs::viz::ui {

void QualityTrendsChart::Render(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::QualityTrends,
                    "QUALITY TRENDS",
                    "Displays model performance metrics across active training domains. Solid lines indicate scores; shaded area indicates the Optimal Strategy Zone (>0.85).",
                    state);

  const auto& trends = loader.GetQualityTrends();
  if (trends.empty()) {
    ImGui::TextDisabled("No quality trend data available");
    return;
  }

  size_t max_len = 0;
  for (const auto& trend : trends) {
    max_len = std::max(max_len, trend.values.size());
  }

  std::vector<float> mean_values;
  if (max_len > 0) {
    mean_values.assign(max_len, 0.0f);
    std::vector<int> counts(max_len, 0);
    for (const auto& trend : trends) {
      if (state.domain_visibility.count(trend.domain) && !state.domain_visibility.at(trend.domain)) continue;
      for (size_t i = 0; i < trend.values.size(); ++i) {
        mean_values[i] += trend.values[i];
        counts[i] += 1;
      }
    }
    for (size_t i = 0; i < max_len; ++i) {
      if (counts[i] > 0) mean_values[i] /= static_cast<float>(counts[i]);
    }
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, true);

  ApplyPremiumPlotStyles("##QualityTrends", state);
  if (ImPlot::BeginPlot("##QualityTrends", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Time Step", "Score (0-1)", axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.1, ImPlotCond_Always);
    if (state.show_plot_legends) {
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
    }
    HandlePlotContextMenu(PlotKind::QualityTrends, state);
    
    // Help markers and goal regions...
    double goal_x[2] = {-100, 1000};
    double goal_y1[2] = {static_cast<double>(state.quality_threshold), static_cast<double>(state.quality_threshold)};
    double goal_y2[2] = {1.1, 1.1};
    ImPlot::SetNextFillStyle(ImVec4(0, 1, 0, 0.05f));
    ImPlot::PlotShaded("Goal Region", goal_x, goal_y1, goal_y2, 2);
    
    ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 0.4f), 1.0f);
    ImPlot::PlotLine("Requirement", goal_x, goal_y1, 2);

    int color_index = 0;
    for (const auto& trend : trends) {
      if (trend.values.empty()) continue;
      if (state.domain_visibility.count(trend.domain) && !state.domain_visibility.at(trend.domain)) {
          color_index++;
          continue;
      }
      std::string label = trend.domain + " (" + trend.metric + ")";
      ImVec4 series_color = GetSeriesColor(color_index++);
      ImPlot::SetNextLineStyle(series_color, 2.2f);
      if (state.show_plot_markers) {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, series_color);
      }
      ImPlot::SetNextFillStyle(series_color, 0.12f);
      ImPlot::PlotLine(label.c_str(), trend.values.data(), (int)trend.values.size());
      
      // Annotation for latest value
      if (!trend.values.empty()) {
        float last_val = trend.values.back();
        ImPlot::Annotation((double)(trend.values.size() - 1), (double)last_val, 
                           series_color, ImVec2(10, -10), true, "%.2f", last_val);
      }
    }

    if (!mean_values.empty()) {
      ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.7f), 2.0f);
      ImPlot::PlotLine("Mean", mean_values.data(), static_cast<int>(mean_values.size()));
    }

    // Custom Tooltip
    if (ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse = ImPlot::GetPlotMousePos();
      int x = (int)std::round(mouse.x);
      if (x >= 0 && x < (int)max_len) {
        ImGui::BeginTooltip();
        ImGui::Text("Step: %d", x);
        ImGui::Separator();
        for (const auto& trend : trends) {
          if (x < (int)trend.values.size()) {
            if (state.domain_visibility.count(trend.domain) && !state.domain_visibility.at(trend.domain)) continue;
            ImGui::ColorButton("##color", GetSeriesColor(&trend - &trends[0]), ImGuiColorEditFlags_NoTooltip);
            ImGui::SameLine();
            ImGui::Text("%s: %.3f", trend.domain.c_str(), trend.values[x]);
          }
        }
        ImGui::EndTooltip();
      }
    }
    
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

} // namespace afs::viz::ui

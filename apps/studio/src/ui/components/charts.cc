#include "charts.h"
#include "../charts/generator_efficiency.h"
#include "../charts/coverage_density.h"
#include "../core.h"
#include "../../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>

namespace afs {
namespace viz {
namespace ui {
void RenderMountsChart(AppState& state, const DataLoader& loader);
void RenderKnowledgeGraphChart(AppState& state, const DataLoader& loader);




void RenderQualityChart(AppState& state, const DataLoader& loader) {
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
      std::string label = trend.domain + " (" + trend.metric + ")";
      ImVec4 series_color = GetSeriesColor(color_index++);
      ImPlot::SetNextLineStyle(series_color, 2.2f);
      if (state.show_plot_markers) {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, series_color);
      }
      ImPlot::SetNextFillStyle(series_color, 0.12f);
      ImPlot::PlotLine(label.c_str(), trend.values.data(), (int)trend.values.size());
    }

    if (!mean_values.empty()) {
      ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.7f), 2.0f);
      ImPlot::PlotLine("Mean", mean_values.data(), static_cast<int>(mean_values.size()));
    }
    
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderGeneratorChart(AppState& state, const DataLoader& loader) {
  GeneratorEfficiencyChart chart;
  chart.Render(state, loader);
}

void RenderCoverageChart(AppState& state, const DataLoader& loader) {
  CoverageDensityChart chart;
  chart.Render(state, loader);
}

void RenderTrainingChart(AppState& state, const DataLoader& loader) {
  auto runs = loader.GetTrainingRuns();
  if (runs.empty()) {
    ImGui::TextDisabled("No training run data available");
    return;
  }

  // Sort by loss (ascending)
  std::sort(runs.begin(), runs.end(),
            [](const TrainingRunData& a, const TrainingRunData& b) {
              return a.final_loss < b.final_loss;
            });

  // Limit to top 10
  if (runs.size() > 10) runs.resize(10);

  std::vector<const char*> labels;
  std::vector<float> losses;
  std::vector<std::string> label_storage;
  labels.reserve(runs.size());
  losses.reserve(runs.size());
  label_storage.reserve(runs.size());

  for (const auto& r : runs) {
    std::string id = r.run_id.substr(0, std::min(r.run_id.size(), size_t(12)));
    label_storage.push_back(id);
    losses.push_back(r.final_loss);
  }

  for (const auto& s : label_storage) {
    labels.push_back(s.c_str());
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  
  ApplyPremiumPlotStyles("##TrainingLoss", state);
  if (ImPlot::BeginPlot("##TrainingLoss", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Run", "Final Loss", axis_flags, axis_flags);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::TrainingLoss, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(3), 0.75f);
    ImPlot::PlotBars("Loss", losses.data(), static_cast<int>(losses.size()), 0.67);
    if (!losses.empty()) {
      float sum = 0.0f;
      for (float value : losses) sum += value;
      float avg = sum / static_cast<float>(losses.size());
      double avg_x[2] = {-1, static_cast<double>(losses.size())};
      double avg_y[2] = {static_cast<double>(avg), static_cast<double>(avg)};
      ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.5f), 1.5f);
      ImPlot::PlotLine("Avg Loss", avg_x, avg_y, 2);
    }

    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        int idx = (int)std::round(mouse.x);
        if (idx >= 0 && idx < (int)losses.size()) {
            ImGui::BeginTooltip();
            ImGui::Text("Run ID: %s", label_storage[idx].c_str());
            ImGui::Text("Final Loss: %.4f", losses[idx]);
            ImGui::EndTooltip();
        }
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderTrainingLossChart(AppState& state, const DataLoader& loader) {
  const auto& runs = loader.GetTrainingRuns();
  if (runs.empty()) {
    ImGui::TextDisabled("No training run data available");
    return;
  }

  std::vector<float> xs;
  std::vector<float> ys;
  xs.reserve(runs.size());
  ys.reserve(runs.size());
  for (const auto& run : runs) {
    xs.push_back(static_cast<float>(run.samples_count));
    ys.push_back(run.final_loss);
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, true);

  ApplyPremiumPlotStyles("##LossVsSamples", state);
  if (ImPlot::BeginPlot("##LossVsSamples", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Samples", "Final Loss", axis_flags, axis_flags);
    HandlePlotContextMenu(PlotKind::LossVsSamples, state);
    
    ImVec4 scatter_color = GetSeriesColor(4);
    if (state.show_plot_markers) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 5.0f, scatter_color);
    }
    ImPlot::SetNextLineStyle(scatter_color, 1.6f);
    ImPlot::PlotScatter("Runs", xs.data(), ys.data(),
                        static_cast<int>(xs.size()));

    if (xs.size() > 1) {
      float sum_x = 0.0f, sum_y = 0.0f;
      for (size_t i = 0; i < xs.size(); ++i) {
        sum_x += xs[i];
        sum_y += ys[i];
      }
      float mean_x = sum_x / static_cast<float>(xs.size());
      float mean_y = sum_y / static_cast<float>(ys.size());
      float num = 0.0f, den = 0.0f;
      for (size_t i = 0; i < xs.size(); ++i) {
        float dx = xs[i] - mean_x;
        num += dx * (ys[i] - mean_y);
        den += dx * dx;
      }
      if (den > 0.0f) {
        float slope = num / den;
        float intercept = mean_y - slope * mean_x;
        float min_x = *std::min_element(xs.begin(), xs.end());
        float max_x = *std::max_element(xs.begin(), xs.end());
        float line_x[2] = {min_x, max_x};
        float line_y[2] = {slope * min_x + intercept, slope * max_x + intercept};
        ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.45f), 1.8f);
        ImPlot::PlotLine("Trend", line_x, line_y, 2);
      }
    }

    // Highlight selected run
    for (const auto& run : runs) {
      if (run.run_id == state.selected_run_id) {
        float sx = static_cast<float>(run.samples_count);
        float sy = run.final_loss;
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, ImVec4(1.0f, 0.4f, 0.1f, 1.0f));
        ImPlot::PlotScatter("Selected", &sx, &sy, 1);
        break;
      }
    }

    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        ImGui::BeginTooltip();
        ImGui::Text("Samples: %.0f", mouse.x);
        ImGui::Text("Loss: %.4f", mouse.y);
        ImGui::EndTooltip();
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderRejectionChart(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::Rejections,
                    "REJECTION REASONS",
                    "Top reasons for sample rejection. High counts in specific categories may indicate issues with data generation or filtering.",
                    state);

  const auto& summary = loader.GetRejectionSummary();
  if (summary.reasons.empty()) {
    ImGui::TextDisabled("No rejection data available");
    return;
  }

  std::vector<std::pair<std::string, int>> sorted(summary.reasons.begin(), summary.reasons.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

  if (sorted.size() > 8) sorted.resize(8);

  std::vector<const char*> labels;
  std::vector<float> counts;
  std::vector<std::string> label_storage;
  labels.reserve(sorted.size());
  counts.reserve(sorted.size());
  label_storage.reserve(sorted.size());

  for (const auto& [reason, count] : sorted) {
    std::string formatted = reason;
    std::replace(formatted.begin(), formatted.end(), '_', ' ');
    label_storage.push_back(formatted);
    counts.push_back(static_cast<float>(count));
  }

  for (const auto& s : label_storage) labels.push_back(s.c_str());

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  
  ApplyPremiumPlotStyles("##Rejections", state);
  if (ImPlot::BeginPlot("##Rejections", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Reason", "Count", axis_flags, axis_flags);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::Rejections, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(7), 0.8f);
    ImPlot::PlotBars("Count", counts.data(), static_cast<int>(counts.size()), 0.67);

    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        int idx = (int)std::round(mouse.x);
        if (idx >= 0 && idx < (int)counts.size()) {
            ImGui::BeginTooltip();
            ImGui::Text("Reason: %s", label_storage[idx].c_str());
            ImGui::Text("Count: %.0f", counts[idx]);
            ImGui::EndTooltip();
        }
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderQualityDirectionChart(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::QualityDirection,
                    "QUALITY TRENDS",
                    "Overview of quality trend directions across all tracked metrics. 'Sparse' indicates insufficient data for a reliable trend.",
                    state);

  const auto& trends = loader.GetQualityTrends();
  if (trends.empty()) {
    ImGui::TextDisabled("No quality trend data available");
    return;
  }

  int improving = 0, declining = 0, stable = 0, insufficient = 0;
  for (const auto& trend : trends) {
    if (trend.trend_direction == "improving") ++improving;
    else if (trend.trend_direction == "declining") ++declining;
    else if (trend.trend_direction == "stable") ++stable;
    else ++insufficient;
  }

  std::array<const char*, 4> labels = {"Improving", "Stable", "Declining", "Sparse"};
  std::array<float, 4> values = {static_cast<float>(improving), static_cast<float>(stable),
                                 static_cast<float>(declining), static_cast<float>(insufficient)};

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  
  ApplyPremiumPlotStyles("##QualityDirection", state);
  if (ImPlot::BeginPlot("##QualityDirection", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Trend", "Count", axis_flags, axis_flags);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 3, 4, labels.data());
    HandlePlotContextMenu(PlotKind::QualityDirection, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(10), 0.75f);
    ImPlot::PlotBars("Trends", values.data(), 4, 0.67);
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderGeneratorMixChart(AppState& state, const DataLoader& loader) {
  const auto& stats = loader.GetGeneratorStats();
  if (stats.empty()) {
    ImGui::TextDisabled("No generator stats available");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> accepted, rejected, quality, xs;
  std::vector<std::string> label_storage;
  labels.reserve(stats.size());
  accepted.reserve(stats.size());
  rejected.reserve(stats.size());
  quality.reserve(stats.size());
  xs.reserve(stats.size());
  label_storage.reserve(stats.size());

  for (const auto& s : stats) {
    std::string name = s.name;
    size_t pos = name.find("DataGenerator");
    if (pos != std::string::npos) name = name.substr(0, pos);
    label_storage.push_back(name);
    accepted.push_back(static_cast<float>(s.samples_accepted));
    rejected.push_back(static_cast<float>(s.samples_rejected));
    quality.push_back(std::max(0.0f, std::min(1.0f, s.avg_quality)) * 100.0f);
  }

  for (size_t i = 0; i < label_storage.size(); ++i) {
    labels.push_back(label_storage[i].c_str());
    xs.push_back(static_cast<float>(i));
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  
  ApplyPremiumPlotStyles("##GeneratorMix", state);
  if (ImPlot::BeginPlot("##GeneratorMix", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Generator", "Samples", axis_flags, axis_flags);
    ImPlot::SetupAxis(ImAxis_Y2, "Avg Quality %", axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0, 100.0, ImPlotCond_Once);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::GeneratorMix, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(1), 0.75f);
    ImPlot::PlotBars("Accepted", accepted.data(), static_cast<int>(accepted.size()), 0.35, -0.2);
    ImPlot::SetNextFillStyle(GetSeriesColor(7), 0.75f);
    ImPlot::PlotBars("Rejected", rejected.data(), static_cast<int>(rejected.size()), 0.35, 0.2);

    ImPlot::SetAxis(ImAxis_Y2);
    ImPlot::SetNextLineStyle(GetSeriesColor(4), 2.0f);
    if (state.show_plot_markers) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, GetSeriesColor(4));
    }
    ImPlot::PlotLine("Avg Quality %", xs.data(), quality.data(), static_cast<int>(quality.size()));
    ImPlot::SetAxis(ImAxis_Y1);
    
    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        int idx = (int)std::round(mouse.x);
        if (idx >= 0 && idx < (int)accepted.size()) {
            ImGui::BeginTooltip();
            ImGui::Text("Generator: %s", label_storage[idx].c_str());
            ImGui::Text("Accepted: %.0f", accepted[idx]);
            ImGui::Text("Rejected: %.0f", rejected[idx]);
            ImGui::Text("Quality: %.1f%%", quality[idx]);
            ImGui::EndTooltip();
        }
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderEmbeddingDensityChart(AppState& state, const DataLoader& loader) {
  const auto& regions = loader.GetEmbeddingRegions();
  if (regions.empty()) {
    ImGui::TextDisabled("No embedding density data available");
    return;
  }

  int min_count = regions.front().sample_count;
  int max_count = regions.front().sample_count;
  for (const auto& region : regions) {
    min_count = std::min(min_count, region.sample_count);
    max_count = std::max(max_count, region.sample_count);
  }

  const int bins = 8;
  std::vector<float> counts(bins, 0.0f);
  int range = std::max(1, max_count - min_count);
  float bin_size = static_cast<float>(range) / static_cast<float>(bins);

  for (const auto& region : regions) {
    int idx = static_cast<int>((static_cast<float>(region.sample_count - min_count)) / bin_size);
    if (idx >= bins) idx = bins - 1;
    counts[idx] += 1.0f;
  }

  std::vector<const char*> labels;
  std::vector<std::string> label_storage;
  for (int i = 0; i < bins; ++i) {
    label_storage.push_back(std::to_string(static_cast<int>(min_count + bin_size * i)) + "-" +
                            std::to_string(static_cast<int>(min_count + bin_size * (i + 1))));
  }
  for (const auto& label : label_storage) labels.push_back(label.c_str());

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##EmbeddingDensity", state);
  if (ImPlot::BeginPlot("##EmbeddingDensity", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Samples", "Regions", axis_flags, axis_flags);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(bins - 1), bins, labels.data());
    HandlePlotContextMenu(PlotKind::EmbeddingDensity, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(11), 0.75f);
    ImPlot::PlotBars("Regions", counts.data(), bins, 0.67);
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderAgentUtilizationChart(AppState& state, const DataLoader& loader) {
  if (state.agents.empty()) {
    ImGui::TextDisabled("No agent utilization data available");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> cpu, mem;
  std::vector<std::string> label_storage;
  labels.reserve(state.agents.size());
  cpu.reserve(state.agents.size());
  mem.reserve(state.agents.size());
  label_storage.reserve(state.agents.size());

  for (const auto& agent : state.agents) {
    label_storage.push_back(agent.name);
    cpu.push_back(agent.cpu_pct);
    mem.push_back(agent.mem_pct);
  }

  for (const auto& label : label_storage) labels.push_back(label.c_str());

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  ApplyPremiumPlotStyles("##AgentUtil", state);
  if (ImPlot::BeginPlot("##AgentUtil", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Agent", "Utilization %", axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Once);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::AgentUtilization, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(0), 0.7f);
    ImPlot::PlotBars("CPU", cpu.data(), static_cast<int>(cpu.size()), 0.35, -0.2);
    ImPlot::SetNextFillStyle(GetSeriesColor(2), 0.7f);
    ImPlot::PlotBars("Mem", mem.data(), static_cast<int>(mem.size()), 0.35, 0.2);

    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        int idx = (int)std::round(mouse.x);
        if (idx >= 0 && idx < (int)cpu.size()) {
            ImGui::BeginTooltip();
            ImGui::Text("Agent: %s", label_storage[idx].c_str());
            ImGui::Text("CPU: %.1f%%", cpu[idx]);
            ImGui::Text("Memory: %.1f%%", mem[idx]);
            ImGui::EndTooltip();
        }
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderMissionProgressChart(AppState& state, const DataLoader& loader) {
  if (state.missions.empty()) {
    ImGui::TextDisabled("No mission progress data available");
    return;
  }

  std::array<float, 4> buckets = {0.0f, 0.0f, 0.0f, 0.0f};
  for (const auto& mission : state.missions) {
    float progress = std::max(0.0f, std::min(1.0f, mission.progress));
    if (progress < 0.25f) buckets[0] += 1.0f;
    else if (progress < 0.5f) buckets[1] += 1.0f;
    else if (progress < 0.75f) buckets[2] += 1.0f;
    else buckets[3] += 1.0f;
  }

  std::array<const char*, 4> labels = {"0-25%", "25-50%", "50-75%", "75-100%"};
  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##MissionProgress", state);
  if (ImPlot::BeginPlot("##MissionProgress", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Progress", "Missions", axis_flags, axis_flags);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 3, 4, labels.data());
    HandlePlotContextMenu(PlotKind::MissionProgress, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(5), 0.75f);
    ImPlot::PlotBars("Missions", buckets.data(), 4, 0.67);
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderEvalMetricsChart(AppState& state, const DataLoader& loader) {
  const auto& runs = loader.GetTrainingRuns();
  if (runs.empty()) {
    ImGui::TextDisabled("No training metrics available");
    return;
  }

  std::map<std::string, std::pair<float, int>> accum;
  for (const auto& run : runs) {
    for (const auto& [metric, value] : run.eval_metrics) {
      auto& slot = accum[metric];
      slot.first += value;
      slot.second += 1;
    }
  }

  if (accum.empty()) {
    ImGui::TextDisabled("No eval metrics reported");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> values;
  std::vector<std::string> label_storage;
  labels.reserve(accum.size());
  values.reserve(accum.size());
  label_storage.reserve(accum.size());
  for (const auto& [metric, data] : accum) {
    label_storage.push_back(metric);
    values.push_back(data.first / static_cast<float>(std::max(1, data.second)));
  }
  for (const auto& label : label_storage) labels.push_back(label.c_str());

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  
  ApplyPremiumPlotStyles("##EvalMetrics", state);
  if (ImPlot::BeginPlot("##EvalMetrics", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Metric", "Avg", axis_flags, axis_flags);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::EvalMetrics, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(6), 0.75f);
    ImPlot::PlotBars("Avg", values.data(), static_cast<int>(values.size()), 0.67);
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderEffectivenessChart(AppState& state, const DataLoader& loader) {
  const auto& optimization = loader.GetOptimizationData();
  const auto& coverage = loader.GetCoverage();
  const auto& effectiveness = optimization.domain_effectiveness;
  if (effectiveness.empty() && coverage.domain_coverage.empty()) {
    ImGui::TextDisabled("No domain effectiveness data available");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> values;
  std::vector<std::string> label_storage;
  size_t entry_count = effectiveness.empty()
                           ? coverage.domain_coverage.size()
                           : effectiveness.size();
  labels.reserve(entry_count);
  values.reserve(entry_count);
  label_storage.reserve(entry_count);
  bool using_proxy = effectiveness.empty();
  if (using_proxy) {
    ImGui::TextDisabled("Using coverage as effectiveness proxy");
  }

  if (!effectiveness.empty()) {
    for (const auto& [domain, value] : effectiveness) {
      label_storage.push_back(domain);
      values.push_back(value);
    }
  } else {
    for (const auto& [domain, value] : coverage.domain_coverage) {
      label_storage.push_back(domain);
      values.push_back(value);
    }
  }
  for (const auto& label : label_storage) labels.push_back(label.c_str());

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##Effectiveness", state);
  if (ImPlot::BeginPlot("##Effectiveness", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Domain", "Effectiveness", axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::Effectiveness, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(2), 0.75f);
    ImPlot::PlotBars("Effectiveness", values.data(), static_cast<int>(values.size()), 0.67);
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderThresholdOptimizationChart(AppState& state, const DataLoader& loader) {
  std::vector<float> xs, ys1, ys2;
  const auto& optimization = loader.GetOptimizationData();
  bool using_simulated = optimization.threshold_sensitivity.empty();
  if (!using_simulated) {
    std::vector<std::pair<float, float>> points;
    points.reserve(optimization.threshold_sensitivity.size());
    for (const auto& [threshold, value] : optimization.threshold_sensitivity) {
      char* end = nullptr;
      float threshold_value = std::strtof(threshold.c_str(), &end);
      if (end == threshold.c_str()) continue;
      points.emplace_back(threshold_value, static_cast<float>(value));
    }
    if (points.empty()) {
      using_simulated = true;
    } else {
      std::sort(points.begin(), points.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      for (const auto& [x, y] : points) {
        xs.push_back(x);
        ys1.push_back(y);
      }
    }
  }
  if (using_simulated) {
    for (int i = 0; i < 20; ++i) {
      float x = 0.5f + 0.02f * i;
      xs.push_back(x);
      ys1.push_back(1.0f - std::pow(x - 0.7f, 2.0f) * 5.0f); // Quality
      ys2.push_back(std::pow(x, 3.0f)); // Rejections
    }
  }

  if (using_simulated) {
    ImGui::TextDisabled("Simulated sensitivity (no data)");
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  ApplyPremiumPlotStyles("##Thresholds", state);
  if (ImPlot::BeginPlot("##Thresholds", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Threshold", "Score", axis_flags, axis_flags);
    HandlePlotContextMenu(PlotKind::Thresholds, state);

  ImPlot::SetNextLineStyle(GetSeriesColor(3), 2.0f);
  if (!xs.empty()) {
    const char* series_label = using_simulated ? "Quality Yield" : "Effectiveness";
    ImPlot::PlotLine(series_label, xs.data(), ys1.data(),
                     static_cast<int>(ys1.size()));
  }
  if (!ys2.empty()) {
    ImPlot::SetNextLineStyle(GetSeriesColor(7), 2.0f);
    ImPlot::PlotLine("Rejection Rate", xs.data(), ys2.data(),
                     static_cast<int>(ys2.size()));
  }
    
    // Highlight current threshold
    float tx[2] = {state.rejection_threshold, state.rejection_threshold};
    float ty[2] = {0.0f, 1.0f};
    ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.6f), 1.5f);
    ImPlot::PlotLine("Current", tx, ty, 2);

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderDomainCoverageChart(AppState& state, const DataLoader& loader) {
  const auto& coverage = loader.GetCoverage();
  if (coverage.domain_coverage.empty()) {
    ImGui::TextDisabled("No domain coverage data available");
    return;
  }

  struct CoverageRow {
    std::string name;
    float value = 0.0f;
  };
  std::vector<CoverageRow> rows;
  rows.reserve(coverage.domain_coverage.size());
  for (const auto& [domain, value] : coverage.domain_coverage) {
    rows.push_back({domain, value * 100.0f});
  }
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.value > b.value; });

  std::vector<const char*> labels;
  std::vector<float> values;
  std::vector<std::string> label_storage;
  labels.reserve(rows.size());
  values.reserve(rows.size());
  label_storage.reserve(rows.size());
  for (const auto& row : rows) {
    label_storage.push_back(row.name);
    values.push_back(row.value);
  }
  for (const auto& label : label_storage) labels.push_back(label.c_str());

  if (!rows.empty()) {
    const auto& top = rows.front();
    const auto& bottom = rows.back();
    ImGui::TextDisabled("Top: %s (%.1f%%)  |  Bottom: %s (%.1f%%)",
                        top.name.c_str(), top.value, bottom.name.c_str(), bottom.value);
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##DomainCoverage", state);
  if (ImPlot::BeginPlot("##DomainCoverage", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Domain", "Coverage %", axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Once);
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::DomainCoverage, state);
    
    ImPlot::SetNextFillStyle(GetSeriesColor(5), 0.75f);
    ImPlot::PlotBars("Coverage", values.data(), static_cast<int>(values.size()), 0.67);
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderEmbeddingQualityChart(AppState& state, const DataLoader& loader) {
  const auto& regions = loader.GetEmbeddingRegions();
  if (regions.empty()) {
    ImGui::TextDisabled("No embedding quality data available");
    return;
  }

  std::vector<float> xs, ys;
  for (const auto& region : regions) {
    xs.push_back(static_cast<float>(region.index));
    ys.push_back(region.avg_quality);
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, true);

  ApplyPremiumPlotStyles("##EmbeddingQuality", state);
  if (ImPlot::BeginPlot("##EmbeddingQuality", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Region", "Avg Quality", axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);
    HandlePlotContextMenu(PlotKind::EmbeddingQuality, state);
    
    ImVec4 scatter_color = GetSeriesColor(6);
    if (state.show_plot_markers) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, scatter_color);
    }
    ImPlot::SetNextLineStyle(scatter_color, 1.6f);
    ImPlot::PlotScatter("Quality", xs.data(), ys.data(), static_cast<int>(xs.size()));
    
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderAgentThroughputChart(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::AgentThroughput,
                    "AGENT THROUGHPUT",
                    "Real-time task processing rate across the swarm. Higher peaks indicate high-availability periods; the dashed line represents the Swarm Target (1.0k).",
                    state);

  if (state.agents.empty()) {
    ImGui::TextDisabled("Data stream offline");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> tasks, queues;
  std::vector<std::string> label_storage;
  labels.reserve(state.agents.size());
  tasks.reserve(state.agents.size());
  queues.reserve(state.agents.size());
  label_storage.reserve(state.agents.size());

  for (size_t i = 0; i < state.agents.size(); ++i) {
    const auto& agent = state.agents[i];
    char buf[16];
    snprintf(buf, sizeof(buf), "A%zu", i + 1);
    label_storage.push_back(buf);
    tasks.push_back(static_cast<float>(agent.tasks_completed));
    queues.push_back(static_cast<float>(agent.queue_depth));
  }

  for (const auto& label : label_storage) labels.push_back(label.c_str());

  ImPlotFlags plot_flags = BasePlotFlags(state, true);
  
  ApplyPremiumPlotStyles("##Throughput", state);
  if (ImPlot::BeginPlot("##AgentThroughput", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Agent Index", "Total Tasks Completed", axis_flags, axis_flags);
    
    if (!labels.empty()) {
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                               static_cast<int>(labels.size()), labels.data());
    }
    HandlePlotContextMenu(PlotKind::AgentThroughput, state);

    ImPlot::SetNextFillStyle(GetSeriesColor(0), 0.7f);
    ImPlot::PlotBars("Tasks Completed", tasks.data(), static_cast<int>(tasks.size()), 0.6);
    ImPlot::SetNextLineStyle(GetSeriesColor(8), 2.0f);
    if (state.show_plot_markers) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 4.0f, GetSeriesColor(8));
    }
    ImPlot::PlotLine("Queue Depth", queues.data(), static_cast<int>(queues.size()));
    
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderMissionQueueChart(AppState& state, const DataLoader& loader) {
  if (state.missions.empty()) {
    ImGui::TextDisabled("No mission data available");
    return;
  }

  int queued = 0, active = 0, complete = 0, other = 0;
  for (const auto& mission : state.missions) {
    if (mission.status == "Queued") ++queued;
    else if (mission.status == "Active") ++active;
    else if (mission.status == "Complete") ++complete;
    else ++other;
  }

  std::array<const char*, 4> labels = {"Queued", "Active", "Complete", "Other"};
  std::array<float, 4> values = {static_cast<float>(queued), static_cast<float>(active),
                                 static_cast<float>(complete), static_cast<float>(other)};

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##MissionQueue", state);
  if (ImPlot::BeginPlot("##MissionQueue", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Status", "Count", axis_flags, axis_flags);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0, 3, 4, labels.data());
    HandlePlotContextMenu(PlotKind::MissionQueue, state);
    
    ImPlot::SetNextFillStyle(GetSeriesColor(9), 0.75f);
    ImPlot::PlotBars("Missions", values.data(), 4, 0.67);
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderLatentSpaceChart(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::LatentSpace,
                    "LATENT TOPOLOGY",
                    "Visualization of the manifold learned by the model. Clusters indicate stable concept representations; voids represent potential logic gaps.",
                    state);

  const auto& regions = loader.GetEmbeddingRegions();
  if (regions.empty()) {
    ImGui::TextDisabled("No embedding data");
    return;
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##LatentSpace", state);
  if (ImPlot::BeginPlot("##LatentSpace", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state)) |
                                 ImPlotAxisFlags_NoDecorations;
    ImPlot::SetupAxes(nullptr, nullptr, axis_flags, axis_flags);
    HandlePlotContextMenu(PlotKind::LatentSpace, state);
    
    std::vector<float> xs, ys;
    for (const auto& r : regions) {
      float angle = (float)r.index * 0.15f;
      float dist = 2.0f + (float)std::sin(r.index * 0.3f) * 1.5f;
      xs.push_back(std::cos(angle) * dist);
      ys.push_back(std::sin(angle) * dist);
    }

    ImVec4 cluster_color = GetSeriesColor(0);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, cluster_color);
    ImPlot::PlotScatter("Embeddings", xs.data(), ys.data(), static_cast<int>(xs.size()));
    
    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        ImGui::BeginTooltip();
        ImGui::Text("Latent X: %.3f", mouse.x);
        ImGui::Text("Latent Y: %.3f", mouse.y);
        ImGui::EndTooltip();
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderPlotByKind(PlotKind kind, AppState& state, const DataLoader& loader) {
  switch (kind) {
    case PlotKind::QualityTrends:
      RenderQualityChart(state, loader);
      break;
    case PlotKind::GeneratorEfficiency:
      RenderGeneratorChart(state, loader);
      break;
    case PlotKind::CoverageDensity:
      RenderCoverageChart(state, loader);
      break;
    case PlotKind::TrainingLoss:
      RenderChartHeader(PlotKind::TrainingLoss, "TRAINING LOSS (TOP RUNS)", "", state);
      RenderTrainingChart(state, loader);
      break;
    case PlotKind::LossVsSamples:
      RenderChartHeader(PlotKind::LossVsSamples, "LOSS VS SAMPLES", "", state);
      RenderTrainingLossChart(state, loader);
      break;
    case PlotKind::DomainCoverage:
      RenderChartHeader(PlotKind::DomainCoverage, "DOMAIN COVERAGE", "", state);
      RenderDomainCoverageChart(state, loader);
      break;
    case PlotKind::EmbeddingQuality:
      RenderChartHeader(PlotKind::EmbeddingQuality, "EMBEDDING QUALITY", "", state);
      RenderEmbeddingQualityChart(state, loader);
      break;
    case PlotKind::AgentThroughput:
      RenderAgentThroughputChart(state, loader);
      break;
    case PlotKind::MissionQueue:
      RenderChartHeader(PlotKind::MissionQueue, "MISSION QUEUE", "", state);
      RenderMissionQueueChart(state, loader);
      break;
    case PlotKind::QualityDirection:
      RenderQualityDirectionChart(state, loader);
      break;
    case PlotKind::GeneratorMix:
      RenderChartHeader(PlotKind::GeneratorMix, "GENERATOR MIX", "", state);
      RenderGeneratorMixChart(state, loader);
      break;
    case PlotKind::EmbeddingDensity:
      RenderChartHeader(PlotKind::EmbeddingDensity, "EMBEDDING DENSITY", "", state);
      RenderEmbeddingDensityChart(state, loader);
      break;
    case PlotKind::AgentUtilization:
      RenderChartHeader(PlotKind::AgentUtilization, "AGENT UTILIZATION", "", state);
      RenderAgentUtilizationChart(state, loader);
      break;
    case PlotKind::MissionProgress:
      RenderChartHeader(PlotKind::MissionProgress, "MISSION PROGRESS", "", state);
      RenderMissionProgressChart(state, loader);
      break;
    case PlotKind::EvalMetrics:
      RenderChartHeader(PlotKind::EvalMetrics, "EVAL METRICS", "", state);
      RenderEvalMetricsChart(state, loader);
      break;
    case PlotKind::Rejections:
      RenderRejectionChart(state, loader); 
      break;
    case PlotKind::KnowledgeGraph:
       RenderKnowledgeGraphChart(state, loader);
      break;
    case PlotKind::LatentSpace:
      RenderLatentSpaceChart(state, loader);
      break;
    case PlotKind::Effectiveness:
      RenderChartHeader(PlotKind::Effectiveness, "DOMAIN EFFECTIVENESS", "", state);
      RenderEffectivenessChart(state, loader);
      break;
    case PlotKind::Thresholds:
      RenderChartHeader(PlotKind::Thresholds, "THRESHOLD SENSITIVITY", "", state);
      RenderThresholdOptimizationChart(state, loader);
      break;
    case PlotKind::MountsStatus:
      RenderMountsChart(state, loader);
      break;
    default:
      break;
  }
}

void RenderMountsChart(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::MountsStatus,
                    "LOCAL MOUNTS STATUS",
                    "Connectivity status for configured project resource paths and local caches.",
                    state);

  const auto& mounts = loader.GetMounts();
  if (mounts.empty()) {
    ImGui::TextDisabled("No mounts configured");
    return;
  }

  std::vector<const char*> labels;
  std::vector<float> values;
  std::vector<std::string> label_storage;
  labels.reserve(mounts.size());
  values.reserve(mounts.size());
  label_storage.reserve(mounts.size());

  for (const auto& m : mounts) {
    label_storage.push_back(m.name);
    values.push_back(m.active ? 1.0f : 0.0f);
  }
  for (const auto& s : label_storage) labels.push_back(s.c_str());

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##MountsStatus", state);
  if (ImPlot::BeginPlot("##MountsStatus", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
    ImPlot::SetupAxes("Mount", "Active", axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_Y1, -0.2, 1.2, ImPlotCond_Always);
    ImPlot::SetupAxisTicks(ImAxis_Y1, 0, 1, 2, nullptr); // Just 0 and 1
    
    if (!labels.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                             static_cast<int>(labels.size()), labels.data());
    }

    // Color bars based on status
    for (int i = 0; i < (int)values.size(); ++i) {
        ImPlot::SetNextFillStyle(values[i] > 0.5f ? ImVec4(0.4f, 1.0f, 0.6f, 0.7f) : ImVec4(1.0f, 0.4f, 0.4f, 0.7f));
        double x = i;
        double y = values[i];
        ImPlot::PlotBars("##Mount", &x, &y, 1, 0.6);
    }

    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        int idx = (int)std::round(mouse.x);
        if (idx >= 0 && idx < (int)values.size()) {
            ImGui::BeginTooltip();
            ImGui::Text("Mount: %s", label_storage[idx].c_str());
            ImGui::Text("Status: %s", values[idx] > 0.5f ? "CONNECTED" : "DISCONNECTED");
            ImGui::EndTooltip();
        }
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

void RenderKnowledgeGraphChart(AppState& state, const DataLoader& loader) {
  RenderChartHeader(PlotKind::KnowledgeGraph, 
                    "CONTEXT GRAPH",
                    "AFS context map showing workspace and mount relationships.",
                    state);

  if (state.knowledge_concepts.empty()) {
    const auto& error = loader.GetContextGraphError();
    if (!error.empty()) {
      ImGui::TextColored(ImVec4(0.9f, 0.5f, 0.2f, 1.0f), "%s", error.c_str());
    }
    ImGui::TextDisabled("No context graph data available");
    return;
  }

  const auto& graph = loader.GetContextGraph();
  if (graph.context_count > 0 || graph.mount_count > 0) {
    ImGui::TextDisabled("Contexts: %d  Mounts: %d", graph.context_count, graph.mount_count);
  }

  ImPlotFlags plot_flags = BasePlotFlags(state, false);
  ApplyPremiumPlotStyles("##KnowledgeGraph", state);
  if (ImPlot::BeginPlot("##KnowledgeGraph", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state)) |
                                 ImPlotAxisFlags_NoDecorations;
    ImPlot::SetupAxes(nullptr, nullptr, axis_flags, axis_flags);
    HandlePlotContextMenu(PlotKind::KnowledgeGraph, state);

    // Draw Edges
    for (const auto& edge : state.knowledge_edges) {
        if (edge.from >= 0 && edge.from < (int)state.knowledge_nodes_x.size() &&
            edge.to >= 0 && edge.to < (int)state.knowledge_nodes_x.size()) {
            double ex[2] = { (double)state.knowledge_nodes_x[edge.from], (double)state.knowledge_nodes_x[edge.to] };
            double ey[2] = { (double)state.knowledge_nodes_y[edge.from], (double)state.knowledge_nodes_y[edge.to] };
            ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.15f), 1.0f);
            ImPlot::PlotLine("##edge", ex, ey, 2);
        }
    }

    // Draw Nodes
    for (size_t i = 0; i < state.knowledge_concepts.size(); ++i) {
        ImVec4 node_color = GetSeriesColor((int)i);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, node_color, 1.0f, node_color);
        ImPlot::PlotScatter(state.knowledge_concepts[i].c_str(), &state.knowledge_nodes_x[i], &state.knowledge_nodes_y[i], 1);
        
        // Label for nodes in high zoom or always?
        ImPlot::Annotation((double)state.knowledge_nodes_x[i], (double)state.knowledge_nodes_y[i], 
                           ImVec4(1,1,1,0.5f), ImVec2(8, -8), false, "%s", state.knowledge_concepts[i].c_str());
    }

    // Tooltip
    if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        for (size_t i = 0; i < state.knowledge_concepts.size(); ++i) {
            float dx = state.knowledge_nodes_x[i] - (float)mouse.x;
            float dy = state.knowledge_nodes_y[i] - (float)mouse.y;
            if (std::sqrt(dx*dx + dy*dy) < 0.2f) {
                ImGui::BeginTooltip();
                ImGui::Text("Concept: %s", state.knowledge_concepts[i].c_str());
                ImGui::EndTooltip();
                break;
            }
        }
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar(6);
}

} // namespace ui
} // namespace viz
} // namespace afs

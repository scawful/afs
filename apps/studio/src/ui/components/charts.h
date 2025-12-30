#pragma once

#include <imgui.h>
#include "../../models/state.h"
#include "../../data_loader.h"

namespace afs {
namespace viz {
namespace ui {

void RenderQualityChart(AppState& state, const DataLoader& loader);
void RenderGeneratorChart(AppState& state, const DataLoader& loader);
void RenderCoverageChart(AppState& state, const DataLoader& loader);
void RenderTrainingChart(AppState& state, const DataLoader& loader);
void RenderTrainingLossChart(AppState& state, const DataLoader& loader);
void RenderRejectionChart(AppState& state, const DataLoader& loader);
void RenderQualityDirectionChart(AppState& state, const DataLoader& loader);
void RenderGeneratorMixChart(AppState& state, const DataLoader& loader);
void RenderEmbeddingDensityChart(AppState& state, const DataLoader& loader);
void RenderAgentUtilizationChart(AppState& state, const DataLoader& loader);
void RenderMissionProgressChart(AppState& state, const DataLoader& loader);
void RenderEvalMetricsChart(AppState& state, const DataLoader& loader);
void RenderEffectivenessChart(AppState& state, const DataLoader& loader);
void RenderThresholdOptimizationChart(AppState& state, const DataLoader& loader);
void RenderMountsChart(AppState& state, const DataLoader& loader);
void RenderDomainCoverageChart(AppState& state, const DataLoader& loader);
void RenderEmbeddingQualityChart(AppState& state, const DataLoader& loader);
void RenderAgentThroughputChart(AppState& state, const DataLoader& loader);
void RenderMissionQueueChart(AppState& state, const DataLoader& loader);
void RenderLatentSpaceChart(AppState& state, const DataLoader& loader);

void RenderPlotByKind(PlotKind kind, AppState& state, const DataLoader& loader);

} // namespace ui
} // namespace viz
} // namespace afs

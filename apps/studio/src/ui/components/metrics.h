#pragma once

#include <imgui.h>
#include <string>
#include "../../models/state.h"
#include "../../data_loader.h"

namespace afs {
namespace viz {
namespace ui {

void RenderMetricCards(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header);
void RenderSummaryRow(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header);
void RenderStatusBar(AppState& state, const DataLoader& loader, const std::string& data_path);

} // namespace ui
} // namespace viz
} // namespace afs

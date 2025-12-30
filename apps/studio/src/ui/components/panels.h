#pragma once

#include <imgui.h>
#include <string>
#include <vector>
#include <functional>
#include "../../models/state.h"
#include "../../data_loader.h"
#include "../shortcuts.h"

namespace afs {
namespace viz {
namespace ui {

void RenderInspectorPanel(AppState& state, const DataLoader& loader, ImFont* font_header, const std::string& data_path);
void RenderDatasetPanel(AppState& state, const DataLoader& loader);
void RenderSystemsPanel(AppState& state, ImFont* font_header, std::function<void(const char*)> refresh_callback);
void RenderMenuBar(AppState& state,
                   std::function<void(const char*)> refresh_callback,
                   std::function<void()> quit_callback,
                   ShortcutManager& shortcuts,
                   bool* show_sample_review,
                   bool* show_shortcuts_window);
void RenderSidebar(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header);

} // namespace ui
} // namespace viz
} // namespace afs

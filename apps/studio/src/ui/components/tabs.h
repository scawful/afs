#pragma once

#include <imgui.h>
#include <string>
#include <vector>
#include <filesystem>
#include "../../models/state.h"
#include "../../data_loader.h"
#include "../../widgets/text_editor.h"
#include "../../widgets/imgui_memory_editor.h"

namespace afs {
namespace viz {
namespace ui {

void RenderKnobsTab(AppState& state, const DataLoader& loader, const std::string& data_path, std::function<void(const char*)> refresh_callback, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);
void RenderAgentsTab(AppState& state, ImFont* font_ui, ImFont* font_header, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);
void RenderMissionsTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);
void RenderLogsTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);
void RenderContextTab(AppState& state, TextEditor& text_editor, MemoryEditorWidget& memory_editor, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);
void RenderAgentPromptTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);
void RenderTablesTab(AppState& state, const DataLoader& loader);
void RenderServicesTab(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);
void RenderComparisonView(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header);

// Helpers
void RefreshBrowserEntries(AppState& state);
void LoadFile(AppState& state, const std::filesystem::path& path, TextEditor& text_editor);
void RenderMarkdown(const std::string& content, ImFont* font_ui, ImFont* font_header, ThemeProfile current_theme);

} // namespace ui
} // namespace viz
} // namespace afs

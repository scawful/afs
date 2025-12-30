#pragma once

#include <array>
#include <string>

#include <imgui.h>

namespace afs {
namespace viz {
namespace ui {

enum class ActionId {
  Refresh,
  Quit,
  ToggleInspector,
  ToggleDatasetPanel,
  ToggleSystemsPanel,
  ToggleStatusBar,
  ToggleControls,
  ToggleAutoRefresh,
  ToggleSimulation,
  ToggleCompactUI,
  ToggleLockLayout,
  ResetLayout,
  ToggleSampleReview,
  ToggleShortcutsWindow,
  ToggleDemoWindow,
  WorkspaceDashboard,
  WorkspaceAnalysis,
  WorkspaceOptimization,
  WorkspaceSystems,
  WorkspaceCustom,
  WorkspaceChat,
  WorkspaceTraining,
  WorkspaceContext,
  // Graph View Navigation
  ToggleGraphBrowser,
  ToggleCompanionPanels,
  NavigateBack,
  NavigateForward,
  BookmarkGraph,
  OpenGraphPalette,
  Count,
};

struct Shortcut {
  ImGuiKey key = ImGuiKey_None;
  bool ctrl = false;
  bool shift = false;
  bool alt = false;
  bool super = false;
};

struct ActionDefinition {
  ActionId id;
  const char* config_key;
  const char* label;
  const char* description;
  Shortcut default_shortcut;
};

constexpr int kActionCount = static_cast<int>(ActionId::Count);

class ShortcutManager {
 public:
  ShortcutManager();

  const std::array<ActionDefinition, kActionCount>& Actions() const;
  const ActionDefinition& GetDefinition(ActionId id) const;
  const Shortcut& GetShortcut(ActionId id) const;
  const std::string& GetConfigPath() const { return config_path_; }
  const std::string& GetLastError() const { return last_error_; }

  void SetShortcut(ActionId id, const Shortcut& shortcut);
  void ResetShortcut(ActionId id);
  void ResetAll();
  void SetConfigPath(std::string path);
  bool LoadFromDisk(std::string* error = nullptr);
  bool SaveToDisk(std::string* error = nullptr);
  bool SaveIfDirty(std::string* error = nullptr);

  bool IsCapturing() const { return capturing_; }
  ActionId CapturingAction() const { return capturing_action_; }
  void BeginCapture(ActionId id);
  void CancelCapture();
  bool HandleCapture(const ImGuiIO& io);

  bool ShouldProcessShortcuts(const ImGuiIO& io) const;
  bool IsTriggered(ActionId id, const ImGuiIO& io) const;

  std::string FormatShortcut(ActionId id, const ImGuiIO& io) const;
  static std::string FormatShortcut(const Shortcut& shortcut, const ImGuiIO& io);

 private:
  static size_t ToIndex(ActionId id);
  static bool IsValidShortcutKey(ImGuiKey key);
  const ActionDefinition* FindDefinition(const std::string& key) const;
  static std::string Trim(const std::string& text);
  static std::string ToLower(const std::string& text);
  static ImGuiKey FindKeyByName(const std::string& name);
  static bool ParseShortcutString(const std::string& text,
                                  Shortcut* shortcut,
                                  std::string* error);
  static std::string ResolveDefaultPath();

  std::array<Shortcut, kActionCount> bindings_{};
  ActionId capturing_action_ = ActionId::Count;
  bool capturing_ = false;
  bool dirty_ = false;
  std::string config_path_;
  std::string last_error_;
};

void RenderShortcutsWindow(ShortcutManager& shortcuts, bool* open);

}  // namespace ui
}  // namespace viz
}  // namespace afs

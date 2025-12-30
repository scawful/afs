#include "shortcuts.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace afs {
namespace viz {
namespace ui {

namespace {

const std::array<ActionDefinition, kActionCount>& ActionDefinitions() {
  static const std::array<ActionDefinition, kActionCount> definitions = {
      ActionDefinition{ActionId::Refresh, "refresh", "Refresh Data",
                       "Reload data from all configured sources.",
                       Shortcut{ImGuiKey_F5, false, false, false, false}},
      ActionDefinition{ActionId::Quit, "quit", "Quit",
                       "Close the visualization application.",
                       Shortcut{ImGuiKey_Q, true, false, false, false}},
      ActionDefinition{ActionId::ToggleInspector, "toggle_inspector",
                       "Inspector Panel",
                       "Show or hide the Inspector panel.",
                       Shortcut{ImGuiKey_I, true, false, false, false}},
      ActionDefinition{ActionId::ToggleDatasetPanel, "toggle_dataset",
                       "Dataset Panel",
                       "Show or hide the Dataset panel.",
                       Shortcut{ImGuiKey_D, true, true, false, false}},
      ActionDefinition{ActionId::ToggleSystemsPanel, "toggle_systems",
                       "Systems Panel",
                       "Show or hide the Systems panel.",
                       Shortcut{ImGuiKey_S, true, true, false, false}},
      ActionDefinition{ActionId::ToggleStatusBar, "toggle_status",
                       "Status Strip",
                       "Show or hide the status strip.",
                       Shortcut{ImGuiKey_B, true, true, false, false}},
      ActionDefinition{ActionId::ToggleControls, "toggle_controls",
                       "Sidebar Controls",
                       "Show or hide the sidebar controls.",
                       Shortcut{ImGuiKey_C, true, true, false, false}},
      ActionDefinition{ActionId::ToggleAutoRefresh, "toggle_auto_refresh",
                       "Auto Refresh",
                       "Enable or disable auto refresh of data.",
                       Shortcut{ImGuiKey_A, true, true, false, false}},
      ActionDefinition{ActionId::ToggleSimulation, "toggle_simulation",
                       "Simulate Activity",
                       "Enable or disable simulated activity streams.",
                       Shortcut{ImGuiKey_Y, true, true, false, false}},
      ActionDefinition{ActionId::ToggleCompactUI, "toggle_compact_ui",
                       "Compact Charts",
                       "Toggle compact chart layout.",
                       Shortcut{ImGuiKey_U, true, true, false, false}},
      ActionDefinition{ActionId::ToggleLockLayout, "toggle_lock_layout",
                       "Lock Layout",
                       "Lock or unlock the docking layout.",
                       Shortcut{ImGuiKey_L, true, true, false, false}},
      ActionDefinition{ActionId::ResetLayout, "reset_layout", "Reset Layout",
                       "Reset the docking layout to defaults.",
                       Shortcut{ImGuiKey_R, true, true, true, false}},
      ActionDefinition{ActionId::ToggleSampleReview, "toggle_sample_review",
                       "Sample Review",
                       "Open the sample review window.",
                       Shortcut{ImGuiKey_R, true, true, false, false}},
      ActionDefinition{ActionId::ToggleShortcutsWindow, "toggle_shortcuts",
                       "Keyboard Shortcuts",
                       "Open the shortcuts editor.",
                       Shortcut{ImGuiKey_Slash, true, false, false, false}},
      ActionDefinition{ActionId::ToggleDemoWindow, "toggle_demo",
                       "ImGui Demo",
                       "Toggle the ImGui demo window.",
                       Shortcut{ImGuiKey_M, true, true, false, false}},
      ActionDefinition{ActionId::WorkspaceDashboard, "workspace_dashboard",
                       "Workspace: Dashboard",
                       "Switch to the Dashboard workspace.",
                       Shortcut{ImGuiKey_1, true, false, false, false}},
      ActionDefinition{ActionId::WorkspaceAnalysis, "workspace_analysis",
                       "Workspace: Analysis",
                       "Switch to the Analysis workspace.",
                       Shortcut{ImGuiKey_2, true, false, false, false}},
      ActionDefinition{ActionId::WorkspaceOptimization,
                       "workspace_optimization",
                       "Workspace: Optimization",
                       "Switch to the Optimization workspace.",
                       Shortcut{ImGuiKey_3, true, false, false, false}},
      ActionDefinition{ActionId::WorkspaceSystems, "workspace_systems",
                       "Workspace: Systems",
                       "Switch to the Systems workspace.",
                       Shortcut{ImGuiKey_4, true, false, false, false}},
      ActionDefinition{ActionId::WorkspaceCustom, "workspace_custom",
                       "Workspace: Custom Grid",
                       "Switch to the Custom Grid workspace.",
                       Shortcut{ImGuiKey_5, true, false, false, false}},
      ActionDefinition{ActionId::WorkspaceChat, "workspace_chat",
                       "Workspace: Chat",
                       "Switch to the Chat workspace.",
                       Shortcut{ImGuiKey_6, true, false, false, false}},
      ActionDefinition{ActionId::WorkspaceTraining, "workspace_training",
                       "Workspace: Training Hub",
                       "Switch to the Training Hub workspace.",
                       Shortcut{ImGuiKey_7, true, false, false, false}},
      ActionDefinition{ActionId::WorkspaceContext, "workspace_context",
                       "Workspace: Context Broker",
                       "Switch to the Context Broker workspace.",
                       Shortcut{ImGuiKey_8, true, false, false, false}},
      // Graph View Navigation
      ActionDefinition{ActionId::ToggleGraphBrowser, "toggle_graph_browser",
                       "Toggle Graph Browser", "Show/hide graph browser sidebar",
                       Shortcut{ImGuiKey_B, true, false, false, false}},
      ActionDefinition{ActionId::ToggleCompanionPanels, "toggle_companion_panels",
                       "Toggle Companion Panels", "Show/hide companion panels sidebar",
                       Shortcut{ImGuiKey_Backslash, true, false, false, false}},
      ActionDefinition{ActionId::NavigateBack, "navigate_back",
                       "Navigate Back", "Go to previous graph in history",
                       Shortcut{ImGuiKey_LeftArrow, false, false, true, false}},
      ActionDefinition{ActionId::NavigateForward, "navigate_forward",
                       "Navigate Forward", "Go to next graph in history",
                       Shortcut{ImGuiKey_RightArrow, false, false, true, false}},
      ActionDefinition{ActionId::BookmarkGraph, "bookmark_graph",
                       "Bookmark Graph", "Add/remove current graph from bookmarks",
                       Shortcut{ImGuiKey_D, true, false, false, false}},
      ActionDefinition{ActionId::OpenGraphPalette, "open_graph_palette",
                       "Open Graph Palette", "Quick graph switcher",
                       Shortcut{ImGuiKey_K, true, false, false, false}},
  };
  return definitions;
}

bool ContainsInsensitive(const std::string& text, const std::string& pattern) {
  if (pattern.empty()) return true;
  auto it = std::search(text.begin(), text.end(), pattern.begin(),
                        pattern.end(), [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                        });
  return it != text.end();
}

}  // namespace

ShortcutManager::ShortcutManager() {
  config_path_ = ResolveDefaultPath();
  const auto& defs = ActionDefinitions();
  for (const auto& def : defs) {
    bindings_[ToIndex(def.id)] = def.default_shortcut;
  }
}

const std::array<ActionDefinition, kActionCount>& ShortcutManager::Actions() const {
  return ActionDefinitions();
}

const ActionDefinition& ShortcutManager::GetDefinition(ActionId id) const {
  return ActionDefinitions()[ToIndex(id)];
}

const Shortcut& ShortcutManager::GetShortcut(ActionId id) const {
  return bindings_[ToIndex(id)];
}

void ShortcutManager::SetShortcut(ActionId id, const Shortcut& shortcut) {
  Shortcut& slot = bindings_[ToIndex(id)];
  if (slot.key == shortcut.key && slot.ctrl == shortcut.ctrl &&
      slot.shift == shortcut.shift && slot.alt == shortcut.alt &&
      slot.super == shortcut.super) {
    return;
  }
  slot = shortcut;
  dirty_ = true;
}

void ShortcutManager::ResetShortcut(ActionId id) {
  SetShortcut(id, GetDefinition(id).default_shortcut);
}

void ShortcutManager::ResetAll() {
  const auto& defs = ActionDefinitions();
  for (const auto& def : defs) {
    bindings_[ToIndex(def.id)] = def.default_shortcut;
  }
  dirty_ = true;
}

void ShortcutManager::SetConfigPath(std::string path) {
  if (path.empty()) {
    config_path_ = ResolveDefaultPath();
    return;
  }
  config_path_ = std::move(path);
}

bool ShortcutManager::LoadFromDisk(std::string* error) {
  last_error_.clear();
  if (config_path_.empty()) {
    config_path_ = ResolveDefaultPath();
  }

  std::filesystem::path path(config_path_);
  if (!std::filesystem::exists(path)) {
    if (error) *error = "";
    return false;
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    last_error_ = "Failed to open shortcut config";
    if (error) *error = last_error_;
    return false;
  }

  std::string line;
  bool ok = true;
  while (std::getline(file, line)) {
    std::string trimmed = Trim(line);
    if (trimmed.empty()) continue;
    if (trimmed[0] == '#' || trimmed[0] == ';') continue;

    size_t pos = trimmed.find('=');
    if (pos == std::string::npos) {
      ok = false;
      continue;
    }
    std::string key = Trim(trimmed.substr(0, pos));
    std::string value = Trim(trimmed.substr(pos + 1));
    const ActionDefinition* def = FindDefinition(key);
    if (!def) {
      ok = false;
      continue;
    }
    Shortcut shortcut;
    std::string parse_error;
    if (!ParseShortcutString(value, &shortcut, &parse_error)) {
      ok = false;
      continue;
    }
    bindings_[ToIndex(def->id)] = shortcut;
  }

  dirty_ = false;
  if (!ok) {
    last_error_ = "Some shortcuts could not be parsed";
    if (error) *error = last_error_;
  }
  return ok;
}

bool ShortcutManager::SaveToDisk(std::string* error) {
  last_error_.clear();
  if (config_path_.empty()) {
    last_error_ = "Missing config path";
    if (error) *error = last_error_;
    return false;
  }

  std::filesystem::path path(config_path_);
  std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      last_error_ = "Failed to create shortcut config directory";
      if (error) *error = last_error_;
      return false;
    }
  }

  std::ofstream file(path, std::ios::trunc);
  if (!file.is_open()) {
    last_error_ = "Failed to write shortcut config";
    if (error) *error = last_error_;
    return false;
  }

  file << "# AFS Viz shortcuts\n";
  file << "# Format: action_key = Ctrl+Shift+Key\n";
  const ImGuiIO& io = ImGui::GetIO();
  for (const auto& def : Actions()) {
    std::string value = FormatShortcut(GetShortcut(def.id), io);
    file << def.config_key << " = " << value << "\n";
  }

  if (!file) {
    last_error_ = "Failed to write shortcut config";
    if (error) *error = last_error_;
    return false;
  }
  dirty_ = false;
  return true;
}

bool ShortcutManager::SaveIfDirty(std::string* error) {
  if (!dirty_) return false;
  std::string save_error;
  if (!SaveToDisk(&save_error)) {
    last_error_ = save_error;
    if (error) *error = last_error_;
    return false;
  }
  dirty_ = false;
  return true;
}

void ShortcutManager::BeginCapture(ActionId id) {
  capturing_action_ = id;
  capturing_ = true;
}

void ShortcutManager::CancelCapture() {
  capturing_action_ = ActionId::Count;
  capturing_ = false;
}

bool ShortcutManager::HandleCapture(const ImGuiIO& io) {
  if (!capturing_) return false;

  if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    CancelCapture();
    return true;
  }

  for (int key = ImGuiKey_NamedKey_BEGIN; key < ImGuiKey_NamedKey_END; ++key) {
    ImGuiKey key_value = static_cast<ImGuiKey>(key);
    if (!IsValidShortcutKey(key_value)) continue;
    if (ImGui::IsKeyPressed(key_value, false)) {
      Shortcut shortcut;
      shortcut.key = key_value;
      shortcut.ctrl = io.KeyCtrl;
      shortcut.shift = io.KeyShift;
      shortcut.alt = io.KeyAlt;
      shortcut.super = io.KeySuper;
      SetShortcut(capturing_action_, shortcut);
      CancelCapture();
      return true;
    }
  }

  return false;
}

bool ShortcutManager::ShouldProcessShortcuts(const ImGuiIO& io) const {
  if (capturing_) return false;
  if (io.WantTextInput) return false;
  return true;
}

bool ShortcutManager::IsTriggered(ActionId id, const ImGuiIO& io) const {
  if (!ShouldProcessShortcuts(io)) return false;
  const Shortcut& shortcut = GetShortcut(id);
  if (shortcut.key == ImGuiKey_None) return false;
  if (!ImGui::IsKeyPressed(shortcut.key, false)) return false;
  if (shortcut.ctrl != io.KeyCtrl) return false;
  if (shortcut.shift != io.KeyShift) return false;
  if (shortcut.alt != io.KeyAlt) return false;
  if (shortcut.super != io.KeySuper) return false;
  return true;
}

std::string ShortcutManager::FormatShortcut(ActionId id, const ImGuiIO& io) const {
  return FormatShortcut(GetShortcut(id), io);
}

std::string ShortcutManager::FormatShortcut(const Shortcut& shortcut,
                                            const ImGuiIO& io) {
  static_cast<void>(io);
  if (shortcut.key == ImGuiKey_None) return "";

  std::string label;
  auto append = [&](const char* part) {
    if (!label.empty()) label += "+";
    label += part;
  };

  if (shortcut.ctrl) append("Ctrl");
  if (shortcut.shift) append("Shift");
  if (shortcut.alt) append("Alt");
  if (shortcut.super) append("Super");

  const char* key_name = ImGui::GetKeyName(shortcut.key);
  append(key_name && key_name[0] != '\0' ? key_name : "?");

  return label;
}

size_t ShortcutManager::ToIndex(ActionId id) {
  return static_cast<size_t>(id);
}

const ActionDefinition* ShortcutManager::FindDefinition(
    const std::string& key) const {
  std::string needle = ToLower(key);
  for (const auto& def : Actions()) {
    if (ToLower(def.config_key) == needle) return &def;
    if (ToLower(def.label) == needle) return &def;
  }
  return nullptr;
}

std::string ShortcutManager::Trim(const std::string& text) {
  size_t start = 0;
  while (start < text.size() &&
         std::isspace(static_cast<unsigned char>(text[start]))) {
    ++start;
  }
  size_t end = text.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(start, end - start);
}

std::string ShortcutManager::ToLower(const std::string& text) {
  std::string out = text;
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return out;
}

ImGuiKey ShortcutManager::FindKeyByName(const std::string& name) {
  std::string target = ToLower(name);
  for (int key = ImGuiKey_NamedKey_BEGIN; key < ImGuiKey_NamedKey_END; ++key) {
    ImGuiKey key_value = static_cast<ImGuiKey>(key);
    const char* key_name = ImGui::GetKeyName(key_value);
    if (!key_name || key_name[0] == '\0') continue;
    if (ToLower(key_name) == target) return key_value;
  }
  return ImGuiKey_None;
}

bool ShortcutManager::ParseShortcutString(const std::string& text,
                                          Shortcut* shortcut,
                                          std::string* error) {
  if (!shortcut) return false;
  Shortcut result;
  std::string value = Trim(text);
  if (value.empty() || ToLower(value) == "none" ||
      ToLower(value) == "unbound") {
    *shortcut = result;
    return true;
  }

  std::stringstream stream(value);
  std::string token;
  bool has_key = false;
  while (std::getline(stream, token, '+')) {
    std::string part = Trim(token);
    if (part.empty()) continue;
    std::string lower = ToLower(part);

    if (lower == "ctrl" || lower == "control") {
      result.ctrl = true;
      continue;
    }
    if (lower == "shift") {
      result.shift = true;
      continue;
    }
    if (lower == "alt" || lower == "option") {
      result.alt = true;
      continue;
    }
    if (lower == "super" || lower == "cmd" || lower == "command" ||
        lower == "meta") {
      result.super = true;
      continue;
    }

    if (has_key) {
      if (error) *error = "Multiple keys specified";
      return false;
    }
    ImGuiKey key_value = FindKeyByName(part);
    if (key_value == ImGuiKey_None) {
      if (error) *error = "Unknown key";
      return false;
    }
    result.key = key_value;
    has_key = true;
  }

  if (!has_key) {
    if (error) *error = "Missing key";
    return false;
  }

  *shortcut = result;
  return true;
}

std::string ShortcutManager::ResolveDefaultPath() {
  const char* override_path = std::getenv("AFS_VIZ_SHORTCUTS_PATH");
  if (override_path && override_path[0] != '\0') {
    return override_path;
  }

  const char* xdg = std::getenv("XDG_CONFIG_HOME");
  const char* home = std::getenv("HOME");
  std::filesystem::path base;
  if (xdg && xdg[0] != '\0') {
    base = xdg;
  } else if (home && home[0] != '\0') {
    base = std::filesystem::path(home) / ".config";
  } else {
    base = std::filesystem::current_path();
  }
  return (base / "afs" / "viz_shortcuts.conf").string();
}

bool ShortcutManager::IsValidShortcutKey(ImGuiKey key) {
  // Exclude modifier keys from being bound as main shortcut keys
  switch (key) {
    case ImGuiKey_None:
    case ImGuiKey_LeftCtrl:
    case ImGuiKey_RightCtrl:
    case ImGuiKey_LeftShift:
    case ImGuiKey_RightShift:
    case ImGuiKey_LeftAlt:
    case ImGuiKey_RightAlt:
    case ImGuiKey_LeftSuper:
    case ImGuiKey_RightSuper:
      return false;
    default:
      return true;
  }
}

void RenderShortcutsWindow(ShortcutManager& shortcuts, bool* open) {
  if (!open || !*open) return;

  if (!ImGui::Begin("Keyboard Shortcuts", open)) {
    ImGui::End();
    return;
  }

  ImGui::Text("Customize keyboard shortcuts.");
  ImGui::TextDisabled("Click Edit, then press the desired key combo.");

  if (ImGui::Button("Save")) {
    shortcuts.SaveToDisk();
  }
  ImGui::SameLine();
  if (ImGui::Button("Reload")) {
    shortcuts.LoadFromDisk();
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset All")) {
    shortcuts.ResetAll();
  }
  if (shortcuts.IsCapturing()) {
    ImGui::SameLine();
    ImGui::TextDisabled("Press keys... (Esc to cancel)");
  }

  ImGui::TextDisabled("Config: %s", shortcuts.GetConfigPath().c_str());
  if (!shortcuts.GetLastError().empty()) {
    ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f),
                       "Shortcut error: %s", shortcuts.GetLastError().c_str());
  }

  static std::array<char, 64> filter{};
  ImGui::InputTextWithHint("##ShortcutFilter", "Filter actions",
                           filter.data(), filter.size());

  if (ImGui::BeginTable("ShortcutTable", 4,
                        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthFixed, 150.0f);
    ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Edit", ImGuiTableColumnFlags_WidthFixed, 140.0f);
    ImGui::TableHeadersRow();

    const std::string filter_text(filter.data());
    const ImGuiIO& io = ImGui::GetIO();
    for (const auto& action : shortcuts.Actions()) {
      if (!ContainsInsensitive(action.label, filter_text) &&
          !ContainsInsensitive(action.description, filter_text)) {
        continue;
      }

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s", action.label);

      ImGui::TableSetColumnIndex(1);
      std::string label = shortcuts.FormatShortcut(action.id, io);
      ImGui::TextDisabled("%s", label.empty() ? "Unbound" : label.c_str());

      ImGui::TableSetColumnIndex(2);
      ImGui::TextWrapped("%s", action.description);

      ImGui::TableSetColumnIndex(3);
      ImGui::PushID(static_cast<int>(action.id));
      bool is_capturing = shortcuts.IsCapturing() &&
                          shortcuts.CapturingAction() == action.id;
      if (is_capturing) {
        if (ImGui::Button("Cancel")) {
          shortcuts.CancelCapture();
        }
      } else if (ImGui::Button("Edit")) {
        shortcuts.BeginCapture(action.id);
      }
      ImGui::SameLine();
      if (ImGui::Button("Clear")) {
        shortcuts.SetShortcut(action.id, Shortcut{});
      }
      ImGui::SameLine();
      if (ImGui::Button("Reset")) {
        shortcuts.ResetShortcut(action.id);
      }
      ImGui::PopID();
    }

    ImGui::EndTable();
  }

  ImGui::End();
}

}  // namespace ui
}  // namespace viz
}  // namespace afs

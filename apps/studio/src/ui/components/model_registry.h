#pragma once

#include "core/registry_reader.h"
#include "deployment_panel.h"

#include <array>
#include <string>

namespace afs {
namespace studio {
namespace ui {

// Model Registry widget for listing and managing trained models
class ModelRegistryWidget {
 public:
  ModelRegistryWidget();

  // Render the widget (call this from main render loop)
  void Render();

  // Reload models from registry
  void Refresh();

  // Get selected model (if any)
  const ModelMetadata* GetSelectedModel() const;

 private:
  void RenderToolbar();
  void RenderModelList();
  void RenderModelDetails();
  void RenderDeploymentPanel();
  void RenderModelCard(const ModelMetadata& model, int index);

  RegistryReader registry_;
  DeploymentPanel deployment_panel_;
  int selected_model_index_ = -1;

  // Filter state
  std::array<char, 64> filter_text_{};
  int filter_role_ = 0;       // 0 = All
  int filter_location_ = 0;   // 0 = All
  int filter_backend_ = 0;    // 0 = All

  // Filter options
  static constexpr const char* kRoleOptions[] = {
      "All", "asm", "debug", "general", "yaze"};
  static constexpr int kRoleCount = 5;

  static constexpr const char* kLocationOptions[] = {"All", "mac", "windows",
                                                      "cloud", "halext"};
  static constexpr int kLocationCount = 5;

  static constexpr const char* kBackendOptions[] = {
      "All", "ollama", "llama.cpp", "vllm", "transformers", "halext-node"};
  static constexpr int kBackendCount = 6;

  // UI state
  bool show_details_ = true;
  bool show_deployment_ = true;
  std::string last_error_;
};

// Render a standalone model registry window
void RenderModelRegistryWindow(ModelRegistryWidget& widget, bool* open);

}  // namespace ui
}  // namespace studio
}  // namespace afs


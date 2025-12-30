#pragma once

#include "core/deployment_actions.h"
#include "core/registry_reader.h"

#include <array>
#include <string>

namespace afs {
namespace studio {
namespace ui {

// Deployment panel for model actions
class DeploymentPanel {
 public:
  DeploymentPanel();

  // Render the panel (typically shown when a model is selected)
  void Render(const ModelMetadata* selected_model);

  // Check if an operation is in progress
  bool IsBusy() const { return is_busy_; }

 private:
  void RenderDeploymentStatus();
  void RenderQuickActions(const ModelMetadata& model);
  void RenderOllamaDeployment(const ModelMetadata& model);
  void RenderConversionOptions(const ModelMetadata& model);
  void RenderTestPrompt(const ModelMetadata& model);

  DeploymentActions actions_;

  // UI State
  bool is_busy_ = false;
  std::string current_operation_;
  float progress_ = 0.0f;
  std::string status_message_;
  std::string last_error_;
  ActionResult last_result_;

  // Deployment options
  int selected_quantization_ = 1;  // Default to Q4_K_M
  std::array<char, 64> ollama_name_buffer_{};
  std::array<char, 512> test_prompt_buffer_{};
  std::string test_output_;
};

// Render as standalone window
void RenderDeploymentPanelWindow(DeploymentPanel& panel,
                                  const ModelMetadata* model,
                                  bool* open);

}  // namespace ui
}  // namespace studio
}  // namespace afs

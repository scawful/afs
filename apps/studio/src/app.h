#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <functional>

#include <imgui.h>
#include "data_loader.h"
#include "models/state.h"
#include "core/context.h"
#include "core/assets.h"
#include "ui/shortcuts.h"
#include "ui/components/model_registry.h"
#include "ui/components/training_dashboard.h"
#include "widgets/text_editor.h"
#include "widgets/imgui_memory_editor.h"
#include "widgets/sample_review.h"
#include "ui/charts/quality_trends.h"
#include "ui/charts/generator_efficiency.h"
#include "ui/charts/coverage_density.h"
#include "ui/components/graph_browser.h"
#include "ui/components/graph_navigator.h"
#include "ui/components/companion_panels.h"
#include "core/llama_client.h"

namespace afs {
namespace viz {

class App {
 public:
  explicit App(const std::string& data_path);
  ~App() = default;

  int Run();

  DataLoader& loader() { return loader_; }
  const DataLoader& loader() const { return loader_; }

 private:
 void RefreshData(const char* reason);
 void SeedDefaultState();
 void SyncDataBackedState();
 void TickSimulatedMetrics(float dt);
 void EnsureActiveGraph();
  
  void RenderFrame();
  void RenderLayout();
  
  // Workspace Views
  void RenderDashboardView();
  void RenderAnalysisView();
  void RenderOptimizationView();
  void RenderSystemsView();
  void RenderCustomGridView();
  void RenderChatView();
  void RenderTrainingView();
  void RenderContextView();
  void RenderModelsView();
  void RenderExpandedPlot();
  void RenderFloaters();

  // Infrastructure
  std::string data_path_;
  DataLoader loader_;
  AppState state_;
  std::unique_ptr<studio::core::GraphicsContext> context_;

  // Editors & Widgets
  TextEditor text_editor_;
  MemoryEditorWidget memory_editor_;
  SampleReviewWidget sample_review_;
  ui::ShortcutManager shortcut_manager_;
  studio::ui::ModelRegistryWidget model_registry_widget_;
  studio::ui::TrainingDashboardWidget training_dashboard_widget_;

  // Modular Charts
  ui::QualityTrendsChart quality_trends_chart_;
  ui::GeneratorEfficiencyChart generator_efficiency_chart_;
  ui::CoverageDensityChart coverage_density_chart_;

  // Graph View System
  ui::GraphBrowser graph_browser_;
  ui::GraphNavigator graph_navigator_;
  ui::CompanionPanels companion_panels_;

  // LLM Chat Client
  LlamaClient llama_client_;

  // State flags
  bool show_sample_review_ = false;
  bool show_shortcuts_window_ = false;

  // Typography
  studio::core::AssetLoader::Fonts fonts_;
};

}  // namespace viz
}  // namespace afs

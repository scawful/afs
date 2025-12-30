#include "training_dashboard.h"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>

namespace afs {
namespace studio {
namespace ui {

TrainingDashboardWidget::TrainingDashboardWidget() {
  Refresh();
}

void TrainingDashboardWidget::Refresh() {
  std::string error;
  if (!monitor_.Poll(&error)) {
    last_error_ = error;
  } else {
    last_error_.clear();
  }
}

void TrainingDashboardWidget::Render() {
  // Auto-refresh check
  if (auto_refresh_ && monitor_.ShouldRefresh()) {
    Refresh();
  }

  // Toolbar
  if (ImGui::Button("Refresh")) {
    Refresh();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Auto", &auto_refresh_);
  ImGui::SameLine();

  const auto& state = monitor_.GetState();

  // Status indicator
  ImVec4 status_color;
  const char* status_text;
  switch (state.status) {
    case TrainingStatus::kRunning:
      status_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
      status_text = "RUNNING";
      break;
    case TrainingStatus::kCompleted:
      status_color = ImVec4(0.3f, 0.6f, 1.0f, 1.0f);
      status_text = "COMPLETED";
      break;
    case TrainingStatus::kFailed:
      status_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
      status_text = "FAILED";
      break;
    case TrainingStatus::kPaused:
      status_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);
      status_text = "PAUSED";
      break;
    default:
      status_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
      status_text = "IDLE";
      break;
  }

  ImGui::TextColored(status_color, "[%s]", status_text);

  if (!last_error_.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "%s",
                       last_error_.c_str());
  }

  ImGui::Separator();

  // Main content
  if (state.status == TrainingStatus::kIdle &&
      state.loss_history.empty()) {
    ImGui::TextDisabled("No active training detected.");
    ImGui::TextDisabled("Mount point: %s",
                        monitor_.GetConfig().windows_mount_path.c_str());
    return;
  }

  // Top section: Status and Progress
  RenderStatusCard();

  ImGui::Spacing();

  // Loss curve chart
  RenderLossCurve();

  ImGui::Spacing();

  // Metrics grid
  RenderMetricsGrid();

  // Source info
  RenderSourceInfo();
}

void TrainingDashboardWidget::RenderStatusCard() {
  const auto& state = monitor_.GetState();

  // Model name
  if (!state.model_name.empty()) {
    ImGui::Text("Model: %s", state.model_name.c_str());
  }

  // Progress bar
  ImGui::Text("Progress:");
  ImGui::SameLine();

  float progress = state.progress_percent / 100.0f;
  char overlay[32];
  std::snprintf(overlay, sizeof(overlay), "%.1f%% (%d/%d)",
                state.progress_percent, state.current_step, state.total_steps);

  ImGui::ProgressBar(progress, ImVec2(-1, 0), overlay);

  // Epoch info
  ImGui::Text("Epoch %d/%d | Step %d/%d", state.current_epoch,
              state.total_epochs, state.current_step, state.total_steps);
}

void TrainingDashboardWidget::RenderLossCurve() {
  const auto& state = monitor_.GetState();

  if (state.loss_history.empty()) {
    ImGui::TextDisabled("No loss data available");
    return;
  }

  // Prepare data for ImPlot
  std::vector<float> steps, losses, eval_losses;
  bool has_eval = false;

  for (const auto& point : state.loss_history) {
    steps.push_back(static_cast<float>(point.step));
    losses.push_back(point.loss);
    if (point.eval_loss > 0.0f) {
      eval_losses.push_back(point.eval_loss);
      has_eval = true;
    } else {
      eval_losses.push_back(point.loss);  // Fallback
    }
  }

  float height = 200.0f;
  if (ImPlot::BeginPlot("Training Loss", ImVec2(-1, height))) {
    ImPlot::SetupAxes("Step", "Loss");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, *std::max_element(losses.begin(), losses.end()) * 1.1, ImPlotCond_Always);

    ImPlot::SetNextLineStyle(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), 2.0f);
    ImPlot::PlotLine("Train Loss", steps.data(), losses.data(),
                     static_cast<int>(steps.size()));

    if (has_eval) {
      ImPlot::SetNextLineStyle(ImVec4(0.9f, 0.5f, 0.2f, 1.0f), 2.0f);
      ImPlot::PlotLine("Eval Loss", steps.data(), eval_losses.data(),
                       static_cast<int>(steps.size()));
    }

    // Mark best loss
    if (state.best_step > 0) {
      float best_x = static_cast<float>(state.best_step);
      float best_y = state.best_loss;
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 8.0f,
                                  ImVec4(0.2f, 0.9f, 0.3f, 1.0f), 2.0f);
      ImPlot::PlotScatter("Best", &best_x, &best_y, 1);
    }

    ImPlot::EndPlot();
  }
}

void TrainingDashboardWidget::RenderMetricsGrid() {
  const auto& state = monitor_.GetState();

  if (ImGui::BeginTable("MetricsGrid", 4,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Value");
    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Value");
    ImGui::TableHeadersRow();

    // Row 1: Current Loss | Best Loss
    ImGui::TableNextColumn();
    ImGui::TextDisabled("Current Loss");
    ImGui::TableNextColumn();
    ImGui::Text("%.4f", state.current_loss);
    ImGui::TableNextColumn();
    ImGui::TextDisabled("Best Loss");
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "%.4f @ %d",
                       state.best_loss, state.best_step);

    // Row 2: Eval Loss | Perplexity
    ImGui::TableNextColumn();
    ImGui::TextDisabled("Eval Loss");
    ImGui::TableNextColumn();
    if (state.eval_loss.has_value()) {
      ImGui::Text("%.4f", state.eval_loss.value());
    } else {
      ImGui::TextDisabled("N/A");
    }
    ImGui::TableNextColumn();
    ImGui::TextDisabled("Perplexity");
    ImGui::TableNextColumn();
    if (state.perplexity.has_value()) {
      ImGui::Text("%.2f", state.perplexity.value());
    } else {
      ImGui::TextDisabled("N/A");
    }

    // Row 3: GPU | ETA
    ImGui::TableNextColumn();
    ImGui::TextDisabled("Device");
    ImGui::TableNextColumn();
    ImGui::Text("%s", state.device.empty() ? "GPU" : state.device.c_str());
    ImGui::TableNextColumn();
    ImGui::TextDisabled("ETA");
    ImGui::TableNextColumn();
    if (state.estimated_remaining_minutes > 0) {
      int hours = state.estimated_remaining_minutes / 60;
      int mins = state.estimated_remaining_minutes % 60;
      if (hours > 0) {
        ImGui::Text("%dh %dm", hours, mins);
      } else {
        ImGui::Text("%dm", mins);
      }
    } else {
      ImGui::TextDisabled("--");
    }

    ImGui::EndTable();
  }
}

void TrainingDashboardWidget::RenderSourceInfo() {
  const auto& state = monitor_.GetState();

  ImGui::Spacing();
  ImGui::TextDisabled("Source: %s (%s)",
                      state.source_location.empty() ? "local"
                                                    : state.source_location.c_str(),
                      state.is_remote ? "remote" : "local");

  if (!state.source_path.empty()) {
    ImGui::TextDisabled("Path: %s", state.source_path.c_str());
  }
}

void RenderTrainingDashboardWindow(TrainingDashboardWidget& widget,
                                    bool* open) {
  if (!open || !*open) return;

  ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);

  if (!ImGui::Begin("Training Monitor", open)) {
    ImGui::End();
    return;
  }

  widget.Render();

  ImGui::End();
}

}  // namespace ui
}  // namespace studio
}  // namespace afs

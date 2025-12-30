#pragma once

#include "core/training_monitor.h"

#include <string>

namespace afs {
namespace studio {
namespace ui {

// Training dashboard widget for real-time training monitoring
class TrainingDashboardWidget {
 public:
  TrainingDashboardWidget();

  // Render the widget
  void Render();

  // Manual refresh
  void Refresh();

  // Get training monitor
  TrainingMonitor& GetMonitor() { return monitor_; }

 private:
  void RenderStatusCard();
  void RenderProgressBar();
  void RenderLossCurve();
  void RenderMetricsGrid();
  void RenderSourceInfo();

  TrainingMonitor monitor_;
  bool auto_refresh_ = true;
  std::string last_error_;
};

// Render as standalone window
void RenderTrainingDashboardWindow(TrainingDashboardWidget& widget, bool* open);

}  // namespace ui
}  // namespace studio
}  // namespace afs

#pragma once

#include "chart.h"
#include "../../data_loader.h"
#include "../../models/state.h"

namespace afs::viz::ui {

class CoverageDensityChart : public Chart {
public:
  void Render(AppState& state, const DataLoader& loader) override;
  std::string GetTitle() const override { return "Density Coverage"; }
  PlotKind GetKind() const override { return PlotKind::CoverageDensity; }
};

} // namespace afs::viz::ui

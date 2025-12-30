#pragma once

#include "chart.h"
#include "../../data_loader.h"
#include "../../models/state.h"

namespace afs::viz::ui {

class GeneratorEfficiencyChart : public Chart {
public:
  void Render(AppState& state, const DataLoader& loader) override;
  std::string GetTitle() const override { return "Generator Efficiency"; }
  PlotKind GetKind() const override { return PlotKind::GeneratorEfficiency; }
};

} // namespace afs::viz::ui

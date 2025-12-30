#pragma once

#include <string>
#include <implot.h>
#include "../../models/state.h"
#include "../../data_loader.h"

namespace afs::viz::ui {

class Chart {
public:
    virtual ~Chart() = default;

    // Render the chart content. 
    // Return true if the chart is visible/active, false if closed.
    virtual void Render(AppState& state, const DataLoader& loader) = 0;

    // Optional: Get the chart title/ID
    virtual std::string GetTitle() const = 0;
    
    // Optional: Get the chart kind
    virtual PlotKind GetKind() const = 0;
};

} // namespace afs::viz::ui

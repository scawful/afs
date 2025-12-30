#pragma once

#include <string>
#include "../../models/state.h"
#include "../../data_loader.h"

namespace afs::viz::ui {

class CompanionPanels {
public:
    CompanionPanels() = default;
    
    // Render all active companion panels for the current graph
    void Render(AppState& state, const DataLoader& loader);
    
private:
    void RenderFilterPanel(AppState& state, const DataLoader& loader);
    void RenderDataQualityPanel(AppState& state, const DataLoader& loader);
    void RenderInspectorPanel(AppState& state, const DataLoader& loader);
    void RenderControlsPanel(AppState& state);
    
    // Helper to determine which panels should be visible
    struct PanelVisibility {
        bool filter = false;
        bool data_quality = false;
        bool inspector = false;
        bool controls = false;
    };
    
    PanelVisibility GetPanelVisibility(PlotKind kind) const;
};

} // namespace afs::viz::ui

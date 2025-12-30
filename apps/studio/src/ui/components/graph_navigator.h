#pragma once

#include <string>
#include "../../models/state.h"

namespace afs::viz::ui {

class GraphBrowser;  // Forward declaration

class GraphNavigator {
public:
    GraphNavigator() = default;
    
    // Render navigation toolbar (breadcrumbs, back/forward)
    void RenderToolbar(AppState& state, const GraphBrowser& browser);
    
    // Navigation actions
    void NavigateBack(AppState& state);
    void NavigateForward(AppState& state);
    void NavigateToGraph(AppState& state, PlotKind kind);
    void ToggleBookmark(AppState& state, PlotKind kind);
    
    // Check navigation state
    bool CanNavigateBack(const AppState& state) const;
    bool CanNavigateForward(const AppState& state) const;
    bool IsBookmarked(const AppState& state, PlotKind kind) const;
    
private:
    void RenderBreadcrumbs(AppState& state, const GraphBrowser& browser);
};

} // namespace afs::viz::ui

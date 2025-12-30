#pragma once

#include <string>
#include <vector>
#include "../../models/state.h"

namespace afs::viz {
class DataLoader;
}

namespace afs::viz::ui {

// Graph metadata for browser display
struct GraphInfo {
    PlotKind kind;
    std::string name;
    std::string description;
    GraphCategory category;
    bool supports_comparison;
    bool needs_filter_panel;
    bool needs_data_quality_panel;
    bool needs_inspector_panel;
};

class GraphBrowser {
public:
    GraphBrowser();
    
    // Render the graph browser sidebar
    void Render(AppState& state, const DataLoader& loader);
    
    // Get all available graphs
    const std::vector<GraphInfo>& GetAllGraphs() const { return all_graphs_; }
    
    // Get filtered graphs based on category and search
    std::vector<GraphInfo> GetFilteredGraphs(GraphCategory category,
                                             const std::string& search) const;
    
    // Get graph info by kind
    const GraphInfo* GetGraphInfo(PlotKind kind) const;

    bool IsGraphAvailable(PlotKind kind,
                          const AppState& state,
                          const DataLoader& loader) const;
    
    // Get category name
    static const char* GetCategoryName(GraphCategory category);
    
private:
    std::vector<GraphInfo> all_graphs_;
    
    void RenderCategorySection(const char* title, GraphCategory category, AppState& state);
    void RenderGraphItem(const GraphInfo& info, AppState& state);
    void InitializeGraphRegistry();
};

} // namespace afs::viz::ui

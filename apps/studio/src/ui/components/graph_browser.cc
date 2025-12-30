#include "graph_browser.h"
#include "../core.h"
#include "../../icons.h"
#include <imgui.h>
#include <algorithm>

namespace afs::viz::ui {

GraphBrowser::GraphBrowser() {
    InitializeGraphRegistry();
}

void GraphBrowser::InitializeGraphRegistry() {
    all_graphs_ = {
        // Training Category
        {PlotKind::TrainingLoss, "Training Loss", "Loss curves over training steps", GraphCategory::Training, true, true, true, true},
        {PlotKind::LossVsSamples, "Loss vs Samples", "Training loss progression by sample count", GraphCategory::Training, false, true, true, false},
        
        // Quality Category
        {PlotKind::QualityTrends, "Quality Trends", "Data quality trends by domain", GraphCategory::Quality, true, true, true, true},
        {PlotKind::QualityDirection, "Quality Direction", "Quality improvement/degradation tracking", GraphCategory::Quality, false, true, true, false},
        {PlotKind::EmbeddingQuality, "Embedding Quality", "Embedding space quality metrics", GraphCategory::Quality, false, true, true, true},
        {PlotKind::Effectiveness, "Effectiveness", "Generator effectiveness analysis", GraphCategory::Quality, false, true, true, false},
        
        // System Category
        {PlotKind::AgentThroughput, "Agent Workload", "Tasks completed and queue depth per agent", GraphCategory::System, false, false, true, true},
        {PlotKind::MountsStatus, "Mounts Status", "Filesystem mount status", GraphCategory::System, false, false, true, false},
        
        // Coverage Category
        {PlotKind::CoverageDensity, "Coverage Density", "Data coverage density heatmap", GraphCategory::Coverage, false, true, true, true},
        {PlotKind::DomainCoverage, "Domain Coverage", "Per-domain coverage analysis", GraphCategory::Coverage, false, true, true, true},
        
        // Embedding Category
        {PlotKind::EmbeddingDensity, "Embedding Density", "Embedding space density visualization", GraphCategory::Embedding, false, true, true, true},
        {PlotKind::LatentSpace, "Embedding Map", "Synthetic layout from embedding regions", GraphCategory::Embedding, false, true, true, true},
        {PlotKind::KnowledgeGraph, "Context Graph", "AFS context and mount relationships", GraphCategory::Embedding, false, false, false, false},
        
        // Optimization Category
        {PlotKind::GeneratorEfficiency, "Generator Efficiency", "Generator performance metrics", GraphCategory::Optimization, true, true, true, true},
        {PlotKind::GeneratorMix, "Generator Mix", "Generator usage distribution", GraphCategory::Optimization, false, true, true, false},
        {PlotKind::Rejections, "Rejections", "Sample rejection analysis", GraphCategory::Optimization, false, true, true, true},
        {PlotKind::Thresholds, "Threshold Optimization", "Quality threshold optimization curves", GraphCategory::Optimization, false, true, true, false},
        {PlotKind::EvalMetrics, "Eval Metrics", "Evaluation metric comparisons", GraphCategory::Optimization, true, true, true, false},
    };
}

const char* GraphBrowser::GetCategoryName(GraphCategory category) {
    switch (category) {
        case GraphCategory::Training: return "Training";
        case GraphCategory::Quality: return "Quality";
        case GraphCategory::System: return "System";
        case GraphCategory::Coverage: return "Coverage";
        case GraphCategory::Embedding: return "Embedding";
        case GraphCategory::Optimization: return "Optimization";
        case GraphCategory::All: return "All Graphs";
        default: return "Unknown";
    }
}

std::vector<GraphInfo> GraphBrowser::GetFilteredGraphs(GraphCategory category, const std::string& search) const {
    std::vector<GraphInfo> filtered;
    
    for (const auto& graph : all_graphs_) {
        // Category filter
        if (category != GraphCategory::All && graph.category != category) {
            continue;
        }
        
        // Search filter
        if (!search.empty()) {
            std::string lower_name = graph.name;
            std::string lower_search = search;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
            std::transform(lower_search.begin(), lower_search.end(), lower_search.begin(), ::tolower);
            
            if (lower_name.find(lower_search) == std::string::npos) {
                continue;
            }
        }
        
        filtered.push_back(graph);
    }
    
    return filtered;
}

const GraphInfo* GraphBrowser::GetGraphInfo(PlotKind kind) const {
    for (const auto& graph : all_graphs_) {
        if (graph.kind == kind) {
            return &graph;
        }
    }
    return nullptr;
}

void GraphBrowser::Render(AppState& state) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
    
    // Search bar
    ImGui::SetNextItemWidth(-FLT_MIN);
    char search_buf[128];
    strncpy(search_buf, state.graph_search_query.c_str(), sizeof(search_buf) - 1);
    search_buf[sizeof(search_buf) - 1] = '\0';
    
    if (ImGui::InputTextWithHint("##GraphSearch", ICON_MD_SEARCH " Search graphs...", 
                                  search_buf, sizeof(search_buf))) {
        state.graph_search_query = search_buf;
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Category filter tabs
    if (ImGui::BeginTabBar("GraphCategories", ImGuiTabBarFlags_FittingPolicyScroll)) {
        const GraphCategory categories[] = {
            GraphCategory::All, GraphCategory::Training, GraphCategory::Quality,
            GraphCategory::System, GraphCategory::Coverage, GraphCategory::Embedding,
            GraphCategory::Optimization
        };
        
        for (auto cat : categories) {
            if (ImGui::BeginTabItem(GetCategoryName(cat))) {
                state.browser_filter = cat;
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
    
    ImGui::Spacing();
    
    // Bookmarks section
    if (!state.graph_bookmarks.empty()) {
        if (ImGui::CollapsingHeader(ICON_MD_BOOKMARK " Bookmarks", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto kind : state.graph_bookmarks) {
                const GraphInfo* info = GetGraphInfo(kind);
                if (info) {
                    RenderGraphItem(*info, state);
                }
            }
        }
        ImGui::Spacing();
    }
    
    // Recent graphs
    if (!state.graph_history.empty()) {
        if (ImGui::CollapsingHeader(ICON_MD_ACCESS_TIME " Recent", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Show last 5 unique graphs
            std::vector<PlotKind> recent_unique;
            for (auto it = state.graph_history.rbegin(); 
                 it != state.graph_history.rend() && recent_unique.size() < 5; ++it) {
                if (std::find(recent_unique.begin(), recent_unique.end(), *it) == recent_unique.end()) {
                    recent_unique.push_back(*it);
                }
            }
            
            for (auto kind : recent_unique) {
                const GraphInfo* info = GetGraphInfo(kind);
                if (info) {
                    RenderGraphItem(*info, state);
                }
            }
        }
        ImGui::Spacing();
    }
    
    // All graphs (filtered)
    ImGui::Separator();
    ImGui::Spacing();
    
    auto filtered = GetFilteredGraphs(state.browser_filter, state.graph_search_query);
    
    if (filtered.empty()) {
        ImGui::TextDisabled("No graphs found");
    } else {
        ImGui::BeginChild("GraphList", ImVec2(0, 0), false);
        for (const auto& graph : filtered) {
            RenderGraphItem(graph, state);
        }
        ImGui::EndChild();
    }
    
    ImGui::PopStyleVar();
}

void GraphBrowser::RenderGraphItem(const GraphInfo& info, AppState& state) {
    bool is_active = state.active_graph == info.kind;
    bool is_bookmarked = std::find(state.graph_bookmarks.begin(), 
                                    state.graph_bookmarks.end(), 
                                    info.kind) != state.graph_bookmarks.end();
    
    ImGui::PushID(static_cast<int>(info.kind));
    
    if (is_active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.59f, 0.98f, 0.40f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.59f, 0.98f, 0.60f));
    }
    
    if (ImGui::Button(info.name.c_str(), ImVec2(-FLT_MIN, 0))) {
        // Navigate to this graph
        if (state.active_graph != info.kind) {
            // Add to history (truncate forward history if we're not at the end)
            if (state.graph_history_index >= 0 && 
                state.graph_history_index < static_cast<int>(state.graph_history.size()) - 1) {
                state.graph_history.erase(
                    state.graph_history.begin() + state.graph_history_index + 1,
                    state.graph_history.end()
                );
            }
            
            state.graph_history.push_back(info.kind);
            state.graph_history_index = static_cast<int>(state.graph_history.size()) - 1;
            state.active_graph = info.kind;
        }
    }
    
    if (is_active) {
        ImGui::PopStyleColor(2);
    }
    
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("%s", info.description.c_str());
        ImGui::TextDisabled("Category: %s", GetCategoryName(info.category));
        if (info.supports_comparison) {
            ImGui::TextDisabled(ICON_MD_COMPARE " Supports comparison");
        }
        ImGui::EndTooltip();
    }
    
    // Context menu for bookmarking
    if (ImGui::BeginPopupContextItem()) {
        if (is_bookmarked) {
            if (ImGui::MenuItem(ICON_MD_BOOKMARK " Remove Bookmark")) {
                state.graph_bookmarks.erase(
                    std::remove(state.graph_bookmarks.begin(), 
                               state.graph_bookmarks.end(), info.kind),
                    state.graph_bookmarks.end()
                );
            }
        } else {
            if (ImGui::MenuItem(ICON_MD_BOOKMARK_BORDER " Add Bookmark")) {
                state.graph_bookmarks.push_back(info.kind);
            }
        }
        ImGui::EndPopup();
    }
    
    ImGui::PopID();
}

} // namespace afs::viz::ui

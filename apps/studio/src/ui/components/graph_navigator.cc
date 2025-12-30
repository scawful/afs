#include "graph_navigator.h"
#include "graph_browser.h"
#include "../core.h"
#include "../../icons.h"
#include <imgui.h>
#include <algorithm>

namespace afs::viz::ui {

bool GraphNavigator::CanNavigateBack(const AppState& state) const {
    return state.graph_history_index > 0;
}

bool GraphNavigator::CanNavigateForward(const AppState& state) const {
    return state.graph_history_index >= 0 && 
           state.graph_history_index < static_cast<int>(state.graph_history.size()) - 1;
}

bool GraphNavigator::IsBookmarked(const AppState& state, PlotKind kind) const {
    return std::find(state.graph_bookmarks.begin(), 
                    state.graph_bookmarks.end(), kind) != state.graph_bookmarks.end();
}

void GraphNavigator::NavigateBack(AppState& state) {
    if (CanNavigateBack(state)) {
        state.graph_history_index--;
        state.active_graph = state.graph_history[state.graph_history_index];
    }
}

void GraphNavigator::NavigateForward(AppState& state) {
    if (CanNavigateForward(state)) {
        state.graph_history_index++;
        state.active_graph = state.graph_history[state.graph_history_index];
    }
}

void GraphNavigator::NavigateToGraph(AppState& state, PlotKind kind) {
    if (state.active_graph == kind) return;
    
    // Truncate forward history
    if (state.graph_history_index >= 0 && 
        state.graph_history_index < static_cast<int>(state.graph_history.size()) - 1) {
        state.graph_history.erase(
            state.graph_history.begin() + state.graph_history_index + 1,
            state.graph_history.end()
        );
    }
    
    state.graph_history.push_back(kind);
    state.graph_history_index = static_cast<int>(state.graph_history.size()) - 1;
    state.active_graph = kind;
}

void GraphNavigator::ToggleBookmark(AppState& state, PlotKind kind) {
    auto it = std::find(state.graph_bookmarks.begin(), state.graph_bookmarks.end(), kind);
    if (it != state.graph_bookmarks.end()) {
        state.graph_bookmarks.erase(it);
    } else {
        state.graph_bookmarks.push_back(kind);
    }
}

void GraphNavigator::RenderToolbar(AppState& state, const GraphBrowser& browser) {
    // Navigation buttons
    ImGui::BeginDisabled(!CanNavigateBack(state));
    if (ImGui::Button(ICON_MD_ARROW_BACK)) {
        NavigateBack(state);
    }
    ImGui::EndDisabled();
    
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Back (Alt+Left)");
    }
    
    ImGui::SameLine();
    
    ImGui::BeginDisabled(!CanNavigateForward(state));
    if (ImGui::Button(ICON_MD_ARROW_FORWARD)) {
        NavigateForward(state);
    }
    ImGui::EndDisabled();
    
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Forward (Alt+Right)");
    }
    
    ImGui::SameLine();
    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();
    
    // Bookmark button (disabled if no graph selected)
    ImGui::BeginDisabled(state.active_graph == PlotKind::None);
    bool is_bookmarked = IsBookmarked(state, state.active_graph);
    if (is_bookmarked) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
    }
    
    if (ImGui::Button(is_bookmarked ? ICON_MD_BOOKMARK : ICON_MD_BOOKMARK_BORDER)) {
        ToggleBookmark(state, state.active_graph);
    }
    
    if (is_bookmarked) {
        ImGui::PopStyleColor();
    }
    ImGui::EndDisabled();
    
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(is_bookmarked ? "Remove Bookmark (Ctrl+D)" : "Add Bookmark (Ctrl+D)");
    }
    
    ImGui::SameLine();
    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();
    
    // Breadcrumbs
    RenderBreadcrumbs(state, browser);
}

void GraphNavigator::RenderBreadcrumbs(AppState& state, const GraphBrowser& browser) {
    if (state.active_graph == PlotKind::None) {
        ImGui::TextDisabled("No graph selected");
        return;
    }
    
    const GraphInfo* info = browser.GetGraphInfo(state.active_graph);
    
    if (!info) {
        ImGui::Text("Unknown Graph");
        return;
    }
    
    // Category > Graph Name
    ImGui::Text(ICON_MD_SHOW_CHART);
    ImGui::SameLine();
    ImGui::TextDisabled("%s", GraphBrowser::GetCategoryName(info->category));
    ImGui::SameLine();
    ImGui::Text(">");
    ImGui::SameLine();
    ImGui::Text("%s", info->name.c_str());
    
    // Show history depth
    if (state.graph_history.size() > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%d / %d)", 
                           state.graph_history_index + 1, 
                           static_cast<int>(state.graph_history.size()));
    }
}

} // namespace afs::viz::ui

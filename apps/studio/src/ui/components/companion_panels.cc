#include "companion_panels.h"
#include "graph_browser.h"
#include "../core.h"
#include "../../icons.h"
#include <imgui.h>

namespace afs::viz::ui {

CompanionPanels::PanelVisibility CompanionPanels::GetPanelVisibility(PlotKind kind) const {
    PanelVisibility vis;
    
    // Define which panels each graph type needs
    switch (kind) {
        case PlotKind::QualityTrends:
        case PlotKind::GeneratorEfficiency:
        case PlotKind::CoverageDensity:
        case PlotKind::TrainingLoss:
        case PlotKind::DomainCoverage:
        case PlotKind::EmbeddingQuality:
        case PlotKind::Rejections:
        case PlotKind::EvalMetrics:
        case PlotKind::EmbeddingDensity:
        case PlotKind::LatentSpace:
            vis.filter = true;
            vis.data_quality = true;
            vis.inspector = true;
            vis.controls = true;
            break;
            
        case PlotKind::GeneratorMix:
        case PlotKind::Effectiveness:
        case PlotKind::Thresholds:
        case PlotKind::LossVsSamples:
        case PlotKind::QualityDirection:
            vis.filter = true;
            vis.data_quality = true;
            vis.controls = true;
            break;
            
        case PlotKind::AgentThroughput:
        case PlotKind::MountsStatus:
            vis.data_quality = true;
            vis.inspector = true;
            break;
            
        case PlotKind::KnowledgeGraph:
            vis.controls = true;
            break;
            
        default:
            break;
    }
    
    return vis;
}

void CompanionPanels::Render(AppState& state, const DataLoader& loader) {
    if (!state.show_companion_panels || state.active_graph == PlotKind::None) {
        return;
    }
    
    auto visibility = GetPanelVisibility(state.active_graph);
    
    // Render each panel in a collapsible section
    if (visibility.filter) {
        if (ImGui::CollapsingHeader(ICON_MD_FILTER_ALT " Filters", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderFilterPanel(state, loader);
        }
        ImGui::Spacing();
    }
    
    if (visibility.data_quality) {
        if (ImGui::CollapsingHeader(ICON_MD_BAR_CHART " Data Quality", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderDataQualityPanel(state, loader);
        }
        ImGui::Spacing();
    }
    
    if (visibility.inspector) {
        if (ImGui::CollapsingHeader(ICON_MD_INFO " Inspector", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderInspectorPanel(state, loader);
        }
        ImGui::Spacing();
    }
    
    if (visibility.controls) {
        if (ImGui::CollapsingHeader(ICON_MD_TUNE " Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderControlsPanel(state);
        }
    }
}

void CompanionPanels::RenderFilterPanel(AppState& state, const DataLoader& loader) {
    // Get or create filter settings for this graph
    auto& filters = state.graph_filters[state.active_graph];
    
    // Domain filtering
    const auto& domain_vis = loader.GetDomainVisibility();
    if (!domain_vis.empty()) {
        ImGui::Text("Domains:");
        ImGui::Indent();
        
        for (const auto& [domain, visible] : domain_vis) {
            bool is_active = std::find(filters.active_domains.begin(), 
                                      filters.active_domains.end(), 
                                      domain) != filters.active_domains.end();
            
            if (filters.active_domains.empty()) {
                // If no explicit filter, show all
                is_active = true;
            }
            
            if (ImGui::Checkbox(domain.c_str(), &is_active)) {
                if (is_active) {
                    if (std::find(filters.active_domains.begin(), 
                                 filters.active_domains.end(), domain) == filters.active_domains.end()) {
                        filters.active_domains.push_back(domain);
                    }
                } else {
                    filters.active_domains.erase(
                        std::remove(filters.active_domains.begin(), 
                                   filters.active_domains.end(), domain),
                        filters.active_domains.end()
                    );
                }
            }
        }
        
        if (ImGui::Button("Clear All")) {
            filters.active_domains.clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Select All")) {
            filters.active_domains.clear();
            for (const auto& [domain, _] : domain_vis) {
                filters.active_domains.push_back(domain);
            }
        }
        
        ImGui::Unindent();
        ImGui::Spacing();
    }
    
    // Run filtering (for training graphs)
    const auto& runs = loader.GetTrainingRuns();
    if (!runs.empty()) {
        ImGui::Text("Training Run:");
        ImGui::SetNextItemWidth(-FLT_MIN);
        
        if (ImGui::BeginCombo("##RunFilter", 
                             filters.active_run_id.empty() ? "All Runs" : filters.active_run_id.c_str())) {
            if (ImGui::Selectable("All Runs", filters.active_run_id.empty())) {
                filters.active_run_id.clear();
            }
            
            for (const auto& run : runs) {
                bool is_selected = filters.active_run_id == run.run_id;
                if (ImGui::Selectable(run.run_id.c_str(), is_selected)) {
                    filters.active_run_id = run.run_id;
                }
            }
            ImGui::EndCombo();
        }
    }
}

void CompanionPanels::RenderDataQualityPanel(AppState& state, const DataLoader& loader) {
    const auto& status = loader.GetLastStatus();
    
    // Data freshness indicator
    ImVec4 freshness_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f); // Green
    const char* freshness_text = "Fresh";
    
    double time_since_refresh = ImGui::GetTime() - state.last_refresh_time;
    if (time_since_refresh > state.refresh_interval_sec * 2) {
        freshness_color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f); // Red
        freshness_text = "Stale";
    } else if (time_since_refresh > state.refresh_interval_sec) {
        freshness_color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f); // Yellow
        freshness_text = "Aging";
    }
    
    ImGui::TextColored(freshness_color, ICON_MD_CIRCLE);
    ImGui::SameLine();
    ImGui::Text("%s", freshness_text);
    
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Last refresh: %.1fs ago", time_since_refresh);
        ImGui::EndTooltip();
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Data source status
    ImGui::Text("Data Sources:");
    ImGui::Indent();
    
    if (status.quality_ok) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_MD_CHECK);
        ImGui::SameLine();
        ImGui::Text("Quality Trends (%d samples)", static_cast<int>(loader.GetQualityTrends().size()));
    }
    
    if (status.active_ok) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_MD_CHECK);
        ImGui::SameLine();
        const auto& cov = loader.GetCoverage();
        ImGui::Text("Coverage (%d samples)", cov.total_samples);
    }
    
    if (status.training_ok) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_MD_CHECK);
        ImGui::SameLine();
        ImGui::Text("Training Runs (%d)", static_cast<int>(loader.GetTrainingRuns().size()));
    }
    
    // Generator stats check removed - not in LoadStatus
    
    ImGui::Unindent();
    
    if (status.error_count > 0) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), ICON_MD_WARNING " %d errors", status.error_count);
        if (!status.last_error.empty() && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", status.last_error.c_str());
            ImGui::EndTooltip();
        }
    }
}

void CompanionPanels::RenderInspectorPanel(AppState& state, const DataLoader& loader) {
    // Note: ImPlot::IsPlotHovered() can only be called within BeginPlot/EndPlot.
    // Since this panel renders separately, we show static info instead.
    ImGui::TextDisabled("Hover over graph for details");

    // Graph-specific inspection data
    ImGui::Separator();
    ImGui::Spacing();

    // Show currently active graph info
    const char* graph_name = "None";
    switch (state.active_graph) {
        case PlotKind::QualityTrends: graph_name = "Quality Trends"; break;
        case PlotKind::GeneratorEfficiency: graph_name = "Generator Efficiency"; break;
        case PlotKind::CoverageDensity: graph_name = "Coverage Density"; break;
        case PlotKind::TrainingLoss: graph_name = "Training Loss"; break;
        case PlotKind::DomainCoverage: graph_name = "Domain Coverage"; break;
        case PlotKind::EmbeddingQuality: graph_name = "Embedding Quality"; break;
        case PlotKind::Rejections: graph_name = "Rejections"; break;
        case PlotKind::EvalMetrics: graph_name = "Eval Metrics"; break;
        case PlotKind::AgentThroughput: graph_name = "Agent Workload"; break;
        case PlotKind::MountsStatus: graph_name = "Mounts Status"; break;
        case PlotKind::KnowledgeGraph: graph_name = "Context Graph"; break;
        case PlotKind::LatentSpace: graph_name = "Embedding Map"; break;
        default: break;
    }
    ImGui::Text("Graph: %s", graph_name);
    ImGui::TextDisabled("Click points for detailed info");
}

void CompanionPanels::RenderControlsPanel(AppState& state) {
    ImGui::Text("Visual Settings:");
    ImGui::Spacing();
    
    ImGui::SliderFloat("Line Weight", &state.line_weight, 1.0f, 5.0f, "%.1f");
    ImGui::Checkbox("Show Markers", &state.show_plot_markers);
    ImGui::Checkbox("Show Legend", &state.show_plot_legends);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    ImGui::Text("Graph Height:");
    ImGui::SliderFloat("##PlotHeight", &state.plot_height, 100.0f, 600.0f, "%.0f px");
}

} // namespace afs::viz::ui

#include "tabs.h"
#include "../../icons.h"
#include "../core.h"
#include <implot.h>
#include <algorithm>

namespace afs {
namespace viz {
namespace ui {

void RenderComparisonView(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header) {
    const auto& runs = loader.GetTrainingRuns();
    
    // Header
    if (font_header) ImGui::PushFont(font_header);
    ImGui::Text(ICON_MD_COMPARE " MODEL COMPARISON");
    if (font_header) ImGui::PopFont();
    ImGui::Separator();
    
    ImGui::Columns(2, "CompareColumns", true);
    
    auto render_selector = [&](int& selected_idx, const char* label) {
        std::string preview = (selected_idx >= 0 && selected_idx < runs.size()) 
                              ? runs[selected_idx].run_id 
                              : "Select Model...";
        if (ImGui::BeginCombo(label, preview.c_str())) {
            for (int i = 0; i < (int)runs.size(); ++i) {
                bool is_selected = (selected_idx == i);
                if (ImGui::Selectable(runs[i].run_id.c_str(), is_selected)) {
                    selected_idx = i;
                }
            }
            ImGui::EndCombo();
        }
    };

    render_selector(state.compare_run_a, "##ModelA");
    ImGui::NextColumn();
    render_selector(state.compare_run_b, "##ModelB");
    ImGui::NextColumn();
    
    ImGui::Separator();
    
    if (state.compare_run_a >= 0 && state.compare_run_b >= 0) {
        const auto& run_a = runs[state.compare_run_a];
        const auto& run_b = runs[state.compare_run_b];
        
        // Detailed diffing logic would go here
        ImGui::Columns(2, "MetricDiff", false);
        
        auto draw_metric = [&](const char* name, float val_a, float val_b) {
            ImGui::Text("%s", name);
            ImGui::NextColumn();
            ImGui::Text("%.4f vs %.4f", val_a, val_b);
            float diff = val_b - val_a;
            ImGui::SameLine();
            if (diff > 0.001f) ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), " (▲%.2f%%)", (diff/val_a)*100.0f);
            else if (diff < -0.001f) ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), " (▼%.2f%%)", (std::abs(diff)/val_a)*100.0f);
            ImGui::NextColumn();
        };

        draw_metric("Final Loss", run_a.final_loss, run_b.final_loss);
        // ... more metrics
        
        ImGui::Columns(1);
    } else {
        ImGui::TextDisabled("Select two models above to compare performance metrics.");
    }
}

} // namespace ui
} // namespace viz
} // namespace afs

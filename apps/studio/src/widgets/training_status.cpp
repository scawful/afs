/**
 * @file training_status.cpp
 * @brief Implementation of training campaign status monitoring widget
 */

#include "training_status.h"
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace afs {
namespace viz {

TrainingStatusWidget::TrainingStatusWidget()
    : last_update_(std::chrono::steady_clock::now()) {
    // Initial update
    Update();
}

void TrainingStatusWidget::Render(bool* p_open) {
    if (!p_open || !*p_open) {
        return;
    }

    // Auto-update check
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update_).count();

    if (auto_update_ && elapsed >= static_cast<int>(update_interval_)) {
        Update();
    }

    // Main window
    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Training Campaign Status", p_open)) {
        ImGui::End();
        return;
    }

    // Header with timestamp
    ImGui::Text("Last Update: %s", FormatTimestamp(health_.campaign ? health_.campaign->last_update : "").c_str());
    ImGui::SameLine(ImGui::GetWindowWidth() - 150);
    if (ImGui::Button("Refresh Now")) {
        Update();
    }

    ImGui::SameLine();
    ImGui::Checkbox("Auto", &auto_update_);

    ImGui::Separator();

    // Error message if any
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        ImGui::TextWrapped("Error: %s", error_message_.c_str());
        ImGui::PopStyleColor();
        ImGui::Separator();
    }

    // Sections in columns
    ImGui::BeginChild("StatusSections", ImVec2(0, 0), false);

    // Campaign section
    RenderCampaignSection();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // System resources section
    RenderSystemResourcesSection();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Services section
    RenderServicesSection();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Issues section
    RenderIssuesSection();

    ImGui::EndChild();

    ImGui::End();
}

void TrainingStatusWidget::Update() {
    if (FetchHealthData()) {
        last_update_ = std::chrono::steady_clock::now();
        error_message_.clear();
    }
}

bool TrainingStatusWidget::FetchHealthData() {
    // Execute Python health check script
    const char* env_root = std::getenv("AFS_ROOT");
    std::string root;
    if (env_root && env_root[0] != '\0') {
        root = env_root;
    } else {
        const char* trunk_root = std::getenv("TRUNK_ROOT");
        if (trunk_root && trunk_root[0] != '\0') {
            root = std::string(trunk_root) + "/lab/afs";
        } else {
            const char* home = std::getenv("HOME");
            if (home && home[0] != '\0') {
                root = std::string(home) + "/src/trunk/lab/afs";
            }
        }
    }

    if (root.empty()) {
        std::error_code ec;
        root = std::filesystem::current_path(ec).string();
    }

    if (root.empty()) {
        error_message_ = "Unable to resolve AFS root path";
        return false;
    }

    std::string cmd = "cd \"" + root + "\" && PYTHONPATH=src .venv/bin/python -m agents.training.health_check --json 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        error_message_ = "Failed to execute health check command";
        return false;
    }

    // Read output
    std::stringstream buffer;
    char line[256];
    while (fgets(line, sizeof(line), pipe)) {
        buffer << line;
    }

    int status = pclose(pipe);
    if (status != 0) {
        error_message_ = "Health check command failed with status " + std::to_string(status);
        return false;
    }

    // Parse JSON
    std::string json_str = buffer.str();
    return ParseHealthJSON(json_str);
}

bool TrainingStatusWidget::ParseHealthJSON(const std::string& json_str) {
    try {
        auto j = json::parse(json_str);

        // Parse campaign status
        if (j.contains("campaign") && !j["campaign"].is_null()) {
            CampaignStatus campaign;
            auto c = j["campaign"];

            if (c.contains("pid") && !c["pid"].is_null()) {
                campaign.pid = c["pid"];
            }

            if (c.contains("log_file") && !c["log_file"].is_null()) {
                campaign.log_file = c["log_file"];
            }

            campaign.running = c.value("running", false);
            campaign.samples_generated = c.value("samples_generated", 0);
            campaign.target_samples = c.value("target_samples", 0);
            campaign.progress_percent = c.value("progress_percent", 0.0f);
            campaign.current_domain = c.value("current_domain", "unknown");
            campaign.samples_per_min = c.value("samples_per_min", 0.0f);
            campaign.eta_hours = c.value("eta_hours", 0.0f);
            campaign.quality_pass_rate = c.value("quality_pass_rate", 0.0f);
            campaign.last_update = c.value("last_update", "");

            health_.campaign = campaign;
        } else {
            health_.campaign = std::nullopt;
        }

        // Parse system health
        health_.embedding_service_running = j.value("embedding_service_running", false);
        health_.knowledge_bases_loaded = j.value("knowledge_bases_loaded", 0);
        health_.cpu_percent = j.value("cpu_percent", 0.0f);
        health_.memory_percent = j.value("memory_percent", 0.0f);
        health_.disk_free_gb = j.value("disk_free_gb", 0.0f);

        // Parse issues
        health_.issues.clear();
        if (j.contains("issues")) {
            for (const auto& issue : j["issues"]) {
                health_.issues.push_back(issue);
            }
        }

        if (j.contains("last_checkpoint") && !j["last_checkpoint"].is_null()) {
            health_.last_checkpoint = j["last_checkpoint"];
        }

        return true;
    } catch (const json::exception& e) {
        error_message_ = std::string("JSON parse error: ") + e.what();
        return false;
    }
}

void TrainingStatusWidget::RenderCampaignSection() {
    ImGui::Text("Campaign Status");
    ImGui::Spacing();

    if (!health_.campaign.has_value()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "🔵 No active campaign");
        return;
    }

    const auto& c = health_.campaign.value();

    // Status indicator
    ImVec4 status_color = c.running ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    const char* status_icon = c.running ? "🟢" : "🔴";
    ImGui::TextColored(status_color, "%s %s", status_icon, c.running ? "Running" : "Stopped");

    if (c.pid > 0) {
        ImGui::Text("PID: %d", c.pid);
    }

    // Progress bar
    ImGui::Text("Progress:");
    ImGui::ProgressBar(c.progress_percent / 100.0f, ImVec2(-1, 0), "");
    ImGui::SameLine();
    ImGui::Text("%d / %d (%.1f%%)", c.samples_generated, c.target_samples, c.progress_percent);

    // Metrics
    ImGui::Text("Domain: %s", c.current_domain.c_str());
    ImGui::Text("Generation Rate: %.1f samples/min", c.samples_per_min);
    ImGui::Text("Quality Pass Rate: %.1f%%", c.quality_pass_rate * 100.0f);

    if (c.eta_hours > 0) {
        ImGui::Text("ETA: %s", FormatDuration(c.eta_hours).c_str());
    }

    if (!c.log_file.empty()) {
        ImGui::Text("Log: %s", c.log_file.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Open")) {
            std::string cmd = "open -a Terminal " + c.log_file;
            system(cmd.c_str());
        }
    }
}

void TrainingStatusWidget::RenderSystemResourcesSection() {
    ImGui::Text("System Resources");
    ImGui::Spacing();

    // CPU
    const char* cpu_icon = GetStatusIcon(health_.cpu_percent, 70.0f, 90.0f);
    ImVec4 cpu_color = GetStatusColor(health_.cpu_percent, 70.0f, 90.0f);
    ImGui::TextColored(cpu_color, "%s CPU: %.1f%%", cpu_icon, health_.cpu_percent);

    // Memory
    const char* mem_icon = GetStatusIcon(health_.memory_percent, 70.0f, 90.0f);
    ImVec4 mem_color = GetStatusColor(health_.memory_percent, 70.0f, 90.0f);
    ImGui::TextColored(mem_color, "%s Memory: %.1f%%", mem_icon, health_.memory_percent);

    // Disk (inverted thresholds - low disk is bad)
    const char* disk_icon = health_.disk_free_gb > 50 ? "🟢" :
                           health_.disk_free_gb > 10 ? "🟡" : "🔴";
    ImVec4 disk_color = health_.disk_free_gb > 50 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :
                       health_.disk_free_gb > 10 ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) :
                                                   ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImGui::TextColored(disk_color, "%s Disk Free: %.1f GB", disk_icon, health_.disk_free_gb);
}

void TrainingStatusWidget::RenderServicesSection() {
    ImGui::Text("Services");
    ImGui::Spacing();

    // Embedding service
    const char* emb_icon = health_.embedding_service_running ? "🟢" : "🔴";
    ImVec4 emb_color = health_.embedding_service_running ?
        ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImGui::TextColored(emb_color, "%s Embedding Service: %s", emb_icon,
                      health_.embedding_service_running ? "Running" : "Stopped");

    ImGui::Text("Knowledge Bases: %d loaded", health_.knowledge_bases_loaded);
}

void TrainingStatusWidget::RenderIssuesSection() {
    if (health_.issues.empty()) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✅ No issues detected");
        return;
    }

    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "⚠️ Issues (%zu)", health_.issues.size());
    ImGui::Spacing();

    for (const auto& issue : health_.issues) {
        ImGui::BulletText("%s", issue.c_str());
    }
}

ImVec4 TrainingStatusWidget::GetStatusColor(float value, float warn_threshold, float critical_threshold) {
    if (value < warn_threshold) {
        return ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green
    } else if (value < critical_threshold) {
        return ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
    } else {
        return ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Red
    }
}

const char* TrainingStatusWidget::GetStatusIcon(float value, float warn_threshold, float critical_threshold) {
    if (value < warn_threshold) {
        return "🟢";
    } else if (value < critical_threshold) {
        return "🟡";
    } else {
        return "🔴";
    }
}

std::string TrainingStatusWidget::FormatDuration(float hours) {
    int h = static_cast<int>(hours);
    int m = static_cast<int>((hours - h) * 60);

    std::ostringstream oss;
    if (h > 0) {
        oss << h << "h ";
    }
    oss << m << "m";
    return oss.str();
}

std::string TrainingStatusWidget::FormatTimestamp(const std::string& iso_timestamp) {
    if (iso_timestamp.empty()) {
        return "Never";
    }

    // Parse ISO timestamp (simplified - just show time part)
    size_t time_pos = iso_timestamp.find('T');
    if (time_pos != std::string::npos && time_pos + 8 < iso_timestamp.size()) {
        return iso_timestamp.substr(time_pos + 1, 8); // HH:MM:SS
    }

    return iso_timestamp;
}

} // namespace viz
} // namespace afs

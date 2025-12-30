/**
 * @file training_status.h
 * @brief Training campaign status monitoring widget for ImGui
 *
 * Displays real-time status of training data generation campaigns:
 * - Campaign progress (samples generated, target, percentage)
 * - Generation rate (samples/min, ETA)
 * - Quality metrics (pass rate, domain breakdown)
 * - System resources (CPU, memory, disk)
 * - Issues and alerts
 *
 * Data source: Python health_check.py via JSON output
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include "imgui.h"

namespace afs {
namespace viz {

struct CampaignStatus {
    int pid = 0;
    std::string log_file;
    bool running = false;
    int samples_generated = 0;
    int target_samples = 0;
    float progress_percent = 0.0f;
    std::string current_domain;
    float samples_per_min = 0.0f;
    float eta_hours = 0.0f;
    float quality_pass_rate = 0.0f;
    std::string last_update;
};

struct SystemHealth {
    std::optional<CampaignStatus> campaign;
    bool embedding_service_running = false;
    int knowledge_bases_loaded = 0;
    float cpu_percent = 0.0f;
    float memory_percent = 0.0f;
    float disk_free_gb = 0.0f;
    std::vector<std::string> issues;
    std::string last_checkpoint;
};

class TrainingStatusWidget {
public:
    TrainingStatusWidget();
    ~TrainingStatusWidget() = default;

    // Render the training status widget
    void Render(bool* p_open = nullptr);

    // Update health data (call periodically)
    void Update();

    // Set update interval (default: 30 seconds)
    void SetUpdateInterval(float seconds) { update_interval_ = seconds; }

private:
    // Fetch health data from Python health_check script
    bool FetchHealthData();

    // Parse JSON health data
    bool ParseHealthJSON(const std::string& json);

    // Render sections
    void RenderCampaignSection();
    void RenderSystemResourcesSection();
    void RenderServicesSection();
    void RenderIssuesSection();

    // Helper functions
    ImVec4 GetStatusColor(float value, float warn_threshold, float critical_threshold);
    const char* GetStatusIcon(float value, float warn_threshold, float critical_threshold);
    std::string FormatDuration(float hours);
    std::string FormatTimestamp(const std::string& iso_timestamp);

    // Data
    SystemHealth health_;
    std::chrono::steady_clock::time_point last_update_;
    float update_interval_ = 30.0f; // seconds
    bool auto_update_ = true;
    std::string error_message_;
};

} // namespace viz
} // namespace afs

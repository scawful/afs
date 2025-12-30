#include "sample_review.h"
#include <imgui.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "../core/filesystem.h"
#include "../core/logger.h"

namespace afs {
namespace viz {

namespace {

std::filesystem::path ResolveContextRoot() {
  const char* env_root = std::getenv("AFS_CONTEXT_ROOT");
  if (env_root && env_root[0] != '\0') {
    auto path = studio::core::FileSystem::ResolvePath(env_root);
    if (studio::core::FileSystem::Exists(path)) {
      return path;
    }
  }

  auto preferred = studio::core::FileSystem::ResolvePath("~/src/context");
  if (studio::core::FileSystem::Exists(preferred)) {
    return preferred;
  }

  auto fallback = studio::core::FileSystem::ResolvePath("~/.context");
  if (studio::core::FileSystem::Exists(fallback)) {
    return fallback;
  }

  return preferred;
}

TrainingSample ParseSampleLine(const std::string& line, bool is_rejected = false) {
  TrainingSample sample;
  sample.is_rejected = is_rejected;

  try {
    auto j = nlohmann::json::parse(line);

    sample.instruction = j.value("instruction", "");
    sample.input = j.value("input", "");
    sample.output = j.value("output", "");
    sample.domain = j.value("domain", "unknown");
    sample.source = j.value("source", "");

    if (is_rejected) {
      if (j.contains("rejection_reason")) {
        sample.rejection_reason = j["rejection_reason"];
      }
      if (j.contains("quality_score")) {
        sample.quality_score = j["quality_score"];
      }
      if (j.contains("rejection_details")) {
        sample.rejection_details = j["rejection_details"];
      }
    }
  } catch (...) {
    // Skip malformed lines
  }

  return sample;
}

}  // namespace

SampleReviewWidget::SampleReviewWidget() {}

bool SampleReviewWidget::LoadDataset(const std::filesystem::path& dataset_dir) {
  dataset_dir_ = dataset_dir;
  accepted_samples_.clear();
  rejected_samples_.clear();

  // Load rejected samples
  auto rejected_file = dataset_dir / "rejected.jsonl";
  if (studio::core::FileSystem::Exists(rejected_file)) {
    std::ifstream f(rejected_file);
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
          if (!line.empty()) {
            rejected_samples_.push_back(ParseSampleLine(line, true));
          }
        }
    }
  }

  // Load accepted samples
  auto train_file = dataset_dir / "train.jsonl";
  if (studio::core::FileSystem::Exists(train_file)) {
    std::ifstream f(train_file);
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
          if (!line.empty()) {
            accepted_samples_.push_back(ParseSampleLine(line, false));
          }
        }
    }
  }

  return !accepted_samples_.empty() || !rejected_samples_.empty();
}

void SampleReviewWidget::Render(bool* p_open) {
  ImGui::SetNextWindowSize(ImVec2(1400, 800), ImGuiCond_FirstUseEver);

  if (!ImGui::Begin("Sample Review & Annotation", p_open)) {
    ImGui::End();
    return;
  }

  // Header stats
  ImGui::Text("Dataset: %s", dataset_dir_.filename().c_str());
  ImGui::SameLine(0, 20);
  ImGui::Text("Accepted: %zu", accepted_samples_.size());
  ImGui::SameLine(0, 20);
  ImGui::Text("Rejected: %zu", rejected_samples_.size());

  ImGui::Separator();

  // Main layout: 3 columns
  ImGui::Columns(3, "review_layout", true);

  // Left: Sample Browser
  RenderSampleBrowser();

  ImGui::NextColumn();

  // Middle: Sample Viewer
  RenderSampleViewer();

  ImGui::NextColumn();

  // Right: Context Reference + Feedback
  RenderContextReference();
  ImGui::Spacing();
  RenderFeedbackPanel();

  ImGui::Columns(1);

  ImGui::End();
}

void SampleReviewWidget::RenderSampleBrowser() {
  ImGui::Text("Sample Browser");
  ImGui::Separator();

  // Filters
  ImGui::Checkbox("Show Rejected", &show_rejected_);
  ImGui::SameLine();
  ImGui::Checkbox("Show Accepted", &show_accepted_);

  ImGui::InputText("Domain", domain_filter_, sizeof(domain_filter_));
  ImGui::InputText("Reason", reason_filter_, sizeof(reason_filter_));
  ImGui::SliderFloat("Min Score", &min_quality_score_, 0.0f, 1.0f);

  ImGui::Separator();

  // Sample list
  ImGui::BeginChild("SampleList", ImVec2(0, -30), true);

  auto render_samples = [&](const std::vector<TrainingSample>& samples, const char* label) {
    if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
      for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];

        // Apply filters
        if (domain_filter_[0] != '\0' && sample.domain.find(domain_filter_) == std::string::npos) continue;
        if (sample.quality_score && *sample.quality_score < min_quality_score_) continue;

        char label_buf[256];
        snprintf(label_buf, sizeof(label_buf), "[%s] %s##%zu",
                 sample.domain.c_str(),
                 sample.instruction.substr(0, 40).c_str(),
                 i);

        if (ImGui::Selectable(label_buf, (int)i == current_sample_idx_)) {
          current_sample_idx_ = i;
        }

        // Show quality score if available
        if (sample.quality_score) {
          ImGui::SameLine();
          ImGui::TextDisabled("%.3f", *sample.quality_score);
        }
      }
      ImGui::TreePop();
    }
  };

  if (show_rejected_) {
    render_samples(rejected_samples_, "Rejected Samples");
  }

  if (show_accepted_) {
    render_samples(accepted_samples_, "Accepted Samples");
  }

  ImGui::EndChild();

  // Navigation
  if (ImGui::Button("< Prev")) {
    current_sample_idx_ = std::max(0, current_sample_idx_ - 1);
  }
  ImGui::SameLine();
  if (ImGui::Button("Next >")) {
    current_sample_idx_++;
  }
}

void SampleReviewWidget::RenderSampleViewer() {
  ImGui::Text("Sample Details");
  ImGui::Separator();

  if (rejected_samples_.empty() && accepted_samples_.empty()) {
    ImGui::TextDisabled("No samples loaded");
    return;
  }

  // Get current sample
  const TrainingSample* sample = nullptr;
  if (current_sample_idx_ < (int)rejected_samples_.size()) {
    sample = &rejected_samples_[current_sample_idx_];
  } else if (current_sample_idx_ < (int)(rejected_samples_.size() + accepted_samples_.size())) {
    sample = &accepted_samples_[current_sample_idx_ - rejected_samples_.size()];
  }

  if (!sample) {
    ImGui::TextDisabled("No sample selected");
    return;
  }

  // Metadata
  ImGui::Text("Domain: %s", sample->domain.c_str());
  ImGui::SameLine(0, 20);
  ImGui::Text("Source: %s", sample->source.c_str());

  if (sample->is_rejected) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "REJECTED");
    if (sample->rejection_reason) {
      ImGui::SameLine();
      ImGui::Text("- %s", sample->rejection_reason->c_str());
    }
    if (sample->quality_score) {
      ImGui::SameLine();
      ImGui::Text("(score: %.3f)", *sample->quality_score);
    }
  } else {
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "ACCEPTED");
  }

  ImGui::Separator();

  // Content
  ImGui::BeginChild("SampleContent", ImVec2(0, 0), true);

  ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Instruction:");
  ImGui::TextWrapped("%s", sample->instruction.c_str());

  ImGui::Spacing();

  if (!sample->input.empty()) {
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.6f, 1.0f), "Input:");
    ImGui::TextWrapped("%s", sample->input.c_str());
    ImGui::Spacing();
  }

  ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.6f, 1.0f), "Output:");
  ImGui::TextWrapped("%s", sample->output.c_str());

  // Rejection details
  if (sample->rejection_details) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.6f, 1.0f), "Rejection Details:");

    auto& details = *sample->rejection_details;
    if (details.contains("threshold")) {
      ImGui::Text("Threshold: %.3f", details["threshold"].get<float>());
    }
    if (details.contains("diversity")) {
      ImGui::Text("Diversity: %.3f", details["diversity"].get<float>());
    }
    if (details.contains("kg_consistency")) {
      ImGui::Text("KG Consistency: %.3f", details["kg_consistency"].get<float>());
    }
    if (details.contains("hallucination_risk")) {
      ImGui::Text("Hallucination Risk: %.3f", details["hallucination_risk"].get<float>());
    }
    if (details.contains("coherence")) {
      ImGui::Text("Coherence: %.3f", details["coherence"].get<float>());
    }
  }

  ImGui::EndChild();
}

void SampleReviewWidget::RenderContextReference() {
  ImGui::Text("Reference Context");
  ImGui::Separator();

  ImGui::InputText("Search ASM", context_search_, sizeof(context_search_));

  if (ImGui::Button("Load My ASM Files")) {
    LoadContextFiles();
  }

  ImGui::BeginChild("ContextFiles", ImVec2(0, 200), true);

  for (const auto& file : context_files_) {
    if (ImGui::Selectable(file.filename().c_str())) {
      std::ifstream f(file);
      std::stringstream ss;
      ss << f.rdbuf();
      selected_context_content_ = ss.str();
    }
  }

  ImGui::EndChild();

  if (!selected_context_content_.empty()) {
    ImGui::Text("Context Preview:");
    ImGui::BeginChild("ContextPreview", ImVec2(0, 150), true);
    ImGui::TextWrapped("%s", selected_context_content_.substr(0, 500).c_str());
    ImGui::EndChild();
  }
}

void SampleReviewWidget::RenderFeedbackPanel() {
  ImGui::Text("User Feedback");
  ImGui::Separator();

  ImGui::InputTextMultiline("##feedback", feedback_buffer_, sizeof(feedback_buffer_),
                            ImVec2(-1, 100));

  if (ImGui::Button("✓ Approve (Use as Golden Example)", ImVec2(-1, 0))) {
    ApproveCurrentSample();
  }

  if (ImGui::Button("✗ Reject (Bad Sample)", ImVec2(-1, 0))) {
    RejectCurrentSample(feedback_buffer_);
  }

  ImGui::Spacing();

  if (ImGui::Button("Save All Annotations", ImVec2(-1, 0))) {
    SaveAnnotations();
  }
}

void SampleReviewWidget::LoadContextFiles() {
  context_files_.clear();

  // Load user's ASM files from alttp disassembly
  auto alttp_path = ResolveContextRoot() / "knowledge" / "alttp";

  if (studio::core::FileSystem::Exists(alttp_path)) {
      try {
        std::error_code ec;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(alttp_path, ec)) {
          std::error_code entry_ec;
          if (!ec && entry.is_regular_file(entry_ec) && entry.path().extension() == ".asm") {
            context_files_.push_back(entry.path());
          }
        }
      } catch (...) {
        LOG_WARN("Error during recursive ASM search in " + alttp_path.string());
      }
  }
}

void SampleReviewWidget::ApproveCurrentSample() {
  // Mark current sample as user-approved golden example
  // This will be saved to a separate file
}

void SampleReviewWidget::RejectCurrentSample(const std::string& reason) {
  // Mark current sample as user-rejected with feedback
}

void SampleReviewWidget::SaveAnnotations() {
  auto annotations_file = dataset_dir_ / "user_annotations.json";

  nlohmann::json annotations = nlohmann::json::array();

  // Save approved/rejected samples with user feedback
  for (const auto& sample : rejected_samples_) {
    if (sample.user_approved || !sample.user_feedback.empty()) {
      annotations.push_back({
        {"instruction", sample.instruction},
        {"domain", sample.domain},
        {"user_approved", sample.user_approved},
        {"user_feedback", sample.user_feedback}
      });
    }
  }

  std::ofstream f(annotations_file);
  f << annotations.dump(2);
}

}  // namespace viz
}  // namespace afs

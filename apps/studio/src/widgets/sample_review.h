#pragma once

#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace afs {
namespace viz {

struct TrainingSample {
  std::string instruction;
  std::string input;
  std::string output;
  std::string domain;
  std::string source;

  // For rejected samples
  std::optional<std::string> rejection_reason;
  std::optional<float> quality_score;
  std::optional<nlohmann::json> rejection_details;

  bool is_rejected = false;
  bool user_approved = false;
  std::string user_feedback;
};

class SampleReviewWidget {
 public:
  SampleReviewWidget();

  /// Load samples from dataset directory
  bool LoadDataset(const std::filesystem::path& dataset_dir);

  /// Render the review window
  void Render(bool* p_open = nullptr);

  /// Save user annotations/approvals
  void SaveAnnotations();

 private:
  void RenderSampleBrowser();
  void RenderSampleViewer();
  void RenderContextReference();
  void RenderFeedbackPanel();

  void LoadContextFiles();
  void ApproveCurrentSample();
  void RejectCurrentSample(const std::string& reason);

  // Data
  std::vector<TrainingSample> accepted_samples_;
  std::vector<TrainingSample> rejected_samples_;
  std::filesystem::path dataset_dir_;

  // UI State
  int current_sample_idx_ = 0;
  bool show_rejected_ = true;
  bool show_accepted_ = true;
  char feedback_buffer_[1024] = {0};
  char context_search_[256] = {0};

  // Filters
  char domain_filter_[128] = {0};
  char reason_filter_[128] = {0};
  float min_quality_score_ = 0.0f;

  // Context files (user's ASM for reference)
  std::vector<std::filesystem::path> context_files_;
  std::string selected_context_content_;
};

}  // namespace viz
}  // namespace afs

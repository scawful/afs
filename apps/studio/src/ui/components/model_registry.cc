#include "model_registry.h"
#include "../../icons.h"

#include <imgui.h>

#include <algorithm>
#include <cstring>

namespace afs {
namespace studio {
namespace ui {

// Define static members
constexpr const char* ModelRegistryWidget::kRoleOptions[];
constexpr const char* ModelRegistryWidget::kLocationOptions[];
constexpr const char* ModelRegistryWidget::kBackendOptions[];

ModelRegistryWidget::ModelRegistryWidget() {
  Refresh();
}

void ModelRegistryWidget::Refresh() {
  std::string error;
  if (!registry_.Load(&error)) {
    last_error_ = error;
  } else {
    last_error_.clear();
  }
  selected_model_index_ = -1;
}

const ModelMetadata* ModelRegistryWidget::GetSelectedModel() const {
  const auto& models = registry_.GetModels();
  if (selected_model_index_ >= 0 &&
      selected_model_index_ < static_cast<int>(models.size())) {
    return &models[selected_model_index_];
  }
  return nullptr;
}

void ModelRegistryWidget::Render() {
  RenderToolbar();

  // Main content with optional details panel
  if (show_details_ && GetSelectedModel()) {
    // Split view: list on left, details on right
    float details_width = 350.0f;
    float list_width = ImGui::GetContentRegionAvail().x - details_width - 10.0f;

    ImGui::BeginChild("ModelList", ImVec2(list_width, 0), true);
    RenderModelList();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("ModelDetails", ImVec2(details_width, 0), true);
    RenderModelDetails();
    ImGui::EndChild();
  } else {
    // Full width list
    ImGui::BeginChild("ModelList", ImVec2(0, 0), true);
    RenderModelList();
    ImGui::EndChild();
  }
}

void ModelRegistryWidget::RenderToolbar() {
  // Refresh button
  if (ImGui::Button("Refresh")) {
    Refresh();
  }
  ImGui::SameLine();

  // Filter text input
  ImGui::SetNextItemWidth(200.0f);
  ImGui::InputTextWithHint("##Filter", "Filter models...", filter_text_.data(),
                           filter_text_.size());
  ImGui::SameLine();

  // Role filter
  ImGui::SetNextItemWidth(100.0f);
  ImGui::Combo("Role", &filter_role_, kRoleOptions, kRoleCount);
  ImGui::SameLine();

  // Location filter
  ImGui::SetNextItemWidth(100.0f);
  ImGui::Combo("Location", &filter_location_, kLocationOptions, kLocationCount);
  ImGui::SameLine();

  // Backend filter
  ImGui::SetNextItemWidth(100.0f);
  ImGui::Combo("Backend", &filter_backend_, kBackendOptions, kBackendCount);
  ImGui::SameLine();

  // Toggle details panel
  ImGui::Checkbox("Details", &show_details_);
  ImGui::SameLine();
  ImGui::Checkbox("Deploy", &show_deployment_);

  // Status line
  const auto& models = registry_.GetModels();
  ImGui::TextDisabled("%zu models registered", models.size());
  if (!registry_.GetLastLoadTime().empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("| Updated: %s",
                        registry_.GetLastLoadTime().c_str());
  }

  // Error display
  if (!last_error_.empty()) {
    ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "Error: %s",
                       last_error_.c_str());
  }

  ImGui::Separator();
}

void ModelRegistryWidget::RenderModelList() {
  const auto& models = registry_.GetModels();
  std::string filter_str(filter_text_.data());

  // Convert filter to lowercase for case-insensitive match
  std::transform(filter_str.begin(), filter_str.end(), filter_str.begin(),
                 ::tolower);

  int display_index = 0;
  for (size_t i = 0; i < models.size(); ++i) {
    const auto& model = models[i];

    // Apply filters
    if (filter_role_ > 0) {
      if (model.role != kRoleOptions[filter_role_]) continue;
    }

    if (filter_location_ > 0) {
      if (model.locations.count(kLocationOptions[filter_location_]) == 0)
        continue;
    }

    if (filter_backend_ > 0) {
      bool found = false;
      for (const auto& backend : model.deployed_backends) {
        if (backend == kBackendOptions[filter_backend_]) {
          found = true;
          break;
        }
      }
      if (!found) continue;
    }

    // Apply text filter
    if (!filter_str.empty()) {
      std::string searchable = model.model_id + " " + model.display_name + " " +
                               model.role + " " + model.base_model;
      std::transform(searchable.begin(), searchable.end(), searchable.begin(),
                     ::tolower);
      if (searchable.find(filter_str) == std::string::npos) continue;
    }

    RenderModelCard(model, static_cast<int>(i));
    ++display_index;
  }

  if (display_index == 0) {
    ImGui::TextDisabled("No models match the current filters.");
  }
}

void ModelRegistryWidget::RenderModelCard(const ModelMetadata& model,
                                           int index) {
  ImGui::PushID(index);

  bool is_selected = (selected_model_index_ == index);

  // Card styling
  ImVec4 header_color =
      is_selected ? ImVec4(0.2f, 0.4f, 0.8f, 1.0f) : ImVec4(0.15f, 0.15f, 0.15f, 1.0f);

  ImGui::PushStyleColor(ImGuiCol_Header, header_color);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                        ImVec4(0.25f, 0.45f, 0.75f, 1.0f));

  // Collapsing header acts as the card
  bool expanded = ImGui::CollapsingHeader(
      model.display_name.empty() ? model.model_id.c_str()
                                 : model.display_name.c_str(),
      ImGuiTreeNodeFlags_DefaultOpen);

  if (ImGui::IsItemClicked()) {
    selected_model_index_ = index;
  }

  ImGui::PopStyleColor(2);

  if (expanded) {
    ImGui::Indent();

    // Role badge
    ImVec4 role_color(0.3f, 0.6f, 0.3f, 1.0f);
    if (model.role == "asm")
      role_color = ImVec4(0.8f, 0.5f, 0.2f, 1.0f);
    else if (model.role == "debug")
      role_color = ImVec4(0.6f, 0.3f, 0.6f, 1.0f);
    else if (model.role == "yaze")
      role_color = ImVec4(0.2f, 0.6f, 0.8f, 1.0f);

    ImGui::TextColored(role_color, "[%s]", model.role.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("%s", model.base_model.c_str());

    // Metrics & Badges
    ImGui::BeginGroup();
    if (model.final_loss.has_value()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), ICON_MD_TRENDING_DOWN " Loss: %.4f", model.final_loss.value());
        ImGui::SameLine();
    }
    if (model.train_samples > 0) {
        ImGui::TextDisabled(ICON_MD_FOLDER " %d samples", model.train_samples);
    }
    ImGui::EndGroup();

    // Locations & Backends
    if (!model.deployed_backends.empty()) {
        ImGui::TextDisabled(ICON_MD_CLOUD_DONE " Deployed:");
        for (const auto& backend : model.deployed_backends) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "[%s]", backend.c_str());
        }
    } else {
        ImGui::TextDisabled(ICON_MD_CLOUD_OFF " Not Deployed");
    }

    // Action Buttons - Premium styling
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.35f, 0.15f, 1.0f));
    if (ImGui::Button(ICON_MD_ROCKET_LAUNCH " Deploy")) {
        show_deployment_ = true;
    }
    ImGui::PopStyleColor();
    
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.25f, 0.45f, 1.0f));
    if (ImGui::Button(ICON_MD_PLAY_ARROW " Test")) {
        show_details_ = true;
        // Scroll or activate test section in details if needed
    }
    ImGui::PopStyleColor();

    ImGui::SameLine();
    if (ImGui::Button(ICON_MD_COMPARE " Compare")) {
        // TODO: Add to comparison view
    }

    ImGui::Unindent();
    ImGui::Spacing();
    ImGui::Separator();
  }

  ImGui::PopID();
}

void ModelRegistryWidget::RenderModelDetails() {
  const ModelMetadata* model = GetSelectedModel();
  if (!model) {
    ImGui::TextDisabled("Select a model to view details.");
    return;
  }

  ImGui::Text("%s", model->display_name.c_str());
  ImGui::Separator();

  // Identity section
  ImGui::TextDisabled("ID:");
  ImGui::SameLine();
  ImGui::TextWrapped("%s", model->model_id.c_str());

  ImGui::TextDisabled("Version:");
  ImGui::SameLine();
  ImGui::Text("%s", model->version.c_str());

  ImGui::TextDisabled("Base Model:");
  ImGui::SameLine();
  ImGui::TextWrapped("%s", model->base_model.c_str());

  ImGui::Spacing();
  ImGui::Separator();

  // Training section
  ImGui::Text("Training");
  ImGui::TextDisabled("Role:");
  ImGui::SameLine();
  ImGui::Text("%s", model->role.c_str());

  ImGui::TextDisabled("Date:");
  ImGui::SameLine();
  ImGui::Text("%s", model->training_date.c_str());

  ImGui::TextDisabled("Duration:");
  ImGui::SameLine();
  ImGui::Text("%d minutes", model->training_duration_minutes);

  ImGui::TextDisabled("Hardware:");
  ImGui::SameLine();
  ImGui::Text("%s (%s)", model->hardware.c_str(), model->device.c_str());

  ImGui::Spacing();
  ImGui::Separator();

  // Dataset section
  ImGui::Text("Dataset");
  ImGui::TextDisabled("Name:");
  ImGui::SameLine();
  ImGui::Text("%s", model->dataset_name.c_str());

  ImGui::TextDisabled("Samples:");
  ImGui::SameLine();
  ImGui::Text("%d train / %d val / %d test", model->train_samples,
              model->val_samples, model->test_samples);

  ImGui::TextDisabled("Acceptance:");
  ImGui::SameLine();
  ImGui::Text("%.1f%%", model->dataset_quality.acceptance_rate * 100.0f);

  ImGui::Spacing();
  ImGui::Separator();

  // Metrics section
  ImGui::Text("Metrics");
  if (model->final_loss.has_value()) {
    ImGui::TextDisabled("Final Loss:");
    ImGui::SameLine();
    ImGui::Text("%.4f", model->final_loss.value());
  }
  if (model->best_loss.has_value()) {
    ImGui::TextDisabled("Best Loss:");
    ImGui::SameLine();
    ImGui::Text("%.4f", model->best_loss.value());
  }
  if (model->perplexity.has_value()) {
    ImGui::TextDisabled("Perplexity:");
    ImGui::SameLine();
    ImGui::Text("%.2f", model->perplexity.value());
  }

  ImGui::Spacing();
  ImGui::Separator();

  // Locations section
  ImGui::Text("Locations");
  for (const auto& [location, path] : model->locations) {
    bool is_primary = (location == model->primary_location);
    ImGui::BulletText("%s%s", location.c_str(), is_primary ? " (primary)" : "");
    ImGui::TextDisabled("    %s", path.c_str());
  }

  ImGui::Spacing();
  ImGui::Separator();

  // Deployment section
  ImGui::Text("Deployment");
  if (model->deployed_backends.empty()) {
    ImGui::TextDisabled("Not deployed");
  } else {
    for (const auto& backend : model->deployed_backends) {
      ImGui::BulletText("%s", backend.c_str());
      if (backend == "ollama" && model->ollama_model_name.has_value()) {
        ImGui::TextDisabled("    Name: %s",
                            model->ollama_model_name.value().c_str());
      }
    }
  }

  // Notes
  if (!model->notes.empty()) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Notes");
    ImGui::TextWrapped("%s", model->notes.c_str());
  }

  // Tags
  if (!model->tags.empty()) {
    ImGui::Spacing();
    ImGui::TextDisabled("Tags:");
    for (const auto& tag : model->tags) {
      ImGui::SameLine();
      ImGui::TextColored(ImVec4(0.5f, 0.7f, 0.9f, 1.0f), "#%s", tag.c_str());
    }
  }

  // Deployment panel section
  if (show_deployment_) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Deployment Actions");
    ImGui::Spacing();
    deployment_panel_.Render(model);
  }
}

void RenderModelRegistryWindow(ModelRegistryWidget& widget, bool* open) {
  if (!open || !*open) return;

  ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);

  if (!ImGui::Begin("Model Registry", open)) {
    ImGui::End();
    return;
  }

  widget.Render();

  ImGui::End();
}

}  // namespace ui
}  // namespace studio
}  // namespace afs

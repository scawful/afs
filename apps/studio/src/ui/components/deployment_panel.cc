#include "deployment_panel.h"

#include <imgui.h>

#include <cstring>

namespace afs {
namespace studio {
namespace ui {

DeploymentPanel::DeploymentPanel() {
  // Default test prompt
  std::strncpy(test_prompt_buffer_.data(),
               "Write a simple NOP instruction in 65816 assembly:",
               test_prompt_buffer_.size() - 1);
}

void DeploymentPanel::Render(const ModelMetadata* selected_model) {
  if (!selected_model) {
    ImGui::TextDisabled("Select a model to view deployment options.");
    return;
  }

  const auto& model = *selected_model;

  // Header
  ImGui::Text("%s", model.display_name.empty() ? model.model_id.c_str()
                                                 : model.display_name.c_str());
  ImGui::Separator();

  // Status display
  if (is_busy_ || !status_message_.empty()) {
    RenderDeploymentStatus();
  }

  // Quick actions
  RenderQuickActions(model);

  ImGui::Spacing();
  ImGui::Separator();

  // Ollama deployment section
  if (ImGui::CollapsingHeader("Deploy to Ollama",
                               ImGuiTreeNodeFlags_DefaultOpen)) {
    RenderOllamaDeployment(model);
  }

  // Conversion options
  if (ImGui::CollapsingHeader("Format Conversion")) {
    RenderConversionOptions(model);
  }

  // Test prompt
  if (ImGui::CollapsingHeader("Test Model")) {
    RenderTestPrompt(model);
  }
}

void DeploymentPanel::RenderDeploymentStatus() {
  if (is_busy_) {
    ImGui::TextColored(ImVec4(0.2f, 0.7f, 0.9f, 1.0f), "%s",
                       current_operation_.c_str());
    ImGui::ProgressBar(progress_, ImVec2(-1, 0));
  }

  if (!status_message_.empty()) {
    bool is_error = !last_error_.empty();
    ImVec4 color = is_error ? ImVec4(0.9f, 0.3f, 0.3f, 1.0f)
                            : ImVec4(0.3f, 0.9f, 0.3f, 1.0f);
    ImGui::TextColored(color, "%s", status_message_.c_str());
  }

  if (!last_error_.empty()) {
    ImGui::TextWrapped("Error: %s", last_error_.c_str());
  }

  ImGui::Spacing();
}

void DeploymentPanel::RenderQuickActions(const ModelMetadata& model) {
  // Check current state
  bool is_local = model.locations.count("mac") > 0;
  bool has_gguf = false;
  for (const auto& fmt : model.formats) {
    if (fmt == "gguf") {
      has_gguf = true;
      break;
    }
  }
  bool in_ollama = false;
  for (const auto& backend : model.deployed_backends) {
    if (backend == "ollama") {
      in_ollama = true;
      break;
    }
  }

  // Quick action buttons
  ImGui::Text("Quick Actions:");

  // Pull button
  ImGui::BeginDisabled(is_busy_ || is_local);
  if (ImGui::Button("Pull to Mac")) {
    is_busy_ = true;
    current_operation_ = "Pulling model...";
    progress_ = 0.1f;
    status_message_.clear();
    last_error_.clear();

    // Execute pull
    last_result_ = actions_.PullModel(model.model_id);
    is_busy_ = false;

    if (last_result_.status == ActionStatus::kCompleted) {
      status_message_ = "Model pulled successfully";
    } else {
      status_message_ = "Pull failed";
      last_error_ = last_result_.error;
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (!is_local) {
    ImGui::TextDisabled("(not local)");
  } else {
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "(local)");
  }

  ImGui::SameLine(0, 20);

  // Convert button
  ImGui::BeginDisabled(is_busy_ || !is_local || has_gguf);
  if (ImGui::Button("Convert to GGUF")) {
    is_busy_ = true;
    current_operation_ = "Converting to GGUF...";
    progress_ = 0.2f;
    status_message_.clear();
    last_error_.clear();

    auto quant_opts = GetQuantizationOptions();
    std::string quant = quant_opts[selected_quantization_].name;

    last_result_ = actions_.ConvertToGGUF(model.model_id, quant);
    is_busy_ = false;

    if (last_result_.status == ActionStatus::kCompleted) {
      status_message_ = "Conversion complete";
    } else {
      status_message_ = "Conversion failed";
      last_error_ = last_result_.error;
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (has_gguf) {
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "(has GGUF)");
  }

  ImGui::SameLine(0, 20);

  // Deploy button
  ImGui::BeginDisabled(is_busy_ || in_ollama);
  if (ImGui::Button("Deploy to Ollama")) {
    is_busy_ = true;
    current_operation_ = "Deploying to Ollama...";
    progress_ = 0.3f;
    status_message_.clear();
    last_error_.clear();

    std::string name(ollama_name_buffer_.data());
    auto quant_opts = GetQuantizationOptions();
    std::string quant = quant_opts[selected_quantization_].name;

    last_result_ = actions_.DeployToOllama(model.model_id, name, quant);
    is_busy_ = false;

    if (last_result_.status == ActionStatus::kCompleted) {
      status_message_ = "Deployed to Ollama";
    } else {
      status_message_ = "Deployment failed";
      last_error_ = last_result_.error;
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (in_ollama) {
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "(in Ollama)");
  }
}

void DeploymentPanel::RenderOllamaDeployment(const ModelMetadata& model) {
  ImGui::Indent();

  // Ollama name input
  ImGui::Text("Ollama Model Name:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(200);
  ImGui::InputTextWithHint("##OllamaName", model.model_id.c_str(),
                           ollama_name_buffer_.data(),
                           ollama_name_buffer_.size());

  // Quantization selection
  ImGui::Text("Quantization:");
  auto quant_opts = GetQuantizationOptions();
  for (int i = 0; i < static_cast<int>(quant_opts.size()); ++i) {
    ImGui::SameLine();
    if (ImGui::RadioButton(quant_opts[i].name.c_str(),
                           selected_quantization_ == i)) {
      selected_quantization_ = i;
    }
  }

  // Current quantization info
  const auto& selected_quant = quant_opts[selected_quantization_];
  ImGui::TextDisabled("%s (%.0f%% of F16 size)",
                      selected_quant.description.c_str(),
                      selected_quant.size_ratio * 100.0f);

  // Status indicators
  ImGui::Spacing();
  bool ollama_running = actions_.IsOllamaRunning();
  bool llama_available = actions_.IsLlamaCppAvailable();

  ImGui::TextDisabled("Prerequisites:");
  ImGui::BulletText("Ollama: %s",
                    ollama_running ? "Running" : "Not running");
  if (!ollama_running) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.2f, 1.0f), "(run: ollama serve)");
  }

  ImGui::BulletText("llama.cpp: %s",
                    llama_available ? "Available" : "Not found");

  ImGui::Unindent();
}

void DeploymentPanel::RenderConversionOptions(const ModelMetadata& model) {
  ImGui::Indent();

  // Current formats
  ImGui::Text("Available formats:");
  for (const auto& fmt : model.formats) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 0.7f, 0.9f, 1.0f), "[%s]", fmt.c_str());
  }

  if (model.formats.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("(none)");
  }

  ImGui::Spacing();

  // Conversion buttons
  ImGui::BeginDisabled(is_busy_);
  if (ImGui::Button("Convert to GGUF (Q4_K_M)")) {
    is_busy_ = true;
    current_operation_ = "Converting to GGUF Q4_K_M...";
    last_result_ = actions_.ConvertToGGUF(model.model_id, "Q4_K_M");
    is_busy_ = false;
    status_message_ = last_result_.status == ActionStatus::kCompleted
                          ? "Conversion complete"
                          : "Conversion failed";
    if (last_result_.status != ActionStatus::kCompleted) {
      last_error_ = last_result_.error;
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Convert to GGUF (Q5_K_M)")) {
    is_busy_ = true;
    current_operation_ = "Converting to GGUF Q5_K_M...";
    last_result_ = actions_.ConvertToGGUF(model.model_id, "Q5_K_M");
    is_busy_ = false;
    status_message_ = last_result_.status == ActionStatus::kCompleted
                          ? "Conversion complete"
                          : "Conversion failed";
    if (last_result_.status != ActionStatus::kCompleted) {
      last_error_ = last_result_.error;
    }
  }
  ImGui::EndDisabled();

  ImGui::Unindent();
}

void DeploymentPanel::RenderTestPrompt(const ModelMetadata& model) {
  ImGui::Indent();

  // Check if model is deployed
  bool can_test = false;
  std::string deployed_to;
  for (const auto& backend : model.deployed_backends) {
    if (backend == "ollama") {
      can_test = true;
      deployed_to = "ollama";
      break;
    }
  }

  if (!can_test) {
    ImGui::TextDisabled("Deploy model to a backend first to test.");
    ImGui::Unindent();
    return;
  }

  // Test prompt input
  ImGui::Text("Test Prompt:");
  ImGui::SetNextItemWidth(-1);
  ImGui::InputTextMultiline("##TestPrompt", test_prompt_buffer_.data(),
                            test_prompt_buffer_.size(), ImVec2(0, 60));

  // Run test button
  ImGui::BeginDisabled(is_busy_);
  if (ImGui::Button("Run Test")) {
    is_busy_ = true;
    current_operation_ = "Testing model...";
    status_message_.clear();
    last_error_.clear();
    test_output_.clear();

    last_result_ = actions_.TestModel(model.model_id, DeploymentBackend::kOllama,
                                       std::string(test_prompt_buffer_.data()));
    is_busy_ = false;

    if (last_result_.status == ActionStatus::kCompleted) {
      status_message_ = "Test completed";
      test_output_ = last_result_.output;
    } else {
      status_message_ = "Test failed";
      last_error_ = last_result_.error;
    }
  }
  ImGui::EndDisabled();

  // Test output
  if (!test_output_.empty()) {
    ImGui::Spacing();
    ImGui::Text("Response:");
    ImGui::BeginChild("TestOutput", ImVec2(0, 100), true);
    ImGui::TextWrapped("%s", test_output_.c_str());
    ImGui::EndChild();
  }

  ImGui::Unindent();
}

void RenderDeploymentPanelWindow(DeploymentPanel& panel,
                                  const ModelMetadata* model,
                                  bool* open) {
  if (!open || !*open) return;

  ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);

  if (!ImGui::Begin("Deployment", open)) {
    ImGui::End();
    return;
  }

  panel.Render(model);

  ImGui::End();
}

}  // namespace ui
}  // namespace studio
}  // namespace afs

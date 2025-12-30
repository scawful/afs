#include "panels.h"
#include "../core.h"
#include "../../icons.h"
#include "core/filesystem.h"
#include <implot.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <GLFW/glfw3.h>

namespace afs {
namespace viz {

std::filesystem::path ResolveHafsScawfulRoot() {
  const char* env_root = std::getenv("AFS_SCAWFUL_ROOT");
  if (env_root && env_root[0] != '\0') {
    auto path = studio::core::FileSystem::ResolvePath(env_root);
    if (studio::core::FileSystem::Exists(path)) {
      return path;
    }
  }

  auto plugin_path = studio::core::FileSystem::ResolvePath("~/.config/afs/plugins/afs_scawful");
  if (studio::core::FileSystem::Exists(plugin_path)) {
    return plugin_path;
  }

  const char* trunk_env = std::getenv("TRUNK_ROOT");
  if (trunk_env && trunk_env[0] != '\0') {
    auto trunk_root = studio::core::FileSystem::ResolvePath(trunk_env);
    auto candidate = trunk_root / "lab" / "afs_scawful";
    if (studio::core::FileSystem::Exists(candidate)) {
      return candidate;
    }
    auto legacy = trunk_root / "scawful" / "research" / "afs_scawful";
    if (studio::core::FileSystem::Exists(legacy)) {
      return legacy;
    }
  }

  auto fallback_path = studio::core::FileSystem::ResolvePath("~/src/trunk/lab/afs_scawful");
  if (studio::core::FileSystem::Exists(fallback_path)) {
    return fallback_path;
  }
  auto legacy_fallback = studio::core::FileSystem::ResolvePath("~/src/trunk/scawful/research/afs_scawful");
  if (studio::core::FileSystem::Exists(legacy_fallback)) {
    return legacy_fallback;
  }

  return {};
}

namespace ui {

static bool ContainsInsensitive(const std::string& str, const std::string& pattern) {
  if (pattern.empty()) return true;
  auto it = std::search(str.begin(), str.end(), pattern.begin(), pattern.end(),
                        [](char ch1, char ch2) {
                          return std::tolower(static_cast<unsigned char>(ch1)) ==
                                 std::tolower(static_cast<unsigned char>(ch2));
                        });
  return it != str.end();
}

static std::string TrimCopy(const std::string& input) {
  auto start = input.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  auto end = input.find_last_not_of(" \t\r\n");
  return input.substr(start, end - start + 1);
}

static std::string UnquoteCopy(const std::string& input) {
  if (input.size() >= 2 && input.front() == '"' && input.back() == '"') {
    return input.substr(1, input.size() - 2);
  }
  return input;
}

struct HackOverrideEdit {
  bool initialized = false;
  bool enabled = false;
  bool dirty = false;
  std::string review_status;
  float weight = 1.0f;
  bool weight_set = false;
  bool notes_set = false;
  bool include_set = false;
  bool exclude_set = false;
  std::string notes;
  std::vector<std::string> include_globs;
  std::vector<std::string> exclude_globs;
};

struct NoteBufferState {
  std::array<char, 512> buffer{};
  bool initialized = false;
};

static bool LoadOverrideFile(const std::filesystem::path& path,
                             std::unordered_map<std::string, HackOverrideEdit>* edits,
                             std::string* error) {
  edits->clear();
  if (error) error->clear();

  if (!studio::core::FileSystem::Exists(path)) {
    if (error) *error = "Overrides file not found";
    return false;
  }

  auto content_opt = studio::core::FileSystem::ReadFile(path);
  if (!content_opt) {
    if (error) *error = "Failed to read overrides file";
    return false;
  }

  std::istringstream stream(*content_opt);
  std::string line;
  HackOverrideEdit current;
  std::string current_name;

  auto flush = [&]() {
    if (current_name.empty()) return;
    current.initialized = true;
    current.enabled = true;
    (*edits)[current_name] = current;
  };

  while (std::getline(stream, line)) {
    std::string trimmed = TrimCopy(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;

    if (trimmed == "[[hack]]") {
      flush();
      current = HackOverrideEdit{};
      current_name.clear();
      continue;
    }

    auto sep = trimmed.find('=');
    if (sep == std::string::npos) continue;
    std::string key = TrimCopy(trimmed.substr(0, sep));
    std::string value = TrimCopy(trimmed.substr(sep + 1));

    if (key == "name") {
      current_name = UnquoteCopy(value);
    } else if (key == "review_status") {
      current.review_status = UnquoteCopy(value);
    } else if (key == "notes") {
      current.notes = UnquoteCopy(value);
      current.notes_set = true;
    } else if (key == "weight") {
      try {
        current.weight = std::stof(value);
        current.weight_set = true;
      } catch (...) {
        // Ignore parse errors
      }
    } else if (key == "include_globs" || key == "exclude_globs") {
      std::vector<std::string> items;
      if (!value.empty() && value.front() == '[' && value.back() == ']') {
        std::string inner = value.substr(1, value.size() - 2);
        std::stringstream list_stream(inner);
        std::string token;
        while (std::getline(list_stream, token, ',')) {
          std::string cleaned = UnquoteCopy(TrimCopy(token));
          if (!cleaned.empty()) items.push_back(cleaned);
        }
      }
      if (key == "include_globs") {
        current.include_globs = std::move(items);
        current.include_set = true;
      } else {
        current.exclude_globs = std::move(items);
        current.exclude_set = true;
      }
    }
  }
  flush();

  return true;
}

template<size_t N>
static void FillBuffer(std::array<char, N>& buffer, const std::string& text) {
  buffer.fill('\0');
  size_t copy_len = std::min(buffer.size() - 1, text.size());
  std::memcpy(buffer.data(), text.data(), copy_len);
  buffer[copy_len] = '\0';
}

static std::string JoinLines(const std::vector<std::string>& lines) {
  std::ostringstream out;
  for (size_t i = 0; i < lines.size(); ++i) {
    out << lines[i];
    if (i + 1 < lines.size()) out << "\n";
  }
  return out.str();
}

static std::vector<std::string> SplitLines(const std::string& input) {
  std::vector<std::string> lines;
  std::istringstream stream(input);
  std::string line;
  while (std::getline(stream, line)) {
    std::string trimmed = TrimCopy(line);
    if (!trimmed.empty()) lines.push_back(trimmed);
  }
  return lines;
}

static std::string EscapeTomlString(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    if (ch == '"') out += "\\\"";
    else out += ch;
  }
  return out;
}

static void WriteStringArray(std::ostringstream& output, const std::string& key,
                             const std::vector<std::string>& values) {
  output << key << " = [";
  for (size_t i = 0; i < values.size(); ++i) {
    output << "\"" << EscapeTomlString(values[i]) << "\"";
    if (i + 1 < values.size()) output << ", ";
  }
  output << "]\n";
}

static bool RunPythonScript(const std::filesystem::path& script_path,
                            const std::filesystem::path& module_root,
                            std::string* output) {
  if (output) output->clear();
  if (!studio::core::FileSystem::Exists(script_path)) {
    if (output) *output = "Script not found";
    return false;
  }

  std::string cmd;
  if (!module_root.empty()) {
    cmd = "AFS_SCAWFUL_ROOT=\"" + module_root.string() + "\" ";
    cmd += "PYTHONPATH=\"" + module_root.string() + "\" ";
  }
  cmd += "python3 \"" + script_path.string() + "\" 2>&1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    if (output) *output = "Failed to launch script";
    return false;
  }

  char buffer[256];
  std::ostringstream result;
  while (fgets(buffer, sizeof(buffer), pipe)) {
    result << buffer;
  }
  int status = pclose(pipe);
  if (output) *output = result.str();
  return status == 0;
}

void RenderInspectorPanel(AppState& state, const DataLoader& loader, ImFont* font_header, const std::string& data_path) {
  const auto& trends = loader.GetQualityTrends();
  const auto& coverage = loader.GetCoverage();
  const auto& runs = loader.GetTrainingRuns();
  const auto& generators = loader.GetGeneratorStats();

  if (font_header) ImGui::PushFont(font_header);
  ImGui::Text(ICON_MD_INSIGHTS " INSPECTOR");
  if (font_header) ImGui::PopFont();
  ImGui::Separator();

  if (state.inspector_context != PlotKind::None) {
      ImGui::PushStyleColor(ImGuiCol_Text, GetThemeColor(ImGuiCol_PlotLines, state.current_theme));
      ImGui::Text(ICON_MD_SETTINGS " CHART PROPERTIES");
      ImGui::PopStyleColor();
      
      const auto& options = PlotOptions();
      const char* label = "Unknown Plot";
      for (const auto& opt : options) {
          if (opt.kind == state.inspector_context) { label = opt.label; break; }
      }
      ImGui::Text("Subject: %s", label);
      ImGui::Separator();
      
      ImGui::Checkbox("Show Markers", &state.show_plot_markers);
      ImGui::Checkbox("Show Legend", &state.show_plot_legends);
      ImGui::SliderFloat("Line Weight", &state.line_weight, 1.0f, 5.0f);
      
      if (state.inspector_context == PlotKind::QualityTrends) {
          ImGui::Spacing();
          ImGui::TextDisabled("Domain Filter");
          for (auto& [domain, visible] : state.domain_visibility) {
              ImGui::Checkbox(domain.c_str(), &visible);
          }
      }
      
      if (ImGui::Button("Reset Context")) state.inspector_context = PlotKind::None;
      ImGui::Separator();
  }

  ImGui::TextDisabled("Data Snapshot");
  ImGui::Text("Runs: %zu", runs.size());
  ImGui::Text("Generators: %zu", generators.size());
  ImGui::Text("Regions: %zu", loader.GetEmbeddingRegions().size());
  ImGui::Text("Sparse Regions: %d", coverage.sparse_regions);
  ImGui::Text("Data Path:");
  ImGui::TextWrapped("%s", data_path.c_str());
  const auto& status = loader.GetLastStatus();
  ImGui::Text("Sources: %d/%d ok", status.OkCount(), status.FoundCount());
  if (status.error_count > 0) {
    ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f),
                       "Load Errors: %d", status.error_count);
    if (!status.last_error.empty()) {
      ImGui::TextDisabled("Last Error:");
      ImGui::TextWrapped("%s (%s)",
                         status.last_error.c_str(),
                         status.last_error_source.empty()
                             ? "unknown"
                             : status.last_error_source.c_str());
    }
  }

  float avg_quality = 0.0f;
  if (!trends.empty()) {
    for (const auto& t : trends) avg_quality += t.mean;
    avg_quality /= static_cast<float>(trends.size());
  }

  ImGui::Spacing();
  ImGui::TextDisabled("Health Signals");
  ImGui::ProgressBar(avg_quality, ImVec2(-1, 0), "Avg Quality");
  ImGui::ProgressBar(coverage.coverage_score, ImVec2(-1, 0), "Coverage Score");

  ImGui::Separator();
  ImGui::TextDisabled("Selected Run");

  if (state.selected_run_index >= 0 && state.selected_run_index < static_cast<int>(runs.size())) {
    const auto& run = runs[state.selected_run_index];
    ImGui::Text("%s", run.run_id.c_str());
    if (!run.model_name.empty()) ImGui::Text("Model: %s", run.model_name.c_str());
    if (!run.base_model.empty()) ImGui::Text("Base: %s", run.base_model.c_str());
    ImGui::Text("Samples: %d", run.samples_count);
    ImGui::Text("Final Loss: %.5f", run.final_loss);
    if (!run.start_time.empty() || !run.end_time.empty()) {
      ImGui::Text("Window: %s -> %s", run.start_time.empty() ? "?" : run.start_time.c_str(), run.end_time.empty() ? "?" : run.end_time.c_str());
    }
    if (!run.dataset_path.empty()) {
      ImGui::Text("Dataset:");
      ImGui::TextWrapped("%s", run.dataset_path.c_str());
    }
    if (!run.notes.empty()) {
      ImGui::Text("Notes:");
      ImGui::TextWrapped("%s", run.notes.c_str());
    }

    if (!run.domain_distribution.empty()) {
      std::vector<std::pair<std::string, int>> domains(run.domain_distribution.begin(), run.domain_distribution.end());
      std::sort(domains.begin(), domains.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      if (domains.size() > 6) domains.resize(6);

      std::vector<const char*> labels;
      std::vector<float> values;
      std::vector<std::string> label_storage;
      for (const auto& [domain, count] : domains) {
        label_storage.push_back(domain);
        values.push_back(static_cast<float>(count));
      }
      for (const auto& label : label_storage) labels.push_back(label.c_str());

      ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus;
      ApplyPremiumPlotStyles("##RunDomains", state);
      if (ImPlot::BeginPlot("##RunDomains", ImVec2(-1, 140), plot_flags)) {
        ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
        ImPlot::SetupAxes("Domain", "Samples", axis_flags, axis_flags);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1), static_cast<int>(labels.size()), labels.data());
        ImPlot::SetNextFillStyle(GetSeriesColor(2), 0.75f);
        ImPlot::PlotBars("Samples", values.data(), static_cast<int>(values.size()), 0.6);
        ImPlot::EndPlot();
      }
      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar(6);
    }

    if (!run.eval_metrics.empty()) {
      std::vector<const char*> labels;
      std::vector<float> values;
      std::vector<std::string> label_storage;
      for (const auto& [metric, value] : run.eval_metrics) {
        label_storage.push_back(metric);
        values.push_back(value);
      }
      for (const auto& label : label_storage) labels.push_back(label.c_str());

      ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus;
      ApplyPremiumPlotStyles("##RunMetrics", state);
      if (ImPlot::BeginPlot("##RunMetrics", ImVec2(-1, 120), plot_flags)) {
        ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
        ImPlot::SetupAxes("Metric", "Score", axis_flags, axis_flags);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1), static_cast<int>(labels.size()), labels.data());
        ImPlot::SetNextFillStyle(GetSeriesColor(4), 0.75f);
        ImPlot::PlotBars("Score", values.data(), static_cast<int>(values.size()), 0.6);
        ImPlot::EndPlot();
      }
      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar(6);
    }
  } else {
    ImGui::TextDisabled("Select a training run in the Dataset panel.");
  }

  ImGui::Separator();
  ImGui::TextDisabled("Selected Generator");
  if (state.selected_generator_index >= 0 && state.selected_generator_index < static_cast<int>(generators.size())) {
    const auto& gen = generators[state.selected_generator_index];
    ImGui::Text("%s", gen.name.c_str());
    ImGui::Text("Accepted: %d", gen.samples_accepted);
    ImGui::Text("Rejected: %d", gen.samples_rejected);
    ImGui::Text("Avg Quality: %.3f", gen.avg_quality);
    ImGui::ProgressBar(gen.acceptance_rate, ImVec2(-1, 0), "Acceptance Rate");

    if (!gen.rejection_reasons.empty()) {
      std::vector<std::pair<std::string, int>> reasons(gen.rejection_reasons.begin(), gen.rejection_reasons.end());
      std::sort(reasons.begin(), reasons.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      if (reasons.size() > 6) reasons.resize(6);

      std::vector<const char*> labels;
      std::vector<float> values;
      std::vector<std::string> label_storage;
      for (const auto& [reason, count] : reasons) {
        std::string formatted = reason;
        std::replace(formatted.begin(), formatted.end(), '_', ' ');
        label_storage.push_back(formatted);
        values.push_back(static_cast<float>(count));
      }
      for (const auto& label : label_storage) labels.push_back(label.c_str());

      ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus;
      ApplyPremiumPlotStyles("##GenRejections", state);
      if (ImPlot::BeginPlot("##GenRejections", ImVec2(-1, 120), plot_flags)) {
        ImPlotAxisFlags axis_flags = static_cast<ImPlotAxisFlags>(GetPlotAxisFlags(state));
        ImPlot::SetupAxes("Reason", "Count", axis_flags, axis_flags);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1), static_cast<int>(labels.size()), labels.data());
        ImPlot::SetNextFillStyle(GetSeriesColor(7), 0.75f);
        ImPlot::PlotBars("Count", values.data(), static_cast<int>(values.size()), 0.6);
        ImPlot::EndPlot();
      }
      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar(6);
    }
  } else {
    ImGui::TextDisabled("Select a generator in the Dataset panel.");
  }

  ImGui::Separator();
  ImGui::Spacing();
  if (font_header) ImGui::PushFont(font_header);
  ImGui::Text(ICON_MD_FILE_DOWNLOAD_DONE " DATA INTEGRITY AUDIT");
  if (font_header) ImGui::PopFont();
  ImGui::Separator();

  auto render_audit_item = [](const char* name, bool found, bool ok, const char* error) {
      ImGui::BeginGroup();
      if (!found) {
          ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_MD_FILE_OPEN " %s", name);
          ImGui::SameLine(); ImGui::TextDisabled("(Not Found)");
      } else if (!ok) {
          ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), ICON_MD_REPORT_GMAILERRORRED " %s", name);
          ImGui::SameLine(); ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "(Error)");
          if (error && error[0] != '\0') {
              ImGui::Indent();
              ImGui::TextDisabled("%s", error);
              ImGui::Unindent();
          }
      } else {
          ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), ICON_MD_CHECK_CIRCLE " %s", name);
          ImGui::SameLine(); ImGui::TextDisabled("(Synced)");
      }
      ImGui::EndGroup();
  };

  render_audit_item("quality_feedback.json", status.quality_found, status.quality_ok, status.quality_ok ? "" : status.last_error.c_str());
  render_audit_item("active_learning.json", status.active_found, status.active_ok, status.active_ok ? "" : status.last_error.c_str());
  render_audit_item("training_feedback.json", status.training_found, status.training_ok, status.training_ok ? "" : status.last_error.c_str());

  ImGui::Spacing();
  ImGui::TextDisabled("Integrity Score: %.1f%%", (status.OkCount() / (float)std::max(1, status.FoundCount())) * 100.0f);
  ImGui::ProgressBar(status.OkCount() / (float)std::max(1, status.FoundCount()), ImVec2(-1, 0));
}

void RenderDatasetPanel(AppState& state, const DataLoader& loader) {
  const auto& runs = loader.GetTrainingRuns();
  const auto& generators = loader.GetGeneratorStats();
  const auto& coverage = loader.GetCoverage();
  const auto& dataset_registry = loader.GetDatasetRegistry();
  const auto& dataset_error = loader.GetDatasetRegistryError();

  if (ImGui::BeginTabBar("DatasetTabs")) {
    if (ImGui::BeginTabItem("Training Runs")) {
      ImGui::InputTextWithHint("##RunFilter", "Filter by run ID or model", state.run_filter.data(), state.run_filter.size());
      ImGui::SameLine();
      if (ImGui::Button("Clear")) state.run_filter[0] = '\0';

      if (ImGui::BeginTable("RunTable", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Run ID", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Samples", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("Loss", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Domains", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < runs.size(); ++i) {
          const auto& run = runs[i];
          if (!ContainsInsensitive(run.run_id, state.run_filter.data()) && !ContainsInsensitive(run.model_name, state.run_filter.data())) continue;

          bool selected = static_cast<int>(i) == state.selected_run_index;
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          if (ImGui::Selectable(run.run_id.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
            state.selected_run_index = static_cast<int>(i);
            state.selected_run_id = run.run_id;
          }
          ImGui::TableNextColumn(); ImGui::Text("%s", run.model_name.empty() ? "-" : run.model_name.c_str());
          ImGui::TableNextColumn(); ImGui::Text("%d", run.samples_count);
          ImGui::TableNextColumn(); ImGui::Text("%.4f", run.final_loss);
          ImGui::TableNextColumn(); ImGui::Text("%zu", run.domain_distribution.size());
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Data Sources")) {
      const auto& resource_index = loader.GetResourceIndex();
      const auto& resource_error = loader.GetResourceIndexError();
      static std::string resource_status;

      if (!resource_error.empty()) {
        ImGui::TextColored(ImVec4(0.9f, 0.5f, 0.2f, 1.0f), "%s", resource_error.c_str());
      }

      if (resource_index.total_files == 0) {
        ImGui::TextDisabled("No resource index loaded.");
      } else {
        ImGui::Text("Total files: %d", resource_index.total_files);
        ImGui::Text("Duplicates: %d", resource_index.duplicates_found);
        if (!resource_index.indexed_at.empty()) {
          ImGui::TextDisabled("Indexed at: %s", resource_index.indexed_at.c_str());
        }
      }

      ImGui::Spacing();
      if (ImGui::Button("Rebuild Resource Index")) {
        auto scawful_root = ResolveHafsScawfulRoot();
        std::filesystem::path script_path = scawful_root.empty()
            ? std::filesystem::current_path() / "rebuild_resource_index.py"
            : scawful_root / "scripts" / "rebuild_resource_index.py";
        std::string build_output;
        bool ok = RunPythonScript(script_path, scawful_root, &build_output);
        if (!build_output.empty()) {
          resource_status = ok ? "Resource index rebuilt (see logs)" : "Resource index rebuild failed (see logs)";
        } else {
          resource_status = ok ? "Resource index rebuilt" : "Resource index rebuild failed";
        }
        state.should_refresh = true;
      }
      if (!resource_status.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", resource_status.c_str());
      }

      if (!resource_index.by_source.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("Sources");

        std::vector<std::pair<std::string, int>> sources;
        sources.reserve(resource_index.by_source.size());
        for (const auto& [name, count] : resource_index.by_source) {
          sources.emplace_back(name, count);
        }
        std::sort(sources.begin(), sources.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (ImGui::BeginTable("ResourceSources", 3,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp)) {
          ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 1.5f);
          ImGui::TableSetupColumn("Files", ImGuiTableColumnFlags_WidthFixed, 80);
          ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthFixed, 70);
          ImGui::TableHeadersRow();

          for (const auto& entry : sources) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", entry.first.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%d", entry.second);
            ImGui::TableNextColumn();
            float share = resource_index.total_files > 0
                ? static_cast<float>(entry.second) / static_cast<float>(resource_index.total_files)
                : 0.0f;
            ImGui::Text("%.1f%%", share * 100.0f);
          }
          ImGui::EndTable();
        }
      }

      if (!resource_index.by_type.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("File Types");

        std::vector<std::pair<std::string, int>> types;
        types.reserve(resource_index.by_type.size());
        for (const auto& [name, count] : resource_index.by_type) {
          types.emplace_back(name, count);
        }
        std::sort(types.begin(), types.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (ImGui::BeginTable("ResourceTypes", 3,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp)) {
          ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 1.5f);
          ImGui::TableSetupColumn("Files", ImGuiTableColumnFlags_WidthFixed, 80);
          ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthFixed, 70);
          ImGui::TableHeadersRow();

          for (const auto& entry : types) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", entry.first.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%d", entry.second);
            ImGui::TableNextColumn();
            float share = resource_index.total_files > 0
                ? static_cast<float>(entry.second) / static_cast<float>(resource_index.total_files)
                : 0.0f;
            ImGui::Text("%.1f%%", share * 100.0f);
          }
          ImGui::EndTable();
        }
      }

      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Datasets")) {
      static std::string dataset_status;
      if (!dataset_error.empty()) {
        ImGui::TextColored(ImVec4(0.9f, 0.5f, 0.2f, 1.0f), "%s", dataset_error.c_str());
      }

      ImGui::InputTextWithHint("##DatasetFilter", "Filter by dataset name", state.dataset_filter.data(), state.dataset_filter.size());
      ImGui::SameLine();
      if (ImGui::Button("Clear##DatasetFilter")) state.dataset_filter[0] = '\0';
      ImGui::SameLine();
      if (ImGui::Button("Rebuild Dataset Registry")) {
        auto scawful_root = ResolveHafsScawfulRoot();
        std::filesystem::path script_path = scawful_root.empty()
            ? std::filesystem::current_path() / "build_dataset_registry.py"
            : scawful_root / "scripts" / "build_dataset_registry.py";
        std::string build_output;
        bool ok = RunPythonScript(script_path, scawful_root, &build_output);
        if (!build_output.empty()) {
          dataset_status = ok ? "Dataset registry rebuilt (see logs)" : "Dataset registry rebuild failed (see logs)";
        } else {
          dataset_status = ok ? "Dataset registry rebuilt" : "Dataset registry rebuild failed";
        }
        state.should_refresh = true;
      }
      if (!dataset_status.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", dataset_status.c_str());
      }

      if (dataset_registry.datasets.empty()) {
        ImGui::TextDisabled("No dataset registry loaded.");
      } else {
        std::uint64_t total_size = 0;
        for (const auto& dataset : dataset_registry.datasets) {
          total_size += dataset.size_bytes;
        }
        double total_mb = static_cast<double>(total_size) / (1024.0 * 1024.0);
        ImGui::Text("Datasets: %zu", dataset_registry.datasets.size());
        ImGui::SameLine();
        ImGui::TextDisabled("Total size: %.2f MB", total_mb);
        if (!dataset_registry.generated_at.empty()) {
          ImGui::TextDisabled("Generated at: %s", dataset_registry.generated_at.c_str());
        }

        if (ImGui::BeginTable("DatasetRegistryTable", 4,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp)) {
          ImGui::TableSetupColumn("Dataset", ImGuiTableColumnFlags_WidthStretch, 1.4f);
          ImGui::TableSetupColumn("Files", ImGuiTableColumnFlags_WidthFixed, 80);
          ImGui::TableSetupColumn("Size (MB)", ImGuiTableColumnFlags_WidthFixed, 100);
          ImGui::TableSetupColumn("Updated", ImGuiTableColumnFlags_WidthStretch, 1.1f);
          ImGui::TableHeadersRow();

          for (size_t i = 0; i < dataset_registry.datasets.size(); ++i) {
            const auto& dataset = dataset_registry.datasets[i];
            if (!ContainsInsensitive(dataset.name, state.dataset_filter.data())) {
              continue;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool selected = static_cast<int>(i) == state.selected_dataset_index;
            if (ImGui::Selectable(dataset.name.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
              state.selected_dataset_index = static_cast<int>(i);
            }
            if (ImGui::IsItemHovered() && !dataset.path.empty()) {
              ImGui::BeginTooltip();
              ImGui::Text("%s", dataset.path.c_str());
              ImGui::EndTooltip();
            }
            ImGui::TableNextColumn();
            ImGui::Text("%zu", dataset.files.size());
            ImGui::TableNextColumn();
            double size_mb = static_cast<double>(dataset.size_bytes) / (1024.0 * 1024.0);
            ImGui::Text("%.2f", size_mb);
            ImGui::TableNextColumn();
            ImGui::Text("%s", dataset.updated_at.empty() ? "-" : dataset.updated_at.c_str());
          }
          ImGui::EndTable();
        }

        ImGui::Separator();
        if (state.selected_dataset_index >= 0 &&
            state.selected_dataset_index < static_cast<int>(dataset_registry.datasets.size())) {
          const auto& selected = dataset_registry.datasets[state.selected_dataset_index];
          ImGui::TextDisabled("Selected Dataset");
          ImGui::Text("%s", selected.name.c_str());
          ImGui::TextDisabled("%s", selected.path.empty() ? "-" : selected.path.c_str());
          ImGui::Text("Files: %zu", selected.files.size());

          ImGui::BeginChild("DatasetFiles", ImVec2(0, 120), true);
          size_t max_files = 12;
          for (size_t i = 0; i < selected.files.size() && i < max_files; ++i) {
            ImGui::BulletText("%s", selected.files[i].c_str());
          }
          if (selected.files.size() > max_files) {
            ImGui::TextDisabled("... and %zu more", selected.files.size() - max_files);
          }
          ImGui::EndChild();
        } else {
          ImGui::TextDisabled("Select a dataset to view details.");
        }
      }

      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Generators")) {
      ImGui::InputTextWithHint("##GenFilter", "Filter by generator name", state.generator_filter.data(), state.generator_filter.size());
      ImGui::SameLine();
      if (ImGui::Button("Clear##Gen")) state.generator_filter[0] = '\0';

      if (ImGui::BeginTable("GeneratorTable", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Generator", ImGuiTableColumnFlags_WidthStretch, 1.5f);
        ImGui::TableSetupColumn("Accepted", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Rejected", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Rate %", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("Quality", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < generators.size(); ++i) {
          const auto& gen = generators[i];
          if (!ContainsInsensitive(gen.name, state.generator_filter.data())) continue;
          bool selected = static_cast<int>(i) == state.selected_generator_index;
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          if (ImGui::Selectable(gen.name.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
            state.selected_generator_index = static_cast<int>(i);
            state.selected_generator_name = gen.name;
          }
          ImGui::TableNextColumn(); ImGui::Text("%d", gen.samples_accepted);
          ImGui::TableNextColumn(); ImGui::Text("%d", gen.samples_rejected);
          ImGui::TableNextColumn(); ImGui::Text("%.1f", gen.acceptance_rate * 100.0f);
          ImGui::TableNextColumn(); ImGui::Text("%.3f", gen.avg_quality);
        }
        ImGui::EndTable();
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Domains")) {
      if (coverage.domain_coverage.empty()) {
        ImGui::TextDisabled("No domain coverage data available.");
      } else {
        if (ImGui::BeginTable("DomainCoverageTable", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Borders)) {
          ImGui::TableSetupColumn("Domain", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Coverage %", ImGuiTableColumnFlags_WidthFixed, 110);
          ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableHeadersRow();

          for (const auto& [domain, value] : coverage.domain_coverage) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%s", domain.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%.1f", value * 100.0f);
            ImGui::TableNextColumn(); ImGui::ProgressBar(value, ImVec2(-1, 0));
          }
          ImGui::EndTable();
        }
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Curated Hacks")) {
      const auto& hacks = loader.GetCuratedHacks();
      const auto& curated_error = loader.GetCuratedHacksError();
      static std::unordered_map<std::string, HackOverrideEdit> override_edits;
      static std::unordered_map<std::string, NoteBufferState> note_buffers;
      static bool overrides_loaded = false;
      static std::string overrides_error;
      static std::string overrides_status;
      static std::string selected_hack_name;
      static std::array<char, 2048> notes_buffer{};
      static std::array<char, 1024> include_buffer{};
      static std::array<char, 1024> exclude_buffer{};

      auto scawful_root = ResolveHafsScawfulRoot();
      std::filesystem::path override_path = scawful_root.empty()
          ? studio::core::FileSystem::ResolvePath("~/.config/afs/curated_hacks_overrides.toml")
          : scawful_root / "config" / "curated_hacks_overrides.toml";

      if (!overrides_loaded) {
        LoadOverrideFile(override_path, &override_edits, &overrides_error);
        overrides_loaded = true;
      }

      if (!curated_error.empty()) {
        ImGui::TextColored(ImVec4(0.9f, 0.5f, 0.2f, 1.0f), "%s", curated_error.c_str());
      }

      ImGui::Spacing();
      if (ImGui::Button("Reload Overrides")) {
        LoadOverrideFile(override_path, &override_edits, &overrides_error);
        overrides_status = overrides_error.empty() ? "Overrides reloaded" : overrides_error;
        note_buffers.clear();
      }
      ImGui::SameLine();
      if (ImGui::Button("Save Overrides")) {
        std::ostringstream output;
        output << "# Curated hack overrides (generated by AFS Studio)\n";
        int saved = 0;

        for (const auto& hack : hacks) {
          auto it = override_edits.find(hack.name);
          if (it == override_edits.end() || !it->second.enabled) continue;

          const auto& edit = it->second;
          output << "\n[[hack]]\n";
          output << "name = \"" << hack.name << "\"\n";
          if (!edit.review_status.empty()) {
            output << "review_status = \"" << edit.review_status << "\"\n";
          }
          output << "weight = " << std::fixed << std::setprecision(2) << edit.weight << "\n";
          if (edit.notes_set) {
            output << "notes = \"" << EscapeTomlString(edit.notes) << "\"\n";
          }
          if (edit.include_set) {
            WriteStringArray(output, "include_globs", edit.include_globs);
          }
          if (edit.exclude_set) {
            WriteStringArray(output, "exclude_globs", edit.exclude_globs);
          }
          saved++;
        }

        if (studio::core::FileSystem::EnsureDirectory(override_path.parent_path()) &&
            studio::core::FileSystem::WriteFile(override_path, output.str())) {
          overrides_status = "Saved " + std::to_string(saved) + " overrides";
          for (auto& [name, edit] : override_edits) {
            edit.dirty = false;
          }
        } else {
          overrides_status = "Failed to save overrides";
        }
      }
      ImGui::SameLine();
      if (ImGui::Button("Save + Rebuild Summary")) {
        std::ostringstream output;
        output << "# Curated hack overrides (generated by AFS Studio)\n";
        int saved = 0;

        for (const auto& hack : hacks) {
          auto it = override_edits.find(hack.name);
          if (it == override_edits.end() || !it->second.enabled) continue;

          const auto& edit = it->second;
          output << "\n[[hack]]\n";
          output << "name = \"" << hack.name << "\"\n";
          if (!edit.review_status.empty()) {
            output << "review_status = \"" << edit.review_status << "\"\n";
          }
          output << "weight = " << std::fixed << std::setprecision(2) << edit.weight << "\n";
          if (edit.notes_set) {
            output << "notes = \"" << EscapeTomlString(edit.notes) << "\"\n";
          }
          if (edit.include_set) {
            WriteStringArray(output, "include_globs", edit.include_globs);
          }
          if (edit.exclude_set) {
            WriteStringArray(output, "exclude_globs", edit.exclude_globs);
          }
          saved++;
        }

        if (studio::core::FileSystem::EnsureDirectory(override_path.parent_path()) &&
            studio::core::FileSystem::WriteFile(override_path, output.str())) {
          std::filesystem::path script_path = override_path.parent_path().parent_path();
          script_path /= "scripts/build_curated_hacks_summary.py";
          std::string build_output;
          bool ok = RunPythonScript(script_path, scawful_root, &build_output);
          overrides_status = ok ? "Saved overrides and rebuilt summary" : "Saved overrides, rebuild failed";
          if (!build_output.empty()) {
            overrides_status += " (see logs)";
          }
          state.should_refresh = true;
          for (auto& [name, edit] : override_edits) {
            edit.dirty = false;
          }
        } else {
          overrides_status = "Failed to save overrides";
        }
      }
      if (!overrides_status.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", overrides_status.c_str());
      }

      ImGui::TextDisabled("Tip: Save + Rebuild Summary refreshes file counts automatically.");

      if (!overrides_error.empty()) {
        ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "%s", overrides_error.c_str());
      }

      if (hacks.empty()) {
        ImGui::TextDisabled("No curated hack summary available.");
        if (!scawful_root.empty()) {
          ImGui::TextDisabled("Run: %s", (scawful_root / "scripts" / "build_curated_hacks_summary.py").string().c_str());
        } else {
          ImGui::TextDisabled("Curated hacks plugin not configured.");
        }
      } else if (ImGui::BeginTable("CuratedHacksTable", 10,
                                   ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                                   ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY |
                                   ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 85);
        ImGui::TableSetupColumn("Review", ImGuiTableColumnFlags_WidthFixed, 85);
        ImGui::TableSetupColumn("Weight", ImGuiTableColumnFlags_WidthFixed, 65);
        ImGui::TableSetupColumn("Eligible", ImGuiTableColumnFlags_WidthFixed, 65);
        ImGui::TableSetupColumn("Select", ImGuiTableColumnFlags_WidthFixed, 65);
        ImGui::TableSetupColumn("Org%", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Addr%", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Comm%", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableHeadersRow();

        auto load_editor_buffers = [&](const auto& hack, HackOverrideEdit& edit) {
          const std::string notes_text = edit.notes_set ? edit.notes : hack.notes;
          const auto& include_list = edit.include_set ? edit.include_globs : hack.include_globs;
          const auto& exclude_list = edit.exclude_set ? edit.exclude_globs : hack.exclude_globs;
          FillBuffer(notes_buffer, notes_text);
          FillBuffer(include_buffer, JoinLines(include_list));
          FillBuffer(exclude_buffer, JoinLines(exclude_list));
          auto& note_state = note_buffers[hack.name];
          FillBuffer(note_state.buffer, notes_text);
          note_state.initialized = true;
        };

        for (const auto& hack : hacks) {
          auto& edit = override_edits[hack.name];
          if (!edit.initialized) {
            edit.review_status = hack.review_status;
            edit.weight = hack.weight;
            edit.weight_set = true;
            edit.notes = hack.notes;
            edit.notes_set = !hack.notes.empty();
            edit.include_globs = hack.include_globs;
            edit.include_set = !hack.include_globs.empty();
            edit.exclude_globs = hack.exclude_globs;
            edit.exclude_set = !hack.exclude_globs.empty();
            edit.initialized = true;
          } else if (!edit.weight_set) {
            edit.weight = hack.weight;
            edit.weight_set = true;
          }

          auto& note_state = note_buffers[hack.name];
          if (!note_state.initialized) {
            const std::string notes_text = edit.notes_set ? edit.notes : hack.notes;
            FillBuffer(note_state.buffer, notes_text);
            note_state.initialized = true;
          }

          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          bool is_selected = (selected_hack_name == hack.name);
          if (ImGui::Selectable(hack.name.empty() ? "-" : hack.name.c_str(), is_selected)) {
            selected_hack_name = hack.name;
            load_editor_buffers(hack, edit);
          }
          if (!hack.authors.empty() && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Authors:");
            for (const auto& author : hack.authors) {
              ImGui::BulletText("%s", author.c_str());
            }
            const auto& include_list = edit.include_set ? edit.include_globs : hack.include_globs;
            const auto& exclude_list = edit.exclude_set ? edit.exclude_globs : hack.exclude_globs;
            const auto& sample_list = hack.sample_files;
            if (!sample_list.empty()) {
              ImGui::Separator();
              ImGui::Text("Sample files:");
              for (const auto& sample : sample_list) {
                ImGui::BulletText("%s", sample.c_str());
              }
            }
            if (!include_list.empty()) {
              ImGui::Separator();
              ImGui::Text("Include globs:");
              for (const auto& glob : include_list) {
                ImGui::BulletText("%s", glob.c_str());
              }
            }
            if (!exclude_list.empty()) {
              ImGui::Separator();
              ImGui::Text("Exclude globs:");
              for (const auto& glob : exclude_list) {
                ImGui::BulletText("%s", glob.c_str());
              }
            }
            ImGui::EndTooltip();
          }

          ImGui::TableNextColumn();
          if (ImGui::Button(("Edit##" + hack.name).c_str())) {
            selected_hack_name = hack.name;
            load_editor_buffers(hack, edit);
          }

          ImGui::TableNextColumn();
          bool enabled = edit.enabled;
          if (ImGui::Checkbox(("##override_" + hack.name).c_str(), &enabled)) {
            edit.enabled = enabled;
            edit.dirty = true;
          }

          ImGui::TableNextColumn();
          if (edit.dirty) {
            ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "dirty");
          } else {
            ImGui::TextDisabled("-");
          }

          ImGui::TableNextColumn();
          ImGui::BeginDisabled(!edit.enabled);
          float weight = edit.weight;
          ImGui::SetNextItemWidth(-1.0f);
          if (ImGui::SliderFloat(("##weight_" + hack.name).c_str(), &weight, 0.0f, 1.0f, "%.2f")) {
            edit.weight = weight;
            edit.weight_set = true;
            edit.dirty = true;
          }
          ImGui::EndDisabled();

          ImGui::TableNextColumn();
          ImGui::BeginDisabled(!edit.enabled);
          const char* status_items[] = {"", "approved", "hold", "rejected"};
          int status_index = 0;
          if (edit.review_status == "approved") status_index = 1;
          else if (edit.review_status == "hold") status_index = 2;
          else if (edit.review_status == "rejected") status_index = 3;
          ImGui::SetNextItemWidth(-1.0f);
          if (ImGui::Combo(("##review_" + hack.name).c_str(), &status_index, status_items, 4)) {
            edit.review_status = status_items[status_index];
            edit.dirty = true;
          }
          ImGui::EndDisabled();

          ImGui::TableNextColumn();
          ImGui::Text("%d/%d", hack.selected_files, hack.eligible_files);

          ImGui::TableNextColumn();
          ImGui::Text("%.0f", hack.org_ratio * 100.0f);

          ImGui::TableNextColumn();
          ImGui::Text("%.0f", hack.address_ratio * 100.0f);

          ImGui::TableNextColumn();
          ImGui::SetNextItemWidth(-1.0f);
          if (ImGui::InputTextWithHint(("##note_" + hack.name).c_str(),
                                       "Notes...", note_state.buffer.data(),
                                       note_state.buffer.size())) {
            edit.notes = TrimCopy(std::string(note_state.buffer.data()));
            edit.notes_set = true;
            edit.enabled = true;
            edit.dirty = true;
            if (selected_hack_name == hack.name) {
              FillBuffer(notes_buffer, edit.notes);
            }
          }
          std::string notes = edit.notes_set ? edit.notes : hack.notes;
          if (notes.empty()) notes = "-";
          if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextWrapped("%s", notes.c_str());
            if (!hack.path.empty()) {
              ImGui::Separator();
              ImGui::TextDisabled("%s", hack.path.c_str());
            }
            ImGui::EndTooltip();
          }
        }
        ImGui::EndTable();
      }

      if (!hacks.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text(ICON_MD_EDIT " Override Editor");

        if (selected_hack_name.empty()) {
          ImGui::TextDisabled("Select a hack row to edit include/exclude globs and notes.");
        } else {
          auto it = std::find_if(hacks.begin(), hacks.end(), [&](const auto& h) {
            return h.name == selected_hack_name;
          });

          if (it == hacks.end()) {
            ImGui::TextDisabled("Selected hack not found.");
          } else {
            auto& edit = override_edits[it->name];
            ImGui::Text("Editing: %s", it->name.c_str());
            ImGui::SameLine();
            if (ImGui::Button("Revert Override")) {
              edit.enabled = false;
              edit.dirty = true;
              edit.review_status = it->review_status;
              edit.weight = it->weight;
              edit.weight_set = false;
              edit.notes = it->notes;
              edit.notes_set = false;
              edit.include_globs = it->include_globs;
              edit.include_set = false;
              edit.exclude_globs = it->exclude_globs;
              edit.exclude_set = false;
              FillBuffer(notes_buffer, it->notes);
              FillBuffer(include_buffer, JoinLines(it->include_globs));
              FillBuffer(exclude_buffer, JoinLines(it->exclude_globs));
              overrides_status = "Cleared override for " + it->name + " (not saved)";
            }
            bool notes_changed = ImGui::InputTextMultiline(
                "Notes", notes_buffer.data(), notes_buffer.size(), ImVec2(-1, 90));
            bool include_changed = ImGui::InputTextMultiline(
                "Include globs (one per line)", include_buffer.data(), include_buffer.size(),
                ImVec2(-1, 70));
            bool exclude_changed = ImGui::InputTextMultiline(
                "Exclude globs (one per line)", exclude_buffer.data(), exclude_buffer.size(),
                ImVec2(-1, 70));

            if (notes_changed || include_changed || exclude_changed) {
              edit.notes = TrimCopy(std::string(notes_buffer.data()));
              edit.notes_set = true;
              edit.include_globs = SplitLines(include_buffer.data());
              edit.include_set = true;
              edit.exclude_globs = SplitLines(exclude_buffer.data());
              edit.exclude_set = true;
              edit.enabled = true;
              edit.dirty = true;
              overrides_status = "Updated overrides for " + it->name + " (not saved)";
            }

            if (ImGui::Button("Reset Editor")) {
              auto load_editor_buffers = [&](const auto& hack, HackOverrideEdit& edit_ref) {
                const std::string notes_text = edit_ref.notes_set ? edit_ref.notes : hack.notes;
                const auto& include_list = edit_ref.include_set ? edit_ref.include_globs : hack.include_globs;
                const auto& exclude_list = edit_ref.exclude_set ? edit_ref.exclude_globs : hack.exclude_globs;
                FillBuffer(notes_buffer, notes_text);
                FillBuffer(include_buffer, JoinLines(include_list));
                FillBuffer(exclude_buffer, JoinLines(exclude_list));
              };
              load_editor_buffers(*it, edit);
            }
          }
        }
      }

      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void RenderSystemsPanel(AppState& state, ImFont* font_header, std::function<void(const char*)> refresh_callback) {
  if (font_header) ImGui::PushFont(font_header);
  ImGui::Text(ICON_MD_ROUTER " SYSTEMS OVERVIEW");
  if (font_header) ImGui::PopFont();
  ImGui::Separator();

  if (ImGui::Button(ICON_MD_REFRESH " Refresh Now", ImVec2(-1, 0))) {
    if (refresh_callback) refresh_callback("ui");
  }
  ImGui::Spacing();

  double seconds_since = std::max(0.0, glfwGetTime() - state.last_refresh_time);
  ImGui::Text("Auto Refresh: %s", state.auto_refresh ? "On" : "Off");
  ImGui::Text("Interval: %.1fs", state.refresh_interval_sec);
  ImGui::Text("Last Refresh: %.0fs ago", seconds_since);
  ImGui::Text("Simulation: %s", state.simulate_activity ? "On" : "Off");
  ImGui::Text("Quality Threshold: %.2f", state.quality_threshold);
  ImGui::Text("Mission Concurrency: %d", state.mission_concurrency);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::TextDisabled("Swarm Snapshot");

  int active_count = 0;
  int total_queue = 0;
  float total_success = 0.0f;
  for (const auto& agent : state.agents) {
    if (agent.enabled) active_count++;
    total_queue += agent.queue_depth;
    total_success += agent.success_rate;
  }
  if (!state.agents.empty()) total_success /= static_cast<float>(state.agents.size());

  if (ImGui::BeginTable("SystemSnapshot", 2, ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableNextColumn();
    ImGui::Text("Active Agents");
    ImGui::Text("Queue Depth");
    ImGui::Text("Avg Success");
    ImGui::TableNextColumn();
    ImGui::Text("%d / %d", active_count, static_cast<int>(state.agents.size()));
    ImGui::Text("%d", total_queue);
    ImGui::Text("%.1f%%", total_success * 100.0f);
    ImGui::EndTable();
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::TextDisabled("Core Services");

  auto render_row = [](const char* name, const char* status) {
    ImVec4 status_color = ImVec4(0.4f, 0.8f, 0.4f, 1.0f);
    if (strcmp(status, "Warning") == 0) status_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);
    else if (strcmp(status, "Error") == 0) status_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);

    ImGui::Bullet(); ImGui::SameLine();
    ImGui::Text("%s", name); ImGui::SameLine();
    ImGui::TextColored(status_color, "%s", status);
  };

  render_row("Orchestrator", "Active");
  render_row("Knowledge Base", "Active");
  render_row("Trainer", "Active");
  render_row("Embedding SVC", "Warning");
}

void RenderMenuBar(AppState& state,
                   std::function<void(const char*)> refresh_callback,
                   std::function<void()> quit_callback,
                   ShortcutManager& shortcuts,
                   bool* show_sample_review,
                   bool* show_shortcuts_window) {
  if (ImGui::BeginMainMenuBar()) {
    auto shortcut_label = [&](ActionId action_id) -> std::string {
      return shortcuts.FormatShortcut(action_id, ImGui::GetIO());
    };

    if (ImGui::BeginMenu("File")) {
      std::string refresh_shortcut = shortcut_label(ActionId::Refresh);
      if (ImGui::MenuItem("Refresh", refresh_shortcut.empty() ? nullptr
                                                              : refresh_shortcut.c_str())) {
        if (refresh_callback) refresh_callback("manual");
      }
      ImGui::Separator();
      std::string quit_shortcut = shortcut_label(ActionId::Quit);
      if (ImGui::MenuItem("Quit", quit_shortcut.empty() ? nullptr
                                                        : quit_shortcut.c_str())) {
        if (quit_callback) quit_callback();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Reset Layout (F1)")) {
        state.force_reset_layout = true;
      }
      ImGui::Separator();
      if (ImGui::BeginMenu("Workspace")) {
        std::string dash_shortcut = shortcut_label(ActionId::WorkspaceDashboard);
        if (ImGui::MenuItem("Dashboard", dash_shortcut.empty() ? nullptr
                                                               : dash_shortcut.c_str(),
                            state.current_workspace == Workspace::Dashboard)) {
          state.current_workspace = Workspace::Dashboard;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string analysis_shortcut = shortcut_label(ActionId::WorkspaceAnalysis);
        if (ImGui::MenuItem("Analysis", analysis_shortcut.empty() ? nullptr
                                                                  : analysis_shortcut.c_str(),
                            state.current_workspace == Workspace::Analysis)) {
          state.current_workspace = Workspace::Analysis;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string optimization_shortcut = shortcut_label(ActionId::WorkspaceOptimization);
        if (ImGui::MenuItem("Optimization",
                            optimization_shortcut.empty() ? nullptr
                                                          : optimization_shortcut.c_str(),
                            state.current_workspace == Workspace::Optimization)) {
          state.current_workspace = Workspace::Optimization;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string systems_shortcut = shortcut_label(ActionId::WorkspaceSystems);
        if (ImGui::MenuItem("Systems", systems_shortcut.empty() ? nullptr
                                                                : systems_shortcut.c_str(),
                            state.current_workspace == Workspace::Systems)) {
          state.current_workspace = Workspace::Systems;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string custom_shortcut = shortcut_label(ActionId::WorkspaceCustom);
        if (ImGui::MenuItem("Custom Grid", custom_shortcut.empty() ? nullptr
                                                                   : custom_shortcut.c_str(),
                            state.current_workspace == Workspace::Custom)) {
          state.current_workspace = Workspace::Custom;
          if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }

        std::string training_shortcut = shortcut_label(ActionId::WorkspaceTraining);
        if (ImGui::MenuItem("Training Hub", training_shortcut.empty() ? nullptr
                                                                      : training_shortcut.c_str(),
                            state.current_workspace == Workspace::Training)) {
            state.current_workspace = Workspace::Training;
            if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        std::string context_shortcut = shortcut_label(ActionId::WorkspaceContext);
        if (ImGui::MenuItem("Context Broker", context_shortcut.empty() ? nullptr
                                                                       : context_shortcut.c_str(),
                            state.current_workspace == Workspace::Context)) {
            state.current_workspace = Workspace::Context;
            if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
        }
        ImGui::EndMenu();
      }
      
      if (ImGui::BeginMenu("Theme Profile")) {
          if (ImGui::MenuItem("Deep Sea (Default)", nullptr, state.current_theme == ThemeProfile::Default)) state.current_theme = ThemeProfile::Default;
          if (ImGui::MenuItem("Cyberpunk", nullptr, state.current_theme == ThemeProfile::Cyberpunk)) state.current_theme = ThemeProfile::Cyberpunk;
          if (ImGui::MenuItem("Monochrome", nullptr, state.current_theme == ThemeProfile::Monochrome)) state.current_theme = ThemeProfile::Monochrome;
          if (ImGui::MenuItem("Solarized", nullptr, state.current_theme == ThemeProfile::Solarized)) state.current_theme = ThemeProfile::Solarized;
          if (ImGui::MenuItem("Nord", nullptr, state.current_theme == ThemeProfile::Nord)) state.current_theme = ThemeProfile::Nord;
          if (ImGui::MenuItem("Dracula", nullptr, state.current_theme == ThemeProfile::Dracula)) state.current_theme = ThemeProfile::Dracula;
          ImGui::EndMenu();
      }

      ImGui::Separator();
      if (ImGui::BeginMenu("Panels")) {
        std::string inspector_shortcut = shortcut_label(ActionId::ToggleInspector);
        ImGui::MenuItem("Inspector", inspector_shortcut.empty() ? nullptr
                                                                : inspector_shortcut.c_str(),
                        &state.show_inspector);
        std::string dataset_shortcut = shortcut_label(ActionId::ToggleDatasetPanel);
        ImGui::MenuItem("Dataset Panel", dataset_shortcut.empty() ? nullptr
                                                                  : dataset_shortcut.c_str(),
                        &state.show_dataset_panel);
        std::string systems_panel_shortcut = shortcut_label(ActionId::ToggleSystemsPanel);
        ImGui::MenuItem("Systems Panel", systems_panel_shortcut.empty() ? nullptr
                                                                        : systems_panel_shortcut.c_str(),
                        &state.show_systems_panel);
        std::string status_shortcut = shortcut_label(ActionId::ToggleStatusBar);
        ImGui::MenuItem("Status Strip", status_shortcut.empty() ? nullptr
                                                                : status_shortcut.c_str(),
                        &state.show_status_strip);
        std::string controls_shortcut = shortcut_label(ActionId::ToggleControls);
        ImGui::MenuItem("Sidebar Controls", controls_shortcut.empty() ? nullptr
                                                                      : controls_shortcut.c_str(),
                        &state.show_controls);
        ImGui::MenuItem("Chat Panel", nullptr, &state.show_chat_panel);
        ImGui::Separator();
        ImGui::MenuItem("Quality Trends", nullptr, &state.show_quality_trends);
        ImGui::MenuItem("Generator Efficiency", nullptr, &state.show_generator_efficiency);
        ImGui::MenuItem("Coverage Density", nullptr, &state.show_coverage_density);
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Windows")) {
        if (show_sample_review) {
          std::string review_shortcut = shortcut_label(ActionId::ToggleSampleReview);
          ImGui::MenuItem("Sample Review", review_shortcut.empty() ? nullptr
                                                                   : review_shortcut.c_str(),
                          show_sample_review);
        }
        if (show_shortcuts_window) {
          std::string shortcuts_shortcut =
              shortcut_label(ActionId::ToggleShortcutsWindow);
          ImGui::MenuItem("Keyboard Shortcuts",
                          shortcuts_shortcut.empty() ? nullptr
                                                     : shortcuts_shortcut.c_str(),
                          show_shortcuts_window);
        }
        ImGui::EndMenu();
      }

      ImGui::Separator();
      std::string compact_shortcut = shortcut_label(ActionId::ToggleCompactUI);
      ImGui::MenuItem("Compact UI", compact_shortcut.empty() ? nullptr
                                                             : compact_shortcut.c_str(),
                      &state.compact_charts);
      std::string lock_shortcut = shortcut_label(ActionId::ToggleLockLayout);
      ImGui::MenuItem("Lock Layout", lock_shortcut.empty() ? nullptr
                                                           : lock_shortcut.c_str(),
                      &state.lock_layout);
      std::string reset_shortcut = shortcut_label(ActionId::ResetLayout);
      if (ImGui::MenuItem("Reset Layout",
                          reset_shortcut.empty() ? nullptr
                                                 : reset_shortcut.c_str())) {
        state.force_reset_layout = true;
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Tools")) {
        std::string auto_refresh_shortcut =
            shortcut_label(ActionId::ToggleAutoRefresh);
        ImGui::MenuItem("Auto Refresh",
                        auto_refresh_shortcut.empty()
                            ? nullptr
                            : auto_refresh_shortcut.c_str(),
                        &state.auto_refresh);
        std::string simulate_shortcut =
            shortcut_label(ActionId::ToggleSimulation);
        ImGui::MenuItem("Simulate Activity",
                        simulate_shortcut.empty()
                            ? nullptr
                            : simulate_shortcut.c_str(),
                        &state.simulate_activity);
        ImGui::Separator();
        std::string demo_shortcut = shortcut_label(ActionId::ToggleDemoWindow);
        ImGui::MenuItem("Show ImGui Demo", demo_shortcut.empty() ? nullptr
                                                                 : demo_shortcut.c_str(),
                        &state.show_demo_window);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help")) {
      if (ImGui::MenuItem("Documentation")) {}
      std::string help_shortcut =
          shortcut_label(ActionId::ToggleShortcutsWindow);
      if (ImGui::MenuItem("Shortcuts",
                          help_shortcut.empty() ? nullptr
                                                : help_shortcut.c_str())) {
        if (show_shortcuts_window) *show_shortcuts_window = true;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("About AFS Studio")) {
        state.show_about_modal = true;
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  if (state.show_about_modal) {
    ImGui::OpenPopup("About AFS Studio");
  }
  if (ImGui::BeginPopupModal("About AFS Studio", &state.show_about_modal,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("AFS Studio");
    ImGui::Separator();
    ImGui::Text("Version: %s", state.studio_version.empty()
                                 ? "(unknown)"
                                 : state.studio_version.c_str());
    ImGui::Text("Data root: %s", state.studio_data_root.empty()
                                  ? "(not set)"
                                  : state.studio_data_root.c_str());
    ImGui::Spacing();
    if (ImGui::Button("Close")) {
      state.show_about_modal = false;
    }
    ImGui::EndPopup();
  }
}

void RenderSidebar(AppState& state, const DataLoader& loader, ImFont* font_ui, ImFont* font_header) {
  // Make the entire sidebar content scrollable to avoid overlaps
  ImGui::BeginChild("SidebarScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_NoBackground);

  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 4));
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.05f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 1, 1, 0.08f));

  auto sidebar_button = [&](const char* label, Workspace ws, const char* icon) {
    bool active = state.current_workspace == ws;
    ImGui::PushID(label);

    ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, 38);
    ImVec2 p_cursor = ImGui::GetCursorScreenPos();
    
    if (ImGui::InvisibleButton("##btn", size)) {
      if (state.current_workspace != ws) {
        state.current_workspace = ws;
        if (state.reset_layout_on_workspace_change) state.force_reset_layout = true;
      }
    }

    bool hovered = ImGui::IsItemHovered();
    bool pressed = ImGui::IsItemActive();
    
    ImDrawList* draw = ImGui::GetWindowDrawList();
    ImU32 bg_col = hovered ? ImGui::GetColorU32(ImGuiCol_ButtonHovered) : (pressed ? ImGui::GetColorU32(ImGuiCol_ButtonActive) : 0);
    if (active) bg_col = ImGui::GetColorU32(ImVec4(1, 1, 1, 0.08f));
    
    if (bg_col != 0) {
        draw->AddRectFilled(p_cursor, ImVec2(p_cursor.x + size.x, p_cursor.y + size.y), bg_col, 4.0f);
    }

    if (active) {
        draw->AddRectFilled(p_cursor, ImVec2(p_cursor.x + 3, p_cursor.y + size.y), ImGui::GetColorU32(ImVec4(0.40f, 0.85f, 1.0f, 1.0f)), 2.0f);
    }

    ImVec2 text_pos = ImVec2(p_cursor.x + 12, p_cursor.y + (size.y - ImGui::GetFontSize()) * 0.5f);
    draw->AddText(font_ui, ImGui::GetFontSize(), text_pos, active ? ImGui::GetColorU32(ImVec4(0.40f, 0.85f, 1.0f, 1.0f)) : ImGui::GetColorU32(ImGuiCol_Text), std::string(std::string(icon) + "  " + label).c_str());

    if (hovered) ImGui::SetTooltip("%s workspace", label);
    ImGui::PopID();
  };

  auto sidebar_header = [&](const char* title, const char* icon = nullptr) {
    ImGui::Spacing(); ImGui::Spacing();
    if (font_header) ImGui::PushFont(font_header);
    ImGui::SetCursorPosX(12);
    if (icon) {
        ImGui::TextDisabled("%s %s", icon, title);
    } else {
        ImGui::TextDisabled("%s", title);
    }
    if (font_header) ImGui::PopFont();
    ImGui::Spacing();
  };

  sidebar_header("WORKSPACES", ICON_MD_VIEW_QUILT);
  sidebar_button("Dashboard", Workspace::Dashboard, ICON_MD_DASHBOARD);
  sidebar_button("Analysis", Workspace::Analysis, ICON_MD_ANALYTICS);
  sidebar_button("Optimization", Workspace::Optimization, ICON_MD_SETTINGS_INPUT_COMPONENT);
  
  sidebar_header("OPERATIONS", ICON_MD_SETTINGS_SUGGEST);
  sidebar_button("Systems", Workspace::Systems, ICON_MD_ROUTER);
  sidebar_button("Training", Workspace::Training, ICON_MD_MODEL_TRAINING);
  sidebar_button("Custom Grid", Workspace::Custom, ICON_MD_DASHBOARD_CUSTOMIZE);
  
  sidebar_header("REGISTRIES", ICON_MD_STORAGE);

  sidebar_button("Context", Workspace::Context, ICON_MD_FOLDER_OPEN);
  sidebar_button("Models", Workspace::Models, ICON_MD_STICKY_NOTE_2);

  // New: Useful Tools Section
  sidebar_header("SYSTEM TOOLS", ICON_MD_HANDYMAN);
  ImGui::Indent(12);
  
  // Health Summary
  {
      float health = 0.92f; // Mock health
      ImGui::TextDisabled("System Health");
      ImGui::ProgressBar(health, ImVec2(-12, 4), "");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overall reliability score: %.0f%%", health * 100.0f);
  }
  
  ImGui::Spacing();
  
  // Quick Toggles
  if (ImGui::Checkbox("Simulate", &state.simulate_activity)) {}
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle background task simulation");
  
  if (ImGui::Checkbox("Auto Refresh", &state.auto_refresh)) {}
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Refresh data every %.1fs", state.refresh_interval_sec);

  ImGui::Unindent(12);
  
  // New: Mounts Management Section
  sidebar_header("LOCAL MOUNTS", ICON_MD_STORAGE);
  ImGui::Indent(12);
  
  const auto& mounts = loader.GetMounts();
  if (mounts.empty()) {
      ImGui::TextDisabled("No mounts discovered");
  } else {
      for (const auto& mount : mounts) {
          ImGui::BeginGroup();
          ImGui::TextColored(mount.active ? ImVec4(0.4f, 1.0f, 0.6f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f), 
                             mount.active ? ICON_MD_DNS : ICON_MD_DASHBOARD_CUSTOMIZE);
          ImGui::SameLine();
          ImGui::Text("%s", mount.name.c_str());
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Path: %s\nStatus: %s", mount.path.c_str(), mount.active ? "Connected" : "Disconnected");
          ImGui::EndGroup();
      }
  }

  if (ImGui::SmallButton(ICON_MD_STORAGE " Mount Drives")) {
      const_cast<DataLoader&>(loader).MountDrive("all");
  }
  ImGui::SameLine();
  if (ImGui::SmallButton(ICON_MD_ADD " Add Mount")) {
      // TODO: Implement mount dialog
  }

  ImGui::Unindent(12);

  ImGui::PopStyleColor(3);
  ImGui::PopStyleVar();

  ImGui::EndChild(); // End SidebarScroll
}

} // namespace ui
} // namespace viz
} // namespace afs

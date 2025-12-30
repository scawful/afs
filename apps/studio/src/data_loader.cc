#include "data_loader.h"
#include "core/logger.h"
#include "core/filesystem.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>
#include <optional>
#include <nlohmann/json.hpp>

namespace afs {
namespace viz {

namespace {

using json = nlohmann::json;

constexpr size_t kTrendWindow = 5;
constexpr float kPi = 3.14159265f;

std::optional<std::filesystem::path> ResolveTrunkRoot();

std::optional<std::filesystem::path> ResolveHafsScawfulRoot() {
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

  auto trunk_root = ResolveTrunkRoot();
  if (trunk_root) {
    auto candidate = *trunk_root / "scawful" / "research" / "afs_scawful";
    if (studio::core::FileSystem::Exists(candidate)) {
      return candidate;
    }
  }

  return std::nullopt;
}

std::optional<std::filesystem::path> ResolveTrunkRoot() {
  const char* env_root = std::getenv("TRUNK_ROOT");
  if (env_root && env_root[0] != '\0') {
    auto path = studio::core::FileSystem::ResolvePath(env_root);
    if (studio::core::FileSystem::Exists(path)) {
      return path;
    }
  }

  auto path = studio::core::FileSystem::ResolvePath("~/src/trunk");
  if (studio::core::FileSystem::Exists(path)) {
    return path;
  }

  return std::nullopt;
}

std::filesystem::path ResolveContextRoot() {
  const char* env_root = std::getenv("AFS_CONTEXT_ROOT");
  if (env_root && env_root[0] != '\0') {
    auto path = studio::core::FileSystem::ResolvePath(env_root);
    if (studio::core::FileSystem::Exists(path)) {
      return path;
    }
  }

  auto candidate = studio::core::FileSystem::ResolvePath("~/src/context");
  if (studio::core::FileSystem::Exists(candidate)) {
    return candidate;
  }

  auto fallback = studio::core::FileSystem::ResolvePath("~/.context");
  if (studio::core::FileSystem::Exists(fallback)) {
    return fallback;
  }

  return candidate;
}

std::filesystem::path ResolveTrainingRoot() {
  const char* env_root = std::getenv("AFS_TRAINING_ROOT");
  if (env_root && env_root[0] != '\0') {
    auto path = studio::core::FileSystem::ResolvePath(env_root);
    if (studio::core::FileSystem::Exists(path)) {
      return path;
    }
  }

  auto candidate = studio::core::FileSystem::ResolvePath("~/src/training");
  if (studio::core::FileSystem::Exists(candidate)) {
    return candidate;
  }

  auto fallback = studio::core::FileSystem::ResolvePath("~/.context/training");
  if (studio::core::FileSystem::Exists(fallback)) {
    return fallback;
  }

  return candidate;
}

std::filesystem::path ResolveContextGraphPath() {
  const char* env_path = std::getenv("AFS_GRAPH_PATH");
  if (env_path && env_path[0] != '\0') {
    return studio::core::FileSystem::ResolvePath(env_path);
  }
  return ResolveContextRoot() / "index" / "afs_graph.json";
}

std::filesystem::path ResolveDatasetRegistryPath() {
  const char* env_path = std::getenv("AFS_DATASET_REGISTRY");
  if (env_path && env_path[0] != '\0') {
    return studio::core::FileSystem::ResolvePath(env_path);
  }
  return ResolveTrainingRoot() / "index" / "dataset_registry.json";
}

std::filesystem::path ResolveResourceIndexPath(const std::string& data_root,
                                               const DataLoader::PathExists& exists) {
  const char* env_path = std::getenv("AFS_RESOURCE_INDEX");
  if (env_path && env_path[0] != '\0') {
    return studio::core::FileSystem::ResolvePath(env_path);
  }

  std::vector<std::filesystem::path> candidates;
  auto training_root = ResolveTrainingRoot();
  if (!training_root.empty()) {
    candidates.push_back(training_root / "index" / "resource_index.json");
    candidates.push_back(training_root / "resource_index.json");
  }
  if (!data_root.empty()) {
    auto data_path = std::filesystem::path(data_root);
    candidates.push_back(data_path / "index" / "resource_index.json");
    candidates.push_back(data_path / "resource_index.json");
  }

  for (const auto& candidate : candidates) {
    if (exists(candidate.string())) {
      return candidate;
    }
  }
  return {};
}

std::filesystem::path ResolveTrainingDataPath(const std::string& filename,
                                              const std::string& data_root,
                                              const DataLoader::PathExists& exists) {
  std::vector<std::filesystem::path> candidates;
  if (!data_root.empty()) {
    auto data_path = std::filesystem::path(data_root);
    candidates.push_back(data_path / filename);
    candidates.push_back(data_path / "index" / filename);
  }

  auto training_root = ResolveTrainingRoot();
  if (!training_root.empty()) {
    candidates.push_back(training_root / filename);
    candidates.push_back(training_root / "index" / filename);
  }

  for (const auto& candidate : candidates) {
    if (exists(candidate.string())) {
      return candidate;
    }
  }
  return {};
}

constexpr float kTrendDeltaThreshold = 0.05f;

bool IsWhitespaceOnly(const std::string& s) {
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isspace(c);
  });
}

} // namespace

DataLoader::DataLoader(const std::string& data_path,
                       FileReader file_reader,
                       PathExists path_exists)
    : data_path_(data_path) {
    
    // Set default handlers if not provided
    if (file_reader) {
        file_reader_ = std::move(file_reader);
    } else {
        file_reader_ = [](const std::string& p, std::string* c, std::string* e) {
            auto content = studio::core::FileSystem::ReadFile(p);
            if (content) {
                *c = *content;
                return true;
            }
            if (e) *e = "Failed to read file";
            return false;
        };
    }

    if (path_exists) {
        path_exists_ = std::move(path_exists);
    } else {
        path_exists_ = [](const std::string& p) {
            try {
                return studio::core::FileSystem::Exists(p);
            } catch (...) {
                return false;
            }
        };
    }
}

bool DataLoader::Refresh() {
  last_error_.clear();
  last_status_ = LoadStatus{};

  const bool base_exists = !data_path_.empty() && path_exists_(data_path_);
  const auto training_root = ResolveTrainingRoot();
  const bool training_exists = !training_root.empty() && path_exists_(training_root.string());
  if (!base_exists && !training_exists) {
    last_error_ = "Data path does not exist: " + data_path_;
    LOG_ERROR(last_error_);
    last_status_.error_count = 1;
    last_status_.last_error = last_error_;
    last_status_.last_error_source = "data_path";
  } else {
    const auto& root = base_exists ? data_path_ : training_root.string();
    if (!root.empty()) {
      LOG_INFO("DataLoader: Refreshing from " + root);
    }
  }

  auto next_quality_trends = quality_trends_;
  auto next_generator_stats = generator_stats_;
  auto next_rejection_summary = rejection_summary_;
  auto next_embedding_regions = embedding_regions_;
  auto next_coverage = coverage_;
  auto next_training_runs = training_runs_;
  auto next_optimization_data = optimization_data_;
  auto next_curated_hacks = curated_hacks_;
  auto next_resource_index = resource_index_;
  auto next_dataset_registry = dataset_registry_;
  auto next_context_graph = context_graph_;

  LoadResult quality = LoadQualityFeedback(&next_quality_trends,
                                           &next_generator_stats,
                                           &next_rejection_summary);
  last_status_.quality_found = quality.found;
  last_status_.quality_ok = quality.ok;
  if (quality.found && !quality.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = quality.error;
      last_status_.last_error_source = "quality_feedback.json";
    }
  }
  if (quality.ok) {
    quality_trends_ = std::move(next_quality_trends);
    generator_stats_ = std::move(next_generator_stats);
    rejection_summary_ = std::move(next_rejection_summary);
    
    // Initialize domain visibility for new domains
    for (const auto& trend : quality_trends_) {
        if (domain_visibility_.find(trend.domain) == domain_visibility_.end()) {
            domain_visibility_[trend.domain] = true;
        }
    }
  }

  LoadResult active = LoadActiveLearning(&next_embedding_regions, &next_coverage);
  last_status_.active_found = active.found;
  last_status_.active_ok = active.ok;
  if (active.found && !active.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = active.error;
      last_status_.last_error_source = "active_learning.json";
    }
  }
  if (active.ok) {
    embedding_regions_ = std::move(next_embedding_regions);
    coverage_ = std::move(next_coverage);
  }

  LoadResult training = LoadTrainingFeedback(&next_training_runs,
                                             &next_optimization_data);
  last_status_.training_found = training.found;
  last_status_.training_ok = training.ok;
  if (training.found && !training.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = training.error;
      last_status_.last_error_source = "training_feedback.json";
    }
  }
  if (training.ok) {
    training_runs_ = std::move(next_training_runs);
    optimization_data_ = std::move(next_optimization_data);
  }

  LoadResult curated = LoadCuratedHacks(&next_curated_hacks);
  if (!curated.found) {
    curated_hacks_.clear();
    curated_hacks_error_ = "curated_hacks.json not found";
  } else if (!curated.ok) {
    curated_hacks_error_ = curated.error;
  } else {
    curated_hacks_ = std::move(next_curated_hacks);
    curated_hacks_error_.clear();
  }

  LoadResult resource = LoadResourceIndex(&next_resource_index);
  last_status_.resource_index_found = resource.found;
  last_status_.resource_index_ok = resource.ok;
  if (resource.found && !resource.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = resource.error;
      last_status_.last_error_source = "resource_index.json";
    }
  }
  if (!resource.found) {
    resource_index_ = ResourceIndexData{};
    resource_index_error_ = "resource_index.json not found";
  } else if (!resource.ok) {
    resource_index_error_ = resource.error;
  } else {
    resource_index_ = std::move(next_resource_index);
    resource_index_error_.clear();
  }

  LoadResult registry = LoadDatasetRegistry(&next_dataset_registry);
  last_status_.dataset_registry_found = registry.found;
  last_status_.dataset_registry_ok = registry.ok;
  if (registry.found && !registry.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = registry.error;
      last_status_.last_error_source = "dataset_registry.json";
    }
  }
  if (!registry.found) {
    dataset_registry_ = DatasetRegistryData{};
    dataset_registry_error_ = "dataset_registry.json not found";
  } else if (!registry.ok) {
    dataset_registry_error_ = registry.error;
  } else {
    dataset_registry_ = std::move(next_dataset_registry);
    dataset_registry_error_.clear();
  }

  LoadResult context_graph = LoadContextGraph(&next_context_graph);
  last_status_.context_graph_found = context_graph.found;
  last_status_.context_graph_ok = context_graph.ok;
  if (context_graph.found && !context_graph.ok) {
    last_status_.error_count += 1;
    if (last_status_.last_error.empty()) {
      last_status_.last_error = context_graph.error;
      last_status_.last_error_source = "afs_graph.json";
    }
  }
  if (!context_graph.found) {
    context_graph_ = ContextGraphData{};
    context_graph_error_ = "afs_graph.json not found";
  } else if (!context_graph.ok) {
    context_graph_error_ = context_graph.error;
  } else {
    context_graph_ = std::move(next_context_graph);
    context_graph_error_.clear();
  }
  
  // Update Mounts status
  mounts_.clear();
  const char* home = std::getenv("HOME");
  std::string home_str = home ? home : "";
  
  auto add_mount = [&](const std::string& name, std::string path) {
      if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
          path = home_str + path.substr(1);
      }
      mounts_.push_back({name, path, path_exists_(path)});
  };

  add_mount("Code", "~/Code");
  auto trunk_root = ResolveTrunkRoot();
  if (trunk_root) {
    add_mount("Trunk", trunk_root->string());
  }
  auto scawful_root = ResolveHafsScawfulRoot();
  if (scawful_root) {
    add_mount("afs_scawful", scawful_root->string());
  }
  add_mount("usdasm", "~/Code/usdasm");
  add_mount("Medical Mechanica (D)", "/Users/scawful/Mounts/mm-d/afs_training");
  if (trunk_root) {
    add_mount("Oracle-of-Secrets", (trunk_root.value() / "scawful/retro/oracle-of-secrets").string());
    add_mount("yaze", (trunk_root.value() / "scawful/retro/yaze").string());
  } else {
    add_mount("Oracle-of-Secrets", "~/Code/Oracle-of-Secrets");
    add_mount("yaze", "~/Code/yaze");
  }
  add_mount("AFS Context", ResolveContextRoot().string());
  add_mount("AFS Training", ResolveTrainingRoot().string());

  has_data_ = !quality_trends_.empty() || !generator_stats_.empty() ||
              !embedding_regions_.empty() || !training_runs_.empty() ||
              !optimization_data_.domain_effectiveness.empty() ||
              !optimization_data_.threshold_sensitivity.empty() ||
              !dataset_registry_.datasets.empty() ||
              resource_index_.total_files > 0 ||
              !context_graph_.labels.empty();
  last_error_ = last_status_.last_error;

  return last_status_.AnyOk() || (!(last_status_.FoundCount() > 0) && has_data_);
}

DataLoader::LoadResult DataLoader::LoadQualityFeedback(
    std::vector<QualityTrendData>* quality_trends,
    std::vector<GeneratorStatsData>* generator_stats,
    RejectionSummary* rejection_summary) {
  
  LoadResult result;
  auto path = ResolveTrainingDataPath("quality_feedback.json", data_path_, path_exists_);
  if (path.empty()) {
    return result;
  }
  LOG_INFO("DataLoader: Loading " + path.string());

  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path.string(), &content, &read_error) || content.empty() || IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "quality_feedback.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    std::vector<QualityTrendData> next_quality_trends;
    std::vector<GeneratorStatsData> next_generator_stats;
    RejectionSummary next_rejection_summary;

    if (data.contains("generator_stats") && data["generator_stats"].is_object()) {
      bool only_one = data["generator_stats"].size() == 1;
      for (auto& [name, stats] : data["generator_stats"].items()) {
        GeneratorStatsData gs;
        std::string processed_name = name;
        if (processed_name == "unknown" && only_one) {
            processed_name = "Core Engine";
        }
        
        // Strip common suffixes for cleaner display
        size_t pos = processed_name.find("DataGenerator");
        if (pos != std::string::npos) {
            processed_name = processed_name.substr(0, pos);
        }
        
        gs.name = processed_name;
        gs.samples_generated = stats.value("samples_generated", 0);
        gs.samples_accepted = stats.value("samples_accepted", 0);
        gs.samples_rejected = stats.value("samples_rejected", 0);
        gs.avg_quality = stats.value("avg_quality_score", 0.0f);

        int total = gs.samples_accepted + gs.samples_rejected;
        gs.acceptance_rate = total > 0 ? static_cast<float>(gs.samples_accepted) / total : 0.0f;

        if (stats.contains("rejection_reasons") && stats["rejection_reasons"].is_object()) {
          for (auto& [reason, count] : stats["rejection_reasons"].items()) {
            int c = count.get<int>();
            gs.rejection_reasons[reason] = c;
            next_rejection_summary.reasons[reason] += c;
            next_rejection_summary.total_rejections += c;
          }
        }
        next_generator_stats.push_back(std::move(gs));
      }
    }

    if (data.contains("rejection_history") && data["rejection_history"].is_array()) {
      std::map<std::pair<std::string, std::string>, QualityTrendData> trends_map;
      for (auto& entry : data["rejection_history"]) {
        std::string domain = entry.value("domain", "unknown");
        if (entry.contains("scores") && entry["scores"].is_object()) {
          for (auto& [metric, value] : entry["scores"].items()) {
            auto key = std::make_pair(domain, metric);
            if (trends_map.find(key) == trends_map.end()) {
              trends_map[key] = QualityTrendData{domain, metric};
            }
            trends_map[key].values.push_back(value.get<float>());
          }
        }
      }

      for (auto& [key, trend] : trends_map) {
        if (!trend.values.empty()) {
          float sum = 0.0f;
          for (float v : trend.values) sum += v;
          trend.mean = sum / trend.values.size();

          if (trend.values.size() < kTrendWindow) {
            trend.trend_direction = "insufficient";
          } else {
            float recent = 0.0f, older = 0.0f;
            for (size_t i = trend.values.size() - kTrendWindow; i < trend.values.size(); ++i) recent += trend.values[i];
            for (size_t i = 0; i < kTrendWindow && i < trend.values.size(); ++i) older += trend.values[i];
            recent /= kTrendWindow;
            older /= std::min((size_t)kTrendWindow, trend.values.size());
            float diff = recent - older;
            if (diff > kTrendDeltaThreshold) trend.trend_direction = "improving";
            else if (diff < -kTrendDeltaThreshold) trend.trend_direction = "declining";
            else trend.trend_direction = "stable";
          }
        }
        next_quality_trends.push_back(std::move(trend));
      }
    }

    if (quality_trends) *quality_trends = std::move(next_quality_trends);
    if (generator_stats) *generator_stats = std::move(next_generator_stats);
    if (rejection_summary) *rejection_summary = std::move(next_rejection_summary);

    LOG_INFO("DataLoader: Successfully loaded data");
    result.ok = true;

  } catch (const json::exception& e) {
    result.ok = false;
    result.error = std::string("JSON error in quality_feedback.json: ") + e.what();
    LOG_ERROR(result.error);
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadActiveLearning(
    std::vector<EmbeddingRegionData>* embedding_regions,
    CoverageData* coverage) {
  
  LoadResult result;
  auto path = ResolveTrainingDataPath("active_learning.json", data_path_, path_exists_);
  if (path.empty()) return result;

  LOG_INFO("DataLoader: Loading " + path.string());
  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path.string(), &content, &read_error) || content.empty() || IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "active_learning.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    std::vector<EmbeddingRegionData> next_embedding_regions;
    CoverageData next_coverage;

    if (data.contains("regions") && data["regions"].is_array()) {
      int idx = 0;
      for (auto& region : data["regions"]) {
        EmbeddingRegionData erd;
        erd.index = idx++;
        erd.sample_count = region.value("sample_count", 0);
        erd.domain = region.value("domain", "unknown");
        erd.avg_quality = region.value("avg_quality", 0.0f);
        next_embedding_regions.push_back(std::move(erd));
      }
    }

    next_coverage.num_regions = data.value("num_regions", 0);
    
    if (embedding_regions) *embedding_regions = std::move(next_embedding_regions);
    if (coverage) *coverage = std::move(next_coverage);

    LOG_INFO("DataLoader: Successfully loaded active learning data");
    result.ok = true;

  } catch (const json::exception& e) {
    result.ok = false;
    result.error = std::string("JSON error in active_learning.json: ") + e.what();
    LOG_ERROR(result.error);
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadTrainingFeedback(
    std::vector<TrainingRunData>* training_runs,
    OptimizationData* optimization_data) {
  
  LoadResult result;
  auto path = ResolveTrainingDataPath("training_feedback.json", data_path_, path_exists_);
  if (path.empty()) return result;

  LOG_INFO("DataLoader: Loading " + path.string());
  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path.string(), &content, &read_error) || content.empty() || IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "training_feedback.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    std::vector<TrainingRunData> next_training_runs;
    OptimizationData next_optimization_data;

    if (data.contains("training_runs") && data["training_runs"].is_object()) {
      for (auto& [id, run] : data["training_runs"].items()) {
        TrainingRunData trd;
        trd.run_id = id;
        trd.model_name = run.value("model_name", "unknown");
        trd.samples_count = run.value("samples_count", 0);
        trd.final_loss = run.value("final_loss", 0.0f);
        trd.start_time = run.value("start_time", "");
        
        if (run.contains("domain_distribution") && run["domain_distribution"].is_object()) {
           for (auto& [domain, count] : run["domain_distribution"].items()) {
             trd.domain_distribution[domain] = count.get<int>();
           }
        }
        next_training_runs.push_back(std::move(trd));
      }
    }

    if (data.contains("domain_effectiveness") && data["domain_effectiveness"].is_object()) {
      for (auto& [domain, val] : data["domain_effectiveness"].items()) {
        next_optimization_data.domain_effectiveness[domain] = val.get<float>();
      }
    }

    if (data.contains("quality_threshold_effectiveness") && data["quality_threshold_effectiveness"].is_object()) {
      for (auto& [thresh, val] : data["quality_threshold_effectiveness"].items()) {
        next_optimization_data.threshold_sensitivity[thresh] = val.get<float>();
      }
    }

    if (training_runs) *training_runs = std::move(next_training_runs);
    if (optimization_data) *optimization_data = std::move(next_optimization_data);

    LOG_INFO("DataLoader: Successfully loaded training feedback data");
    result.ok = true;

  } catch (const json::exception& e) {
    result.ok = false;
    result.error = std::string("JSON error in training_feedback.json: ") + e.what();
    LOG_ERROR(result.error);
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadCuratedHacks(
    std::vector<CuratedHackEntry>* curated_hacks) {
  LoadResult result;
  auto path = ResolveTrainingDataPath("curated_hacks.json", data_path_, path_exists_);
  if (path.empty()) {
    return result;
  }
  LOG_INFO("DataLoader: Loading " + path.string());

  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path.string(), &content, &read_error) || content.empty() ||
      IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error =
        read_error.empty() ? "curated_hacks.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    if (!data.contains("hacks") || !data["hacks"].is_array()) {
      result.ok = false;
      result.error = "curated_hacks.json missing 'hacks' array";
      return result;
    }

    curated_hacks->clear();
    for (const auto& hack : data["hacks"]) {
      CuratedHackEntry entry;
      entry.name = hack.value("name", "");
      entry.path = hack.value("path", "");
      entry.notes = hack.value("notes", "");
      entry.review_status = hack.value("review_status", "");
      entry.weight = hack.value("weight", 1.0f);
      entry.eligible_files = hack.value("eligible_files", 0);
      entry.selected_files = hack.value("selected_files", 0);
      entry.org_ratio = hack.value("org_ratio", 0.0f);
      entry.address_ratio = hack.value("address_ratio", 0.0f);
      entry.avg_comment_ratio = hack.value("avg_comment_ratio", 0.0f);
      entry.status = hack.value("status", "");
      entry.error = hack.value("error", "");

      auto read_string_array = [](const json& arr) {
        std::vector<std::string> out;
        if (!arr.is_array()) return out;
        for (const auto& value : arr) {
          if (value.is_string()) out.push_back(value.get<std::string>());
        }
        return out;
      };

      if (hack.contains("authors")) entry.authors = read_string_array(hack["authors"]);
      if (hack.contains("include_globs")) entry.include_globs = read_string_array(hack["include_globs"]);
      if (hack.contains("exclude_globs")) entry.exclude_globs = read_string_array(hack["exclude_globs"]);
      if (hack.contains("sample_files")) entry.sample_files = read_string_array(hack["sample_files"]);

      curated_hacks->push_back(std::move(entry));
    }

    result.ok = true;
  } catch (const std::exception& e) {
    result.ok = false;
    result.error = std::string("Failed to parse curated_hacks.json: ") + e.what();
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadResourceIndex(ResourceIndexData* resource_index) {
  LoadResult result;
  auto path = ResolveResourceIndexPath(data_path_, path_exists_);
  if (path.empty()) {
    return result;
  }
  LOG_INFO("DataLoader: Loading " + path.string());

  result.found = true;
  std::string content;
  std::string read_error;
  if (!file_reader_(path.string(), &content, &read_error) || content.empty() ||
      IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "resource_index.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    if (!data.contains("metadata")) {
      result.ok = false;
      result.error = "resource_index.json missing metadata";
      return result;
    }

    const auto& meta = data["metadata"];
    resource_index->total_files = meta.value("total_files", 0);
    resource_index->duplicates_found = meta.value("duplicates_found", 0);
    resource_index->duration_seconds = meta.value("duration_seconds", 0.0f);
    resource_index->indexed_at = meta.value("indexed_at", "");
    resource_index->by_source.clear();
    resource_index->by_type.clear();

    if (meta.contains("by_source")) {
      for (auto it = meta["by_source"].begin(); it != meta["by_source"].end(); ++it) {
        resource_index->by_source[it.key()] = it.value().get<int>();
      }
    }
    if (meta.contains("by_type")) {
      for (auto it = meta["by_type"].begin(); it != meta["by_type"].end(); ++it) {
        resource_index->by_type[it.key()] = it.value().get<int>();
      }
    }

    result.ok = true;
  } catch (const std::exception& e) {
    result.ok = false;
    result.error = std::string("Failed to parse resource_index.json: ") + e.what();
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadDatasetRegistry(DatasetRegistryData* dataset_registry) {
  LoadResult result;
  std::filesystem::path path = ResolveDatasetRegistryPath();
  if (!path_exists_(path.string())) {
    return result;
  }
  result.found = true;
  LOG_INFO("DataLoader: Loading " + path.string());

  std::string content;
  std::string read_error;
  if (!file_reader_(path.string(), &content, &read_error) || content.empty() ||
      IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "dataset_registry.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    dataset_registry->generated_at = data.value("generated_at", "");
    dataset_registry->datasets.clear();

    if (!data.contains("datasets") || !data["datasets"].is_array()) {
      result.ok = false;
      result.error = "dataset_registry.json missing datasets array";
      return result;
    }

    for (const auto& entry : data["datasets"]) {
      DatasetEntry dataset;
      dataset.name = entry.value("name", "");
      dataset.path = entry.value("path", "");
      dataset.size_bytes = static_cast<std::uint64_t>(entry.value("size_bytes", 0));
      dataset.updated_at = entry.value("updated_at", "");
      if (entry.contains("files") && entry["files"].is_array()) {
        for (const auto& file : entry["files"]) {
          if (file.is_string()) {
            dataset.files.push_back(file.get<std::string>());
          }
        }
      }
      dataset_registry->datasets.push_back(std::move(dataset));
    }

    result.ok = true;
  } catch (const std::exception& e) {
    result.ok = false;
    result.error = std::string("Failed to parse dataset_registry.json: ") + e.what();
  }

  return result;
}

DataLoader::LoadResult DataLoader::LoadContextGraph(ContextGraphData* context_graph) {
  LoadResult result;
  std::filesystem::path path = ResolveContextGraphPath();
  if (!path_exists_(path.string())) {
    return result;
  }
  result.found = true;
  LOG_INFO("DataLoader: Loading " + path.string());

  std::string content;
  std::string read_error;
  if (!file_reader_(path.string(), &content, &read_error) || content.empty() ||
      IsWhitespaceOnly(content)) {
    result.ok = false;
    result.error = read_error.empty() ? "afs_graph.json is empty" : read_error;
    return result;
  }

  try {
    json data = json::parse(content);
    if (!data.contains("contexts") || !data["contexts"].is_array()) {
      result.ok = false;
      result.error = "afs_graph.json missing contexts array";
      return result;
    }

    context_graph->labels.clear();
    context_graph->nodes_x.clear();
    context_graph->nodes_y.clear();
    context_graph->edges.clear();
    context_graph->context_count = 0;
    context_graph->mount_count = 0;
    context_graph->source_path = path.string();

    const auto& contexts = data["contexts"];
    const size_t context_total = contexts.size();
    context_graph->context_count = static_cast<int>(context_total);

    for (size_t i = 0; i < context_total; ++i) {
      const auto& ctx = contexts[i];
      std::string name = ctx.value("name", "context");
      float angle = (context_total > 0)
          ? (2.0f * kPi * static_cast<float>(i) / static_cast<float>(context_total))
          : 0.0f;
      float cx = std::cos(angle);
      float cy = std::sin(angle);

      int ctx_index = static_cast<int>(context_graph->labels.size());
      context_graph->labels.push_back(name);
      context_graph->nodes_x.push_back(cx);
      context_graph->nodes_y.push_back(cy);

      if (!ctx.contains("mounts") || !ctx["mounts"].is_array()) {
        continue;
      }

      const auto& mounts = ctx["mounts"];
      const size_t mount_total = mounts.size();
      if (mount_total == 0) {
        continue;
      }

      float ring = 0.35f + 0.02f * static_cast<float>(mount_total);
      for (size_t j = 0; j < mount_total; ++j) {
        const auto& mount = mounts[j];
        std::string mount_name = mount.value("name", "mount");
        std::string mount_type = mount.value("mount_type", "");
        std::string label = mount_type.empty() ? mount_name : (mount_type + ":" + mount_name);
        float local_angle = (2.0f * kPi * static_cast<float>(j) / static_cast<float>(mount_total));
        float mx = cx + std::cos(local_angle) * ring;
        float my = cy + std::sin(local_angle) * ring;

        int mount_index = static_cast<int>(context_graph->labels.size());
        context_graph->labels.push_back(label);
        context_graph->nodes_x.push_back(mx);
        context_graph->nodes_y.push_back(my);
        context_graph->edges.push_back({ctx_index, mount_index});
        context_graph->mount_count += 1;
      }
    }

    result.ok = true;
  } catch (const std::exception& e) {
    result.ok = false;
    result.error = std::string("Failed to parse afs_graph.json: ") + e.what();
  }

  return result;
}

void DataLoader::MountDrive(const std::string& name) {
  auto scawful_root = ResolveHafsScawfulRoot();
  std::filesystem::path script_path;
  if (scawful_root) {
    script_path = *scawful_root / "scripts" / "mount_windows.sh";
  } else {
    auto trunk_root = ResolveTrunkRoot();
    if (trunk_root) {
      script_path = *trunk_root / "scawful" / "research" / "afs_scawful" / "scripts" / "mount_windows.sh";
    } else {
      script_path = studio::core::FileSystem::ResolvePath("~/src/trunk/scawful/research/afs_scawful/scripts/mount_windows.sh");
    }
  }

  if (studio::core::FileSystem::Exists(script_path)) {
    LOG_INFO("DataLoader: Triggering mount using " + script_path.string());
    // The script takes an optional argument, but 'mount' is default.
    std::string cmd = "bash \"" + script_path.string() + "\" mount 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
      char buffer[256];
      while (fgets(buffer, sizeof(buffer), pipe)) {
          std::string line(buffer);
          if (!line.empty() && line.back() == '\n') line.pop_back();
          LOG_INFO("Mount output: " + line);
      }
      pclose(pipe);
    }
  } else {
    LOG_ERROR("DataLoader: Mount script not found: " + script_path.string());
  }
}

} // namespace viz
} // namespace afs

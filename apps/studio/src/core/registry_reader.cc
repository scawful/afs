#include "registry_reader.h"
#include "logger.h"
#include "filesystem.h"

#include <cstdlib>
#include <fstream>

namespace {

std::filesystem::path ResolveContextRoot() {
  const char* env_root = std::getenv("AFS_CONTEXT_ROOT");
  if (env_root && env_root[0] != '\0') {
    auto path = afs::studio::core::FileSystem::ResolvePath(env_root);
    if (afs::studio::core::FileSystem::Exists(path)) {
      return path;
    }
  }

  auto preferred = afs::studio::core::FileSystem::ResolvePath("~/src/context");
  if (afs::studio::core::FileSystem::Exists(preferred)) {
    return preferred;
  }

  auto fallback = afs::studio::core::FileSystem::ResolvePath("~/.context");
  if (afs::studio::core::FileSystem::Exists(fallback)) {
    return fallback;
  }

  return preferred;
}

}  // namespace

namespace afs {
namespace studio {

RegistryReader::RegistryReader() : registry_path_(ResolveDefaultPath()) {}

RegistryReader::RegistryReader(const std::filesystem::path& registry_path)
    : registry_path_(registry_path) {}

std::filesystem::path RegistryReader::ResolveDefaultPath() const {
  return ResolveContextRoot() / "models" / "registry.json";
}

bool RegistryReader::Exists() const {
  return core::FileSystem::Exists(registry_path_);
}

bool RegistryReader::Load(std::string* error) {
  last_error_.clear();
  models_.clear();

  if (!Exists()) {
    last_error_ = "Registry file not found: " + registry_path_.string();
    LOG_ERROR(last_error_);
    if (error) *error = last_error_;
    return false;
  }
  LOG_INFO("RegistryReader: Loading from " + registry_path_.string());

  std::ifstream file(registry_path_);
  if (!file.is_open()) {
    last_error_ = "Failed to open registry file";
    if (error) *error = last_error_;
    return false;
  }

  models_.clear();

  try {
    nlohmann::json root = nlohmann::json::parse(file);

    // Parse registry metadata
    if (root.contains("updated_at")) {
      last_load_time_ = root["updated_at"].get<std::string>();
    }

    // Parse models
    if (root.contains("models") && root["models"].is_object()) {
      for (auto& [model_id, model_json] : root["models"].items()) {
        ModelMetadata model;
        if (ParseModel(model_json, &model)) {
          model.model_id = model_id;  // Ensure ID matches key
          models_.push_back(std::move(model));
        }
      }
    }

    LOG_INFO("RegistryReader: Successfully loaded " + std::to_string(models_.size()) + " models");
    return true;

  } catch (const nlohmann::json::exception& e) {
    last_error_ = std::string("JSON parse error: ") + e.what();
    if (error) *error = last_error_;
    return false;
  }
}

bool RegistryReader::Reload(std::string* error) {
  return Load(error);
}

const ModelMetadata* RegistryReader::GetModel(
    const std::string& model_id) const {
  for (const auto& model : models_) {
    if (model.model_id == model_id) {
      return &model;
    }
  }
  return nullptr;
}

std::vector<const ModelMetadata*> RegistryReader::FilterByRole(
    const std::string& role) const {
  std::vector<const ModelMetadata*> result;
  for (const auto& model : models_) {
    if (model.role == role) {
      result.push_back(&model);
    }
  }
  return result;
}

std::vector<const ModelMetadata*> RegistryReader::FilterByLocation(
    const std::string& location) const {
  std::vector<const ModelMetadata*> result;
  for (const auto& model : models_) {
    if (model.locations.count(location) > 0) {
      result.push_back(&model);
    }
  }
  return result;
}

std::vector<const ModelMetadata*> RegistryReader::FilterByBackend(
    const std::string& backend) const {
  std::vector<const ModelMetadata*> result;
  for (const auto& model : models_) {
    for (const auto& b : model.deployed_backends) {
      if (b == backend) {
        result.push_back(&model);
        break;
      }
    }
  }
  return result;
}

bool RegistryReader::ParseModel(const nlohmann::json& json,
                                 ModelMetadata* model) const {
  if (!json.is_object()) return false;

  // Helper to safely get string
  auto get_string = [&](const char* key) -> std::string {
    if (json.contains(key) && json[key].is_string()) {
      return json[key].get<std::string>();
    }
    return "";
  };

  // Helper to safely get optional string
  auto get_optional_string =
      [&](const char* key) -> std::optional<std::string> {
    if (json.contains(key) && json[key].is_string()) {
      return json[key].get<std::string>();
    }
    return std::nullopt;
  };

  // Helper to safely get int
  auto get_int = [&](const char* key) -> int {
    if (json.contains(key) && json[key].is_number()) {
      return json[key].get<int>();
    }
    return 0;
  };

  // Helper to safely get optional float
  auto get_optional_float = [&](const char* key) -> std::optional<float> {
    if (json.contains(key) && json[key].is_number()) {
      return json[key].get<float>();
    }
    return std::nullopt;
  };

  // Identity
  model->model_id = get_string("model_id");
  model->display_name = get_string("display_name");
  model->version = get_string("version");

  // Training info
  model->base_model = get_string("base_model");
  model->role = get_string("role");
  model->group = get_string("group");
  model->training_date = get_string("training_date");
  model->training_duration_minutes = get_int("training_duration_minutes");

  // Dataset info
  model->dataset_name = get_string("dataset_name");
  model->dataset_path = get_string("dataset_path");
  model->train_samples = get_int("train_samples");
  model->val_samples = get_int("val_samples");
  model->test_samples = get_int("test_samples");

  // Dataset quality
  if (json.contains("dataset_quality") && json["dataset_quality"].is_object()) {
    const auto& q = json["dataset_quality"];
    if (q.contains("acceptance_rate") && q["acceptance_rate"].is_number()) {
      model->dataset_quality.acceptance_rate = q["acceptance_rate"].get<float>();
    }
    if (q.contains("rejection_rate") && q["rejection_rate"].is_number()) {
      model->dataset_quality.rejection_rate = q["rejection_rate"].get<float>();
    }
    if (q.contains("avg_diversity") && q["avg_diversity"].is_number()) {
      model->dataset_quality.avg_diversity = q["avg_diversity"].get<float>();
    }
  }

  // Training metrics
  model->final_loss = get_optional_float("final_loss");
  model->best_loss = get_optional_float("best_loss");
  model->eval_loss = get_optional_float("eval_loss");
  model->perplexity = get_optional_float("perplexity");

  // Hardware
  model->hardware = get_string("hardware");
  model->device = get_string("device");

  // Files
  model->model_path = get_string("model_path");
  model->checkpoint_path = get_optional_string("checkpoint_path");
  model->adapter_path = get_optional_string("adapter_path");

  // Formats
  if (json.contains("formats") && json["formats"].is_array()) {
    for (const auto& fmt : json["formats"]) {
      if (fmt.is_string()) {
        model->formats.push_back(fmt.get<std::string>());
      }
    }
  }

  // Locations
  if (json.contains("locations") && json["locations"].is_object()) {
    for (auto& [loc, path] : json["locations"].items()) {
      if (path.is_string()) {
        model->locations[loc] = path.get<std::string>();
      }
    }
  }
  model->primary_location = get_string("primary_location");

  // Serving
  if (json.contains("deployed_backends") &&
      json["deployed_backends"].is_array()) {
    for (const auto& backend : json["deployed_backends"]) {
      if (backend.is_string()) {
        model->deployed_backends.push_back(backend.get<std::string>());
      }
    }
  }
  model->ollama_model_name = get_optional_string("ollama_model_name");
  model->halext_node_id = get_optional_string("halext_node_id");

  // Metadata
  model->git_commit = get_optional_string("git_commit");
  model->notes = get_string("notes");

  if (json.contains("tags") && json["tags"].is_array()) {
    for (const auto& tag : json["tags"]) {
      if (tag.is_string()) {
        model->tags.push_back(tag.get<std::string>());
      }
    }
  }

  model->created_at = get_string("created_at");
  model->updated_at = get_string("updated_at");

  return true;
}

}  // namespace studio
}  // namespace afs

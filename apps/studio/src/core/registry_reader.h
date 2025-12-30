#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace afs {
namespace studio {

// Model location types
enum class ModelLocation { kMac, kWindows, kCloud, kHalext };

// Serving backend types
enum class ServingBackend { kOllama, kLlamaCpp, kVllm, kTransformers, kHalextNode };

// Dataset quality metrics
struct DatasetQuality {
  float acceptance_rate = 0.0f;
  float rejection_rate = 0.0f;
  float avg_diversity = 0.0f;
};

// Model metadata matching Python ModelMetadata dataclass
struct ModelMetadata {
  // Identity
  std::string model_id;
  std::string display_name;
  std::string version;

  // Training info
  std::string base_model;
  std::string role;
  std::string group;
  std::string training_date;
  int training_duration_minutes = 0;

  // Dataset info
  std::string dataset_name;
  std::string dataset_path;
  int train_samples = 0;
  int val_samples = 0;
  int test_samples = 0;
  DatasetQuality dataset_quality;

  // Training metrics
  std::optional<float> final_loss;
  std::optional<float> best_loss;
  std::optional<float> eval_loss;
  std::optional<float> perplexity;

  // Hardware
  std::string hardware;
  std::string device;

  // Files
  std::string model_path;
  std::optional<std::string> checkpoint_path;
  std::optional<std::string> adapter_path;

  // Formats available
  std::vector<std::string> formats;

  // Locations (location -> path)
  std::map<std::string, std::string> locations;
  std::string primary_location;

  // Serving
  std::vector<std::string> deployed_backends;
  std::optional<std::string> ollama_model_name;
  std::optional<std::string> halext_node_id;

  // Metadata
  std::optional<std::string> git_commit;
  std::string notes;
  std::vector<std::string> tags;
  std::string created_at;
  std::string updated_at;
};

// Registry reader class
class RegistryReader {
 public:
  RegistryReader();
  explicit RegistryReader(const std::filesystem::path& registry_path);

  // Load registry from disk
  bool Load(std::string* error = nullptr);

  // Reload registry (refresh from disk)
  bool Reload(std::string* error = nullptr);

  // Get all models
  const std::vector<ModelMetadata>& GetModels() const { return models_; }

  // Get model by ID
  const ModelMetadata* GetModel(const std::string& model_id) const;

  // Filter models by role
  std::vector<const ModelMetadata*> FilterByRole(const std::string& role) const;

  // Filter models by location
  std::vector<const ModelMetadata*> FilterByLocation(
      const std::string& location) const;

  // Filter models by backend
  std::vector<const ModelMetadata*> FilterByBackend(
      const std::string& backend) const;

  // Get registry path
  const std::filesystem::path& GetPath() const { return registry_path_; }

  // Check if registry exists
  bool Exists() const;

  // Get last load time
  const std::string& GetLastLoadTime() const { return last_load_time_; }

  // Get last error
  const std::string& GetLastError() const { return last_error_; }

 private:
  std::filesystem::path ResolveDefaultPath() const;
  bool ParseModel(const nlohmann::json& json, ModelMetadata* model) const;

  std::filesystem::path registry_path_;
  std::vector<ModelMetadata> models_;
  std::string last_load_time_;
  std::string last_error_;
};

}  // namespace studio
}  // namespace afs

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace afs {
namespace viz {

/// Quality trend for a single domain/metric.
struct QualityTrendData {
  std::string domain;
  std::string metric;
  std::vector<float> values;
  float mean = 0.0f;
  std::string trend_direction;  // "improving", "declining", "stable", "insufficient"
};

/// Statistics for a single generator.
struct GeneratorStatsData {
  std::string name;
  int samples_generated = 0;
  int samples_accepted = 0;
  int samples_rejected = 0;
  std::map<std::string, int> rejection_reasons;
  float acceptance_rate = 0.0f;
  float avg_quality = 0.0f;
};

/// Embedding space region data.
struct EmbeddingRegionData {
  int index = 0;
  int sample_count = 0;
  std::string domain;
  float avg_quality = 0.0f;
};

/// Training run metadata.
struct TrainingRunData {
  std::string run_id;
  std::string model_name;
  std::string base_model;
  std::string dataset_path;
  std::string start_time;
  std::string end_time;
  std::string notes;
  float final_loss = 0.0f;
  int samples_count = 0;
  std::map<std::string, int> domain_distribution;
  std::map<std::string, float> eval_metrics;
};

/// Embedding coverage summary.
struct CoverageData {
  int total_samples = 0;
  int num_regions = 0;
  float coverage_score = 0.0f;
  int sparse_regions = 0;
  std::map<std::string, float> domain_coverage;
};

/// Aggregated rejection reasons across all generators.
struct RejectionSummary {
  std::map<std::string, int> reasons;
  int total_rejections = 0;
};

/// Curated hack entry metadata.
struct CuratedHackEntry {
  std::string name;
  std::string path;
  std::vector<std::string> authors;
  std::string notes;
  std::string review_status;
  float weight = 1.0f;
  std::vector<std::string> include_globs;
  std::vector<std::string> exclude_globs;
  int eligible_files = 0;
  int selected_files = 0;
  float org_ratio = 0.0f;
  float address_ratio = 0.0f;
  float avg_comment_ratio = 0.0f;
  std::vector<std::string> sample_files;
  std::string status;
  std::string error;
};

/// Resource index summary for data sources.
struct ResourceIndexData {
  int total_files = 0;
  int duplicates_found = 0;
  float duration_seconds = 0.0f;
  std::string indexed_at;
  std::map<std::string, int> by_source;
  std::map<std::string, int> by_type;
};

/// Dataset registry entry.
struct DatasetEntry {
  std::string name;
  std::string path;
  std::uint64_t size_bytes = 0;
  std::string updated_at;
  std::vector<std::string> files;
};

/// Dataset registry summary.
struct DatasetRegistryData {
  std::string generated_at;
  std::vector<DatasetEntry> datasets;
};

/// Context graph edge.
struct ContextGraphEdge {
  int from = -1;
  int to = -1;
};

/// Context graph data (contexts + mounts).
struct ContextGraphData {
  std::string source_path;
  int context_count = 0;
  int mount_count = 0;
  std::vector<std::string> labels;
  std::vector<float> nodes_x;
  std::vector<float> nodes_y;
  std::vector<ContextGraphEdge> edges;
};

/// Optimization metrics.
struct OptimizationData {
  std::map<std::string, float> domain_effectiveness;
  std::map<std::string, float> threshold_sensitivity;
};

/// Local mount information for the system.
struct MountData {
  std::string name;
  std::string path;
  bool active = false;
};

/// Data load status for the last refresh cycle.
struct LoadStatus {
  bool quality_found = false;
  bool quality_ok = false;
  bool active_found = false;
  bool active_ok = false;
  bool training_found = false;
  bool training_ok = false;
  bool resource_index_found = false;
  bool resource_index_ok = false;
  bool dataset_registry_found = false;
  bool dataset_registry_ok = false;
  bool context_graph_found = false;
  bool context_graph_ok = false;
  int error_count = 0;
  std::string last_error;
  std::string last_error_source;

  int FoundCount() const {
    return static_cast<int>(quality_found) + static_cast<int>(active_found) +
           static_cast<int>(training_found) + static_cast<int>(resource_index_found) +
           static_cast<int>(dataset_registry_found) + static_cast<int>(context_graph_found);
  }
  int OkCount() const {
    return static_cast<int>(quality_ok) + static_cast<int>(active_ok) +
           static_cast<int>(training_ok) + static_cast<int>(resource_index_ok) +
           static_cast<int>(dataset_registry_ok) + static_cast<int>(context_graph_ok);
  }
  bool AnyOk() const {
    return quality_ok || active_ok || training_ok || resource_index_ok ||
           dataset_registry_ok || context_graph_ok;
  }
};

/// Loads training data from JSON files.
class DataLoader {
 public:
  using FileReader = std::function<bool(const std::string&,
                                        std::string*,
                                        std::string*)>;
  using PathExists = std::function<bool(const std::string&)>;

  explicit DataLoader(const std::string& data_path,
                      FileReader file_reader = {},
                      PathExists path_exists = {});

  /// Reload all data from disk. Returns true on success.
  bool Refresh();

  // Accessors
  const std::vector<QualityTrendData>& GetQualityTrends() const {
    return quality_trends_;
  }
  const std::vector<GeneratorStatsData>& GetGeneratorStats() const {
    return generator_stats_;
  }
  const std::vector<EmbeddingRegionData>& GetEmbeddingRegions() const {
    return embedding_regions_;
  }
  const std::vector<TrainingRunData>& GetTrainingRuns() const {
    return training_runs_;
  }
  const CoverageData& GetCoverage() const { return coverage_; }
  const RejectionSummary& GetRejectionSummary() const {
    return rejection_summary_;
  }
  const OptimizationData& GetOptimizationData() const {
    return optimization_data_;
  }
  const std::vector<CuratedHackEntry>& GetCuratedHacks() const {
    return curated_hacks_;
  }
  const std::string& GetCuratedHacksError() const {
    return curated_hacks_error_;
  }
  const ResourceIndexData& GetResourceIndex() const { return resource_index_; }
  const std::string& GetResourceIndexError() const { return resource_index_error_; }
  const DatasetRegistryData& GetDatasetRegistry() const { return dataset_registry_; }
  const std::string& GetDatasetRegistryError() const { return dataset_registry_error_; }
  const ContextGraphData& GetContextGraph() const { return context_graph_; }
  const std::string& GetContextGraphError() const { return context_graph_error_; }
  const std::vector<MountData>& GetMounts() const {
    return mounts_;
  }
  const std::map<std::string, bool>& GetDomainVisibility() const { return domain_visibility_; }
  const LoadStatus& GetLastStatus() const { return last_status_; }

  bool HasData() const { return has_data_; }
  std::string GetLastError() const { return last_error_; }

  /// Trigger mounting of external drives.
  void MountDrive(const std::string& name);

 private:
  struct LoadResult {
    bool found = false;
    bool ok = false;
    std::string error;
  };

  LoadResult LoadQualityFeedback(std::vector<QualityTrendData>* quality_trends,
                                 std::vector<GeneratorStatsData>* generator_stats,
                                 RejectionSummary* rejection_summary);
  LoadResult LoadActiveLearning(
      std::vector<EmbeddingRegionData>* embedding_regions,
      CoverageData* coverage);
  LoadResult LoadTrainingFeedback(std::vector<TrainingRunData>* training_runs,
                                  OptimizationData* optimization_data);
  LoadResult LoadCuratedHacks(std::vector<CuratedHackEntry>* curated_hacks);
  LoadResult LoadResourceIndex(ResourceIndexData* resource_index);
  LoadResult LoadDatasetRegistry(DatasetRegistryData* dataset_registry);
  LoadResult LoadContextGraph(ContextGraphData* context_graph);

  std::string data_path_;
  FileReader file_reader_;
  PathExists path_exists_;
  bool has_data_ = false;
  std::string last_error_;
  LoadStatus last_status_;

  std::vector<QualityTrendData> quality_trends_;
  std::vector<GeneratorStatsData> generator_stats_;
  std::vector<EmbeddingRegionData> embedding_regions_;
  std::vector<TrainingRunData> training_runs_;
  CoverageData coverage_;
  RejectionSummary rejection_summary_;
  OptimizationData optimization_data_;
  std::vector<CuratedHackEntry> curated_hacks_;
  std::string curated_hacks_error_;
  ResourceIndexData resource_index_;
  std::string resource_index_error_;
  DatasetRegistryData dataset_registry_;
  std::string dataset_registry_error_;
  ContextGraphData context_graph_;
  std::string context_graph_error_;
  std::vector<MountData> mounts_;
  std::map<std::string, bool> domain_visibility_;
};

}  // namespace viz
}  // namespace afs

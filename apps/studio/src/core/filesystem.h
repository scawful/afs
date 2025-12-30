#pragma once

#include <filesystem>
#include <string>
#include <optional>
#include <nlohmann/json.hpp>

namespace afs {
namespace studio {
namespace core {

class FileSystem {
public:
    /// Resolve a home-relative path (e.g., ~/.context) to an absolute path.
    static std::filesystem::path ResolvePath(const std::string& path_str);

    /// Check if a path exists.
    static bool Exists(const std::filesystem::path& path);

    /// Read the entire content of a file as a string.
    static std::optional<std::string> ReadFile(const std::filesystem::path& path);

    /// Read and parse a JSON file.
    static std::optional<nlohmann::json> ReadJson(const std::filesystem::path& path, std::string* error = nullptr);

    /// Write a string to a file.
    static bool WriteFile(const std::filesystem::path& path, const std::string& content);

    /// Ensure a directory exists.
    static bool EnsureDirectory(const std::filesystem::path& path);
};

} // namespace core
} // namespace studio
} // namespace afs

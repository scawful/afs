#include "filesystem.h"
#include "logger.h"

#include <fstream>
#include <sstream>
#include <cstdlib>

namespace afs {
namespace studio {
namespace core {

std::filesystem::path FileSystem::ResolvePath(const std::string& path_str) {
    if (path_str.empty()) return {};
    
    if (path_str[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            return std::filesystem::path(home) / path_str.substr(2);
        }
    }
    
    return std::filesystem::path(path_str);
}

bool FileSystem::Exists(const std::filesystem::path& path) {
    try {
        std::error_code ec;
        bool exists = std::filesystem::exists(path, ec);
        if (ec) {
            // "Device not configured" and other disk errors should just return false
            return false;
        }
        return exists;
    } catch (...) {
        return false;
    }
}

std::optional<std::string> FileSystem::ReadFile(const std::filesystem::path& path) {
    if (!Exists(path)) {
        LOG_WARN("File not found: " + path.string());
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file: " + path.string());
        return std::nullopt;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::optional<nlohmann::json> FileSystem::ReadJson(const std::filesystem::path& path, std::string* error) {
    auto content = ReadFile(path);
    if (!content) {
        if (error) *error = "Could not read file";
        return std::nullopt;
    }

    try {
        return nlohmann::json::parse(*content);
    } catch (const nlohmann::json::exception& e) {
        std::string err = "JSON parse error in " + path.string() + ": " + e.what();
        LOG_ERROR(err);
        if (error) *error = err;
        return std::nullopt;
    }
}

bool FileSystem::WriteFile(const std::filesystem::path& path, const std::string& content) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for writing: " + path.string());
        return false;
    }
    file << content;
    return true;
}

bool FileSystem::EnsureDirectory(const std::filesystem::path& path) {
    if (Exists(path)) return true;
    try {
        return std::filesystem::create_directories(path);
    } catch (const std::filesystem::filesystem_error& e) {
        LOG_ERROR("Failed to create directory " + path.string() + ": " + e.what());
        return false;
    }
}

} // namespace core
} // namespace studio
} // namespace afs

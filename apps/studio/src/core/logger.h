#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <iostream>

namespace afs {
namespace studio {
namespace core {

enum class LogLevel {
    kTrace,
    kDebug,
    kInfo,
    kWarn,
    kError
};

struct LogEntry {
    LogLevel level;
    std::string message;
    std::string timestamp;
};

class Logger {
public:
    static Logger& GetInstance();

    void Log(LogLevel level, const std::string& message);
    
    void Trace(const std::string& message) { Log(LogLevel::kTrace, message); }
    void Debug(const std::string& message) { Log(LogLevel::kDebug, message); }
    void Info(const std::string& message) { Log(LogLevel::kInfo, message); }
    void Warn(const std::string& message) { Log(LogLevel::kWarn, message); }
    void Error(const std::string& message) { Log(LogLevel::kError, message); }

    const std::vector<LogEntry>& GetEntries() const { return entries_; }
    void Clear() { std::lock_guard<std::mutex> lock(mutex_); entries_.clear(); }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string GetTimestamp();
    const char* LevelToString(LogLevel level);

    std::vector<LogEntry> entries_;
    mutable std::mutex mutex_;
};

// Global convenience macros
#define LOG_TRACE(msg) ::afs::studio::core::Logger::GetInstance().Trace(msg)
#define LOG_DEBUG(msg) ::afs::studio::core::Logger::GetInstance().Debug(msg)
#define LOG_INFO(msg)  ::afs::studio::core::Logger::GetInstance().Info(msg)
#define LOG_WARN(msg)  ::afs::studio::core::Logger::GetInstance().Warn(msg)
#define LOG_ERROR(msg) ::afs::studio::core::Logger::GetInstance().Error(msg)

} // namespace core
} // namespace studio
} // namespace afs

#include "logger.h"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace afs {
namespace studio {
namespace core {

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::Log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string ts = GetTimestamp();
    entries_.push_back({level, message, ts});
    
    // Output to console as well
    std::ostream& out = (level == LogLevel::kError || level == LogLevel::kWarn) ? std::cerr : std::cout;
    out << "[" << ts << "] [" << LevelToString(level) << "] " << message << std::endl;

    // Cap entries to prevent memory leaks in long-running app
    if (entries_.size() > 1000) {
        entries_.erase(entries_.begin(), entries_.begin() + 100);
    }
}

std::string Logger::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%H:%M:%S");
    return ss.str();
}

const char* Logger::LevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::kTrace: return "TRACE";
        case LogLevel::kDebug: return "DEBUG";
        case LogLevel::kInfo:  return "INFO";
        case LogLevel::kWarn:  return "WARN";
        case LogLevel::kError: return "ERROR";
        default: return "UNKNOWN";
    }
}

} // namespace core
} // namespace studio
} // namespace afs

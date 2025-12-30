#pragma once

namespace afs {
namespace studio {
namespace core {

#ifndef AFS_STUDIO_VERSION
#define AFS_STUDIO_VERSION "0.0.0"
#endif

inline const char* StudioVersion() { return AFS_STUDIO_VERSION; }

}  // namespace core
}  // namespace studio
}  // namespace afs

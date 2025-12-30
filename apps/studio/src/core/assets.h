#pragma once

#include <string>
#include <filesystem>
#include <imgui.h>

namespace afs {
namespace studio {
namespace core {

class AssetLoader {
public:
    struct Fonts {
        ImFont* ui = nullptr;
        ImFont* header = nullptr;
        ImFont* mono = nullptr;
        ImFont* icons = nullptr;
    };

    static Fonts LoadFonts();

private:
    static std::filesystem::path FindFontDir();
};

} // namespace core
} // namespace studio
} // namespace afs

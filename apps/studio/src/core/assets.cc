#include "assets.h"
#include "logger.h"
#include "icons.h"

#include <vector>

namespace afs {
namespace studio {
namespace core {

AssetLoader::Fonts AssetLoader::LoadFonts() {
    ImGuiIO& io = ImGui::GetIO();
    std::filesystem::path font_dir = FindFontDir();
    
    if (font_dir.empty()) {
        LOG_WARN("Could not find font directory, using default ImGui font.");
        return Fonts{};
    }

    Fonts fonts;
    float base_size = 15.0f;
    float header_size = 18.0f;
    float mono_size = 14.0f;

    auto MergeIcons = [&](float size) {
        static const ImWchar icons_ranges[] = { (ImWchar)ICON_MIN_MD, (ImWchar)0xFFFF, 0 };
        ImFontConfig icons_config;
        icons_config.MergeMode = true;
        icons_config.PixelSnapH = true;
        icons_config.GlyphMinAdvanceX = size;
        return io.Fonts->AddFontFromFileTTF((font_dir / "MaterialIcons-Regular.ttf").string().c_str(), size, &icons_config, icons_ranges);
    };

    fonts.ui = io.Fonts->AddFontFromFileTTF((font_dir / "Karla-Regular.ttf").string().c_str(), base_size);
    MergeIcons(base_size);
    
    fonts.header = io.Fonts->AddFontFromFileTTF((font_dir / "Roboto-Medium.ttf").string().c_str(), header_size);
    MergeIcons(header_size);
    
    fonts.mono = io.Fonts->AddFontFromFileTTF((font_dir / "Cousine-Regular.ttf").string().c_str(), mono_size);
    MergeIcons(mono_size);
    
    fonts.icons = fonts.ui;

    LOG_INFO("AssetLoader: Successfully loaded fonts from " + font_dir.string());
    return fonts;
}

std::filesystem::path AssetLoader::FindFontDir() {
    std::filesystem::path current = std::filesystem::current_path();
    std::vector<std::filesystem::path> search_paths = {
        current / "assets" / "font",
        current / "src" / "assets" / "font",
        current / ".." / ".." / ".." / "apps" / "studio" / "src" / "assets" / "font",
        current / ".." / ".." / "apps" / "studio" / "src" / "assets" / "font"
    };

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path / "Roboto-Medium.ttf")) {
            return path;
        }
    }
    return "";
}

} // namespace core
} // namespace studio
} // namespace afs

#pragma once

#include <string>
#include <imgui.h>

struct GLFWwindow;
struct ImPlotContext;

namespace afs {
namespace studio {
namespace core {

class GraphicsContext {
public:
    GraphicsContext(const std::string& title, int width, int height);
    ~GraphicsContext();

    bool IsValid() const { return window_ != nullptr; }
    GLFWwindow* GetWindow() const { return window_; }
    
    bool ShouldClose() const;
    void SwapBuffers();
    void PollEvents();

private:
    bool InitGLFW(const std::string& title, int width, int height);
    bool InitImGui();
    void Shutdown();

    GLFWwindow* window_ = nullptr;
    ImGuiContext* imgui_ctx_ = nullptr;
    ImPlotContext* implot_ctx_ = nullptr;
};

} // namespace core
} // namespace studio
} // namespace afs

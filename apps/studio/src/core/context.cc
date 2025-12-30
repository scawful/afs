#include "context.h"
#include "logger.h"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

namespace afs {
namespace studio {
namespace core {

namespace {
void GlfwErrorCallback(int error, const char* description) {
    LOG_ERROR("GLFW Error " + std::to_string(error) + ": " + description);
}
} // namespace

GraphicsContext::GraphicsContext(const std::string& title, int width, int height) {
    if (!InitGLFW(title, width, height)) return;
    if (!InitImGui()) return;
}

GraphicsContext::~GraphicsContext() {
    Shutdown();
}

bool GraphicsContext::InitGLFW(const std::string& title, int width, int height) {
    glfwSetErrorCallback(GlfwErrorCallback);
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        LOG_ERROR("Failed to create GLFW window.");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    return true;
}

bool GraphicsContext::InitImGui() {
    IMGUI_CHECKVERSION();
    imgui_ctx_ = ImGui::CreateContext();
    implot_ctx_ = ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    if (!ImGui_ImplGlfw_InitForOpenGL(window_, true)) return false;
    if (!ImGui_ImplOpenGL3_Init("#version 150")) return false;

    return true;
}

bool GraphicsContext::ShouldClose() const {
    return window_ ? glfwWindowShouldClose(window_) : true;
}

void GraphicsContext::SwapBuffers() {
    if (window_) glfwSwapBuffers(window_);
}

void GraphicsContext::PollEvents() {
    glfwPollEvents();
}

void GraphicsContext::Shutdown() {
    if (imgui_ctx_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext(implot_ctx_);
        ImGui::DestroyContext(imgui_ctx_);
    }
    if (window_) {
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
}

} // namespace core
} // namespace studio
} // namespace afs

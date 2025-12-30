cmake_minimum_required(VERSION 3.15)
project(afs_studio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# M1/ARM64 optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  add_compile_options(-march=armv8-a+simd+dotprod -O3 -ffast-math)
  add_compile_definitions(AFS_ARM64=1)
endif()

# =============================================================================
# Dependencies
# =============================================================================

option(AFS_FETCH_GLFW "Fetch GLFW automatically if missing" ON)

# Fetch Dear ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)

# Fetch ImPlot
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG master
)
FetchContent_MakeAvailable(implot)

# Fetch nlohmann/json (header-only)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Find GLFW
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND AND AFS_FETCH_GLFW)
  message(STATUS "glfw3 not found - fetching with FetchContent")
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
  )
  FetchContent_MakeAvailable(glfw)
  set(glfw3_FOUND TRUE)
endif()
if(NOT glfw3_FOUND)
  message(FATAL_ERROR "glfw3 not found. Install it or set AFS_FETCH_GLFW=ON.")
endif()

find_package(OpenGL REQUIRED)

# =============================================================================
# Build afs_studio
# =============================================================================

# ImGui sources
set(IMGUI_SOURCES
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# ImPlot sources
set(IMPLOT_SOURCES
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  ${implot_SOURCE_DIR}/implot_demo.cpp
)

# afs_studio application
add_executable(afs_studio
  src/main.cc
  src/app.cc
  src/data_loader.cc
  src/core/registry_reader.cc
  src/core/training_monitor.cc
  src/core/deployment_actions.cc
  src/core/filesystem.cc
  src/core/logger.cc
  src/core/context.cc
  src/core/assets.cc
  src/core/llama_client.cc
  src/ui/core.cc
  src/ui/components/metrics.cc
  src/ui/components/charts.cc
  src/ui/components/tabs.cc
  src/ui/components/panels.cc
  src/ui/components/model_registry.cc
  src/ui/components/training_dashboard.cc
  src/ui/components/deployment_panel.cc
  src/ui/components/comparison_view.cc
  src/ui/components/graph_browser.cc
  src/ui/components/graph_navigator.cc
  src/ui/components/companion_panels.cc
  src/ui/charts/quality_trends.cc
  src/ui/charts/generator_efficiency.cc
  src/ui/charts/coverage_density.cc
  src/ui/panels/chat_panel.cc
  src/ui/shortcuts.cc
  src/widgets/text_editor.cc
  src/widgets/sample_review.cc
  ${IMGUI_SOURCES}
  ${IMPLOT_SOURCES}
)

target_include_directories(afs_studio PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${implot_SOURCE_DIR}
)

target_compile_definitions(afs_studio PRIVATE
  IMGUI_IMPL_OPENGL_LOADER_GLAD=0
)

if(TARGET OpenGL::OpenGL)
  set(AFS_OPENGL_TARGET OpenGL::OpenGL)
else()
  set(AFS_OPENGL_TARGET OpenGL::GL)
endif()

target_link_libraries(afs_studio PRIVATE
  glfw
  ${AFS_OPENGL_TARGET}
  nlohmann_json::nlohmann_json
)

# macOS specific
if(APPLE)
  target_link_libraries(afs_studio PRIVATE
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
  )
endif()

# Install
install(TARGETS afs_studio RUNTIME DESTINATION bin)

message(STATUS "afs_studio will be built")

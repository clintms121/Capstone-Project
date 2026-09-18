cmake_minimum_required(VERSION 3.16)
project(CapstoneProject CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# --- GLFW ---
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- Dear ImGui ---
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.1
)
FetchContent_MakeAvailable(imgui)

set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# --- Executable ---
add_executable(capstone
    graphics/main.cpp
    graphics/fonts.cpp
    graphics/main_screen.cpp
    graphics/menu_screen.cpp
    graphics/research_screen.cpp
    graphics/under_the_hood_screen.cpp
    graphics/spacetime_simulation_screen.cpp
    graphics/simulation_screen.cpp
    graphics/manifest.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(capstone PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(capstone PRIVATE glfw opengl32 shell32)

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

# --- stb_image (header-only JPEG/PNG loader for the slide images in imgs/) ---
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        2c980bb59875b0d32144a71867fbdebb2f77cd20
)
FetchContent_MakeAvailable(stb)

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
    graphics/back_button.cpp
    graphics/main_screen.cpp
    graphics/menu_screen.cpp
    graphics/research_screen.cpp
    graphics/under_the_hood_screen.cpp
    graphics/cnn_animation.cpp
    graphics/earth_scene.cpp
    graphics/spacetime_simulation_screen.cpp
    graphics/simulation_screen.cpp
    graphics/trajectory.cpp
    graphics/merger_scene.cpp
    graphics/demo_screen.cpp
    graphics/demo_data.cpp
    graphics/manifest.cpp
    graphics/image_texture.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(capstone PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${stb_SOURCE_DIR}
)

target_link_libraries(capstone PRIVATE glfw opengl32 glu32 shell32)

# With MinGW, build the C++ runtime into the .exe so it runs on its own
# (otherwise it needs MSYS2's DLLs on PATH to start).
if(MINGW)
    target_link_options(capstone PRIVATE -static)
endif()

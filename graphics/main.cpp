// Program entry point: opens the window, loads the fonts and the data the
// screens show, then runs the main loop, drawing whichever screen is current.
// Each screen's render function draws one frame and returns the screen to
// show next (itself, unless a button was clicked).
#include "screen.h"
#include "simulation_params.h"
#include "manifest.h"
#include "demo_data.h"
#include "fonts.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <vector>

// One render function per screen, each in its own *_screen.cpp file.
Screen render_title_screen(GLFWwindow* window);
Screen render_menu_screen(GLFWwindow* window);
Screen render_research_screen(GLFWwindow* window);
Screen render_under_the_hood_screen(GLFWwindow* window);
Screen render_spacetime_simulation_screen(GLFWwindow* window);
Screen render_demo_screen(GLFWwindow* window, DemoData& demo,
                          const std::vector<SimulationParams>& available_events,
                          SimulationParams& selected);
Screen render_simulation_screen(GLFWwindow* window, SimulationParams& selected);

// The app loads fonts/, imgs/ and data/ from the project folder. If it was
// started from somewhere else (say, by double-clicking build\capstone.exe),
// look upward from the .exe for the folder that has fonts/ in it and move there.
static void change_to_project_folder(const char* exe_path) {
    namespace fs = std::filesystem;
    std::error_code error;
    if (fs::exists("fonts", error))
        return;
    fs::path dir = fs::absolute(exe_path, error).parent_path();
    for (int levels = 0; levels < 4 && !dir.empty(); ++levels, dir = dir.parent_path()) {
        if (fs::exists(dir / "fonts", error)) {
            fs::current_path(dir, error);
            return;
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 0)
        change_to_project_folder(argv[0]);

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // The "compatibility" profile (rather than "core") keeps the old OpenGL
    // drawing calls available. The 3D scenes (earth_scene.cpp,
    // merger_scene.cpp) use them, while ImGui works either way.
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    // 4x multisampling: smooth edges on the 3D spheres and grid lines.
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Capstone Project", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // wait for the monitor's refresh (vsync)

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    load_fonts();

    // Results computed ahead of time by the Python pipeline (see README.md).
    std::vector<SimulationParams> available_events = load_trajectory_manifest("data/trajectories/index.csv");
    DemoData demo = load_demo_data("data/demo");

    Screen current = Screen::Title;
    SimulationParams selected_event; // the merger picked in the Classifier, for the simulation

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        switch (current) {
        case Screen::Title:
            current = render_title_screen(window);
            break;
        case Screen::Menu:
            current = render_menu_screen(window);
            break;
        case Screen::Research:
            current = render_research_screen(window);
            break;
        case Screen::UnderTheHood:
            current = render_under_the_hood_screen(window);
            break;
        case Screen::SpacetimeSimulation:
            current = render_spacetime_simulation_screen(window);
            break;
        case Screen::Demo:
            current = render_demo_screen(window, demo, available_events, selected_event);
            break;
        case Screen::Simulation:
            current = render_simulation_screen(window, selected_event);
            break;
        }

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// sim entry point
#include "screen.h"
#include "simulation_params.h"
#include "manifest.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <vector>

Screen render_title_screen(GLFWwindow* window);
Screen render_menu_screen(GLFWwindow* window);
Screen render_research_screen(GLFWwindow* window);
Screen render_under_the_hood_screen(GLFWwindow* window);
Screen render_simulation_screen(GLFWwindow* window,
                                 const std::vector<SimulationParams>& available_events,
                                 SimulationParams& selected);

int main() {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Capstone Project", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    Screen current = Screen::Title;
    SimulationParams selected_event;
    std::vector<SimulationParams> available_events =
        load_trajectory_manifest("data/trajectories/index.csv");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        switch (current) {
            case Screen::Title:      current = render_title_screen(window);      break;
            case Screen::Menu:       current = render_menu_screen(window);       break;
            case Screen::Research:   current = render_research_screen(window);   break;
            case Screen::UnderTheHood: current = render_under_the_hood_screen(window); break;
            case Screen::Simulation:
                current = render_simulation_screen(window, available_events, selected_event);
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

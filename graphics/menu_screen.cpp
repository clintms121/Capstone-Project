#include "screen.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

// Renders the post-Begin menu: choose between reading about the research
// or jumping into the merger simulation.
Screen render_menu_screen(GLFWwindow* window) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_w, screen_h), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar  |
        ImGuiWindowFlags_NoResize    |
        ImGuiWindowFlags_NoMove      |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings;

    Screen next = Screen::Menu;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##menu", nullptr, flags);

    const char* heading = "Gravitational Wave Classification & Merger Visualization";
    float heading_w = ImGui::CalcTextSize(heading).x;
    ImGui::SetCursorPos(ImVec2((screen_w - heading_w) * 0.5f, screen_h * 0.35f));
    ImGui::Text("%s", heading);

    float btn_w = 200.0f;
    float btn_h = 32.0f;
    float gap = 20.0f;
    float total_w = btn_w * 2 + gap;
    float start_x = (screen_w - total_w) * 0.5f;
    float btn_y = screen_h * 0.35f + 50.0f;

    ImGui::SetCursorPos(ImVec2(start_x, btn_y));
    if (ImGui::Button("View Research", ImVec2(btn_w, btn_h)))
        next = Screen::Research;

    ImGui::SetCursorPos(ImVec2(start_x + btn_w + gap, btn_y));
    if (ImGui::Button("Start Simulation", ImVec2(btn_w, btn_h)))
        next = Screen::Simulation;

    ImGui::End();
    ImGui::PopStyleColor(5);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

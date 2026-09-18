#include "screen.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

// Placeholder for a spacetime-curvature visualization -- e.g. a warped grid
// showing how mass curves space and how a passing gravitational wave
// stretches/squeezes it, illustrating the "How the Data is Collected"
// section of the Under the Hood screen that links here.
Screen render_spacetime_simulation_screen(GLFWwindow* window) {
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

    Screen next = Screen::SpacetimeSimulation;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##spacetime_simulation", nullptr, flags);

    const char* msg = "Spacetime simulation not implemented yet.";
    float msg_w = ImGui::CalcTextSize(msg).x;
    ImGui::SetCursorPos(ImVec2((screen_w - msg_w) * 0.5f, screen_h * 0.45f));
    ImGui::Text("%s", msg);

    float btn_w = 90.0f;
    float btn_h = 28.0f;
    ImGui::SetCursorPos(ImVec2(40.0f, screen_h - btn_h - 40.0f));
    if (ImGui::Button("Back", ImVec2(btn_w, btn_h)))
        next = Screen::UnderTheHood;

    ImGui::End();
    ImGui::PopStyleColor(5);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

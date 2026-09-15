#include "screen.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

// Path to the compiled writeup. Nothing generates this yet -- see
// docs/README.md for where to drop it once it exists.
const char* kResearchPaperPath = "docs/research_paper.pdf";

// Opens the paper in the OS's default PDF viewer. Returns a status string
// to show the user (empty on success).
std::string open_research_paper() {
    if (!std::filesystem::exists(kResearchPaperPath))
        return std::string("Not found: ") + kResearchPaperPath +
               " (see docs/README.md)";

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(nullptr, "open", kResearchPaperPath,
                                      nullptr, nullptr, SW_SHOWNORMAL);
    if (reinterpret_cast<intptr_t>(result) <= 32)
        return "Failed to open the paper with the default PDF viewer.";
    return "";
#else
    return "Opening files is only implemented on Windows right now.";
#endif
}

} // namespace

// Research summary screen. Content here should track the writeup in
// ml/README.md and physics/README.md as the project develops.
Screen render_research_screen(GLFWwindow* window) {
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

    Screen next = Screen::Research;
    static std::string download_status;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##research", nullptr, flags);

    ImGui::SetCursorPos(ImVec2(40.0f, 40.0f));
    ImGui::TextWrapped(
        "Gravitational Wave Classification + Merger Visualization\n\n"
        "1. Classifier: LIGO open strain data (GWOSC) + the Gravity Spy glitch "
        "dataset are turned into Q-transform spectrograms and used to train a "
        "CNN that separates real compact-binary signals from ~20 classes of "
        "detector glitches.\n\n"
        "2. Parameter estimation: chirp mass and mass ratio are estimated from "
        "the identified signal's time-frequency track.\n\n"
        "3. Visualization: the estimated masses drive a post-Newtonian inspiral "
        "trajectory and a damped-sinusoid ringdown, animated in 3D in the "
        "Simulation screen."
    );

    float btn_w = 220.0f;
    float btn_h = 32.0f;
    float gap = 20.0f;
    float total_w = btn_w * 2 + gap;
    float start_x = (screen_w - total_w) * 0.5f;
    float btn_y = screen_h * 0.65f;

    ImGui::SetCursorPos(ImVec2(start_x, btn_y));
    if (ImGui::Button("Download Research Paper", ImVec2(btn_w, btn_h)))
        download_status = open_research_paper();

    ImGui::SetCursorPos(ImVec2(start_x + btn_w + gap, btn_y));
    if (ImGui::Button("Under the Hood", ImVec2(btn_w, btn_h)))
        next = Screen::UnderTheHood;

    if (!download_status.empty()) {
        float status_w = ImGui::CalcTextSize(download_status.c_str()).x;
        ImGui::SetCursorPos(ImVec2((screen_w - status_w) * 0.5f, btn_y + btn_h + 12.0f));
        ImGui::TextWrapped("%s", download_status.c_str());
    }

    float back_w = 90.0f;
    float back_h = 28.0f;
    ImGui::SetCursorPos(ImVec2(40.0f, screen_h - back_h - 40.0f));
    if (ImGui::Button("Back", ImVec2(back_w, back_h))) {
        download_status.clear();
        next = Screen::Menu;
    }

    ImGui::End();
    ImGui::PopStyleColor(5);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

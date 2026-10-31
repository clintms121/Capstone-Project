// The menu after Begin: View Research (the science and how the model works)
// or Classifier (the model's results on real LIGO data).
#include "screen.h"
#include "back_button.h"
#include "fonts.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <string>

namespace {

struct MenuOption {
    const char* label;
    const char* description;
    Screen target;
};

const MenuOption k_options[] = {
    { "View Research",
      "Learn how black holes merge and how LIGO detects the waves they send out.",
      Screen::Research },
    { "Classifier",
      "Run my machine learning model on real LIGO detector data.",
      Screen::Demo }, // the merger simulation opens from here, per event
};

// Draws `text` word-wrapped to `width`, with each line centered on the
// column that starts at `x`. Returns the y just below the last line.
float draw_centered_wrapped(const char* text, float x, float y, float width, ImVec4 color) {
    float line_h = ImGui::GetTextLineHeightWithSpacing();
    std::string line;
    std::string word;
    auto flush = [&]() {
        float w = ImGui::CalcTextSize(line.c_str()).x;
        ImGui::SetCursorPos(ImVec2(x + (width - w) * 0.5f, y));
        ImGui::TextColored(color, "%s", line.c_str());
        y += line_h;
        line.clear();
    };

    for (const char* p = text;; ++p) {
        if (*p == ' ' || *p == '\0') {
            std::string candidate = line.empty() ? word : line + " " + word;
            if (!line.empty() && ImGui::CalcTextSize(candidate.c_str()).x > width) {
                flush();
                line = word;
            } else {
                line = candidate;
            }
            word.clear();
            if (*p == '\0') break;
        } else {
            word += *p;
        }
    }
    if (!line.empty()) flush();
    return y;
}

} // namespace

// Renders the post-Begin menu: read about the research, or try the
// classifier on real data (and from there, watch a merger).
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
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.12f, 0.16f, 0.26f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.20f, 0.28f, 0.45f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.28f, 0.38f, 0.60f, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);

    ImGui::Begin("##menu", nullptr, flags);

    // --- Large option buttons with a short description under each ---
    const int n = (int)(sizeof(k_options) / sizeof(k_options[0]));
    float gap = 40.0f;
    float btn_w = std::min(300.0f, (screen_w - 80.0f - gap * (n - 1)) / n);
    float btn_h = 90.0f;
    float total_w = btn_w * n + gap * (n - 1);
    float start_x = (screen_w - total_w) * 0.5f;
    float btn_y = (screen_h - btn_h) * 0.45f; // roughly centered, leaving room for descriptions

    for (int i = 0; i < n; ++i) {
        const MenuOption& opt = k_options[i];
        float x = start_x + i * (btn_w + gap);

        ImGui::PushFont(g_font_subtitle);
        ImGui::SetCursorPos(ImVec2(x, btn_y));
        if (ImGui::Button(opt.label, ImVec2(btn_w, btn_h)))
            next = opt.target;
        ImGui::PopFont();

        draw_centered_wrapped(opt.description, x, btn_y + btn_h + 12.0f, btn_w,
                              ImVec4(0.75f, 0.75f, 0.75f, 1.0f));
    }

    // --- Back button ---
    if (back_button("< Back to Title"))
        next = Screen::Title;

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(5);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

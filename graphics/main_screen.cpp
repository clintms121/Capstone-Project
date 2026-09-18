#include "screen.h"
#include "fonts.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

struct WarpStar {
    float angle;
    float radius;
    float speed;
};

float rand_float(float lo, float hi) {
    return lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

// Streaks stars outward from the center of the window to suggest flying
// through a star field at warp speed.
void draw_warp_stars(ImDrawList* draw_list, ImVec2 center, float max_radius, float dt) {
    static std::vector<WarpStar> stars;
    if (stars.empty()) {
        stars.resize(250);
        for (auto& s : stars) {
            s.angle = rand_float(0.0f, 6.2831853f);
            s.radius = rand_float(0.0f, max_radius);
            s.speed = rand_float(200.0f, 600.0f);
        }
    }

    for (auto& s : stars) {
        float prev_radius = s.radius;
        s.radius += s.speed * dt;
        if (s.radius > max_radius) {
            s.angle = rand_float(0.0f, 6.2831853f);
            s.radius = 0.0f;
            prev_radius = 0.0f;
        }

        float cos_a = cosf(s.angle);
        float sin_a = sinf(s.angle);
        ImVec2 p0(center.x + cos_a * prev_radius, center.y + sin_a * prev_radius);
        ImVec2 p1(center.x + cos_a * s.radius, center.y + sin_a * s.radius);

        float t = s.radius / max_radius;
        ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, t));
        draw_list->AddLine(p0, p1, color, 1.0f + t * 2.0f);
    }
}

} // namespace

// Renders the title screen for one frame.
// Returns Screen::Menu once the user clicks "Begin", otherwise Screen::Title.
Screen render_title_screen(GLFWwindow* window) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;

    // Full-screen invisible window so we can draw anywhere
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_w, screen_h), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar  |
        ImGuiWindowFlags_NoResize    |
        ImGuiWindowFlags_NoMove      |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings;

    Screen next = Screen::Title;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##title", nullptr, flags);

    // --- Warp-speed star field background ---
    ImVec2 center(screen_w * 0.5f, screen_h * 0.5f);
    float max_radius = sqrtf(screen_w * screen_w + screen_h * screen_h) * 0.5f;
    draw_warp_stars(ImGui::GetWindowDrawList(), center, max_radius, io.DeltaTime);

    // --- Title (baked at 4x pixel size -- crisp, not runtime-scaled) ---
    const char* title = "Capstone Project";
    float title_y = screen_h * 0.28f;
    ImGui::PushFont(g_font_title);
    float title_w = ImGui::CalcTextSize(title).x;
    float title_line_h = ImGui::GetTextLineHeightWithSpacing();
    ImGui::SetCursorPos(ImVec2((screen_w - title_w) * 0.5f, title_y));
    ImGui::Text("%s", title);
    ImGui::PopFont();

    // --- Subtitle (baked at 2x pixel size) ---
    const char* subtitle = "Black Holes and Machine Learning";
    ImGui::PushFont(g_font_subtitle);
    float subtitle_w = ImGui::CalcTextSize(subtitle).x;
    float subtitle_y = title_y + title_line_h + 10.0f;
    ImGui::SetCursorPos(ImVec2((screen_w - subtitle_w) * 0.5f, subtitle_y));
    ImGui::TextColored(ImVec4(0.65f, 0.75f, 1.0f, 1.0f), "%s", subtitle);
    float subtitle_line_h = ImGui::GetTextLineHeightWithSpacing();
    ImGui::PopFont();

    // --- Begin button ---
    float btn_w = 90.0f;
    float btn_h = 28.0f;
    float btn_y = subtitle_y + subtitle_line_h + 20.0f;
    ImGui::SetCursorPos(ImVec2((screen_w - btn_w) * 0.5f, btn_y));
    if (ImGui::Button("Begin", ImVec2(btn_w, btn_h)))
        next = Screen::Menu;

    ImGui::End();
    ImGui::PopStyleColor(5);

    // Render
    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

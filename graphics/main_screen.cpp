// The title screen: the project's name over a warp-speed star field, and
// the Begin button that opens the menu.
#include "screen.h"
#include "fonts.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
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

    // The menu's button blues, like the rest of the app.
    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.12f, 0.16f, 0.26f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.20f, 0.28f, 0.45f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.28f, 0.38f, 0.60f, 1));

    ImGui::Begin("##title", nullptr, flags);

    // --- Warp-speed star field background ---
    ImVec2 center(screen_w * 0.5f, screen_h * 0.5f);
    float max_radius = sqrtf(screen_w * screen_w + screen_h * screen_h) * 0.5f;
    draw_warp_stars(ImGui::GetWindowDrawList(), center, max_radius, io.DeltaTime);

    // --- Title, subtitle and Begin button, centered as one block ---
    // Fonts and sizes are for a 1080p-tall window and shrink to fit smaller
    // ones, so the block fills the same share of any window. Shrinking a
    // baked font stays sharp; stretching one past its size would blur.
    const float scale = std::min(1.0f, screen_h / 1080.0f);

    const char* title = "Black Hole Merger Classifier";
    const char* subtitle = "Capstone Machine Learning Project";

    // On narrow windows the title also shrinks to fit 90% of the width.
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PushFont(g_font_title);
    const float title_scale = std::min(scale, screen_w * 0.9f / ImGui::CalcTextSize(title).x);
    ImGui::PopFont();
    ImGui::SetWindowFontScale(title_scale);
    ImGui::PushFont(g_font_title);
    ImVec2 title_size = ImGui::CalcTextSize(title);
    ImGui::PopFont();
    ImGui::SetWindowFontScale(scale);
    ImGui::PushFont(g_font_tagline);
    ImVec2 subtitle_size = ImGui::CalcTextSize(subtitle);
    ImGui::PopFont();

    const float subtitle_gap = 20.0f * scale;
    const float button_gap = 64.0f * scale;
    const ImVec2 btn_size(floorf(300.0f * scale), floorf(72.0f * scale));
    float block_h = title_size.y + subtitle_gap + subtitle_size.y + button_gap + btn_size.y;
    float y = floorf(screen_h * 0.47f - block_h * 0.5f);

    ImGui::SetWindowFontScale(title_scale);
    ImGui::PushFont(g_font_title);
    ImGui::SetCursorPos(ImVec2(floorf((screen_w - title_size.x) * 0.5f), y));
    ImGui::TextUnformatted(title);
    ImGui::PopFont();
    ImGui::SetWindowFontScale(scale);
    y += title_size.y + subtitle_gap;

    ImGui::PushFont(g_font_tagline);
    ImGui::SetCursorPos(ImVec2(floorf((screen_w - subtitle_size.x) * 0.5f), y));
    ImGui::TextColored(ImVec4(0.62f, 0.71f, 0.86f, 1.0f), "%s", subtitle);
    y += subtitle_size.y + button_gap;

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scale);
    ImGui::SetCursorPos(ImVec2(floorf((screen_w - btn_size.x) * 0.5f), y));
    if (ImGui::Button("Begin", btn_size))
        next = Screen::Menu;
    ImGui::PopStyleVar();
    ImGui::PopFont();

    ImGui::End();
    ImGui::PopStyleColor(5);

    // Render
    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

// The Space Time Simulation screen (described above its render function at
// the bottom), plus the physics behind the numbers it shows.
#include "screen.h"
#include "back_button.h"
#include "earth_scene.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cstdio>

namespace {

constexpr double kEarthRadiusKm = 6371.0;
constexpr double kEarthMeanDensity = 5514.0;         // kg/m^3
constexpr double kEarthMassKg = 5.972e24;
constexpr double kSolarMassKg = 1.989e30;
constexpr double kSchwarzschildRadiusPerSolarMassKm = 2.95; // r_s of the Sun
constexpr double kG = 6.674e-11;                     // m^3 kg^-1 s^-2
constexpr double kC = 2.998e8;                        // m/s

// World-unit-to-km scales chosen so the slider's default (2.0) lands on a
// real reference body: Earth's actual radius in Earth mode, and a
// relatable ~10 solar mass black hole (well within LIGO's detectable
// range) in black hole mode. The two modes are different physical regimes
// -- an ordinary body vs. a compact relativistic object -- so they don't
// share one scale; see real_mass_kg() below for the physics each one uses.
constexpr double kKmPerUnitEarth = kEarthRadiusKm / 2.0;
constexpr double kKmPerUnitBlackHole = (10.0 * kSchwarzschildRadiusPerSolarMassKm) / 2.0;

// The actual mass a body of this type and slider-controlled radius would
// have. Earth: an ordinary body, so mass scales with volume at Earth's
// real mean density (M = 4/3 * pi * R^3 * rho) -- exact at the slider's
// default (R = Earth's real radius -> M = Earth's real mass), not just an
// approximation. Black hole: `size` IS the Schwarzschild radius, so mass
// comes directly from r_s = 2GM/c^2 -- real GR, not a stand-in number.
double real_mass_kg(float size, bool black_hole_mode) {
    if (black_hole_mode) {
        double r_s_m = size * kKmPerUnitBlackHole * 1000.0;
        return r_s_m * kC * kC / (2.0 * kG);
    }
    double r_m = size * kKmPerUnitEarth * 1000.0;
    return (4.0 / 3.0) * 3.14159265358979323846 * r_m * r_m * r_m * kEarthMeanDensity;
}

// Slider ranges. Black holes stop at 3 units (~15 solar masses) so the
// accretion disk, which starts at 3 r_s, still fits on the grid.
constexpr float kEarthSizeMin = 0.5f, kEarthSizeMax = 6.0f;
constexpr float kBlackHoleSizeMin = 0.6f, kBlackHoleSizeMax = 3.0f;

// How strongly an Earth-density body of this size really curves space:
// r_s / R = (8 pi G rho / 3 c^2) R^2. For the real Earth that's ~1.4e-9 --
// a dent about 9 mm deep on a 6,371 km planet, far too small to see.
double earth_compactness(float size) {
    double r_m = size * kKmPerUnitEarth * 1000.0;
    return 8.0 * 3.14159265358979323846 * kG * kEarthMeanDensity / (3.0 * kC * kC) * r_m * r_m;
}

// One fixed exaggeration for Earth mode, chosen so the largest planet on
// the slider reaches r_s / R = 0.6 -- visible, but still under the 8/9
// limit (Buchdahl's) past which no static body can hold itself up. Being
// fixed, bigger planets still dent space more, in the right proportion.
double earth_curvature_exaggeration() {
    return 0.6 / earth_compactness(kEarthSizeMax);
}

} // namespace

// The Space Time Simulation screen, opened from the "Spacetime and
// Gravitational Waves" slide: a 3D body (Earth or a black hole, toggled with
// a checkbox) sitting in a grid that shows how it curves space around it.
// The grid's shape is the standard general-relativity picture, Flamm's
// paraboloid; earth_scene.cpp draws it. A slider sets the body's size, and
// the text above it gives the real mass that size implies.
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
    static float body_size = 2.0f;       // sphere radius in world units; real_mass_kg() above turns it into a mass
    static float cam_azimuth = 0.0f;     // radians, around Y
    static float cam_elevation = 0.616f; // radians, above the XZ plane
    static float cam_distance = 24.0f; // world units from the look-at target
    static bool black_hole_mode = false;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##spacetime_simulation", nullptr, flags);

    // Dark backing panel so the labels stay readable over the glowing disk.
    float panel_w = 620.0f;
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2((screen_w - panel_w) * 0.5f, 18.0f),
                                              ImVec2((screen_w + panel_w) * 0.5f, 166.0f),
                                              IM_COL32(0, 0, 0, 170), 10.0f);

    const char* msg = black_hole_mode
        ? "A black hole curves space into a funnel that ends at its event horizon"
        : "Earth curves space into a gentle bowl";
    float msg_w = ImGui::CalcTextSize(msg).x;
    ImGui::SetCursorPos(ImVec2((screen_w - msg_w) * 0.5f, 30.0f));
    ImGui::Text("%s", msg);

    const char* toggle_label = "Black Hole";
    float toggle_w = ImGui::CalcTextSize(toggle_label).x + ImGui::GetFrameHeight() + 8.0f;
    ImGui::SetCursorPos(ImVec2((screen_w - toggle_w) * 0.5f, 60.0f));
    ImGui::Checkbox(toggle_label, &black_hole_mode);

    const float size_min = black_hole_mode ? kBlackHoleSizeMin : kEarthSizeMin;
    const float size_max = black_hole_mode ? kBlackHoleSizeMax : kEarthSizeMax;
    const float size_step = 0.1f;
    body_size = body_size < size_min ? size_min : (body_size > size_max ? size_max : body_size);
    float slider_w = 260.0f;
    float arrow_w = ImGui::GetFrameHeight();
    float gap = 8.0f;
    float row_w = arrow_w + gap + slider_w + gap + arrow_w;
    ImGui::SetCursorPos(ImVec2((screen_w - row_w) * 0.5f, 90.0f));

    if (ImGui::ArrowButton("##size_dec", ImGuiDir_Left)) {
        body_size -= size_step;
        if (body_size < size_min) body_size = size_min;
    }
    ImGui::SameLine(0.0f, gap);
    ImGui::SetNextItemWidth(slider_w);
    ImGui::SliderFloat("Size", &body_size, size_min, size_max, "%.2f");
    ImGui::SameLine(0.0f, gap);
    if (ImGui::ArrowButton("##size_inc", ImGuiDir_Right)) {
        body_size += size_step;
        if (body_size > size_max) body_size = size_max;
    }

    // Real physical mass implied by that size -- see real_mass_kg() above
    // for why Earth and black hole mode use different physics here.
    double mass_kg = real_mass_kg(body_size, black_hole_mode);
    char mass_text[128];
    if (black_hole_mode)
        snprintf(mass_text, sizeof(mass_text), "Mass: %.3g kg  (%.1f solar masses)",
                 mass_kg, mass_kg / kSolarMassKg);
    else
        snprintf(mass_text, sizeof(mass_text), "Mass: %.3g kg  (%.2f Earth masses)",
                 mass_kg, mass_kg / kEarthMassKg);
    float mass_text_w = ImGui::CalcTextSize(mass_text).x;
    ImGui::SetCursorPos(ImVec2((screen_w - mass_text_w) * 0.5f, 120.0f));
    ImGui::Text("%s", mass_text);

    // How strongly it curves space (r_s = 2GM/c^2, in world units). A black
    // hole's sphere IS its horizon, so that's exact; Earth's real curvature
    // is invisible at this scale, so it's scaled up by a fixed, stated factor.
    float schwarzschild_radius = body_size;
    char curvature_text[160];
    if (black_hole_mode) {
        snprintf(curvature_text, sizeof(curvature_text),
                 "Event horizon radius: %.0f km. Grid drawn to true scale.",
                 body_size * kKmPerUnitBlackHole);
    } else {
        double exaggeration = earth_curvature_exaggeration();
        schwarzschild_radius = (float)(body_size * earth_compactness(body_size) * exaggeration);
        snprintf(curvature_text, sizeof(curvature_text),
                 "Curvature exaggerated %.0f million times -- the real dent is too small to see.",
                 exaggeration / 1e6);
    }
    float curvature_text_w = ImGui::CalcTextSize(curvature_text).x;
    ImGui::SetCursorPos(ImVec2((screen_w - curvature_text_w) * 0.5f, 142.0f));
    ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f), "%s", curvature_text);

    if (back_button("< Back to Under the Hood"))
        next = Screen::UnderTheHood;

    // --- Orbit camera controls: drag to rotate, scroll to zoom ---
    // This is an invisible button covering the whole screen, and it must be
    // added after every other widget: ImGui gives a click to the first
    // widget under the mouse, so added earlier, it would steal clicks from
    // the checkbox, slider and buttons. Last, it only gets clicks on empty
    // space.
    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
    ImGui::InvisibleButton("##viewport", ImVec2(screen_w, screen_h));
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        cam_azimuth -= io.MouseDelta.x * 0.01f;
        cam_elevation += io.MouseDelta.y * 0.01f;
        if (cam_elevation < 0.05f) cam_elevation = 0.05f; // low enough to see the disk nearly edge-on
        if (cam_elevation > 1.45f) cam_elevation = 1.45f;
    }
    if (ImGui::IsItemHovered()) {
        cam_distance -= io.MouseWheel * 1.0f;
        if (cam_distance < 4.0f) cam_distance = 4.0f;
        if (cam_distance > 60.0f) cam_distance = 60.0f;
    }

    ImGui::End();
    ImGui::PopStyleColor(5);

    ImGui::Render();

    glViewport(0, 0, (GLsizei)screen_w, (GLsizei)screen_h);
    glClearColor(0.0f, 0.0f, 0.01f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    render_earth_scene(screen_w, screen_h, (float)glfwGetTime(), body_size, schwarzschild_radius,
                       cam_azimuth, cam_elevation, cam_distance,
                       black_hole_mode ? CelestialBody::BlackHole : CelestialBody::Earth);
    glDisable(GL_DEPTH_TEST);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

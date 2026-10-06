#include "screen.h"
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

} // namespace

// Spacetime-curvature visualization: a rotating 3D body (toggle between
// Earth and a black hole) sitting in a grid warps it downward around
// itself, following Flamm's paraboloid -- the standard GR embedding
// diagram for how a mass curves the spatial metric (see
// graphics/earth_scene.cpp). Real 3D geometry with a real depth-tested
// camera, not a hand-projected 2D approximation -- see main.cpp for the
// compatibility-profile context this needs. Static curvature around a
// fixed mass; animating an orbit or a passing wave through this grid is
// future work. Linked from the "How the Data is Collected" section of the
// Under the Hood screen.
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
    static float object_mass = 2.0f;   // sphere radius in world units -- see earth_scene.h and real_mass_kg() above
    static float cam_azimuth = 0.0f;   // radians, around Y
    static float cam_elevation = 0.616f; // radians, above the XZ plane
    static float cam_distance = 14.7f; // world units from the look-at target
    static bool black_hole_mode = false;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##spacetime_simulation", nullptr, flags);

    const char* msg = black_hole_mode
        ? "A black hole bends the fabric of space (Flamm's paraboloid)"
        : "Earth bends the fabric of space (Flamm's paraboloid)";
    float msg_w = ImGui::CalcTextSize(msg).x;
    ImGui::SetCursorPos(ImVec2((screen_w - msg_w) * 0.5f, 30.0f));
    ImGui::Text("%s", msg);

    const char* toggle_label = "Black Hole";
    float toggle_w = ImGui::CalcTextSize(toggle_label).x + ImGui::GetFrameHeight() + 8.0f;
    ImGui::SetCursorPos(ImVec2((screen_w - toggle_w) * 0.5f, 60.0f));
    ImGui::Checkbox(toggle_label, &black_hole_mode);

    const float mass_min = 0.5f, mass_max = 6.0f, mass_step = 0.1f;
    float slider_w = 260.0f;
    float arrow_w = ImGui::GetFrameHeight();
    float gap = 8.0f;
    float row_w = arrow_w + gap + slider_w + gap + arrow_w;
    ImGui::SetCursorPos(ImVec2((screen_w - row_w) * 0.5f, 90.0f));

    if (ImGui::ArrowButton("##mass_dec", ImGuiDir_Left)) {
        object_mass -= mass_step;
        if (object_mass < mass_min) object_mass = mass_min;
    }
    ImGui::SameLine(0.0f, gap);
    ImGui::SetNextItemWidth(slider_w);
    ImGui::SliderFloat("Size", &object_mass, mass_min, mass_max, "%.2f");
    ImGui::SameLine(0.0f, gap);
    if (ImGui::ArrowButton("##mass_inc", ImGuiDir_Right)) {
        object_mass += mass_step;
        if (object_mass > mass_max) object_mass = mass_max;
    }

    // Real physical mass implied by that size -- see real_mass_kg() above
    // for why Earth and black hole mode use different physics here.
    double mass_kg = real_mass_kg(object_mass, black_hole_mode);
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

    float btn_w = 90.0f;
    float btn_h = 28.0f;
    ImGui::SetCursorPos(ImVec2(40.0f, screen_h - btn_h - 40.0f));
    if (ImGui::Button("Back", ImVec2(btn_w, btn_h)))
        next = Screen::UnderTheHood;

    // --- Orbit camera controls: drag to rotate, scroll to zoom ---
    // Submitted LAST (not first, as it originally was): Dear ImGui grabs a
    // widget's "active" (clicked/held) state during that widget's own
    // submission call, in submission order -- a full-screen invisible
    // button submitted *before* the checkbox/slider/arrows/Back button
    // would claim the click for itself the instant the mouse went down
    // anywhere on screen, before those widgets were even processed that
    // frame. That's exactly why they stopped responding once this was
    // added. Submitting it last means real widgets get first claim on
    // clicks landing on themselves; this only picks up drags/scroll that
    // land on empty space they didn't already claim.
    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
    ImGui::InvisibleButton("##viewport", ImVec2(screen_w, screen_h));
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        cam_azimuth -= io.MouseDelta.x * 0.01f;
        cam_elevation += io.MouseDelta.y * 0.01f;
        if (cam_elevation < 0.15f) cam_elevation = 0.15f;
        if (cam_elevation > 1.45f) cam_elevation = 1.45f;
    }
    if (ImGui::IsItemHovered()) {
        cam_distance -= io.MouseWheel * 1.0f;
        if (cam_distance < 4.0f) cam_distance = 4.0f;
        if (cam_distance > 40.0f) cam_distance = 40.0f;
    }

    ImGui::End();
    ImGui::PopStyleColor(5);

    ImGui::Render();

    glViewport(0, 0, (GLsizei)screen_w, (GLsizei)screen_h);
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    render_earth_scene(screen_w, screen_h, (float)glfwGetTime(), object_mass,
                        cam_azimuth, cam_elevation, cam_distance,
                        black_hole_mode ? CelestialBody::BlackHole : CelestialBody::Earth);
    glDisable(GL_DEPTH_TEST);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

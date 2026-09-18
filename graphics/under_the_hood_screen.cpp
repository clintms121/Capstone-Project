#include "screen.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

namespace
{

    void section(const char *heading, const char *body)
    {
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%s", heading);
        ImGui::TextWrapped("%s", body);
        ImGui::Dummy(ImVec2(0.0f, 14.0f));
    }

} // namespace

// Technical breakdown of the ML pipeline. Keep this in sync with the actual
// implementation in ml/ as it's built out -- it's currently describing the
// intended design (see ml/README.md), not a trained model.
Screen render_under_the_hood_screen(GLFWwindow *window)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO &io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_w, screen_h), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings;

    Screen next = Screen::UnderTheHood;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##under_the_hood", nullptr, flags);

    ImGui::SetCursorPos(ImVec2(40.0f, 30.0f));
    ImGui::Text("Under the Hood");

    float back_w = 90.0f;
    float back_h = 28.0f;
    float content_top = 70.0f;
    float content_h = screen_h - content_top - back_h - 60.0f;

    ImGui::SetCursorPos(ImVec2(40.0f, content_top));
    ImGui::BeginChild("##uth_scroll", ImVec2(screen_w - 80.0f, content_h), true);

    section(
        "Input Data",
        "Each clip starts as a short strain recording from one LIGO detector -- "
        "basically a sensor reading of how much the detector's arms stretched "
        "and squeezed over a few seconds around a possible event. That raw "
        "signal looks like pure noise to the eye, so it gets cleaned up and "
        "turned into a 128x128 image: one axis is time, the other is "
        "frequency, and brightness shows signal strength. That image is what "
        "the model actually looks at.");

    section(
        "How the Data is Collected",
        "LIGO isn't a telescope -- it doesn't look at light. A gravitational "
        "wave is a literal ripple in spacetime, produced when massive objects "
        "like two black holes accelerate around each other, and it stretches "
        "space in one direction while squeezing it in another as it passes "
        "through. LIGO measures that with two 4-kilometer laser arms at right "
        "angles: normally the light from both arms cancels out perfectly, but "
        "a passing wave shifts one arm relative to the other by about a "
        "thousandth the width of a proton -- tiny, but enough to show up as a "
        "change in the laser's interference pattern. There are two of these "
        "detectors, in Washington and Louisiana, 3,000 km apart -- a real "
        "wave hits both within milliseconds of each other, which is one of "
        "the first checks used to tell a real signal from local noise.");

    if (ImGui::Button("Space Time Simulation", ImVec2(220.0f, 32.0f)))
        next = Screen::SpacetimeSimulation;
    ImGui::Dummy(ImVec2(0.0f, 14.0f));

    ImGui::EndChild();

    ImGui::SetCursorPos(ImVec2(40.0f, screen_h - back_h - 40.0f));
    if (ImGui::Button("Back", ImVec2(back_w, back_h)))
        next = Screen::Research;

    ImGui::End();
    ImGui::PopStyleColor(5);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

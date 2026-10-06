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

// Plain-language breakdown of the ML pipeline. Keep this in sync with the
// actual implementation and results in ml/README.md.
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
        "turned into a 140x170 image: one axis is time, the other is "
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

    section(
        "How the Classifier Decides",
        "The images go to a convolutional neural network (CNN), a model built "
        "for pictures. Its first layers learn simple features like edges and "
        "bright spots; later layers combine them into shapes like \"a streak "
        "that curves upward\" -- the chirp a merger makes as the two black "
        "holes spiral in faster and faster. It sorts every image into one of "
        "22 labels: a real signal, or one of 21 kinds of glitch or noise "
        "labeled by Gravity Spy volunteers. On images it never saw in "
        "training it gets 97.9% right.");

    section(
        "Making Sure It Isn't Cheating",
        "The first version scored a perfect test -- and also called every "
        "image of pure noise a real signal. Every image our code produced had "
        "been a merger, so it had learned to recognize our code, not the "
        "chirp. Adding noise images made the same way fixed that. The real "
        "test is the one LIGO uses: hide simulated chirps in real noise and "
        "count how many it finds. It catches about half at a signal-to-noise "
        "ratio of 8 and nine in ten at 15, while flagging plain noise only "
        "1.8% of the time.");

    section(
        "Weighing the Black Holes",
        "For real signals, the tool compares the recording against about "
        "3,000 chirps predicted by Einstein's equations for different masses "
        "-- the same matched-filtering idea LIGO's searches use -- and keeps "
        "the best match. That gives the chirp mass, which sets how fast the "
        "chirp sweeps up. Noise alone can produce a match strength up to 6.4, "
        "so only clearer matches count: for those 76 mergers the result is "
        "within 2.2% of LIGO's published value (median).");

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

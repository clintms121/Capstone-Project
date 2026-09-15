#include "screen.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

namespace {

void section(const char* heading, const char* body) {
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%s", heading);
    ImGui::TextWrapped("%s", body);
    ImGui::Dummy(ImVec2(0.0f, 14.0f));
}

} // namespace

// Technical breakdown of the ML pipeline. Keep this in sync with the actual
// implementation in ml/ as it's built out -- it's currently describing the
// intended design (see ml/README.md), not a trained model.
Screen render_under_the_hood_screen(GLFWwindow* window) {
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

    Screen next = Screen::UnderTheHood;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

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
        "1. Input: raw strain -> spectrogram",
        "Every clip -- a LIGO strain segment around a candidate event, or a "
        "labeled Gravity Spy glitch -- is whitened against the detector's "
        "noise power spectral density (so every frequency contributes "
        "equally instead of the spectrum being dominated by low-frequency "
        "seismic noise), band-passed, then run through a Q-transform to "
        "produce a fixed-size time-frequency image. Real signals and "
        "glitches go through the identical transform so the classifier "
        "never sees a systematic difference in preprocessing, only in "
        "content."
    );

    section(
        "2. Classifier: CNN over the spectrogram",
        "A convolutional network (a small ResNet-style stack, single-channel "
        "input) maps that image to one of ~21 classes: ~20 Gravity Spy "
        "glitch morphologies (Blip, Koi Fish, Scattered Light, Whistle, "
        "...) plus a 'real signal' class. The glitch classes are heavily "
        "imbalanced in the dataset, so the training loss is class-weighted "
        "and evaluated with a full confusion matrix, not just accuracy -- a "
        "model that just always guesses the majority glitch class would "
        "score deceptively well otherwise."
    );

    section(
        "3. Mass estimation: reading the chirp",
        "Only spectrograms the classifier calls 'real signal' move on. A "
        "regression model reads the chirp's frequency-vs-time slope off the "
        "spectrogram -- how fast the instantaneous frequency sweeps upward "
        "is set almost entirely by the chirp mass, via the leading-order "
        "post-Newtonian relation "
        "f_GW(t) ~ (5 / (256*(tc - t)))^(3/8) * (G*Mchirp/c^3)^(-5/8). "
        "The model outputs a point estimate of chirp mass and mass ratio -- "
        "this is a simplified stand-in for full Bayesian parameter "
        "estimation, validated against the published GWTC catalog values "
        "for confirmed events rather than derived from first principles."
    );

    section(
        "4. Handoff to the visualization",
        "The estimated (chirp_mass, mass_ratio) pair is written to "
        "data/trajectories/index.csv alongside a trajectory file computed "
        "by physics/inspiral.py and physics/ringdown.py. No ML runs at "
        "visualization time -- the Simulation screen only reads that "
        "trajectory back and plays it."
    );

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

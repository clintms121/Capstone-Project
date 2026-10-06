#include "screen.h"
#include "simulation_params.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>

// Renders the simulation screen -- the project's final, still in-progress
// part. With no event selected yet, this is a picker over `available_events`
// (the parsed data/trajectories/index.csv manifest written by the Python
// pipeline). Once one is picked, `selected` holds the parameters (event id,
// chirp mass, mass ratio, trajectory path) that the eventual playback code
// needs -- see physics/README.md for the trajectory file contract.
Screen render_simulation_screen(GLFWwindow* window,
                                 const std::vector<SimulationParams>& available_events,
                                 SimulationParams& selected) {
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

    Screen next = Screen::Simulation;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##simulation", nullptr, flags);

    // --- In-progress banner ---
    ImGui::SetCursorPos(ImVec2(40.0f, 24.0f));
    ImGui::TextColored(ImVec4(1.0f, 0.70f, 0.30f, 1.0f), "MERGER SIMULATION -- IN PROGRESS");
    ImGui::SetCursorPos(ImVec2(40.0f, 44.0f));
    ImGui::PushTextWrapPos(screen_w - 40.0f);
    ImGui::TextColored(ImVec4(0.62f, 0.62f, 0.66f, 1.0f),
        "This is the final part of the project and is still being built. The physics is done: "
        "each merger below already has its orbit, merger and ringdown computed from the chirp "
        "mass the tool measured (physics/build_trajectories.py). The 3D playback of those "
        "trajectories is what's coming next.");
    ImGui::PopTextWrapPos();

    if (!selected.is_selected()) {
        // --- Event picker ---
        ImGui::SetCursorPos(ImVec2(40.0f, 100.0f));
        ImGui::Text("Mergers the tool classified as real and could weigh (%d)", (int)available_events.size());

        if (available_events.empty()) {
            ImGui::SetCursorPos(ImVec2(40.0f, 130.0f));
            ImGui::TextWrapped(
                "No events available yet. The ml/ + physics/ pipeline hasn't "
                "produced data/trajectories/index.csv -- see ml/README.md for "
                "the steps to run."
            );
        } else {
            ImGui::SetCursorPos(ImVec2(40.0f, 130.0f));
            ImGui::BeginChild("##event_list", ImVec2(screen_w - 80.0f, screen_h - 240.0f), true);

            const float card_w = 280.0f;
            const float card_h = 160.0f;
            const float gap = 20.0f;
            float region_w = ImGui::GetContentRegionAvail().x;
            int cards_per_row = (int)((region_w + gap) / (card_w + gap));
            if (cards_per_row < 1)
                cards_per_row = 1;

            for (int i = 0; i < (int)available_events.size(); ++i) {
                const SimulationParams& ev = available_events[i];

                if (i % cards_per_row != 0)
                    ImGui::SameLine(0.0f, gap);

                ImGui::PushID(i);
                ImGui::BeginChild("##card", ImVec2(card_w, card_h), true);

                ImGui::SetWindowFontScale(1.5f);
                ImGui::TextUnformatted(ev.event_id.c_str());
                ImGui::SetWindowFontScale(1.0f);

                ImGui::Dummy(ImVec2(0.0f, 10.0f));
                ImGui::TextWrapped("Chirp mass: %.1f Msun (measured)", ev.chirp_mass_msun);
                ImGui::TextWrapped("Mass ratio: %.2f (typical value)", ev.mass_ratio);

                ImGui::EndChild();

                bool card_clicked = ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
                ImGui::PopID();

                if (card_clicked)
                    selected = ev;
            }

            ImGui::EndChild();
        }
    } else {
        // --- Selected event (playback not implemented yet) ---
        ImGui::SetCursorPos(ImVec2(40.0f, 110.0f));
        ImGui::Text("Event: %s", selected.event_id.c_str());

        ImGui::SetCursorPos(ImVec2(40.0f, 140.0f));
        ImGui::Text("Chirp mass: %.2f Msun (measured)   Mass ratio: %.2f (typical value)",
                    selected.chirp_mass_msun, selected.mass_ratio);

        ImGui::SetCursorPos(ImVec2(40.0f, 170.0f));
        ImGui::TextWrapped("Computed trajectory: %s", selected.trajectory_path.c_str());

        ImGui::SetCursorPos(ImVec2(40.0f, 210.0f));
        ImGui::TextDisabled("3D playback of this merger is in progress.");

        ImGui::SetCursorPos(ImVec2(40.0f, 240.0f));
        if (ImGui::Button("Choose another event", ImVec2(180.0f, 28.0f)))
            selected = SimulationParams{};
    }

    float btn_w = 90.0f;
    float btn_h = 28.0f;
    ImGui::SetCursorPos(ImVec2(40.0f, screen_h - btn_h - 40.0f));
    if (ImGui::Button("Back", ImVec2(btn_w, btn_h))) {
        selected = SimulationParams{}; // clear so re-entering shows the picker again
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

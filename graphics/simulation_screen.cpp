#include "screen.h"
#include "simulation_params.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>

// Renders the simulation screen. With no event selected yet, this is a
// picker over `available_events` (the parsed data/trajectories/index.csv
// manifest written by the Python pipeline). Once one is picked, `selected`
// holds the parameters (event id, chirp mass, mass ratio, trajectory path)
// that the eventual playback code needs -- see physics/README.md for the
// trajectory file contract.
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

    if (!selected.is_selected()) {
        // --- Event picker ---
        ImGui::SetCursorPos(ImVec2(40.0f, 40.0f));
        ImGui::Text("Choose a classified event to visualize");

        if (available_events.empty()) {
            ImGui::SetCursorPos(ImVec2(40.0f, 80.0f));
            ImGui::TextWrapped(
                "No events available yet. The ml/ + physics/ pipeline hasn't "
                "produced data/trajectories/index.csv -- run the classifier "
                "and mass_estimation steps on a real event first."
            );
        } else {
            ImGui::SetCursorPos(ImVec2(40.0f, 80.0f));
            ImGui::BeginChild("##event_list", ImVec2(screen_w - 80.0f, screen_h - 200.0f), true);

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
                ImGui::TextWrapped("Chirp mass: %.1f Msun", ev.chirp_mass_msun);
                ImGui::TextWrapped("Mass ratio: %.2f", ev.mass_ratio);

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
        ImGui::SetCursorPos(ImVec2(40.0f, 40.0f));
        ImGui::Text("Event: %s", selected.event_id.c_str());

        ImGui::SetCursorPos(ImVec2(40.0f, 70.0f));
        ImGui::Text("Chirp mass: %.2f Msun   Mass ratio: %.2f",
                    selected.chirp_mass_msun, selected.mass_ratio);

        ImGui::SetCursorPos(ImVec2(40.0f, 100.0f));
        ImGui::TextWrapped("Trajectory file: %s", selected.trajectory_path.c_str());

        ImGui::SetCursorPos(ImVec2(40.0f, 140.0f));
        ImGui::TextDisabled("Playback not implemented yet.");
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

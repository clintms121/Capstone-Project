// The "< Back to ..." button every screen draws in its bottom-left corner,
// kept in one place so they all look the same.
#include "back_button.h"
#include "imgui.h"

bool back_button(const char* label) {
    const float width = 200.0f;
    const float margin = 40.0f; // from the window's left and bottom edges

    ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.12f, 0.16f, 0.26f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.28f, 0.45f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.28f, 0.38f, 0.60f, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);

    ImGui::SetCursorPos(ImVec2(margin, ImGui::GetWindowHeight() - k_back_button_height - margin));
    bool clicked = ImGui::Button(label, ImVec2(width, k_back_button_height));

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(4);
    return clicked;
}

#include "fonts.h"

ImFont* g_font_title = nullptr;
ImFont* g_font_subtitle = nullptr;

void load_fonts() {
    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->AddFontDefault(); // normal body/button text, stays Fonts[0]

    ImFontConfig subtitle_cfg;
    subtitle_cfg.SizePixels = 26.0f; // ~2x default
    g_font_subtitle = io.Fonts->AddFontDefault(&subtitle_cfg);

    ImFontConfig title_cfg;
    title_cfg.SizePixels = 52.0f; // ~4x default
    g_font_title = io.Fonts->AddFontDefault(&title_cfg);
}

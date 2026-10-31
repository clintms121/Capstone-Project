// Loads the Inter font (fonts/) at each size the screens use.
#include "fonts.h"

#include <cstdio>

ImFont* g_font_title = nullptr;
ImFont* g_font_tagline = nullptr;
ImFont* g_font_subtitle = nullptr;
ImFont* g_font_heading = nullptr;
ImFont* g_font_body = nullptr;

namespace {

// Inter (SIL Open Font License, fonts/LICENSE.txt), loaded relative to the
// repo root like imgs/. Regular for text, SemiBold for titles and headings.
const char* k_regular = "fonts/Inter-Regular.ttf";
const char* k_semibold = "fonts/Inter-SemiBold.ttf";

// The TTF at `size` pixels, or ImGui's built-in ProggyClean if the file is
// missing (AddFontFromFileTTF asserts on a bad path), so the app still runs.
ImFont* add_font(const char* path, float size) {
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig cfg;
    cfg.OversampleH = size >= 48.0f ? 1 : 2; // large glyphs don't need it; keeps the atlas small
    if (FILE* f = std::fopen(path, "rb")) {
        std::fclose(f);
        return io.Fonts->AddFontFromFileTTF(path, size, &cfg);
    }
    cfg.SizePixels = size;
    return io.Fonts->AddFontDefault(&cfg);
}

} // namespace

void load_fonts() {
    add_font(k_regular, 16.0f); // normal body/button text, stays Fonts[0]
    g_font_subtitle = add_font(k_regular, 26.0f);
    g_font_title = add_font(k_semibold, 96.0f);
    g_font_tagline = add_font(k_regular, 32.0f);
    g_font_heading = add_font(k_semibold, 40.0f);
    g_font_body = add_font(k_regular, 20.0f);
}

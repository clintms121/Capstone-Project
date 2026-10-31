#pragma once

#include "imgui.h"

// Inter, baked at each target pixel size (not runtime-scaled) so large text
// stays crisp -- SetWindowFontScale() just stretches the 16px default and
// blurs. Valid after load_fonts() has run.
extern ImFont* g_font_title;   // title screen, sized for a 1080p window
extern ImFont* g_font_tagline; // title screen subtitle and Begin button, same
extern ImFont* g_font_subtitle;
extern ImFont* g_font_heading; // section headings on slide-style screens
extern ImFont* g_font_body;    // large, readable paragraph text

// Bakes the extra font sizes into the atlas. Call once, after
// ImGui::CreateContext() and before the first frame is rendered.
void load_fonts();

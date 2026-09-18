#pragma once

#include "imgui.h"

// Baked at their target pixel size (not runtime-scaled) so large text stays
// crisp -- SetWindowFontScale() just stretches the default 13px bitmap and
// blurs at 4x. Valid after load_fonts() has run.
extern ImFont* g_font_title;
extern ImFont* g_font_subtitle;

// Bakes the extra font sizes into the atlas. Call once, after
// ImGui::CreateContext() and before the first frame is rendered.
void load_fonts();

#pragma once

// Height of the back button, for screens that lay out around it.
constexpr float k_back_button_height = 40.0f;

// The "< Back to ..." button in the bottom-left corner of the current
// full-screen window. Every screen uses this one so they all look the same:
// rounded, dark blue, the menu's original style. Returns true when clicked.
bool back_button(const char* label);

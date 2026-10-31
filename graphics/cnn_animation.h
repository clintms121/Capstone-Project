#pragma once

#include "imgui.h"

// Draws one frame of the looping "Model Demonstration" animation (a Q-scan
// flowing through the classifier network) into the box at `pos` (screen
// coordinates) of `size`, faded by `alpha` (0..1). `time` is in seconds and
// only needs to keep increasing; the animation repeats every 7 seconds.
void draw_classifier_animation(ImDrawList* dl, ImVec2 pos, ImVec2 size, float alpha, float time);

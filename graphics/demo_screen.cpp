// The Classifier screen: pick a real LIGO merger (or noise the model
// wrongly flagged) from the lists on the left, see the Q-scan image the
// model looked at in the middle, and its verdict on the right -- with a
// button that opens the 3D simulation of that merger.
#include "screen.h"
#include "back_button.h"
#include "demo_data.h"
#include "fonts.h"
#include "simulation_params.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

const ImVec4 kAccent(0.40f, 0.70f, 1.00f, 1.0f);
const ImVec4 kQuiet(0.62f, 0.62f, 0.66f, 1.0f);

// The menu's button blues, and the brighter fill for the selected list.
const ImVec4 kButton(0.12f, 0.16f, 0.26f, 1.0f);
const ImVec4 kButtonHovered(0.20f, 0.28f, 0.45f, 1.0f);
const ImVec4 kButtonActive(0.28f, 0.38f, 0.60f, 1.0f);
const ImVec4 kButtonSelected(0.20f, 0.44f, 0.88f, 1.0f);

constexpr float kFreqLo = 10.0f, kFreqHi = 2048.0f; // the Q-scan's frequency range, bottom to top (Hz)
constexpr float kFillSeconds = 0.6f; // confidence count-up after picking a sample
constexpr float kColumnGap = 32.0f;  // whitespace between the three columns, instead of borders

// Dev only: show whether the model trained on the selected recording. Turn
// off before presenting.
constexpr bool kShowTrainingSplit = true;

// The two lists on the left, the sample kind each one shows, and its button.
enum Tab { kMergers, kFalsePositives, kTabCount };
const char* const kTabKinds[kTabCount] = {"event", "false_positive"};
const char* const kTabLabels[kTabCount] = {"Valid black hole mergers", "False positives"};

ImU32 col(const ImVec4& c) { return ImGui::ColorConvertFloat4ToU32(c); }

void text_wrapped_colored(const ImVec4& c, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ImGui::PushStyleColor(ImGuiCol_Text, c);
    ImGui::TextWrappedV(fmt, args);
    ImGui::PopStyleColor();
    va_end(args);
}

bool is_real(const std::string& label) { return label == "real_signal"; }

const char* detector_name(const std::string& detector) {
    return detector == "H1" ? "Hanford" : detector == "L1" ? "Livingston" : detector.c_str();
}

// ------------------------------------------------------------- Left: lists

// Button in the menu's style; the selected one is filled brighter.
bool list_button(const char* label, bool selected, float width) {
    ImGui::PushStyleColor(ImGuiCol_Button, selected ? kButtonSelected : kButton);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, selected ? kButtonSelected : kButtonHovered);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, selected ? kButtonSelected : kButtonActive);
    bool clicked = ImGui::Button(label, ImVec2(width, k_back_button_height));
    ImGui::PopStyleColor(3);
    return clicked;
}

// The samples one list shows, in display order: mergers newest first (event
// names start with the date, GWyymmdd, so they sort by time), false
// positives as exported.
std::vector<DemoSample*> list_samples(DemoData& demo, Tab tab) {
    std::vector<DemoSample*> out;
    for (DemoSample& s : demo.samples)
        if (s.kind == kTabKinds[tab])
            out.push_back(&s);
    if (tab == kMergers)
        std::stable_sort(out.begin(), out.end(), [](const DemoSample* a, const DemoSample* b) {
            return a->name != b->name ? a->name > b->name : a->detector < b->detector;
        });
    return out;
}

// One clickable list row: text on a soft rounded highlight when hovered or
// selected, with an optional note right-aligned in gray.
bool list_row(const std::string& text, const char* note, bool selected) {
    ImVec2 size(ImGui::GetContentRegionAvail().x, ImGui::GetFontSize() + 14.0f);
    ImVec2 p0 = ImGui::GetCursorScreenPos(), p1(p0.x + size.x, p0.y + size.y);
    bool clicked = ImGui::InvisibleButton("##row", size);
    bool hovered = ImGui::IsItemHovered();
    if (hovered)
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (selected)
        dl->AddRectFilled(p0, p1, col(kButtonHovered), 8.0f);
    else if (hovered)
        dl->AddRectFilled(p0, p1, col(ImVec4(1.0f, 1.0f, 1.0f, 0.06f)), 8.0f);
    float text_y = floorf(p0.y + (size.y - ImGui::GetFontSize()) * 0.5f);
    dl->AddText(ImVec2(p0.x + 12.0f, text_y), col(ImVec4(1, 1, 1, 1)), text.c_str());
    if (note) {
        float w = ImGui::CalcTextSize(note).x;
        dl->AddText(ImVec2(p1.x - w - 12.0f, text_y), col(kQuiet), note);
    }
    return clicked;
}

void draw_sample_list(DemoData& demo, Tab tab, int& selected_id) {
    text_wrapped_colored(kQuiet, tab == kMergers ? "Most recent mergers"
                                                 : "Held-out detector noise the model wrongly called a merger.");
    ImGui::Dummy(ImVec2(0.0f, 4.0f));

    ImGui::BeginChild(tab == kMergers ? "##merger_items" : "##false_positive_items");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 2.0f));
    std::vector<DemoSample*> samples = list_samples(demo, tab);
    for (DemoSample* s : samples) {
        ImGui::PushID(s->id);
        // Every false positive is held out, so only mergers need the note.
        const char* note = tab == kMergers && s->split == "test" ? "held out" : nullptr;
        if (list_row(s->name + "  (" + s->detector + ")", note, s->id == selected_id))
            selected_id = s->id;
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
    if (samples.empty())
        text_wrapped_colored(kQuiet, "None in data/demo/. From the project folder, rerun: python -m ml.export_demo");
    ImGui::EndChild();
}

// ---------------------------------------------------------- Middle: Q-scan

void draw_qscan_panel(DemoSample* s) {
    const float left = 64.0f, bottom = 44.0f, gap = 14.0f;

    // Title over the image: "Q-scan", plus how far away the merger was when
    // that's published (noise has no source, so no distance).
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + left);
    ImGui::PushFont(g_font_body);
    ImGui::TextUnformatted("Q-scan");
    if (s && s->kind == "event" && !std::isnan(s->distance_ly)) {
        ImGui::SameLine();
        ImGui::TextColored(kQuiet, "\xc2\xb7  %s away", format_light_years(s->distance_ly).c_str());
    }
    ImGui::PopFont();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float w = avail.x - left - 28.0f, h = w * 140.0f / 170.0f; // right margin fits the "+0.50" label
    if (h > avail.y - bottom - gap) {
        h = avail.y - bottom - gap;
        w = h * 170.0f / 140.0f;
    }
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 p0(origin.x + left, origin.y + gap), p1(p0.x + w, p0.y + h);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    unsigned int tex = s ? demo_texture(*s) : 0;
    if (tex) {
        dl->AddImageRounded((ImTextureID)(intptr_t)tex, p0, p1, ImVec2(0, 0), ImVec2(1, 1),
                            col(ImVec4(1, 1, 1, 1)), 10.0f);
    } else {
        dl->AddRectFilled(p0, p1, col(ImVec4(0.07f, 0.07f, 0.09f, 1.0f)), 10.0f);
        const char* msg = s ? "image not found -- rerun ml/export_demo.py" : "";
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2((p0.x + p1.x - ts.x) * 0.5f, (p0.y + p1.y - ts.y) * 0.5f), col(kQuiet), msg);
    }

    const float freqs[] = {16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f};
    for (float f : freqs) {
        float y = p1.y - std::log(f / kFreqLo) / std::log(kFreqHi / kFreqLo) * h;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%g Hz", f);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddText(ImVec2(p0.x - 10.0f - ts.x, y - ts.y * 0.5f), col(kQuiet), buf);
    }
    for (int i = 0; i <= 4; ++i) {
        float t = -0.5f + 0.25f * i, x = p0.x + (t + 0.5f) * w;
        char buf[16];
        std::snprintf(buf, sizeof(buf), i == 2 ? "0" : "%+.2f", t);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddText(ImVec2(x - ts.x * 0.5f, p1.y + 8.0f), col(kQuiet), buf);
    }
    const char* xlab = s && s->kind == "event" ? "seconds from the merger" : "seconds from the center of the window";
    ImVec2 xs = ImGui::CalcTextSize(xlab);
    dl->AddText(ImVec2((p0.x + p1.x - xs.x) * 0.5f, p1.y + 26.0f), col(kQuiet), xlab);

    ImGui::Dummy(ImVec2(avail.x, h + bottom + gap));
}

// ----------------------------------------------------------- Right: details

// Confidence that `s` is a real merger: a big percentage over a bar, both
// counting up from 0 as `fill` goes 0 -> 1.
void draw_confidence(const DemoSample& s, float fill) {
    ImGui::TextColored(kQuiet, "Confidence it's a real black hole merger");
    float shown = s.p_real * fill;
    char pct[16];
    if (shown >= 0.9995f)
        std::snprintf(pct, sizeof(pct), ">99.9%%");
    else
        std::snprintf(pct, sizeof(pct), "%.1f%%", 100.0f * shown);
    ImGui::PushFont(g_font_heading);
    ImGui::TextUnformatted(pct);
    ImGui::PopFont();

    ImVec2 p0 = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x, h = 12.0f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, ImVec2(p0.x + w, p0.y + h), col(ImVec4(1.0f, 1.0f, 1.0f, 0.08f)), h * 0.5f);
    if (shown > 0.0f)
        dl->AddRectFilled(p0, ImVec2(p0.x + std::max(h, w * shown), p0.y + h), col(kAccent), h * 0.5f);
    ImGui::Dummy(ImVec2(w, h));
}

void draw_details(const DemoSample& s, float fill, const std::vector<SimulationParams>& events,
                  SimulationParams& selected_event, Screen& next) {
    const bool merger = s.kind == "event";
    const float section_gap = 22.0f;

    ImGui::PushFont(g_font_subtitle);
    ImGui::TextWrapped("%s", merger ? s.name.c_str() : "Detector noise");
    ImGui::PopFont();
    if (merger) {
        text_wrapped_colored(kQuiet, "Recorded by the %s detector (%s).", detector_name(s.detector),
                             s.detector.c_str());
    } else {
        const std::string prefix = "Noise "; // export names these "Noise 6 s after GW..."
        std::string when = s.name.rfind(prefix, 0) == 0 ? s.name.substr(prefix.size()) : s.name;
        text_wrapped_colored(kQuiet, "Recorded by the %s detector (%s), %s.", detector_name(s.detector),
                             s.detector.c_str(), when.c_str());
    }
    ImGui::Dummy(ImVec2(0.0f, section_gap));

    draw_confidence(s, fill);
    ImGui::Dummy(ImVec2(0.0f, 6.0f));
    if (!merger)
        ImGui::TextWrapped("Wrongly classified as a black hole merger: there's no merger in this recording, "
                           "only detector noise.");
    else if (is_real(s.predicted_label))
        ImGui::TextWrapped("Classified as a black hole merger.");
    else
        ImGui::TextWrapped("Missed: classified as %s.", pretty_label(s.predicted_label).c_str());

    if (kShowTrainingSplit) {
        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        if (s.split == "test")
            text_wrapped_colored(kQuiet, "Dev: held out -- the model never trained on this recording.");
        else if (s.split == "val")
            text_wrapped_colored(kQuiet, "Dev: validation set -- used to pick the best model, not trained on.");
        else
            text_wrapped_colored(kQuiet, "Dev: in the training set -- the model has seen this recording.");
    }

    // The merger simulation, for mergers it was built for (the ones whose
    // masses the matched filter could measure; see physics/build_trajectories.py).
    if (!merger)
        return;
    const SimulationParams* sim = nullptr;
    for (const SimulationParams& ev : events)
        if (ev.event_id == s.name)
            sim = &ev;
    ImGui::Dummy(ImVec2(0.0f, section_gap));
    if (!sim) {
        text_wrapped_colored(kQuiet, "No simulation for this merger: the tool couldn't measure its black "
                                     "holes' masses reliably, and the simulation is built from them.");
        return;
    }
    ImGui::PushStyleColor(ImGuiCol_Button, kButtonSelected);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.52f, 0.95f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.17f, 0.38f, 0.78f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    if (ImGui::Button("Watch both of these black holes merge", ImVec2(-1.0f, 44.0f))) {
        selected_event = *sim;
        selected_event.distance_ly = std::isnan(s.distance_ly) ? 0.0 : s.distance_ly;
        next = Screen::Simulation;
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);
}

// Lists on the left, the selected sample's Q-scan (titled with its distance)
// in the middle, and the classifier's confidence on the right -- with, for a
// merger, the button that opens its simulation.
void draw_classifier(DemoData& demo, const std::vector<SimulationParams>& events,
                     SimulationParams& selected_event, Screen& next) {
    static Tab tab = kMergers;
    static int selected_id[kTabCount] = {-1, -1};
    static int shown_id = -1;
    static double shown_since = 0.0;

    // Start each list on its top row (for mergers, the newest).
    for (int t = 0; t < kTabCount; ++t) {
        if (selected_id[t] >= 0)
            continue;
        std::vector<DemoSample*> samples = list_samples(demo, (Tab)t);
        if (!samples.empty())
            selected_id[t] = samples.front()->id;
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float list_w = 380.0f, info_w = 330.0f;
    float mid_w = avail.x - list_w - info_w - 2.0f * kColumnGap;

    ImGui::BeginChild("##list", ImVec2(list_w, avail.y));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    float button_w = floorf((list_w - ImGui::GetStyle().ItemSpacing.x) * 0.5f);
    for (int t = 0; t < kTabCount; ++t) {
        if (t > 0)
            ImGui::SameLine();
        if (list_button(kTabLabels[t], t == tab, button_w))
            tab = (Tab)t;
    }
    ImGui::PopStyleVar();
    ImGui::Dummy(ImVec2(0.0f, 6.0f));
    draw_sample_list(demo, tab, selected_id[tab]);
    ImGui::EndChild();

    DemoSample* selected = nullptr;
    for (DemoSample& s : demo.samples)
        if (s.id == selected_id[tab])
            selected = &s;
    if (selected && selected->id != shown_id) {
        shown_id = selected->id;
        shown_since = ImGui::GetTime();
    }
    // The confidence bar counts up after a new pick, easing out as it lands.
    float progress = std::min(1.0f, (float)(ImGui::GetTime() - shown_since) / kFillSeconds);
    float fill = 1.0f - (1.0f - progress) * (1.0f - progress) * (1.0f - progress);

    ImGui::SameLine(0.0f, kColumnGap);
    ImGui::BeginChild("##qscan", ImVec2(mid_w, avail.y), false, ImGuiWindowFlags_NoScrollbar);
    draw_qscan_panel(selected);
    ImGui::EndChild();

    ImGui::SameLine(0.0f, kColumnGap);
    ImGui::BeginChild("##details", ImVec2(info_w, avail.y));
    if (selected)
        draw_details(*selected, fill, events, selected_event, next);
    ImGui::EndChild();
}

} // namespace

// Classifier: real LIGO mergers and the held-out noise the model mistook for
// one, each with the Q-scan the CNN saw and its confidence that it's a real
// merger. Results come from ml/export_demo.py; nothing is computed here.
Screen render_demo_screen(GLFWwindow* window, DemoData& demo, const std::vector<SimulationParams>& events,
                          SimulationParams& selected_event) {
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

    Screen next = Screen::Demo;

    ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,            ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,          kButton);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   kButtonHovered);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    kButtonActive);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg,     ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0f);

    ImGui::Begin("##demo", nullptr, flags);

    ImGui::SetCursorPos(ImVec2(40.0f, 24.0f));
    ImGui::PushFont(g_font_heading);
    ImGui::TextUnformatted("Classifier");
    ImGui::PopFont();

    const float body_top = 24.0f + g_font_heading->FontSize + 24.0f;
    ImGui::SetCursorPos(ImVec2(40.0f, body_top));
    ImGui::BeginChild("##demo_body", ImVec2(screen_w - 80.0f, screen_h - body_top - k_back_button_height - 56.0f));
    if (!demo.loaded())
        ImGui::TextWrapped("No demo data found in data/demo/. From the project folder, run the pipeline "
                           "(see ml/README.md), then:\n\n    python -m ml.export_demo");
    else
        draw_classifier(demo, events, selected_event, next);
    ImGui::EndChild();

    if (back_button("< Back to Menu"))
        next = Screen::Menu;

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(6);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

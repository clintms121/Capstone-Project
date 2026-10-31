// The merger simulation screen (a prototype), opened from the Classifier:
// replays one merger in 3D from its trajectory file in data/trajectories/,
// computed ahead of time from the chirp mass the tool measured. Around the
// 3D view: an info panel with live numbers in real units, a clock, camera
// hints, and play controls over a timeline of the gravitational-wave signal.
// merger_scene.cpp draws the 3D view; trajectory.cpp reads the file.
#include "screen.h"
#include "back_button.h"
#include "demo_data.h"
#include "fonts.h"
#include "merger_scene.h"
#include "simulation_params.h"
#include "trajectory.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kKmPerSolarMass = 1.476625; // G M_sun / c^2: a solar mass as a length
constexpr double kSecondsPerYear = 365.25 * 24.0 * 3600.0;

// Overlay layout: the info column down the left, inside its panel padding.
constexpr float kInfoWidth = 360.0f;
const ImVec2 kPanelPadding(18.0f, 14.0f);

// Playback runs on its own clock, which ticks `clock_rate` per second at 1x
// speed -- set so the first orbit takes kFirstOrbitSeconds. Without slow
// motion the clock is the trajectory's own time t_m, true to life: the
// orbits speed up ~20x by the end and the merger is over in a blink. With
// it, each stretch of t_m is drawn out by (a0 / a)^kSlowMotionPower for
// separation a (a0 at the start), so the orbits still visibly quicken but
// the merger lasts long enough to watch. The ringdown keeps the stretch
// from the moment of merger.
constexpr double kFirstOrbitSeconds = 2.0;
constexpr double kSlowMotionPower = 0.9;

// The classifier screen's palette: gray labels, white values, one accent
// blue, and the menu's button blues.
const ImVec4 kQuiet(0.62f, 0.62f, 0.66f, 1.0f);
const ImVec4 kAccent(0.40f, 0.70f, 1.00f, 1.0f);
const ImVec4 kButton(0.12f, 0.16f, 0.26f, 1.0f);
const ImVec4 kButtonHovered(0.20f, 0.28f, 0.45f, 1.0f);
const ImVec4 kButtonActive(0.28f, 0.38f, 0.60f, 1.0f);

ImU32 col(const ImVec4& c) { return ImGui::ColorConvertFloat4ToU32(c); }

struct Playback {
    std::string event_id; // event `trajectory` was loaded for; empty = none
    Trajectory trajectory;
    std::vector<double> times; // t_m of each sample
    std::vector<double> clock; // playback-clock reading at each sample
    double clock_rate = 1.0;   // clock units per second at 1x speed
    double now = 0.0;          // current clock reading
    bool playing = true;
    bool slow_motion = true;
    float speed = 1.0f;
    float cam_azimuth = 0.5f;
    float cam_elevation = 0.55f;
    float cam_distance = 42.0f; // units of M
};

Playback g_playback;

// y at x along the polyline (xs, ys), xs ascending; clamped at the ends.
double interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    if (x <= xs.front())
        return ys.front();
    if (x >= xs.back())
        return ys.back();
    size_t i = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin(); // xs[i-1] <= x < xs[i]
    double span = xs[i] - xs[i - 1];
    double u = span > 0.0 ? (x - xs[i - 1]) / span : 1.0;
    return ys[i - 1] + (ys[i] - ys[i - 1]) * u;
}

void build_clock(Playback& p) {
    const Trajectory& traj = p.trajectory;
    const double a0 = traj.samples.front().separation();
    auto stretch = [&](size_t i) {
        if (!p.slow_motion)
            return 1.0;
        double a = traj.samples[std::min((int)i, traj.merge_index)].separation();
        return pow(a0 / a, kSlowMotionPower);
    };
    p.clock.assign(p.times.size(), 0.0);
    for (size_t i = 1; i < p.times.size(); ++i)
        p.clock[i] = p.clock[i - 1] + (p.times[i] - p.times[i - 1]) * 0.5 * (stretch(i - 1) + stretch(i));
}

// Clock units per M of trajectory time at t_m: how drawn out playback is there.
double clock_stretch(const Playback& p, double t_m) {
    int i = p.trajectory.index_at(t_m);
    double span = p.times[i + 1] - p.times[i];
    return span > 0.0 ? (p.clock[i + 1] - p.clock[i]) / span : 1.0;
}

double current_time(const Playback& p) { return interpolate(p.clock, p.times, p.now); }

void start_event(Playback& p, const SimulationParams& ev) {
    Playback fresh;
    fresh.slow_motion = p.slow_motion; // keep the viewer's playback choices
    fresh.speed = p.speed;
    p = std::move(fresh);
    p.event_id = ev.event_id;
    p.trajectory = load_trajectory(ev.trajectory_path);
    if (!p.trajectory.loaded())
        return;

    for (const TrajectorySample& s : p.trajectory.samples)
        p.times.push_back(s.t_m);
    build_clock(p);
    double a0 = p.trajectory.samples.front().separation();
    p.clock_rate = 2.0 * kPi * pow(a0, 1.5) / kFirstOrbitSeconds; // Kepler: period 2 pi a^1.5, units of M
}

// A real-world duration, in milliseconds below a second -- a whole ringdown
// can be over in a few.
void format_duration(char* out, size_t size, double seconds) {
    if (seconds < 1.0)
        snprintf(out, size, "%.1f ms", seconds * 1000.0);
    else
        snprintf(out, size, "%.2f s", seconds);
}

// Clears the selection and forgets the playback, so opening an event from
// the classifier -- even the same one -- starts it from the top.
void stop_playback(SimulationParams& selected) {
    selected = SimulationParams{};
    g_playback.event_id.clear();
}

// A soft, rounded, translucent panel over the 3D view -- just enough backing
// to keep text readable over the grid. Pair with ImGui::EndChild().
void begin_panel(const char* id, ImVec2 size, ImGuiChildFlags child_flags) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.02f, 0.03f, 0.06f, 0.72f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 14.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, kPanelPadding);
    ImGui::BeginChild(id, size, child_flags | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

// The strain strip chart, which doubles as the timeline: click or drag on
// it to scrub. Its time axis is the playback clock, so with slow motion on
// the merger gets as much room as it gets screen time.
void draw_waveform(Playback& p, ImVec2 pos, ImVec2 size) {
    const Trajectory& traj = p.trajectory;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 end(pos.x + size.x, pos.y + size.y);
    dl->AddRectFilled(pos, end, IM_COL32(8, 12, 22, 255), 10.0f);

    const double total = p.clock.back();
    const float pad = 8.0f;
    auto x_at = [&](double clock) { return pos.x + pad + (float)(clock / total) * (size.x - 2.0f * pad); };
    const float mid = pos.y + size.y * 0.5f;
    const float amp = size.y * 0.3f;

    const float merge_x = x_at(p.clock[traj.merge_index]);
    dl->AddLine(ImVec2(merge_x, pos.y + 4.0f), ImVec2(merge_x, end.y - 4.0f), IM_COL32(255, 255, 255, 70));
    dl->AddText(ImVec2(merge_x + 5.0f, end.y - 20.0f), col(kQuiet), "Merger");
    dl->AddText(ImVec2(pos.x + pad + 2.0f, pos.y + 4.0f), col(kQuiet), "Gravitational-wave signal");

    // Bright where it has played, dim where it hasn't.
    for (size_t i = 1; i < traj.samples.size(); ++i) {
        ImVec2 a(x_at(p.clock[i - 1]), mid - traj.samples[i - 1].h * amp);
        ImVec2 b(x_at(p.clock[i]), mid - traj.samples[i].h * amp);
        bool played = p.clock[i] <= p.now;
        dl->AddLine(a, b, played ? IM_COL32(140, 210, 255, 255) : IM_COL32(90, 105, 135, 170), 1.5f);
    }

    const float play_x = x_at(p.now);
    dl->AddLine(ImVec2(play_x, pos.y), ImVec2(play_x, end.y), IM_COL32(255, 255, 255, 230), 2.0f);

    ImGui::SetCursorScreenPos(pos);
    ImGui::InvisibleButton("##timeline", size);
    if (ImGui::IsItemActive()) {
        float f = (ImGui::GetIO().MousePos.x - pos.x - pad) / (size.x - 2.0f * pad);
        p.now = std::max(0.0f, std::min(1.0f, f)) * total;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
}

// Bottom panel: play controls over the waveform timeline.
void draw_controls(Playback& p, ImVec2 pos, ImVec2 size) {
    ImGui::SetCursorPos(pos);
    begin_panel("##controls", size, ImGuiChildFlags_None);

    bool at_end = p.now >= p.clock.back();
    const char* play_label = p.playing ? "Pause###play" : (at_end ? "Replay###play" : "Play###play");
    if (ImGui::Button(play_label, ImVec2(90.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
        if (at_end)
            p.now = 0.0;
        p.playing = at_end || !p.playing;
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart", ImVec2(90.0f, 0.0f))) {
        p.now = 0.0;
        p.playing = true;
    }
    ImGui::SameLine(0.0f, 20.0f);
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderFloat("Speed", &p.speed, 0.25f, 4.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine(0.0f, 20.0f);
    if (ImGui::Checkbox("Slow motion near merger", &p.slow_motion)) {
        double t = current_time(p); // rebuild the clock without jumping
        build_clock(p);
        p.now = interpolate(p.times, p.clock, t);
    }

    // How much slower than reality this moment is playing.
    double seconds_per_wall_second = p.clock_rate * p.speed / clock_stretch(p, current_time(p)) *
                                     p.trajectory.seconds_per_m;
    ImGui::SameLine(0.0f, 20.0f);
    ImGui::TextColored(kQuiet, "Now %.0fx slower than real time", 1.0 / seconds_per_wall_second);

    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ImVec2 strip_pos = ImGui::GetCursorScreenPos();
    draw_waveform(p, strip_pos, ImGui::GetContentRegionAvail());
    ImGui::EndChild();
}

// Top-left: the event name as the screen's header, then a panel with its
// masses, the current stage with live numbers in real units, and how far
// the waves have come on their trip to Earth.
void draw_info_panel(const Playback& p, const SimulationParams& ev, double t) {
    const Trajectory& traj = p.trajectory;
    const double km_per_m = traj.total_mass_msun() * kKmPerSolarMass;
    const double seconds_per_m = traj.seconds_per_m;

    ImGui::SetCursorPos(ImVec2(40.0f, 24.0f));
    ImGui::PushFont(g_font_heading);
    ImGui::TextUnformatted(ev.event_id.c_str());
    ImGui::PopFont();
    ImGui::SetCursorPosX(40.0f);
    ImGui::TextColored(kQuiet, "Merger simulation (prototype)");

    ImGui::SetCursorPos(ImVec2(24.0f, ImGui::GetCursorPosY() + 14.0f));
    begin_panel("##info", ImVec2(kInfoWidth, 0.0f), ImGuiChildFlags_AutoResizeY);

    const float value_x = 130.0f;
    char value[64];
    auto row = [&](const char* label) {
        ImGui::TextColored(kQuiet, "%s", label);
        ImGui::SameLine(value_x);
        ImGui::TextUnformatted(value);
    };
    auto section_gap = [] { ImGui::Dummy(ImVec2(0.0f, 12.0f)); };
    auto stage = [](const char* name, const char* about) {
        ImGui::PushFont(g_font_body);
        ImGui::TextColored(kAccent, "%s", name);
        ImGui::PopFont();
        ImGui::PushStyleColor(ImGuiCol_Text, kQuiet);
        ImGui::TextWrapped("%s", about);
        ImGui::PopStyleColor();
    };

    snprintf(value, sizeof(value), "%.1f Msun (measured)", ev.chirp_mass_msun);
    row("Chirp mass");
    snprintf(value, sizeof(value), "%.2f (typical value)", ev.mass_ratio);
    row("Mass ratio");
    snprintf(value, sizeof(value), "%.1f Msun", traj.total_mass_msun());
    row("Total mass");

    section_gap();
    if (t <= traj.merge_time()) {
        TrajectorySample s = traj.at(t);
        float separation = s.separation();
        float wave_omega = traj.wave_omega(t); // twice the orbit's angular velocity
        if (separation > 6.0f)                 // outside the innermost stable orbit, 6 M
            stage("Inspiral", "The two black holes orbit ever closer as gravitational waves carry energy away.");
        else
            stage("Merger", "Inside the last stable orbit, the two black holes plunge together.");
        snprintf(value, sizeof(value), "%.0f km", separation * km_per_m);
        row("Separation");
        snprintf(value, sizeof(value), "%.0f%% of light speed", 100.0 * separation * 0.5 * wave_omega);
        row("Relative speed");
        snprintf(value, sizeof(value), "%.0f Hz", wave_omega / (2.0 * kPi * seconds_per_m));
        row("Wave frequency");
        format_duration(value, sizeof(value), (traj.merge_time() - t) * seconds_per_m);
        row("Until merger");
    } else {
        float amplitude, phase;
        traj.wave_at(t, amplitude, phase);
        stage("Ringdown", "One black hole now, ringing like a struck bell as it settles.");
        snprintf(value, sizeof(value), "%.0f km", traj.samples[traj.merge_index + 1].r1 * km_per_m);
        row("Horizon radius");
        snprintf(value, sizeof(value), "%.0f%% of peak", 100.0 * amplitude);
        row("Ringing");
        snprintf(value, sizeof(value), "%.0f Hz", traj.ringdown_omega / (2.0 * kPi * seconds_per_m));
        row("Wave frequency");
        format_duration(value, sizeof(value), (t - traj.merge_time()) * seconds_per_m);
        row("Since merger");
    }

    // Light-years the waves have covered so far, with the playback standing in
    // for their whole trip to Earth (it really took as many years).
    if (ev.distance_ly > 0.0) {
        section_gap();
        double f = p.clock.back() > 0.0 ? std::min(1.0, p.now / p.clock.back()) : 0.0;
        ImGui::TextColored(kQuiet, "Journey to Earth");
        ImGui::PushFont(g_font_body);
        ImGui::TextUnformatted(format_light_years(f * ev.distance_ly).c_str());
        ImGui::PopFont();

        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 b0 = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x, h = 6.0f;
        dl->AddRectFilled(b0, ImVec2(b0.x + w, b0.y + h), col(ImVec4(1.0f, 1.0f, 1.0f, 0.08f)), h * 0.5f);
        if (f > 0.0)
            dl->AddRectFilled(b0, ImVec2(b0.x + std::max(h, w * (float)f), b0.y + h), col(kAccent), h * 0.5f);
        ImGui::Dummy(ImVec2(w, h + 2.0f));

        ImGui::PushStyleColor(ImGuiCol_Text, kQuiet);
        ImGui::TextWrapped("of %s. The real trip takes %s years; this playback compresses it.",
                           format_light_years(ev.distance_ly).c_str(), format_count(ev.distance_ly).c_str());
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();
}

// Top-right: how to drive the camera, and what's exaggerated. Returns the
// panel's left edge and bottom, for laying out around it.
ImVec2 draw_hints(float screen_w) {
    const char* lines[] = {"Drag to rotate, scroll to zoom, space to pause.",
                           "Not to scale: the wells and ripples are exaggerated.",
                           "The ripples spread out at the speed of light."};
    float text_w = 0.0f;
    for (const char* line : lines)
        text_w = std::max(text_w, ImGui::CalcTextSize(line).x);
    const float panel_w = text_w + 2.0f * kPanelPadding.x;
    const float x = screen_w - panel_w - 24.0f;
    ImGui::SetCursorPos(ImVec2(x, 18.0f));
    begin_panel("##hints", ImVec2(panel_w, 0.0f), ImGuiChildFlags_AutoResizeY);
    for (const char* line : lines)
        ImGui::TextColored(kQuiet, "%s", line);
    ImGui::EndChild();
    return ImVec2(x, ImGui::GetItemRectMax().y);
}

// "or 5.2 × 10^-8 years", the exponent marked for tabular_text() to raise.
void format_years(char* out, size_t size, double seconds) {
    double years = seconds / kSecondsPerYear;
    if (years <= 0.0) {
        snprintf(out, size, "or 0 years");
        return;
    }
    int exponent = (int)floor(log10(years));
    double mantissa = years / pow(10.0, exponent);
    if (mantissa >= 9.95) { // would print as 10.0
        mantissa /= 10.0;
        ++exponent;
    }
    snprintf(out, size, "or %.1f \xc3\x97 10^%d years", mantissa, exponent);
}

// Draws `text` at `pos` with every digit in a cell as wide as the font's
// widest digit, so a running number holds still as its digits change
// (Inter's digits are proportional: a "1" is much narrower than a "4").
// A '^' sets what follows -- a sign and digits -- as a smaller, raised
// exponent. Text may hold 2-byte UTF-8 (Latin-1, like the multiplication
// sign). Returns the width; with dl == nullptr it only measures.
float tabular_text(ImDrawList* dl, ImFont* font, ImVec2 pos, ImU32 color, const char* text) {
    float digit_w = 0.0f;
    for (char c = '0'; c <= '9'; ++c)
        digit_w = std::max(digit_w, font->GetCharAdvance((ImWchar)c));
    float x = pos.x;
    bool exponent = false;
    for (const char* s = text; *s;) {
        unsigned char b = (unsigned char)*s;
        if (b == '^') {
            exponent = true;
            ++s;
            continue;
        }
        int len = b >= 0xC0 && s[1] ? 2 : 1;
        unsigned int c = len == 2 ? ((b & 0x1Fu) << 6) | ((unsigned char)s[1] & 0x3Fu) : b;
        bool digit = c >= '0' && c <= '9';
        exponent = exponent && (digit || c == '-');
        float scale = exponent ? 0.7f : 1.0f; // top-aligned, so it sits raised
        float advance = font->GetCharAdvance((ImWchar)c) * scale;
        float cell = digit ? digit_w * scale : advance;
        if (dl)
            dl->AddText(font, font->FontSize * scale, ImVec2(floorf(x + (cell - advance) * 0.5f), pos.y), color, s,
                        s + len);
        x += cell;
        s += len;
    }
    return x - pos.x;
}

// Top center, above the holes: the real time that has passed since the
// first orbit shown, ticking with the playback -- slowed, paused and scrubbed
// along with it. It's seconds, not years: these last orbits and the merger
// take about a second or two (a few hundredths for the lightest events),
// and the years figure shows just how small that is. Sits between the info
// column and the hints panel if there's room, otherwise below the hints.
void draw_clock(const Playback& p, double t, float screen_w, ImVec2 hints) {
    const Trajectory& traj = p.trajectory;
    const double start_s = traj.samples.front().t_s;
    const double total_s = traj.samples.back().t_s - start_s;
    const double elapsed_s = traj.at(t).t_s - start_s;

    // About four significant digits over this event's whole span, with the
    // decimals fixed so the digits don't shift as the clock runs.
    int decimals = std::max(3, std::min(6, 3 - (int)floor(log10(std::max(total_s, 1e-6)))));
    char digits[32], widest_digits[32], years[48];
    snprintf(digits, sizeof(digits), "%.*f s", decimals, elapsed_s);
    snprintf(widest_digits, sizeof(widest_digits), "%.*f s", decimals, total_s);
    format_years(years, sizeof(years), elapsed_s);
    const char* label = "Real time elapsed";

    // Sized for the widest the text gets, so the panel doesn't twitch.
    ImFont* big = g_font_heading;
    ImFont* small = ImGui::GetFont();
    float digits_w = tabular_text(nullptr, big, ImVec2(), 0, digits);
    float years_w = tabular_text(nullptr, small, ImVec2(), 0, years);
    float content_w = std::max({ImGui::CalcTextSize(label).x, tabular_text(nullptr, big, ImVec2(), 0, widest_digits),
                                tabular_text(nullptr, small, ImVec2(), 0, "or 0.0 \xc3\x97 10^-10 years")});
    float panel_w = content_w + 2.0f * kPanelPadding.x;

    const float left = 24.0f + kInfoWidth + 16.0f; // clear of the info column
    const float right = hints.x - 16.0f;           // and of the hints panel
    float x = screen_w * 0.5f - panel_w * 0.5f;     // centered over the holes
    float y = 18.0f;
    if (right - left >= panel_w) {
        x = std::max(left, std::min(x, right - panel_w));
    } else {
        y = hints.y + 12.0f;
        x = std::max(left, std::min(x, screen_w - 24.0f - panel_w));
    }

    auto centered = [&](float w) { ImGui::SetCursorPosX(kPanelPadding.x + (content_w - w) * 0.5f); };
    auto tabular_line = [&](ImFont* font, ImU32 color, const char* text, float w) {
        centered(w);
        tabular_text(ImGui::GetWindowDrawList(), font, ImGui::GetCursorScreenPos(), color, text);
        ImGui::Dummy(ImVec2(w, font->FontSize));
    };
    ImGui::SetCursorPos(ImVec2(x, y));
    begin_panel("##clock", ImVec2(panel_w, 0.0f), ImGuiChildFlags_AutoResizeY);
    centered(ImGui::CalcTextSize(label).x);
    ImGui::TextColored(kQuiet, "%s", label);
    tabular_line(big, col(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), digits, digits_w);
    tabular_line(small, col(kQuiet), years, years_w);
    ImGui::EndChild();
}

} // namespace

// Draws one frame of the simulation screen. `selected` is the merger the
// Classifier opened; its trajectory file is loaded the first time it's shown.
// Back returns to the Classifier.
Screen render_simulation_screen(GLFWwindow* window, SimulationParams& selected) {
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
        ImGuiWindowFlags_NoTitleBar        |
        ImGuiWindowFlags_NoResize          |
        ImGuiWindowFlags_NoMove            |
        ImGuiWindowFlags_NoScrollbar       |
        ImGuiWindowFlags_NoScrollWithMouse | // the wheel zooms the camera
        ImGuiWindowFlags_NoSavedSettings;

    Screen next = Screen::Simulation;
    Playback& p = g_playback;

    if (selected.is_selected() && p.event_id != selected.event_id)
        start_event(p, selected);
    const bool playing_back = selected.is_selected() && p.trajectory.loaded();
    if (playing_back && p.playing) {
        p.now += io.DeltaTime * p.clock_rate * p.speed;
        if (p.now >= p.clock.back()) {
            p.now = p.clock.back();
            p.playing = false;
        }
    }

    // Rounded menu-blue controls with accent highlights, like the classifier.
    ImGui::PushStyleColor(ImGuiCol_WindowBg,         ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text,             ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button,           kButton);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,    kButtonHovered);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,     kButtonActive);
    ImGui::PushStyleColor(ImGuiCol_FrameBg,          kButton);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered,   kButtonHovered);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive,    kButtonActive);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab,       ImVec4(0.40f, 0.70f, 1.00f, 0.55f)); // the value text sits on it
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0.60f, 0.82f, 1.00f, 0.75f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark,        kAccent);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 6.0f));

    ImGui::Begin("##simulation", nullptr, flags);

    if (!playing_back) {
        ImGui::SetCursorPos(ImVec2(40.0f, 24.0f));
        ImGui::PushFont(g_font_heading);
        ImGui::TextUnformatted("Merger simulation");
        ImGui::PopFont();
        ImGui::SetCursorPos(ImVec2(40.0f, ImGui::GetCursorPosY() + 8.0f));
        ImGui::PushTextWrapPos(screen_w - 40.0f);
        if (selected.is_selected())
            ImGui::TextWrapped("Couldn't read %s. Rebuild the trajectory files with "
                               "\"python -m physics.build_trajectories\" (see physics/README.md).",
                               selected.trajectory_path.c_str());
        else
            ImGui::TextWrapped("Pick a merger in the Classifier to watch it.");
        ImGui::PopTextWrapPos();
    } else {
        const float panel_x = 40.0f + 200.0f + 24.0f; // right of the back button
        const float panel_h = 140.0f;
        draw_controls(p, ImVec2(panel_x, screen_h - 40.0f - panel_h), ImVec2(screen_w - panel_x - 40.0f, panel_h));
        draw_info_panel(p, selected, current_time(p));
        ImVec2 hints = draw_hints(screen_w);
        draw_clock(p, current_time(p), screen_w, hints);
    }

    if (back_button("< Back to Classifier")) {
        stop_playback(selected); // so the next merger opened starts from the top
        next = Screen::Demo;
    }

    // --- Orbit camera: drag to rotate, scroll to zoom ---
    // Submitted last so the real widgets get first claim on clicks (see
    // spacetime_simulation_screen.cpp).
    if (playing_back) {
        ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
        ImGui::InvisibleButton("##viewport", ImVec2(screen_w, screen_h));
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            p.cam_azimuth -= io.MouseDelta.x * 0.01f;
            p.cam_elevation = std::max(0.05f, std::min(1.45f, p.cam_elevation + io.MouseDelta.y * 0.01f));
        }
        if (ImGui::IsItemHovered())
            p.cam_distance = std::max(8.0f, std::min(120.0f, p.cam_distance - io.MouseWheel * 3.0f));
    }

    ImGui::End();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(11);

    ImGui::Render();
    glViewport(0, 0, (GLsizei)screen_w, (GLsizei)screen_h);
    glClearColor(0.0f, 0.0f, playing_back ? 0.01f : 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (playing_back) {
        glEnable(GL_DEPTH_TEST);
        render_merger_scene(screen_w, screen_h, p.trajectory, current_time(p),
                            p.cam_azimuth, p.cam_elevation, p.cam_distance);
        glDisable(GL_DEPTH_TEST);
    }
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

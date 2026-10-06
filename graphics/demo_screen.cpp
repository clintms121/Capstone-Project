#include "screen.h"
#include "demo_data.h"
#include "fonts.h"
#include "simulation_params.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

const ImVec4 kAccent(0.40f, 0.70f, 1.00f, 1.0f);
const ImVec4 kGood(0.40f, 0.85f, 0.50f, 1.0f);
const ImVec4 kWarn(1.00f, 0.70f, 0.30f, 1.0f);
const ImVec4 kBad(1.00f, 0.45f, 0.40f, 1.0f);
const ImVec4 kQuiet(0.62f, 0.62f, 0.66f, 1.0f);
const ImVec4 kGrid(0.28f, 0.28f, 0.32f, 1.0f);

constexpr float kStageSeconds = 0.35f;
constexpr float kFreqLo = 10.0f, kFreqHi = 2048.0f;

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

ImVec4 verdict_color(const std::string& label) {
    if (is_real(label)) return kGood;
    if (label == "No_Glitch") return kQuiet;
    return kWarn;
}

std::string list_label(const DemoSample& s) {
    return s.kind == "event" ? s.name + "  (" + s.detector + ")" : s.name;
}

// The pipeline steps this kind of input went through, in order. Index of the
// step after which the Q-scan exists is returned through `qscan_step`.
std::vector<const char*> stages_for(const DemoSample& s, int& qscan_step) {
    if (s.kind == "glitch") {
        qscan_step = 0;
        return {"Load Gravity Spy's Q-scan", "Run the CNN classifier"};
    }
    qscan_step = 1;
    if (s.kind == "noise")
        return {"Whiten the strain", "Build the Q-scan", "Run the CNN classifier"};
    return {"Whiten the strain", "Build the Q-scan", "Run the CNN classifier",
            "Matched-filter ~3,000 predicted chirps"};
}

void draw_dashed_hline(ImDrawList* dl, float x0, float x1, float y, ImU32 c) {
    for (float x = x0; x < x1; x += 10.0f)
        dl->AddLine(ImVec2(x, y), ImVec2(std::fmin(x + 5.0f, x1), y), c, 1.0f);
}

// ---------------------------------------------------------------- Try it tab

void draw_sample_list(DemoData& demo, int& selected_id, int& filter, bool& held_out_only) {
    ImGui::TextColored(kAccent, "Choose an input");
    ImGui::RadioButton("Real mergers", &filter, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Glitches", &filter, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Noise", &filter, 2);
    ImGui::Checkbox("Held-out only (never seen in training)", &held_out_only);
    ImGui::Separator();

    const char* kinds[] = {"event", "glitch", "noise"};
    ImGui::BeginChild("##list_items");
    for (DemoSample& s : demo.samples) {
        if (s.kind != kinds[filter] || (held_out_only && s.split != "test"))
            continue;
        ImGui::PushID(s.id);
        std::string label = "      " + list_label(s);
        if (ImGui::Selectable(label.c_str(), s.id == selected_id))
            selected_id = s.id;
        ImVec2 r0 = ImGui::GetItemRectMin(), r1 = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddCircleFilled(
            ImVec2(r0.x + 9.0f, (r0.y + r1.y) * 0.5f), 4.5f, col(verdict_color(s.pred[0])));
        if (s.kind == "event" && s.split == "test") // glitch and noise samples are all held out
            ImGui::GetWindowDrawList()->AddText(ImVec2(r1.x - 62.0f, r0.y), col(kQuiet), "held out");
        ImGui::PopID();
    }
    ImGui::EndChild();
}

void draw_qscan_panel(DemoSample* s, bool visible, bool working) {
    ImGui::TextColored(kAccent, "Q-scan: what the classifier sees");
    text_wrapped_colored(kQuiet, "1 second of data: time across, pitch up the side, brightness = loudness");

    ImVec2 avail = ImGui::GetContentRegionAvail();
    const float left = 64.0f, bottom = 44.0f, gap = 10.0f;
    float w = avail.x - left - 16.0f, h = w * 140.0f / 170.0f;
    if (h > avail.y - bottom - gap) {
        h = avail.y - bottom - gap;
        w = h * 170.0f / 140.0f;
    }
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 p0(origin.x + left, origin.y + gap), p1(p0.x + w, p0.y + h);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    unsigned int tex = (s && visible) ? demo_texture(*s) : 0;
    if (tex) {
        dl->AddImage((ImTextureID)(intptr_t)tex, p0, p1);
    } else {
        dl->AddRectFilled(p0, p1, col(ImVec4(0.07f, 0.07f, 0.09f, 1.0f)));
        const char* msg = working ? "building the Q-scan..." : (s ? "image not found -- rerun ml/export_demo.py" : "");
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2((p0.x + p1.x - ts.x) * 0.5f, (p0.y + p1.y - ts.y) * 0.5f), col(kQuiet), msg);
    }
    dl->AddRect(p0, p1, col(kGrid));

    const float freqs[] = {16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f};
    for (float f : freqs) {
        float y = p1.y - std::log(f / kFreqLo) / std::log(kFreqHi / kFreqLo) * h;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%g Hz", f);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddLine(ImVec2(p0.x - 4.0f, y), ImVec2(p0.x, y), col(kQuiet));
        dl->AddText(ImVec2(p0.x - 8.0f - ts.x, y - ts.y * 0.5f), col(kQuiet), buf);
    }
    for (int i = 0; i <= 4; ++i) {
        float t = -0.5f + 0.25f * i, x = p0.x + (t + 0.5f) * w;
        char buf[16];
        std::snprintf(buf, sizeof(buf), i == 2 ? "0" : "%+.2f", t);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddLine(ImVec2(x, p1.y), ImVec2(x, p1.y + 4.0f), col(kQuiet));
        dl->AddText(ImVec2(x - ts.x * 0.5f, p1.y + 6.0f), col(kQuiet), buf);
    }
    const char* xlab = "seconds from the event";
    ImVec2 xs = ImGui::CalcTextSize(xlab);
    dl->AddText(ImVec2((p0.x + p1.x - xs.x) * 0.5f, p1.y + 24.0f), col(kQuiet), xlab);

    ImGui::Dummy(ImVec2(avail.x, h + bottom + gap));
}

void describe_sample(const DemoSample& s) {
    if (s.kind == "event")
        ImGui::TextWrapped("What it really is: a confirmed merger from LIGO's catalog, seen by the %s detector.",
                           s.detector == "H1" ? "Hanford" : "Livingston");
    else if (is_real(s.true_label))
        ImGui::TextWrapped("What it really is: a fake merger signal LIGO injected into the detector on purpose.");
    else if (s.kind == "glitch")
        ImGui::TextWrapped("What it really is: a glitch Gravity Spy volunteers labeled \"%s\".",
                           pretty_label(s.true_label).c_str());
    else
        ImGui::TextWrapped("What it really is: plain detector noise, a few seconds away from a merger.");

    if (s.split == "test")
        text_wrapped_colored(kGood, "Held out: the model never saw this during training.");
    else if (s.split == "val")
        text_wrapped_colored(kWarn, "Used to pick the best model during training.");
    else
        text_wrapped_colored(kWarn, "Used in training, so the model has seen it before.");
}

void draw_steps(const std::vector<const char*>& stages, int done) {
    for (int i = 0; i < (int)stages.size(); ++i) {
        if (i < done)
            ImGui::TextColored(kGood, "  [done]  %s", stages[i]);
        else if (i == done)
            ImGui::TextColored(kAccent, "  [ .. ]  %s", stages[i]);
        else
            ImGui::TextColored(kQuiet, "  [    ]  %s", stages[i]);
    }
}

void draw_verdict(const DemoSample& s) {
    ImGui::TextColored(kQuiet, "Classifier verdict");
    ImGui::PushFont(g_font_subtitle);
    ImGui::TextColored(verdict_color(s.pred[0]), "%s", pretty_label(s.pred[0]).c_str());
    ImGui::PopFont();

    if (s.pred[0] == s.true_label)
        ImGui::TextColored(kGood, "Correct");
    else
        text_wrapped_colored(kBad, "Wrong: it's really %s", pretty_label(s.true_label).c_str());

    ImGui::Dummy(ImVec2(0.0f, 4.0f));
    ImGui::TextColored(kQuiet, "Top 3 guesses (confidence)");
    for (int k = 0; k < 3; ++k) {
        char overlay[96];
        std::snprintf(overlay, sizeof(overlay), "%s  %.1f%%", pretty_label(s.pred[k]).c_str(), 100.0f * s.prob[k]);
        ImVec4 c = verdict_color(s.pred[k]);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(c.x * 0.6f, c.y * 0.6f, c.z * 0.6f, 1.0f));
        ImGui::ProgressBar(s.prob[k], ImVec2(-1.0f, 20.0f), overlay);
        ImGui::PopStyleColor();
    }
}

void draw_mass(const DemoSample& s, double threshold) {
    ImGui::TextColored(kQuiet, "Chirp mass (as seen from Earth)");
    if (std::isnan(s.mc_est)) {
        ImGui::TextWrapped("No matched-filter result for this recording.");
        return;
    }
    if (!s.measurable) {
        text_wrapped_colored(kWarn, "Too faint here to weigh reliably");
        ImGui::TextWrapped("Best match strength (SNR) is %.1f, but pure noise alone reaches %.1f, so the "
                           "best-matching mass could be chance. LIGO's published value: %.1f solar masses.",
                           s.snr, threshold, s.mc_pub);
        return;
    }
    float err = 100.0f * std::fabs(s.mc_est / s.mc_pub - 1.0f);
    ImGui::PushFont(g_font_subtitle);
    ImGui::Text("%.2f solar masses", s.mc_est);
    ImGui::PopFont();
    ImGui::Text("LIGO's published value: %.2f", s.mc_pub);
    ImGui::TextColored(err < 10.0f ? kGood : kWarn, "Off by %.1f%%", err);
    text_wrapped_colored(kQuiet, "Match strength (SNR) %.1f, above the %.1f that noise alone can reach.", s.snr,
                         threshold);
}

void draw_analysis(DemoData& demo, DemoSample& s, double elapsed, const std::vector<SimulationParams>& events,
                   SimulationParams& selected_event, Screen& next) {
    ImGui::PushFont(g_font_subtitle);
    ImGui::TextWrapped("%s", s.kind == "event" ? s.name.c_str() : pretty_label(s.true_label).c_str());
    ImGui::PopFont();
    describe_sample(s);
    ImGui::Separator();

    int qscan_step = 0;
    std::vector<const char*> stages = stages_for(s, qscan_step);
    int done = (int)(elapsed / kStageSeconds);
    draw_steps(stages, done);
    int classify_step = qscan_step + 1;
    ImGui::Separator();

    if (done > classify_step)
        draw_verdict(s);
    if (s.kind == "event" && done > classify_step + 1) {
        ImGui::Separator();
        draw_mass(s, demo.metric("mass_threshold_snr"));

        for (const SimulationParams& ev : events) {
            if (ev.event_id != s.name)
                continue;
            ImGui::Dummy(ImVec2(0.0f, 6.0f));
            if (ImGui::Button("View this merger's simulation (in progress)", ImVec2(-1.0f, 28.0f))) {
                selected_event = ev;
                next = Screen::Simulation;
            }
            break;
        }
    }
}

void draw_try_it(DemoData& demo, const std::vector<SimulationParams>& events, SimulationParams& selected_event,
                 Screen& next) {
    static int selected_id = -1;
    static int filter = 0;
    static bool held_out_only = false;
    static int revealed_id = -1;
    static double reveal_start = 0.0;

    if (selected_id < 0) {
        for (const DemoSample& s : demo.samples)
            if (s.kind == "event" && s.name == "GW150914" && s.detector == "H1")
                selected_id = s.id;
        if (selected_id < 0)
            selected_id = demo.samples.front().id;
    }
    DemoSample* sel = nullptr;
    for (DemoSample& s : demo.samples)
        if (s.id == selected_id)
            sel = &s;
    if (selected_id != revealed_id) {
        revealed_id = selected_id;
        reveal_start = ImGui::GetTime();
    }
    double elapsed = ImGui::GetTime() - reveal_start;

    float spacing = ImGui::GetStyle().ItemSpacing.x;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float list_w = 310.0f, info_w = 390.0f;
    float mid_w = avail.x - list_w - info_w - 2.0f * spacing;

    ImGui::BeginChild("##list", ImVec2(list_w, avail.y), true);
    draw_sample_list(demo, selected_id, filter, held_out_only);
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("##qscan", ImVec2(mid_w, avail.y), true, ImGuiWindowFlags_NoScrollbar);
    int qscan_step = 0;
    if (sel)
        stages_for(*sel, qscan_step);
    bool qscan_ready = sel && elapsed / kStageSeconds > qscan_step + 1;
    draw_qscan_panel(sel, qscan_ready, sel && !qscan_ready);
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("##analysis", ImVec2(info_w, avail.y), true);
    if (sel) {
        draw_analysis(demo, *sel, elapsed, events, selected_event, next);
        ImGui::Dummy(ImVec2(0.0f, 8.0f));
        if (ImGui::SmallButton("Replay steps"))
            reveal_start = ImGui::GetTime();
    }
    ImGui::EndChild();
}

// ---------------------------------------------------- How well it works tab

void metric_card(const char* id, const char* value, const char* caption, float w, float h) {
    ImGui::BeginChild(id, ImVec2(w, h), true, ImGuiWindowFlags_NoScrollbar);
    ImGui::PushFont(g_font_subtitle);
    ImGui::TextColored(kAccent, "%s", value);
    ImGui::PopFont();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(kQuiet, "%s", caption);
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
}

struct PlotArea {
    ImVec2 p0, p1;
};

// Reserves the rest of the child window for a plot, leaving room for axis
// labels, and returns the inner plotting rectangle.
PlotArea reserve_plot(float left, float bottom) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 o = ImGui::GetCursorScreenPos();
    ImGui::Dummy(avail);
    return {ImVec2(o.x + left, o.y + 8.0f), ImVec2(o.x + avail.x - 16.0f, o.y + avail.y - bottom)};
}

void draw_efficiency_plot(const DemoData& demo) {
    ImGui::TextColored(kAccent, "How loud a chirp has to be to get caught");
    ImGui::TextColored(kQuiet, "Simulated chirps hidden in real held-out noise (injection test)");

    PlotArea a = reserve_plot(48.0f, 44.0f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    auto X = [&](float snr) { return a.p0.x + snr / 32.0f * (a.p1.x - a.p0.x); };
    auto Y = [&](float frac) { return a.p1.y - frac * (a.p1.y - a.p0.y); };

    for (int i = 0; i <= 4; ++i) {
        float y = Y(0.25f * i);
        dl->AddLine(ImVec2(a.p0.x, y), ImVec2(a.p1.x, y), col(kGrid));
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%d%%", 25 * i);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddText(ImVec2(a.p0.x - 6.0f - ts.x, y - ts.y * 0.5f), col(kQuiet), buf);
    }
    for (int s = 0; s <= 30; s += 5) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%d", s);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddText(ImVec2(X((float)s) - ts.x * 0.5f, a.p1.y + 4.0f), col(kQuiet), buf);
    }
    const char* xlab = "signal-to-noise ratio of the hidden chirp";
    ImVec2 xs = ImGui::CalcTextSize(xlab);
    dl->AddText(ImVec2((a.p0.x + a.p1.x - xs.x) * 0.5f, a.p1.y + 22.0f), col(kQuiet), xlab);

    ImVec2 prev;
    bool have_prev = false;
    for (const InjectionPoint& p : demo.injection) {
        if (p.n <= 0)
            continue;
        float frac = (float)p.caught / p.n;
        if (p.snr == 0.0f) {
            draw_dashed_hline(dl, a.p0.x, a.p1.x, Y(frac), col(kQuiet));
            char buf[48];
            std::snprintf(buf, sizeof(buf), "plain noise flagged %.1f%%", 100.0f * frac);
            ImVec2 ts = ImGui::CalcTextSize(buf);
            dl->AddText(ImVec2(a.p1.x - ts.x, Y(frac) - ts.y - 2.0f), col(kQuiet), buf);
            continue;
        }
        ImVec2 pt(X(p.snr), Y(frac));
        if (have_prev)
            dl->AddLine(prev, pt, col(kAccent), 2.0f);
        dl->AddCircleFilled(pt, 4.5f, col(kAccent));
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%.0f%%", 100.0f * frac);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddText(ImVec2(pt.x - ts.x - 6.0f, pt.y - ts.y - 2.0f), col(ImVec4(1, 1, 1, 1)), buf);
        prev = pt;
        have_prev = true;
    }
}

void draw_mass_plot(const DemoData& demo) {
    ImGui::TextColored(kAccent, "Measured vs. published chirp mass");
    ImGui::TextColored(kQuiet, "One dot per merger (its clearer detector); blue = clear enough to weigh");

    PlotArea a = reserve_plot(48.0f, 44.0f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float lo = std::log10(0.8f), hi = std::log10(250.0f);
    auto X = [&](float m) { return a.p0.x + (std::log10(m) - lo) / (hi - lo) * (a.p1.x - a.p0.x); };
    auto Y = [&](float m) { return a.p1.y - (std::log10(m) - lo) / (hi - lo) * (a.p1.y - a.p0.y); };

    const float ticks[] = {1.0f, 10.0f, 100.0f};
    for (float t : ticks) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%g", t);
        ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddLine(ImVec2(a.p0.x, Y(t)), ImVec2(a.p1.x, Y(t)), col(kGrid));
        dl->AddLine(ImVec2(X(t), a.p0.y), ImVec2(X(t), a.p1.y), col(kGrid));
        dl->AddText(ImVec2(a.p0.x - 6.0f - ts.x, Y(t) - ts.y * 0.5f), col(kQuiet), buf);
        dl->AddText(ImVec2(X(t) - ts.x * 0.5f, a.p1.y + 4.0f), col(kQuiet), buf);
    }
    dl->AddLine(ImVec2(X(0.8f), Y(0.8f)), ImVec2(X(250.0f), Y(250.0f)), col(kQuiet), 1.0f);
    const char* xlab = "LIGO's published value (solar masses); vertical: this tool's estimate";
    ImVec2 xs = ImGui::CalcTextSize(xlab);
    dl->AddText(ImVec2((a.p0.x + a.p1.x - xs.x) * 0.5f, a.p1.y + 22.0f), col(kQuiet), xlab);

    std::map<std::string, const DemoSample*> best;
    for (const DemoSample& s : demo.samples) {
        if (s.kind != "event" || std::isnan(s.mc_pub) || std::isnan(s.snr))
            continue;
        auto it = best.find(s.name);
        if (it == best.end() || s.snr > it->second->snr)
            best[s.name] = &s;
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& kv : best) {
            const DemoSample& s = *kv.second;
            if (s.measurable != (pass == 1))
                continue;
            float est = std::fmin(std::fmax(s.mc_est, 0.8f), 250.0f);
            ImVec4 c = s.measurable ? kAccent : ImVec4(0.45f, 0.45f, 0.50f, 1.0f);
            dl->AddCircleFilled(ImVec2(X(s.mc_pub), Y(est)), s.measurable ? 3.5f : 2.5f, col(c));
        }
    }
}

void draw_performance(const DemoData& demo) {
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float card_w = (ImGui::GetContentRegionAvail().x - 3.0f * spacing) / 4.0f, card_h = 96.0f;
    char v[4][32], c[4][160];
    std::snprintf(v[0], sizeof(v[0]), "%.1f%%", 100.0 * demo.metric("accuracy"));
    std::snprintf(c[0], sizeof(c[0]), "overall accuracy on %d held-out images, %d classes",
                  (int)demo.metric("test_images"), (int)demo.metric("classes"));
    std::snprintf(v[1], sizeof(v[1]), "%.1f%%", 100.0 * demo.metric("real_recall"));
    std::snprintf(c[1], sizeof(c[1]), "of held-out real signals caught (%.0f%% of its \"real\" calls are right)",
                  100.0 * demo.metric("real_precision"));
    std::snprintf(v[2], sizeof(v[2]), "%d of %d", (int)demo.metric("noise_called_real"),
                  (int)demo.metric("noise_total"));
    std::snprintf(c[2], sizeof(c[2]), "pure-noise images mistaken for a signal");
    std::snprintf(v[3], sizeof(v[3]), "%.1f%%", 100.0 * demo.metric("mass_median_error"));
    std::snprintf(c[3], sizeof(c[3]), "median chirp-mass error; %d of %d mergers clear enough to weigh",
                  (int)demo.metric("mass_measurable"), (int)demo.metric("mass_events"));
    for (int i = 0; i < 4; ++i) {
        if (i)
            ImGui::SameLine();
        char id[16];
        std::snprintf(id, sizeof(id), "##card%d", i);
        metric_card(id, v[i], c[i], card_w, card_h);
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float plot_w = (avail.x - spacing) * 0.5f;
    ImGui::BeginChild("##eff", ImVec2(plot_w, avail.y), true, ImGuiWindowFlags_NoScrollbar);
    draw_efficiency_plot(demo);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##mass", ImVec2(plot_w, avail.y), true, ImGuiWindowFlags_NoScrollbar);
    draw_mass_plot(demo);
    ImGui::EndChild();
}

} // namespace

// Classifier demo: browse real LIGO mergers, Gravity Spy glitches and detector
// noise, and see what the trained pipeline made of each -- the Q-scan, the
// CNN's verdict, and for mergers the matched-filter chirp mass. Results come
// from ml/export_demo.py; nothing is computed here.
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
    ImGui::PushStyleColor(ImGuiCol_Button,          ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,   ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,    ImVec4(0.38f, 0.38f, 0.38f, 1));

    ImGui::Begin("##demo", nullptr, flags);

    ImGui::SetCursorPos(ImVec2(40.0f, 24.0f));
    ImGui::PushFont(g_font_subtitle);
    ImGui::Text("Classifier Demo");
    ImGui::PopFont();
    ImGui::SetCursorPos(ImVec2(40.0f, 58.0f));
    ImGui::TextColored(kQuiet, "Real LIGO data through the trained model. Results were computed by the Python "
                               "pipeline (ml/export_demo.py) and are replayed here step by step.");

    float back_h = 28.0f;
    ImGui::SetCursorPos(ImVec2(40.0f, 86.0f));
    ImGui::BeginChild("##demo_body", ImVec2(screen_w - 80.0f, screen_h - 86.0f - back_h - 56.0f));
    if (!demo.loaded()) {
        ImGui::TextWrapped("No demo data found in data/demo/. From the project folder, run the pipeline "
                           "(see ml/README.md), then:\n\n    python -m ml.export_demo");
    } else if (ImGui::BeginTabBar("##demo_tabs")) {
        if (ImGui::BeginTabItem("Try it on real data")) {
            draw_try_it(demo, events, selected_event, next);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("How well it works")) {
            draw_performance(demo);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    ImGui::SetCursorPos(ImVec2(40.0f, screen_h - back_h - 40.0f));
    if (ImGui::Button("Back", ImVec2(90.0f, back_h)))
        next = Screen::Menu;

    ImGui::End();
    ImGui::PopStyleColor(5);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

// The Under the Hood screen: five slides that explain the science and the
// model in plain language -- black holes, LIGO's data, spacetime, how the
// classifier works, and how well it does. Arrows (or the arrow keys) move
// between slides; on the last one, the next arrow opens the Classifier.
//
// What's where: the slide text and images (k_slides), text layout helpers,
// the Space Time Simulation button, the "How the ML Model Works" slide
// (its animation is in cnn_animation.cpp), the accuracy chart, the general
// slide layout (draw_slide), and the screen itself at the bottom.
#include "screen.h"
#include "back_button.h"
#include "cnn_animation.h"
#include "fonts.h"
#include "image_texture.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{

    struct SlideImage
    {
        const char *path; // relative to the repo root
        const char *caption;
        ImVec2 uv0 = ImVec2(0.0f, 0.0f); // crop rectangle within the image, 0..1
        ImVec2 uv1 = ImVec2(1.0f, 1.0f);
    };

    // Custom drawn visual for slides that need more than text and images.
    enum class SlideVisual
    {
        None,
        ClassifierAnimation, // text on the left; CNN animation over the equations image on the right
        AccuracyChart,       // text on the left; bar chart of the test results on the right
    };

    struct Slide
    {
        const char *heading;
        const char *body;
        bool spacetime_button;              // show the "Space Time Simulation" link under the text
        const SlideImage *images = nullptr; // shown side by side between heading and text
        int image_count = 0;
        SlideVisual visual = SlideVisual::None;
    };

    const SlideImage k_black_hole_images[] = {
        {"imgs/jTVzX3HFWeZTNJJVtLESbU.jpg",
         "M87*: the first real photo of a black hole (EHT Collaboration, 2019)"},
        {"imgs/Black_hole.jpg",
         "Gargantua from Interstellar (2014): a simulation built from Einstein's equations"},
    };

    const SlideImage k_ligo_images[] = {
        {"imgs/labeled_quad.jpg",
         "The four-stage pendulum each mirror hangs from (LIGO Lab)"},
        // The figure has black bars on both sides; crop to the plot itself.
        {"imgs/data_H1L1-H1_DCS-CALIB_STRAIN_C01-L1_DCS-CALIB_STRAIN_C01-qscan.jpg",
         "Real chirp GW170104 seen at both LIGO sites (LIGO/Virgo, 2017)",
         ImVec2(108.0f / 620.0f, 0.0f), ImVec2(513.0f / 620.0f, 1.0f)},
    };

    const Slide k_slides[] = {
        {"What is a Black Hole?",
         "A black hole is a place where so much mass is packed into so small a "
         "space that nothing can escape its gravity -- not even light. You "
         "might notice the bottom image is from the movie Interstellar. "
         "The black hole in Interstellar took up to 100 hours to render a "
         "single frame, because the effects team traced how its gravity bends "
         "light using equations from physicist Kip Thorne -- who later shared "
         "the 2017 Nobel Prize for his work on LIGO. The edge "
         "of no return is called the event horizon. Many form when a huge star "
         "collapses at the end of its life; others, like M87* (6.5 billion times "
         "the Sun's mass), sit at the centers of galaxies. The black hole itself "
         "is invisible, but the glowing ring around it is superheated gas whose "
         "light is bent by its gravity. When two black holes collide, they send "
         "ripples through spacetime -- the signals LIGO detects.",
         false,
         k_black_hole_images,
         (int)(sizeof(k_black_hole_images) / sizeof(k_black_hole_images[0]))},

        {"Input Data",
         "LIGO measures strain: how much its two 4 km arms stretch and squeeze. "
         "At the end of each arm hangs a mirror -- a 40 kg block of ultra-pure "
         "glass -- and the laser bounces between the mirrors about 280 times, so "
         "tiny stretches add up. To block everyday shaking from traffic, ocean "
         "waves and earthquakes, each mirror hangs from a four-stage pendulum, "
         "the last stage on glass fibers just 0.4 mm thick, all inside one of "
         "the largest vacuum systems in the world. The public data I use "
         "records the strain 4,096 times per second. It looks like pure noise, "
         "so I clean it up and turn each second into a 140x170 time-frequency "
         "image where brightness shows signal strength -- the image the model "
         "actually looks at.",
         false,
         k_ligo_images,
         (int)(sizeof(k_ligo_images) / sizeof(k_ligo_images[0]))},

        {"Spacetime and Gravitational Waves",
         "Spacetime is space and time woven together into one stretchy fabric. "
         "Mass bends it, and gravity is just objects following those bends: "
         "Earth circles the Sun because the Sun dents the fabric around it. A "
         "black hole dents it so deeply that nothing can climb back out -- "
         "compare the two in the Space Time Simulation below.\n"
         "When two black holes spiral together, they shake the fabric and send "
         "out ripples called gravitational waves. Each LIGO detector is an L "
         "with two 4 km arms. A passing wave stretches one arm and squeezes the "
         "other by about a thousandth of a proton's width. Lasers track that "
         "change thousands of times per second, and that recording is what my "
         "model analyzes.",
         true},

        {"How the ML Model Works",
         "The classifier is a convolutional neural network (CNN), a type of "
         "model built for pictures. It reads the 140x170 Q-scan through small "
         "filters that slide across the image, each one searching for its own "
         "pattern. The first layers react to simple features like edges and "
         "bright spots; deeper layers -- more filters, coarser resolution -- "
         "combine those into shapes, like the upward-sweeping chirp of two "
         "black holes spiraling together. The 256 final feature maps are each "
         "averaged down to one number, and those numbers become a probability "
         "for each of 22 labels: a real signal, or one of 21 kinds of glitch "
         "or noise. On images it never saw in training, it picks the right "
         "label 97.9% of the time.",
         false,
         nullptr,
         0,
         SlideVisual::ClassifierAnimation},

        // Last slide: its next arrow turns into the link to the Classifier screen.
        {"Accuracy and Application",
         "I tested the model on 1,559 images it never saw while training. It "
         "picked the right label 97.9% of the time and caught 64 of 65 real "
         "black hole mergers. When it says \"merger,\" it is right about 9 "
         "times out of 10 -- every mistake was plain noise. For mergers clear "
         "enough to measure, the mass it estimates lands within 10% of LIGO's "
         "official value 91% of the time.\n"
         "This can be used in practice as a first filter for detecting black "
         "hole mergers. It isn't a replacement for LIGO's own searches, but it "
         "can flag the most likely mergers for a closer look and sort out "
         "glitches, helping engineers track down what's causing faulty "
         "signals -- all while picking out black hole mergers automatically.",
         false,
         nullptr,
         0,
         SlideVisual::AccuracyChart},
    };

    const int k_slide_count = (int)(sizeof(k_slides) / sizeof(k_slides[0]));
    const float k_slide_seconds = 0.4f;

    const ImVec2 k_spacetime_btn_size(320.0f, 54.0f);

    // Slide state persists across frames (and across a visit to the
    // Space Time Simulation, so coming back lands on the same slide).
    int g_current = 0;
    int g_from = -1; // slide being animated out, or -1 when idle
    int g_dir = 1;   // +1 = moving forward (content slides left)
    float g_anim_t = 1.0f;
    float g_flip = 0.0f; // 0..1: next arrow turning over into the demo link on the last slide

    const float k_flip_seconds = 0.5f;
    const char *k_demo_link_label = "See the model in action";

    float ease_in_out_cubic(float t)
    {
        return t < 0.5f ? 4.0f * t * t * t
                        : 1.0f - powf(-2.0f * t + 2.0f, 3.0f) * 0.5f;
    }

    void go_to(int index)
    {
        if (g_from >= 0 || index < 0 || index >= k_slide_count || index == g_current)
            return;
        g_dir = index > g_current ? 1 : -1;
        g_from = g_current;
        g_current = index;
        g_anim_t = 0.0f;
    }

    // Word-wraps `text` to a column `wrap_w` wide starting at `x`. Each line
    // is placed by `align` (0 = left, 0.5 = centered). A '\n' starts a new
    // paragraph, half a line further down. With draw == false it only
    // measures. Returns the total height.
    float wrapped_text(ImDrawList *dl, ImFont *font, float x, float y, float wrap_w,
                       float align, ImU32 col, const char *text, bool draw)
    {
        const float size = font->FontSize;
        const float line_h = floorf(size * 1.35f);
        const char *s = text;
        const char *end = text + strlen(text);
        float top = y;

        while (s < end)
        {
            const char *para_end = s;
            while (para_end < end && *para_end != '\n')
                ++para_end;

            while (s < para_end)
            {
                const char *line_end = font->CalcWordWrapPositionA(1.0f, s, para_end, wrap_w);
                if (line_end == s)
                    line_end = s + 1;

                const char *trimmed = line_end;
                while (trimmed > s && trimmed[-1] == ' ')
                    --trimmed;

                if (draw)
                {
                    float w = font->CalcTextSizeA(size, FLT_MAX, 0.0f, s, trimmed).x;
                    dl->AddText(font, size, ImVec2(floorf(x + (wrap_w - w) * align), floorf(y)), col, s, trimmed);
                }
                y += line_h;

                s = line_end;
                while (s < para_end && *s == ' ')
                    ++s;
            }

            if (para_end < end)
            {
                s = para_end + 1;
                y += floorf(line_h * 0.5f);
            }
        }
        return y - top;
    }

    float centered_text(ImDrawList *dl, ImFont *font, float cx, float y,
                        float wrap_w, ImU32 col, const char *text, bool draw)
    {
        return wrapped_text(dl, font, cx - wrap_w * 0.5f, y, wrap_w, 0.5f, col, text, draw);
    }

    // "Space Time Simulation" link centered under the slide text at `y`, if
    // the slide has one. It's the slide's call to action, so it's drawn to
    // stand out: accent blue with a play icon and a softly pulsing halo.
    void draw_spacetime_button(const Slide &slide, int index, float cx, float y,
                               float alpha, bool interactive, Screen &next)
    {
        if (!slide.spacetime_button)
            return;
        auto col = [alpha](float r, float g, float b, float a)
        { return ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a * alpha)); };
        const ImVec2 size = k_spacetime_btn_size;
        const ImVec2 pos(floorf(cx - size.x * 0.5f), floorf(y + 32.0f));
        const ImVec2 end(pos.x + size.x, pos.y + size.y);
        const float rounding = 12.0f;

        ImGui::PushID(index);
        ImGui::SetCursorScreenPos(pos);
        if (ImGui::InvisibleButton("##spacetime", size) && interactive)
            next = Screen::SpacetimeSimulation;
        bool hovered = ImGui::IsItemHovered();
        bool held = ImGui::IsItemActive();
        ImGui::PopID();
        if (hovered && interactive)
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

        ImDrawList *dl = ImGui::GetWindowDrawList();
        // Halo: widening outlines that fade outward and breathe slowly.
        float pulse = 0.5f + 0.5f * sinf((float)ImGui::GetTime() * 2.5f);
        for (int i = 1; i <= 6; ++i)
        {
            float grow = 2.0f * (float)i + 3.0f * pulse;
            dl->AddRect(ImVec2(pos.x - grow, pos.y - grow), ImVec2(end.x + grow, end.y + grow),
                        col(0.45f, 0.70f, 1.0f, (0.24f - 0.035f * (float)i) * (0.55f + 0.45f * pulse)),
                        rounding + grow, 0, 2.0f);
        }

        float shade = held ? 0.85f : (hovered ? 1.15f : 1.0f);
        dl->AddRectFilled(pos, end, col(0.20f * shade, 0.44f * shade, 0.88f * shade, 1.0f), rounding);
        dl->AddRect(pos, end, col(0.72f, 0.86f, 1.0f, 0.9f), rounding, 0, 1.5f);

        // Play icon and label, centered together.
        const char *label = "Space Time Simulation";
        ImFont *font = g_font_body;
        ImVec2 text = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, label);
        float icon = floorf(font->FontSize * 0.6f);
        float gap = 10.0f;
        float x = floorf(cx - (icon + gap + text.x) * 0.5f);
        float mid_y = pos.y + size.y * 0.5f;
        dl->AddTriangleFilled(ImVec2(x, mid_y - icon * 0.5f), ImVec2(x, mid_y + icon * 0.5f),
                              ImVec2(x + icon * 0.87f, mid_y), col(1.0f, 1.0f, 1.0f, 1.0f));
        dl->AddText(font, font->FontSize, ImVec2(x + icon + gap, floorf(mid_y - text.y * 0.5f)),
                    col(1.0f, 1.0f, 1.0f, 1.0f), label);
    }

    // ---- "How the ML Model Works" visual --------------------------------

    // Drop the equations image here (either extension); until it exists the
    // slide shows a placeholder.
    const char *k_equation_image_paths[] = {"imgs/model_equations.png", "imgs/model_equations.jpg"};

    // Shown under the equations panel.
    const char *k_equations_note =
        "Math used to tune the model. Top: each training step nudges every "
        "weight in the direction that makes the model less wrong. Bottom: the "
        "nudges start big and shrink over 30 passes through the data, so the "
        "model settles in.";

    float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
    float lerpf(float a, float b, float t) { return a + (b - a) * t; }
    float smoothstep01(float t)
    {
        t = clamp01(t);
        return t * t * (3.0f - 2.0f * t);
    }

    ImU32 rgba(float r, float g, float b, float a)
    {
        return ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    }

    // The first equations image that loads, or nullptr if none has been added.
    const ImageTexture *equations_texture()
    {
        for (const char *path : k_equation_image_paths)
        {
            const ImageTexture &tex = load_image_texture(path);
            if (tex.texture)
                return &tex;
        }
        return nullptr;
    }

    // Height the equations panel actually takes in a box `size`: the fitted
    // image's height, or the whole box for the placeholder.
    float equations_panel_height(ImVec2 size)
    {
        const ImageTexture *tex = equations_texture();
        if (!tex)
            return size.y;
        return floorf(tex->height * std::min(size.x / tex->width, size.y / tex->height));
    }

    // The equations image, fitted into the panel -- or, until it's been
    // added, a dashed placeholder saying where to put it.
    void draw_equations_panel(ImDrawList *dl, ImVec2 pos, ImVec2 size, float alpha)
    {
        if (const ImageTexture *tex = equations_texture())
        {
            float scale = std::min(size.x / tex->width, size.y / tex->height);
            float w = floorf(tex->width * scale), h = floorf(tex->height * scale);
            ImVec2 a(floorf(pos.x + (size.x - w) * 0.5f), floorf(pos.y + (size.y - h) * 0.5f));
            dl->AddImageRounded((ImTextureID)(intptr_t)tex->texture, a, ImVec2(a.x + w, a.y + h),
                                ImVec2(0, 0), ImVec2(1, 1), rgba(1.0f, 1.0f, 1.0f, alpha), 8.0f);
            return;
        }

        ImU32 dash_col = rgba(0.4f, 0.5f, 0.7f, 0.8f * alpha);
        const float dash = 8.0f, space = 6.0f;
        ImVec2 corners[4] = {pos, ImVec2(pos.x + size.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y),
                             ImVec2(pos.x, pos.y + size.y)};
        for (int e = 0; e < 4; ++e)
        {
            ImVec2 a = corners[e], b = corners[(e + 1) % 4];
            float len = sqrtf((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
            for (float d = 0.0f; d < len; d += dash + space)
            {
                float d1 = std::min(d + dash, len);
                dl->AddLine(ImVec2(lerpf(a.x, b.x, d / len), lerpf(a.y, b.y, d / len)),
                            ImVec2(lerpf(a.x, b.x, d1 / len), lerpf(a.y, b.y, d1 / len)), dash_col, 1.5f);
            }
        }

        const char *title = "Model equations";
        const char *note = "Image coming soon -- save it as imgs/model_equations.png";
        ImFont *small = ImGui::GetFont();
        float title_w = g_font_body->CalcTextSizeA(g_font_body->FontSize, FLT_MAX, 0.0f, title).x;
        float note_w = small->CalcTextSizeA(small->FontSize, FLT_MAX, 0.0f, note).x;
        float block_h = g_font_body->FontSize + 8.0f + small->FontSize;
        float y = floorf(pos.y + (size.y - block_h) * 0.5f);
        dl->AddText(g_font_body, g_font_body->FontSize, ImVec2(floorf(pos.x + (size.x - title_w) * 0.5f), y),
                    rgba(0.75f, 0.8f, 0.9f, alpha), title);
        float note_y = y + g_font_body->FontSize + 8.0f;
        dl->AddText(small, small->FontSize, ImVec2(floorf(pos.x + (size.x - note_w) * 0.5f), note_y),
                    rgba(0.55f, 0.55f, 0.55f, alpha), note);
    }

    // ---- "Accuracy and Application" visual -------------------------------

    // Results from data/demo/metrics.csv (ml/export_demo.py; see ml/README.md).
    // Hardcoded like the slide text -- update both if the models are retrained.
    struct AccuracyBar
    {
        const char *label;
        float value; // 0..1
        const char *detail;
    };

    const AccuracyBar k_accuracy_bars[] = {
        {"Picks the right label", 0.979f, "1,559 test images, 22 labels"},
        {"Real mergers caught", 0.985f, "64 of 65"},
        {"Right when it says \"merger\"", 0.889f, "8 noise images were mistaken for mergers"},
        {"Mass within 10% of LIGO's value", 0.908f, "69 of 76 measurable mergers"},
    };

    // One horizontal bar per result on a 0-100% track, value to the right of
    // the track. Bars grow in with the slide (alpha 0 -> 1). With draw ==
    // false it only measures. Returns the chart's height.
    float draw_accuracy_chart(ImDrawList *dl, ImVec2 pos, float width, float alpha, bool draw)
    {
        ImFont *small = ImGui::GetFont();
        ImFont *body = g_font_body;
        const float title_gap = 22.0f;
        const float label_gap = 8.0f;
        const float bar_h = 18.0f;
        const float detail_gap = 6.0f;
        const float row_gap = 24.0f;
        const float row_h = body->FontSize + label_gap + bar_h + detail_gap + small->FontSize;
        const int rows = (int)(sizeof(k_accuracy_bars) / sizeof(k_accuracy_bars[0]));
        const float height = body->FontSize + title_gap + rows * row_h + (rows - 1) * row_gap;
        if (!draw)
            return height;

        const float value_w = body->CalcTextSizeA(body->FontSize, FLT_MAX, 0.0f, "100.0%").x;
        const float track_w = floorf(width - value_w - 12.0f);
        const float grow = smoothstep01(alpha);

        dl->AddText(body, body->FontSize, pos, rgba(0.92f, 0.92f, 0.92f, alpha), "Tested on unseen data");
        float y = pos.y + body->FontSize + title_gap;
        for (const AccuracyBar &bar : k_accuracy_bars)
        {
            dl->AddText(body, body->FontSize, ImVec2(pos.x, y), rgba(0.85f, 0.85f, 0.85f, alpha), bar.label);
            float bar_y = y + body->FontSize + label_gap;
            ImVec2 track0(pos.x, bar_y), track1(pos.x + track_w, bar_y + bar_h);
            dl->AddRectFilled(track0, track1, rgba(1.0f, 1.0f, 1.0f, 0.08f * alpha), 4.0f,
                              ImDrawFlags_RoundCornersRight);
            dl->AddRectFilled(track0, ImVec2(pos.x + floorf(track_w * bar.value * grow), track1.y),
                              rgba(0.45f, 0.70f, 1.0f, alpha), 4.0f, ImDrawFlags_RoundCornersRight);

            char value[16];
            std::snprintf(value, sizeof(value), "%.1f%%", 100.0f * bar.value);
            float value_y = floorf(bar_y + (bar_h - body->FontSize) * 0.5f);
            dl->AddText(body, body->FontSize, ImVec2(track1.x + 12.0f, value_y),
                        rgba(1.0f, 1.0f, 1.0f, alpha), value);
            dl->AddText(small, small->FontSize, ImVec2(pos.x, track1.y + detail_gap), rgba(0.6f, 0.6f, 0.6f, alpha),
                        bar.detail);
            y += row_h + row_gap;
        }
        return height;
    }

    struct Column
    {
        float x, w;
    };

    // Left column of a two-column slide: heading and body text, centered
    // vertically. Returns the right column.
    Column draw_text_column(const Slide &slide, float cx, float alpha, float area_top, float area_bottom,
                            float content_w)
    {
        ImDrawList *dl = ImGui::GetWindowDrawList();
        const float left = floorf(cx - content_w * 0.5f);
        const float col_gap = 36.0f;
        const float heading_w =
            g_font_heading->CalcTextSizeA(g_font_heading->FontSize, FLT_MAX, 0.0f, slide.heading).x;
        const float text_w = std::min(content_w * 0.55f, std::max(heading_w + 4.0f, content_w * 0.42f));
        const float avail_h = area_bottom - area_top;

        const float heading_gap = 20.0f;
        float heading_h = wrapped_text(dl, g_font_heading, 0.0f, 0.0f, text_w, 0.0f, 0, slide.heading, false);
        float body_h = wrapped_text(dl, g_font_body, 0.0f, 0.0f, text_w, 0.0f, 0, slide.body, false);
        float y = area_top + std::max(0.0f, (avail_h - heading_h - heading_gap - body_h) * 0.5f);
        wrapped_text(dl, g_font_heading, left, y, text_w, 0.0f, rgba(0.6f, 0.8f, 1.0f, alpha), slide.heading,
                     true);
        wrapped_text(dl, g_font_body, left, y + heading_h + heading_gap, text_w, 0.0f,
                     rgba(0.92f, 0.92f, 0.92f, alpha), slide.body, true);
        return {left + text_w + col_gap, content_w - text_w - col_gap};
    }

    // Two-column slide: heading and text on the left, the test-result bar
    // chart on the right.
    void draw_accuracy_slide(const Slide &slide, float cx, float alpha, float area_top, float area_bottom,
                             float content_w)
    {
        ImDrawList *dl = ImGui::GetWindowDrawList();
        const Column right = draw_text_column(slide, cx, alpha, area_top, area_bottom, content_w);
        float chart_h = draw_accuracy_chart(dl, ImVec2(0.0f, 0.0f), right.w, alpha, false);
        float y = floorf(area_top + std::max(0.0f, (area_bottom - area_top - chart_h) * 0.5f));
        draw_accuracy_chart(dl, ImVec2(right.x, y), right.w, alpha, true);
    }

    // Two-column slide: heading and text on the left, the classifier
    // animation above the equations panel (and its note) on the right.
    void draw_model_slide(const Slide &slide, float cx, float alpha, float area_top, float area_bottom,
                          float content_w)
    {
        ImDrawList *dl = ImGui::GetWindowDrawList();
        const Column right = draw_text_column(slide, cx, alpha, area_top, area_bottom, content_w);
        const float right_x = right.x;
        const float right_w = right.w;
        const float avail_h = area_bottom - area_top;

        const float panel_gap = 16.0f;
        const float note_gap = 8.0f;
        ImFont *note_font = ImGui::GetFont();
        float note_h = centered_text(dl, note_font, 0.0f, 0.0f, right_w, 0, k_equations_note, false);
        float anim_h = floorf(std::min(320.0f, avail_h * 0.6f));
        float eq_h = floorf(std::min(220.0f, avail_h - anim_h - panel_gap - note_gap - note_h));
        eq_h = equations_panel_height(ImVec2(right_w, eq_h));
        float right_h = anim_h + panel_gap + eq_h + note_gap + note_h;
        float ry = floorf(area_top + std::max(0.0f, (avail_h - right_h) * 0.5f));
        float eq_y = ry + anim_h + panel_gap;
        draw_classifier_animation(dl, ImVec2(right_x, ry), ImVec2(right_w, anim_h), alpha,
                                  (float)ImGui::GetTime());
        draw_equations_panel(dl, ImVec2(right_x, eq_y), ImVec2(right_w, eq_h), alpha);
        centered_text(dl, note_font, right_x + right_w * 0.5f, eq_y + eq_h + note_gap, right_w,
                      rgba(0.65f, 0.65f, 0.65f, alpha), k_equations_note, true);
    }

    // Draws one slide horizontally offset by `offset_x`, faded to `alpha`.
    // Only the settled (non-animating) slide accepts clicks. Text-only slides
    // use a `text_w` column; image slides spread across `content_w`.
    void draw_slide(int index, float offset_x, float alpha, bool interactive,
                    float screen_w, float area_top, float area_bottom, float text_w,
                    float content_w, Screen &next)
    {
        const Slide &slide = k_slides[index];
        ImDrawList *dl = ImGui::GetWindowDrawList();
        float cx = screen_w * 0.5f + offset_x;

        if (slide.visual == SlideVisual::ClassifierAnimation)
        {
            draw_model_slide(slide, cx, alpha, area_top, area_bottom, content_w);
            return;
        }
        if (slide.visual == SlideVisual::AccuracyChart)
        {
            draw_accuracy_slide(slide, cx, alpha, area_top, area_bottom, content_w);
            return;
        }

        ImU32 heading_col = ImGui::ColorConvertFloat4ToU32(ImVec4(0.6f, 0.8f, 1.0f, alpha));
        ImU32 body_col = ImGui::ColorConvertFloat4ToU32(ImVec4(0.92f, 0.92f, 0.92f, alpha));
        ImU32 caption_col = ImGui::ColorConvertFloat4ToU32(ImVec4(0.65f, 0.65f, 0.65f, alpha));
        ImU32 image_tint = ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, alpha));

        float heading_h = g_font_heading->FontSize;
        float heading_gap = 28.0f;
        float button_block = slide.spacetime_button ? 32.0f + k_spacetime_btn_size.y : 0.0f;

        const ImageTexture *textures[4] = {};
        int shown = 0;
        for (int i = 0; i < slide.image_count && i < 4; ++i)
        {
            const ImageTexture &tex = load_image_texture(slide.images[i].path);
            if (tex.texture)
            {
                textures[i] = &tex;
                ++shown;
            }
        }

        if (shown == 0)
        {
            // Text-only slide: heading and body centered, block centered vertically.
            float body_h = centered_text(dl, g_font_body, cx, 0.0f, text_w, 0, slide.body, false);
            float block_h = heading_h + heading_gap + body_h + button_block;
            float y = area_top + std::max(0.0f, (area_bottom - area_top - block_h) * 0.5f);

            centered_text(dl, g_font_heading, cx, y, screen_w, heading_col, slide.heading, true);
            y += heading_h + heading_gap;
            y += centered_text(dl, g_font_body, cx, y, text_w, body_col, slide.body, true);
            draw_spacetime_button(slide, index, cx, y, alpha, interactive, next);
            return;
        }

        // Image slide: images in a column on the left, text on the right.
        // Landscape images stack vertically; portrait ones sit side by side so
        // they don't get squeezed thin. Pick the widest image column for which
        // both columns still fit under the heading.
        ImFont *caption_font = ImGui::GetFont();
        const float col_gap = 40.0f;
        const float image_gap = 18.0f;
        const float caption_gap = 8.0f;
        const float avail_h = area_bottom - area_top - heading_h - heading_gap - button_block;

        float aspect[4] = {}; // displayed width / height, after cropping
        bool all_portrait = true;
        for (int i = 0; i < 4; ++i)
            if (textures[i])
            {
                const SlideImage &img = slide.images[i];
                aspect[i] = (textures[i]->width * (img.uv1.x - img.uv0.x)) /
                            (textures[i]->height * (img.uv1.y - img.uv0.y));
                all_portrait = all_portrait && aspect[i] < 1.0f;
            }
        const bool side_by_side = all_portrait && shown > 1;

        // Fills each image's size for a column `col_w` wide and returns the
        // column's total height including captions.
        ImVec2 sizes[4];
        float cap_h[4];
        auto layout_images = [&](float col_w)
        {
            float aspect_sum = 0.0f;
            for (int i = 0; i < 4; ++i)
                aspect_sum += aspect[i];
            float row_h = floorf((col_w - image_gap * (shown - 1)) / aspect_sum);

            float total = side_by_side ? 0.0f : image_gap * (shown - 1);
            float row_caption_h = 0.0f;
            for (int i = 0; i < 4; ++i)
            {
                if (!textures[i])
                    continue;
                sizes[i] = side_by_side ? ImVec2(floorf(row_h * aspect[i]), row_h)
                                        : ImVec2(col_w, floorf(col_w / aspect[i]));
                cap_h[i] = centered_text(dl, caption_font, 0.0f, 0.0f, sizes[i].x, 0,
                                         slide.images[i].caption, false);
                if (side_by_side)
                    row_caption_h = std::max(row_caption_h, cap_h[i]);
                else
                    total += sizes[i].y + caption_gap + cap_h[i];
            }
            return side_by_side ? row_h + caption_gap + row_caption_h : total;
        };

        float max_w = side_by_side ? floorf(content_w * 0.55f) : 380.0f;
        float image_w = 160.0f;
        for (float w = max_w; w >= 160.0f; w -= 10.0f)
        {
            float text_h = wrapped_text(dl, g_font_body, 0.0f, 0.0f, content_w - w - col_gap, 0.0f, 0,
                                        slide.body, false);
            if (layout_images(w) <= avail_h && text_h <= avail_h)
            {
                image_w = w;
                break;
            }
        }

        float body_w = content_w - image_w - col_gap;
        float images_h = layout_images(image_w);
        float body_h = wrapped_text(dl, g_font_body, 0.0f, 0.0f, body_w, 0.0f, 0, slide.body, false);
        float columns_h = std::max(images_h, body_h);
        float block_h = heading_h + heading_gap + columns_h + button_block;
        float y = area_top + std::max(0.0f, (area_bottom - area_top - block_h) * 0.5f);

        centered_text(dl, g_font_heading, cx, y, screen_w, heading_col, slide.heading, true);
        y += heading_h + heading_gap;

        // The shorter column is centered against the taller one.
        float left = floorf(cx - content_w * 0.5f);
        float img_x = left;
        float img_y = floorf(y + (columns_h - images_h) * 0.5f);
        for (int i = 0; i < 4; ++i)
        {
            if (!textures[i])
                continue;
            const SlideImage &img = slide.images[i];
            dl->AddImageRounded((ImTextureID)(intptr_t)textures[i]->texture,
                                ImVec2(img_x, img_y), ImVec2(img_x + sizes[i].x, img_y + sizes[i].y),
                                img.uv0, img.uv1, image_tint, 8.0f);
            centered_text(dl, caption_font, img_x + sizes[i].x * 0.5f, img_y + sizes[i].y + caption_gap,
                          sizes[i].x, caption_col, img.caption, true);
            if (side_by_side)
                img_x += sizes[i].x + image_gap;
            else
                img_y += sizes[i].y + caption_gap + cap_h[i] + image_gap;
        }

        wrapped_text(dl, g_font_body, left + image_w + col_gap, y + (columns_h - body_h) * 0.5f,
                     body_w, 0.0f, body_col, slide.body, true);
        y += columns_h;
        draw_spacetime_button(slide, index, cx, y, alpha, interactive, next);
    }

    // Round clickable arrow. `dir` is +1 for a right-pointing arrow, -1 for left.
    // `flip` (0..1) turns it upside down like a coin flipped top over bottom;
    // past halfway it shows its accent-blue back face.
    bool arrow_button(const char *id, ImVec2 center, float radius, int dir, float flip = 0.0f)
    {
        ImGui::SetCursorScreenPos(ImVec2(center.x - radius, center.y - radius));
        bool clicked = ImGui::InvisibleButton(id, ImVec2(radius * 2.0f, radius * 2.0f));
        bool hovered = ImGui::IsItemHovered();
        bool held = ImGui::IsItemActive();

        ImDrawList *dl = ImGui::GetWindowDrawList();
        float turn = cosf(flip * 3.14159265f); // 1 face up, -1 upside down
        float shade = held ? 0.85f : hovered ? 1.15f
                                             : 1.0f;
        ImVec2 radii(radius, std::max(1.0f, radius * fabsf(turn)));
        if (turn >= 0.0f)
        {
            float fill = held ? 0.38f : hovered ? 0.28f
                                                : 0.15f;
            dl->AddEllipseFilled(center, radii, rgba(fill, fill, fill, 1.0f), 0.0f, 48);
        }
        else
        {
            dl->AddEllipseFilled(center, radii, rgba(0.20f * shade, 0.44f * shade, 0.88f * shade, 1.0f), 0.0f, 48);
            dl->AddEllipse(center, radii, rgba(0.72f, 0.86f, 1.0f, 0.9f), 0.0f, 48, 1.5f);
        }

        float s = radius * 0.32f;
        float d = (float)dir;
        ImVec2 pts[3] = {
            ImVec2(center.x - d * s * 0.5f, center.y - s * turn),
            ImVec2(center.x + d * s * 0.5f, center.y),
            ImVec2(center.x - d * s * 0.5f, center.y + s * turn),
        };
        dl->AddPolyline(pts, 3, IM_COL32(255, 255, 255, 255), ImDrawFlags_None, 4.0f);

        if (hovered)
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        return clicked;
    }

} // namespace

// Plain-language breakdown of the ML pipeline, shown one section at a time
// as slides. Keep this in sync with the actual implementation and results
// in ml/README.md.
Screen render_under_the_hood_screen(GLFWwindow *window)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO &io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_w, screen_h), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse |
        ImGuiWindowFlags_NoSavedSettings;

    Screen next = Screen::UnderTheHood;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.28f, 0.28f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.38f, 0.38f, 0.38f, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);

    ImGui::Begin("##under_the_hood", nullptr, flags);

    ImGui::SetCursorPos(ImVec2(40.0f, 30.0f));
    ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f), "Under the Hood");

    // --- Advance the slide animation ---
    if (g_from >= 0)
    {
        g_anim_t += io.DeltaTime / k_slide_seconds;
        if (g_anim_t >= 1.0f)
        {
            g_anim_t = 1.0f;
            g_from = -1;
        }
    }

    // --- Slides ---
    float arrow_radius = 36.0f;
    float arrow_margin = 50.0f;
    float side_space = arrow_margin + arrow_radius * 2.0f + 40.0f;
    float text_w = std::min(900.0f, screen_w - side_space * 2.0f);
    float content_w = std::min(1000.0f, screen_w - side_space * 2.0f);
    float area_top = 70.0f;
    float area_bottom = screen_h - 110.0f;

    if (g_from >= 0)
    {
        float e = ease_in_out_cubic(g_anim_t);
        float travel = screen_w * (float)g_dir;
        draw_slide(g_from, floorf(-e * travel), 1.0f - e, false,
                   screen_w, area_top, area_bottom, text_w, content_w, next);
        draw_slide(g_current, floorf((1.0f - e) * travel), e, false,
                   screen_w, area_top, area_bottom, text_w, content_w, next);
    }
    else
    {
        draw_slide(g_current, 0.0f, 1.0f, true,
                   screen_w, area_top, area_bottom, text_w, content_w, next);
    }

    // --- Arrows (and keyboard left/right) ---
    // On the last slide the next arrow flips upside down and becomes the link
    // to the Classifier screen, labeled underneath.
    const bool on_last = g_current == k_slide_count - 1;
    g_flip = on_last ? std::min(1.0f, g_flip + io.DeltaTime / k_flip_seconds)
                     : std::max(0.0f, g_flip - io.DeltaTime / k_flip_seconds);
    bool open_demo = false;

    float arrow_y = (area_top + area_bottom) * 0.5f;
    ImVec2 next_center(screen_w - arrow_margin - arrow_radius, arrow_y);
    if (arrow_button("##next", next_center, arrow_radius, 1, ease_in_out_cubic(g_flip)))
    {
        if (on_last)
            open_demo = true;
        else
            go_to(g_current + 1);
    }
    if (g_flip > 0.5f)
    {
        // Fades in once the blue back face is showing; clicking it works too.
        ImDrawList *dl = ImGui::GetWindowDrawList();
        float label_a = smoothstep01((g_flip - 0.5f) * 2.0f);
        float label_w = 2.0f * arrow_radius + 80.0f;
        float label_y = arrow_y + arrow_radius + 12.0f;
        float label_h =
            centered_text(dl, g_font_body, next_center.x, label_y, label_w, 0, k_demo_link_label, false);
        ImGui::SetCursorScreenPos(ImVec2(next_center.x - label_w * 0.5f, label_y));
        if (ImGui::InvisibleButton("##demo_label", ImVec2(label_w, label_h)) && on_last)
            open_demo = true;
        bool hovered = ImGui::IsItemHovered() && on_last;
        if (hovered)
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        centered_text(dl, g_font_body, next_center.x, label_y, label_w,
                      hovered ? rgba(0.8f, 0.9f, 1.0f, label_a) : rgba(0.6f, 0.8f, 1.0f, label_a),
                      k_demo_link_label, true);
    }
    if (open_demo)
    {
        g_current = 0;
        g_from = -1;
        g_flip = 0.0f;
        next = Screen::Demo;
    }
    if (g_current > 0 &&
        arrow_button("##prev", ImVec2(arrow_margin + arrow_radius, arrow_y), arrow_radius, -1))
        go_to(g_current - 1);

    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false))
        go_to(g_current + 1);
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false))
        go_to(g_current - 1);

    // --- Progress dots ---
    {
        ImDrawList *dl = ImGui::GetWindowDrawList();
        float dot_r = 5.0f;
        float dot_gap = 22.0f;
        float dots_w = dot_gap * (k_slide_count - 1);
        float dot_y = screen_h - 60.0f;
        for (int i = 0; i < k_slide_count; ++i)
        {
            ImVec2 c(screen_w * 0.5f - dots_w * 0.5f + i * dot_gap, dot_y);
            if (i == g_current)
                dl->AddCircleFilled(c, dot_r + 1.0f, IM_COL32(153, 204, 255, 255), 24);
            else
                dl->AddCircle(c, dot_r, IM_COL32(150, 150, 150, 255), 24, 1.5f);
        }
    }

    if (back_button("< Back to Research"))
    {
        g_current = 0;
        g_from = -1;
        g_flip = 0.0f;
        next = Screen::Research;
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(5);

    ImGui::Render();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return next;
}

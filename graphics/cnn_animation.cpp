// The looping "Model Demonstration" animation on the "How the ML Model
// Works" slide: a stylized picture of one Q-scan image passing through the
// classifier network (ml/classifier/model.py), drawn with ImGui shapes.
// It's an illustration -- nothing here runs the model.
#include "cnn_animation.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace
{

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

    // Matplotlib's viridis (5 stops) -- the colormap the real Q-scans use.
    ImU32 viridis(float t, float alpha)
    {
        static const float stops[5][3] = {{0.267f, 0.005f, 0.329f},
                                          {0.231f, 0.322f, 0.545f},
                                          {0.129f, 0.569f, 0.549f},
                                          {0.369f, 0.788f, 0.384f},
                                          {0.993f, 0.906f, 0.144f}};
        t = clamp01(t) * 4.0f;
        int i = std::min(3, (int)t);
        float f = t - (float)i;
        return rgba(lerpf(stops[i][0], stops[i + 1][0], f), lerpf(stops[i][1], stops[i + 1][1], f),
                    lerpf(stops[i][2], stops[i + 1][2], f), alpha);
    }

    // Repeatable per-cell noise in [0, 1].
    float hash01(int a, int b)
    {
        unsigned int h = (unsigned int)a * 374761393u + (unsigned int)b * 668265263u;
        h = (h ^ (h >> 13)) * 1274126177u;
        return (float)((h ^ (h >> 16)) & 0xFFFFFF) / 16777215.0f;
    }

    // Brightness of a synthetic merger chirp at (x, y) in a unit Q-scan:
    // x is time left to right, y = 0 is the top (high frequency, matching the
    // real images). The track sweeps up in frequency faster and faster and
    // gets louder until the merger, then cuts off. `blur` widens it, standing
    // in for deeper layers seeing coarser, larger-scale patterns.
    float chirp_energy(float x, float y, float blur)
    {
        const float merge_x = 0.78f;
        if (x > merge_x + 0.04f)
            return 0.0f;
        float u = std::min(x / merge_x, 1.0f);
        float track_y = 0.9f - 0.72f * u * u * u;
        float width = 0.055f + blur;
        float loudness = 0.35f + 0.65f * u;
        float cutoff = x > merge_x ? 1.0f - (x - merge_x) / 0.04f : 1.0f;
        return loudness * cutoff * expf(-powf((y - track_y) / width, 2.0f));
    }

    // Draws a polyline and, when 0 <= f <= 1, a pulse dot that fraction of
    // the way along it.
    void path_with_pulse(ImDrawList *dl, const ImVec2 *pts, int count, float f, float alpha)
    {
        float total = 0.0f;
        for (int i = 1; i < count; ++i)
        {
            dl->AddLine(pts[i - 1], pts[i], rgba(0.3f, 0.4f, 0.6f, 0.6f * alpha), 1.0f);
            total += fabsf(pts[i].x - pts[i - 1].x) + fabsf(pts[i].y - pts[i - 1].y);
        }
        if (f < 0.0f || f > 1.0f || total <= 0.0f)
            return;
        float d = f * total;
        for (int i = 1; i < count; ++i)
        {
            float seg = fabsf(pts[i].x - pts[i - 1].x) + fabsf(pts[i].y - pts[i - 1].y);
            if (d <= seg || i == count - 1)
            {
                float u = seg > 0.0f ? std::min(1.0f, d / seg) : 0.0f;
                ImVec2 p(lerpf(pts[i - 1].x, pts[i].x, u), lerpf(pts[i - 1].y, pts[i].y, u));
                dl->AddCircleFilled(p, 3.5f, rgba(0.6f, 0.95f, 1.0f, alpha), 10);
                return;
            }
            d -= seg;
        }
    }

    // Point at fraction u along a cubic Bezier curve.
    ImVec2 bezier_point(const ImVec2 (&p)[4], float u)
    {
        float a = 1.0f - u;
        float b0 = a * a * a, b1 = 3.0f * a * a * u, b2 = 3.0f * a * u * u, b3 = u * u * u;
        return ImVec2(b0 * p[0].x + b1 * p[1].x + b2 * p[2].x + b3 * p[3].x,
                      b0 * p[0].y + b1 * p[1].y + b2 * p[2].y + b3 * p[3].y);
    }

    // A glowing particle at fraction u along a curve, trailed by a few
    // fainter dots behind it.
    void curve_particle(ImDrawList *dl, const ImVec2 (&p)[4], float u, float radius,
                        float r, float g, float b, float a)
    {
        for (int k = 3; k >= 0; --k)
        {
            float uk = u - 0.045f * k;
            if (uk < 0.0f)
                continue;
            float fade = k == 0 ? 1.0f : 0.45f / k;
            dl->AddCircleFilled(bezier_point(p, uk), k == 0 ? radius : radius * 0.8f, rgba(r, g, b, a * fade), 10);
        }
    }

} // namespace

// Looping illustration of one Q-scan going through the network, drawn
// to match ml/classifier/model.py: a filter sweeps the input, and four
// convolution stages (32/64/128/256 filters, each at lower resolution)
// light up in turn, compacting the image. After the 256 maps the drawing
// turns organic: curved strands carry the last maps down into glowing
// beads (global average pooling -- one number per map), and a fan of
// curves carries every bead into every label's score, whose node swells
// with its probability. Two rows -- input and feature maps on top,
// pooling and prediction below, flowing back right to left -- so it
// stays large in a narrow column. Timeline, in seconds of a 7 s loop:
// 0-2.2 filter sweep, 2.2-3.5 deeper stages, 3.45-4.2 into the pool,
// 4.0-5.3 into the scores, hold, fade out from 6.4.
void draw_classifier_animation(ImDrawList *dl, ImVec2 pos, ImVec2 size, float alpha, float time)
{
    ImFont *small = ImGui::GetFont();
    const float fs = small->FontSize;

    dl->AddText(small, fs, ImVec2(pos.x + 12.0f, pos.y + 9.0f), rgba(0.6f, 0.6f, 0.6f, alpha),
                "Model Demonstration");

    // --- Timeline ---
    float t = fmodf(time, 7.0f);
    float reset = 1.0f - clamp01((t - 6.4f) / 0.6f);
    float sweep = clamp01(t / 2.2f);
    auto ramp = [&](float start, float length)
    { return clamp01((t - start) / length) * reset; };
    const float stage_level[4] = {reset, ramp(2.6f, 0.35f), ramp(3.05f, 0.35f), ramp(3.5f, 0.35f)};
    const float pool_level = ramp(3.95f, 0.3f);
    const float out_level = smoothstep01(ramp(4.4f, 0.9f));
    auto pulse = [&](float start)
    { return reset > 0.0f ? (t - start) / 0.4f : -1.0f; };

    // --- Layout ---
    const float pad = 14.0f;
    const float label_h = fs + 6.0f;
    const float content_top = pos.y + pad + fs + 6.0f;
    const float content_bottom = pos.y + size.y - pad;
    const float content_h = content_bottom - content_top;
    const float inner_x = pos.x + pad;
    const float inner_w = size.x - 2.0f * pad;
    const float split = content_top + floorf(content_h * 0.62f);

    // Row 1: input and feature maps (filter counts above, labels below).
    const float row1_top = content_top + label_h;
    const float row1_bottom = split - label_h;
    const float row1_mid = floorf((row1_top + row1_bottom) * 0.5f);
    // Row 2: pooled numbers and the prediction (labels below).
    const float row2_top = split + 10.0f;
    const float row2_bottom = content_bottom - label_h;
    const float row2_mid = floorf((row2_top + row2_bottom) * 0.5f);

    float input_h = floorf(std::min(row1_bottom - row1_top, inner_w * 0.32f * 140.0f / 170.0f));
    float input_w = floorf(input_h * 170.0f / 140.0f);
    const float input_x = inner_x;
    const float input_y = floorf(row1_mid - input_h * 0.5f);
    const float gap = 18.0f;

    // Input Q-scan: noise plus a chirp, in viridis.
    const int in_cols = 34, in_rows = 28;
    for (int r = 0; r < in_rows; ++r)
        for (int c = 0; c < in_cols; ++c)
        {
            float xn = (c + 0.5f) / in_cols, yn = (r + 0.5f) / in_rows;
            float v = 0.12f + 0.25f * hash01(c, r) + 0.85f * chirp_energy(xn, yn, 0.0f);
            ImVec2 p0(floorf(input_x + c * input_w / in_cols), floorf(input_y + r * input_h / in_rows));
            ImVec2 p1(floorf(input_x + (c + 1) * input_w / in_cols),
                      floorf(input_y + (r + 1) * input_h / in_rows));
            dl->AddRectFilled(p0, p1, viridis(v, alpha));
        }

    // Feature-map stacks: more filters (cards) and smaller maps per stage.
    const int cards[4] = {3, 4, 5, 6};
    const char *filters[4] = {"32", "64", "128", "256"};
    const float card_scale[4] = {0.85f, 0.68f, 0.52f, 0.38f};
    const int map_rows[4] = {12, 8, 6, 4}, map_cols[4] = {14, 10, 7, 5};
    const float map_blur[4] = {0.0f, 0.05f, 0.11f, 0.2f};
    const float depth = 3.0f, stack_gap = 10.0f;
    const float stacks_x = input_x + input_w + gap;
    const float stacks_w = inner_x + inner_w - stacks_x;

    float base_w_sum = 0.0f, depth_sum = 0.0f;
    for (int s = 0; s < 4; ++s)
    {
        base_w_sum += input_h * card_scale[s] * 170.0f / 140.0f;
        depth_sum += (cards[s] - 1) * depth;
    }
    const float k = std::min(1.0f, (stacks_w - 3.0f * stack_gap - depth_sum) / base_w_sum);

    ImVec2 stage_left[4], stage_right[4], front_min[4], front_max[4];
    float sx = stacks_x;
    for (int s = 0; s < 4; ++s)
    {
        float ch = floorf(input_h * card_scale[s] * k);
        float cw = floorf(ch * 170.0f / 140.0f);
        float stack_h = ch + (cards[s] - 1) * depth;
        float fy = floorf(row1_mid - stack_h * 0.5f + (cards[s] - 1) * depth); // front card top
        float lit = stage_level[s];

        for (int j = cards[s] - 1; j >= 1; --j)
        {
            ImVec2 a(sx + j * depth, fy - j * depth), b(a.x + cw, a.y + ch);
            dl->AddRectFilled(a, b, rgba(0.07f, 0.10f, 0.18f, alpha));
            ImU32 edge = rgba(lerpf(0.25f, 0.4f, lit), lerpf(0.35f, 0.75f, lit), lerpf(0.55f, 1.0f, lit), 0.8f * alpha);
            dl->AddRect(a, b, edge);
        }

        ImVec2 a(sx, fy), b(sx + cw, fy + ch);
        dl->AddRectFilled(a, b, rgba(0.07f, 0.10f, 0.18f, alpha));
        int revealed = s == 0 ? (int)(sweep * map_rows[0] * map_cols[0]) : map_rows[s] * map_cols[s];
        for (int r = 0; r < map_rows[s]; ++r)
            for (int c = 0; c < map_cols[s]; ++c)
            {
                if (r * map_cols[s] + c >= revealed)
                    continue;
                float xn = (c + 0.5f) / map_cols[s], yn = (r + 0.5f) / map_rows[s];
                float act = std::min(1.0f, 1.2f * chirp_energy(xn, yn, map_blur[s])) * lit;
                if (act < 0.02f)
                    continue;
                float hot = clamp01((act - 0.6f) / 0.4f);
                ImVec2 c0(floorf(a.x + c * cw / map_cols[s]), floorf(a.y + r * ch / map_rows[s]));
                ImVec2 c1(floorf(a.x + (c + 1) * cw / map_cols[s]), floorf(a.y + (r + 1) * ch / map_rows[s]));
                // Dark blue when quiet, through bright blue, to white where it's hottest.
                float red = lerpf(0.07f, lerpf(0.2f, 1.0f, hot), act);
                float green = lerpf(0.10f, lerpf(0.75f, 1.0f, hot), act);
                float blue = lerpf(0.18f, 1.0f, act);
                dl->AddRectFilled(c0, c1, rgba(red, green, blue, alpha));
            }
        dl->AddRect(a, b, rgba(lerpf(0.3f, 0.5f, lit), lerpf(0.45f, 0.85f, lit), lerpf(0.65f, 1.0f, lit), alpha));

        float stack_w = cw + (cards[s] - 1) * depth;
        float label_w = small->CalcTextSizeA(fs, FLT_MAX, 0.0f, filters[s]).x;
        float label_y = fy - (cards[s] - 1) * depth - fs - 4.0f;
        dl->AddText(small, fs, ImVec2(floorf(sx + (stack_w - label_w) * 0.5f), label_y),
                    rgba(0.6f, 0.6f, 0.6f, alpha), filters[s]);

        front_min[s] = a;
        front_max[s] = b;
        stage_left[s] = ImVec2(sx, row1_mid);
        stage_right[s] = ImVec2(sx + stack_w, row1_mid);
        sx += stack_w + stack_gap;
    }
    const float stacks_end = sx - stack_gap;

    // The sliding filter on the input and the feature-map cell it produces.
    if (sweep < 1.0f && reset > 0.0f)
    {
        int idx = (int)(sweep * map_rows[0] * map_cols[0]);
        int r = idx / map_cols[0], c = idx % map_cols[0];
        float box = floorf(input_w * 0.16f);
        float bx = floorf(input_x + (c + 0.5f) / map_cols[0] * input_w - box * 0.5f);
        float by = floorf(input_y + (r + 0.5f) / map_rows[0] * input_h - box * 0.5f);
        bx = std::min(std::max(bx, input_x), input_x + input_w - box);
        by = std::min(std::max(by, input_y), input_y + input_h - box);

        float cw = front_max[0].x - front_min[0].x, ch = front_max[0].y - front_min[0].y;
        ImVec2 c0(floorf(front_min[0].x + c * cw / map_cols[0]), floorf(front_min[0].y + r * ch / map_rows[0]));
        ImVec2 c1(floorf(front_min[0].x + (c + 1) * cw / map_cols[0]),
                  floorf(front_min[0].y + (r + 1) * ch / map_rows[0]));

        ImU32 kernel_col = rgba(1.0f, 0.85f, 0.3f, alpha);
        dl->AddLine(ImVec2(bx + box, by), c0, rgba(1.0f, 0.85f, 0.3f, 0.45f * alpha));
        dl->AddLine(ImVec2(bx + box, by + box), ImVec2(c0.x, c1.y), rgba(1.0f, 0.85f, 0.3f, 0.45f * alpha));
        dl->AddRect(ImVec2(bx, by), ImVec2(bx + box, by + box), kernel_col, 0.0f, 0, 2.0f);
        dl->AddRect(c0, c1, kernel_col, 0.0f, 0, 1.5f);
    }

    // Row-1 connectors, with a pulse traveling along each as the signal
    // moves from one stage to the next.
    for (int s = 0; s < 4; ++s)
    {
        ImVec2 a = s == 0 ? ImVec2(input_x + input_w + 2.0f, row1_mid)
                           : ImVec2(stage_right[s - 1].x + 2.0f, row1_mid);
        ImVec2 b(stage_left[s].x - 2.0f, row1_mid);
        if (b.x > a.x)
        {
            ImVec2 seg[2] = {a, b};
            path_with_pulse(dl, seg, 2, s == 0 ? -1.0f : pulse(2.2f + 0.45f * (s - 1)), alpha);
        }
    }

    // --- After the 256 maps: pooling and prediction ---
    const float pi = 3.14159265f;

    // Pooled values: a gently curved column of beads under the 256 maps,
    // sized by value and breathing softly once lit.
    const int beads = 8;
    const float col_x = floorf((front_min[3].x + front_max[3].x) * 0.5f + 4.0f);
    ImVec2 bead_pos[beads];
    float bead_r[beads], bead_v[beads];
    for (int i = 0; i < beads; ++i)
    {
        float f = (float)i / (beads - 1);
        bead_v[i] = 0.35f + 0.65f * hash01(i, 7);
        float breathe = 1.0f + 0.07f * sinf(time * 2.2f + i * 0.9f) * pool_level;
        bead_r[i] = lerpf(2.2f, 4.3f, bead_v[i]) * breathe;
        bead_pos[i] = ImVec2(col_x - 6.0f * sinf(f * pi), lerpf(row2_top + 4.0f, row2_bottom - 4.0f, f));
    }

    // Scores: one node per label on the left, with its label and
    // percentage beside it. Row 2 reads right to left, back under the
    // feature maps.
    struct Guess
    {
        const char *label;
        float p;
    };
    const Guess guesses[] = {{"Real signal", 0.94f}, {"Blip", 0.03f}, {"Whistle", 0.02f}, {"No glitch", 0.01f}};
    const int guess_count = (int)(sizeof(guesses) / sizeof(guesses[0]));
    const float node_r_max = 7.0f;
    const float label_w = small->CalcTextSizeA(fs, FLT_MAX, 0.0f, "Real signal").x;
    const float pct_w = small->CalcTextSizeA(fs, FLT_MAX, 0.0f, "100%").x;
    const float node_x = floorf(inner_x + label_w + 10.0f + pct_w + 8.0f + node_r_max);
    const float node_step = std::min(20.0f, (row2_bottom - row2_top) / guess_count);
    float node_y[guess_count];
    for (int j = 0; j < guess_count; ++j)
        node_y[j] = floorf(row2_mid + (j - (guess_count - 1) * 0.5f) * node_step);

    // Strands: the last feature maps drawn down into the beads.
    const float src_y = front_max[3].y + 2.0f;
    for (int i = 0; i < beads; ++i)
    {
        float src_x = lerpf(front_min[3].x + 3.0f, front_max[3].x - 3.0f, (float)i / (beads - 1));
        ImVec2 b = bead_pos[i];
        ImVec2 p[4] = {ImVec2(src_x, src_y), ImVec2(src_x, lerpf(src_y, b.y, 0.55f)),
                       ImVec2(b.x + 18.0f, b.y - 4.0f), ImVec2(b.x + bead_r[i] + 1.0f, b.y)};
        float strand_alpha = (0.18f + 0.3f * stage_level[3]) * alpha;
        dl->AddBezierCubic(p[0], p[1], p[2], p[3], rgba(0.35f, 0.55f, 0.85f, strand_alpha), 1.0f);
        float u = (t - 3.45f - 0.25f * hash01(i, 3)) / 0.5f;
        if (u > 0.0f && u < 1.0f && reset > 0.0f)
            curve_particle(dl, p, u, 1.8f, 0.6f, 0.95f, 1.0f, sinf(u * pi) * alpha);
    }

    // Fan: every pooled number feeds every score, with the strongest
    // links running into "Real signal".
    for (int i = 0; i < beads; ++i)
        for (int j = 0; j < guess_count; ++j)
        {
            bool to_top = j == 0;
            float w = to_top ? 0.45f + 0.55f * hash01(i, 20) : 0.6f * hash01(i, j + 20);
            ImVec2 a(bead_pos[i].x - bead_r[i] - 1.0f, bead_pos[i].y);
            ImVec2 b(node_x + node_r_max + 3.0f, node_y[j]);
            float reach = (a.x - b.x) * 0.45f;
            ImVec2 p[4] = {a, ImVec2(a.x - reach, a.y), ImVec2(b.x + reach, b.y), b};
            float line_a = (0.05f + 0.16f * w + (to_top ? 0.18f * w * out_level : 0.0f)) * alpha;
            dl->AddBezierCubic(p[0], p[1], p[2], p[3],
                               to_top ? rgba(0.4f, 0.85f, 0.7f, line_a) : rgba(0.35f, 0.55f, 0.85f, line_a), 1.0f);
            float u = (t - 4.0f - 0.35f * hash01(i * 7 + j, 11)) / 0.6f;
            if (u > 0.0f && u < 1.0f && reset > 0.0f)
                curve_particle(dl, p, u, 1.6f, to_top ? 0.55f : 0.6f, to_top ? 1.0f : 0.9f, to_top ? 0.75f : 1.0f,
                               sinf(u * pi) * (0.3f + 0.7f * w) * alpha);
        }

    for (int i = 0; i < beads; ++i)
    {
        float lv = pool_level * bead_v[i];
        if (lv > 0.01f)
            dl->AddCircleFilled(bead_pos[i], bead_r[i] + 3.0f, rgba(0.35f, 0.8f, 1.0f, 0.18f * lv * alpha), 16);
        dl->AddCircleFilled(bead_pos[i], bead_r[i],
                            rgba(lerpf(0.14f, 0.45f, lv), lerpf(0.18f, 0.9f, lv), lerpf(0.28f, 1.0f, lv), alpha), 16);
    }

    for (int j = 0; j < guess_count; ++j)
    {
        const Guess &gu = guesses[j];
        bool top_pick = j == 0;
        ImVec2 c(node_x, node_y[j]);
        float r = lerpf(2.5f, node_r_max, sqrtf(gu.p * out_level));
        float glow = clamp01(out_level * (0.35f + 0.65f * sqrtf(gu.p)));
        if (top_pick && out_level > 0.0f)
        {
            float breathe = 1.0f + 0.06f * sinf(time * 2.0f);
            dl->AddCircleFilled(c, (r + 8.0f) * breathe, rgba(0.35f, 0.9f, 0.6f, 0.08f * out_level * alpha), 24);
            dl->AddCircleFilled(c, (r + 4.0f) * breathe, rgba(0.35f, 0.9f, 0.6f, 0.18f * out_level * alpha), 24);
        }
        float nr = top_pick ? 0.35f : 0.45f, ng = top_pick ? 0.9f : 0.6f, nb = top_pick ? 0.6f : 0.9f;
        ImU32 node_col = rgba(lerpf(0.12f, nr, glow), lerpf(0.16f, ng, glow), lerpf(0.25f, nb, glow), alpha);
        dl->AddCircleFilled(c, r, node_col, 20);
        dl->AddCircle(c, r + 0.5f, rgba(0.35f, 0.5f, 0.75f, 0.6f * alpha), 20, 1.0f);

        float shade = top_pick ? lerpf(0.75f, 1.0f, out_level) : 0.6f;
        ImU32 text_col = rgba(shade, shade, shade, alpha);
        float ty = floorf(node_y[j] - fs * 0.5f);
        dl->AddText(small, fs, ImVec2(inner_x, ty), text_col, gu.label);
        char pct[16];
        snprintf(pct, sizeof(pct), "%.0f%%", gu.p * out_level * 100.0f);
        float w = small->CalcTextSizeA(fs, FLT_MAX, 0.0f, pct).x;
        dl->AddText(small, fs, ImVec2(node_x - node_r_max - 8.0f - w, ty), text_col, pct);
    }

    // Labels under each part, kept inside the panel.
    auto label_under = [&](float x0, float x1, float y, const char *text)
    {
        float w = small->CalcTextSizeA(fs, FLT_MAX, 0.0f, text).x;
        float x = std::min(std::max((x0 + x1 - w) * 0.5f, inner_x), inner_x + inner_w - w);
        dl->AddText(small, fs, ImVec2(floorf(x), y), rgba(0.75f, 0.75f, 0.75f, alpha), text);
    };
    label_under(input_x, input_x + input_w, row1_bottom + 3.0f, "Q-scan in");
    label_under(stacks_x, stacks_end, row1_bottom + 3.0f, "Feature maps");
    label_under(col_x - 20.0f, col_x + 20.0f, row2_bottom + 3.0f, "Average pool");
    label_under(inner_x, node_x + node_r_max, row2_bottom + 3.0f, "Prediction");
}

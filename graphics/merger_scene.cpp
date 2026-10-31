// The 3D view for the merger simulation: the two black holes orbiting and
// merging, their trails, and a grid that dips under them and ripples with
// the gravitational waves they send out. Drawn with old-style OpenGL calls.
#include "merger_scene.h"
#ifdef _WIN32
#include <windows.h> // must precede GL/glu.h so CALLBACK etc. are defined
#endif
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

// World units are the binary's total mass M (G = c = 1), the trajectory's
// own units, so the horizons, their orbits and the waves' wavelength are all
// to scale with one another -- and a wave crosses the grid at one unit per M
// of trajectory time: the speed of light.
constexpr float kGridExtent = 60.0f; // radius of the round grid
constexpr float kGridSpacing = 2.0f;

// The sheet itself is illustrative. The spacetime screen's grid is an exact
// embedding diagram, but a merging binary's spacetime has no such picture,
// so here each hole dents the sheet with a softened 1/r well and the waves
// ripple it as they pass -- both heights hugely exaggerated so they show.
// Ripples much taller than this get steeper than the camera's line of sight
// on the far side of the grid and fold over into spikes.
constexpr float kWellDepth = 6.0f;     // depth scale of the well under mass M
constexpr float kWellSoftening = 2.0f; // keeps each well finite under its hole
constexpr float kRippleHeight = 1.0f;  // crest height of the merger's wave out to kRippleNear,
constexpr float kRippleNear = 8.0f;    // beyond which waves weaken as 1/r
constexpr float kWaveStep = 0.25f;     // radial resolution of the wave lookup table

// The file goes from two horizons to the remnant between one sample and the
// next; the picture instead fuses them over this long (units of M).
constexpr float kMergeBlend = 4.0f;

struct Vec3 {
    float x, y, z;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalize(Vec3 a) {
    float len = sqrtf(dot(a, a));
    return len > 0.0f ? a * (1.0f / len) : a;
}

float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
float lerp(float a, float b, float t) { return a + (b - a) * t; }
float smoothstep01(float t) {
    t = clamp01(t);
    return t * t * (3.0f - 2.0f * t);
}

// A horizon as drawn. The trajectory's orbital x-y plane becomes the
// scene's x-z plane (y is up) via (x, y) -> (x, -y), which keeps the orbit
// counterclockwise seen from above.
struct Body {
    float x = 0.0f, z = 0.0f;
    float radius = 0.0f;
    float mass = 0.0f;    // fraction of M; sets the depth of its well
    float stretch = 0.0f; // fractional elongation along `angle`
    float angle = 0.0f;   // radians, counterclockwise seen from above
    float glow = 1.0f;    // 0..1 strength of its photon-sphere glow
};

struct Scene {
    Body bodies[3]; // the two holes, then (after merger) the remnant
    int count = 2;
    bool merged = false;
    float since_merger = 0.0f; // units of M
    // The wave passing radius k * kWaveStep: its height there times the
    // cosine and sine of its phase.
    std::vector<float> wave_cos, wave_sin;

    float well(float x, float z) const {
        float w = 0.0f;
        for (int i = 0; i < count; ++i) {
            if (bodies[i].mass <= 0.0f)
                continue;
            float dx = x - bodies[i].x, dz = z - bodies[i].z;
            w -= kWellDepth * bodies[i].mass / sqrtf(dx * dx + dz * dz + kWellSoftening * kWellSoftening);
        }
        return w;
    }

    // A binary radiates a two-armed spiral: in its orbital plane the wave
    // goes as cos(phase - 2 * azimuth), dipping along the line through the
    // two holes. The azimuth is taken in the trajectory's plane, (x, -z).
    float ripple(float x, float z) const {
        float r2 = x * x + z * z;
        if (r2 < 1e-6f)
            return 0.0f;
        float f = sqrtf(r2) / kWaveStep;
        int k = std::min((int)f, (int)wave_cos.size() - 2);
        float u = std::min(1.0f, f - (float)k);
        float c = lerp(wave_cos[k], wave_cos[k + 1], u);
        float s = lerp(wave_sin[k], wave_sin[k + 1], u);
        float cos2 = (x * x - z * z) / r2;
        float sin2 = -2.0f * x * z / r2;
        return -(c * cos2 + s * sin2);
    }

    float height(float x, float z) const { return well(x, z) + ripple(x, z); }
};

Scene build_scene(const Trajectory& traj, double t) {
    Scene sc;
    const TrajectorySample& last = traj.samples[traj.merge_index];
    const float m1 = 0.5f * last.r1, m2 = 0.5f * last.r2; // a horizon's radius is 2 m

    if (t <= traj.merge_time() || traj.merge_index + 1 >= (int)traj.samples.size()) {
        TrajectorySample s = traj.at(t);
        sc.bodies[0] = {s.x1, -s.y1, s.r1, m1};
        sc.bodies[1] = {s.x2, -s.y2, s.r2, m2};
    } else {
        // From merger on the file has only the remnant. The two holes keep
        // turning with the wave as they sink into each other while the
        // remnant swells out of them; its horizon starts out stretched along
        // the line they fell in on and rings down to a sphere as the wave
        // dies away. The file has no horizon shapes, so the stretch is
        // illustrative -- its rotation and decay follow the ringdown.
        sc.merged = true;
        sc.since_merger = (float)(t - traj.merge_time());
        float blend = smoothstep01(sc.since_merger / kMergeBlend);
        float amplitude, phase;
        traj.wave_at(t, amplitude, phase);
        float angle = 0.5f * phase; // the l = m = 2 wave turns twice per turn of the orbit or bulge
        float c = cosf(angle), s = sinf(angle);
        float d1 = hypotf(last.x1, last.y1) * (1.0f - blend);
        float d2 = hypotf(last.x2, last.y2) * (1.0f - blend);
        float r_f = traj.samples[traj.merge_index + 1].r1;
        sc.bodies[0] = {d1 * c, -d1 * s, last.r1, m1, 0.0f, 0.0f, 1.0f - blend};
        sc.bodies[1] = {-d2 * c, d2 * s, last.r2, m2, 0.0f, 0.0f, 1.0f - blend};
        sc.bodies[2] = {0.0f, 0.0f, r_f * lerp(0.55f, 1.0f, blend), 0.0f, 0.25f * amplitude, angle, blend};
        sc.count = 3;
    }

    // Radius r sees the wave the binary sent out r / c earlier (c = 1).
    // Within a few M of the center there's no wave yet -- that region is the
    // holes' own pull, already shown by the wells -- so it fades in.
    int bins = (int)(kGridExtent / kWaveStep) + 2;
    sc.wave_cos.resize(bins);
    sc.wave_sin.resize(bins);
    for (int k = 0; k < bins; ++k) {
        float r = (float)k * kWaveStep;
        float amplitude, phase;
        traj.wave_at(t - r, amplitude, phase);
        float near_fade = smoothstep01((r - 2.0f) / 8.0f);
        float height = kRippleHeight * amplitude * near_fade * std::min(1.0f, kRippleNear / std::max(r, 1e-3f));
        sc.wave_cos[k] = height * cosf(phase);
        sc.wave_sin[k] = height * sinf(phase);
    }
    return sc;
}

// Deterministic generator so the star field doesn't depend on (or disturb)
// rand(), which the title screen's warp stars use.
float next_random(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) / 16777216.0f;
}

// Fixed background stars on a sphere around the camera, so they show no
// parallax as it orbits. Same seed as the spacetime screen: the same sky.
void draw_stars(Vec3 eye) {
    struct Star {
        Vec3 dir;
        float r, g, b;
    };
    static std::vector<Star> stars;
    if (stars.empty()) {
        unsigned int seed = 12345u;
        stars.resize(2200);
        for (Star& s : stars) {
            float z = next_random(seed) * 2.0f - 1.0f;
            float a = next_random(seed) * 2.0f * kPi;
            float ring = sqrtf(1.0f - z * z);
            s.dir = {ring * cosf(a), z, ring * sinf(a)};
            float brightness = 0.15f + 0.85f * powf(next_random(seed), 6.0f);
            float tint = next_random(seed); // a few bluish and yellowish stars
            s.r = brightness * (tint < 0.15f ? 0.75f : 1.0f);
            s.g = brightness * 0.92f;
            s.b = brightness * (tint > 0.85f ? 0.7f : 1.0f);
        }
    }

    glDisable(GL_DEPTH_TEST);
    glPointSize(1.6f);
    glBegin(GL_POINTS);
    for (const Star& s : stars) {
        Vec3 p = eye + s.dir * 90.0f;
        glColor3f(s.r, s.g, s.b);
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
    glEnable(GL_DEPTH_TEST);
}

// How far (x, z) lies outside a horizon's footprint on the plane, roughly
// in units of M; negative inside. Stretched horizons have an elliptical
// footprint.
float outside_horizon(const Body& b, float axis_cos, float axis_sin, float x, float z) {
    float dx = x - b.x, dz = z - b.z;
    float along = dx * axis_cos - dz * axis_sin;
    float across = dx * axis_sin + dz * axis_cos;
    float semi_along = b.radius * (1.0f + b.stretch);
    float semi_across = b.radius / sqrtf(1.0f + b.stretch);
    float q = sqrtf((along * along) / (semi_along * semi_along) + (across * across) / (semi_across * semi_across));
    return (q - 1.0f) * b.radius;
}

// Square grid of lines draped over the sheet and clipped to a round rim.
// Lines brighten where they sink into a well and along wave crests, dim in
// the troughs, fade out toward the rim, and stop at the horizons.
void draw_grid(const Scene& sc) {
    struct Vertex {
        float x, y, z, r, g, b, a;
        bool hidden;
    };
    const int samples = 240;
    std::vector<Vertex> line(samples + 1);
    float axis_cos[3], axis_sin[3];
    for (int i = 0; i < sc.count; ++i) {
        axis_cos[i] = cosf(sc.bodies[i].angle);
        axis_sin[i] = sinf(sc.bodies[i].angle);
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);

    int steps = (int)(kGridExtent / kGridSpacing);
    for (int axis = 0; axis < 2; ++axis) {
        for (int i = -steps; i <= steps; ++i) {
            float c = (float)i * kGridSpacing;
            if (fabsf(c) >= kGridExtent)
                continue;
            float half = sqrtf(kGridExtent * kGridExtent - c * c);

            for (int k = 0; k <= samples; ++k) {
                float t = -half + 2.0f * half * (float)k / (float)samples;
                float x = axis == 0 ? c : t;
                float z = axis == 0 ? t : c;
                float well = sc.well(x, z);
                float ripple = sc.ripple(x, z);

                float fade = clamp01((kGridExtent - sqrtf(x * x + z * z)) / (kGridExtent * 0.3f));
                bool hidden = false;
                for (int b = 0; b < sc.count; ++b) {
                    float outside = outside_horizon(sc.bodies[b], axis_cos[b], axis_sin[b], x, z);
                    hidden = hidden || outside < 0.0f;
                    fade *= clamp01(outside / 0.6f);
                }

                float sink = clamp01(-well / 3.0f);
                float crest = std::max(-1.0f, std::min(1.0f, ripple / (0.4f * kRippleHeight)));
                float lift = std::max(crest, 0.0f);
                line[k] = {x,
                           well + ripple,
                           z,
                           std::min(1.0f, lerp(0.30f, 0.80f, sink) + 0.45f * lift),
                           std::min(1.0f, lerp(0.50f, 0.90f, sink) + 0.30f * lift),
                           1.0f,
                           fade * clamp01(lerp(0.35f, 0.85f, sink) + 0.35f * crest),
                           hidden};
            }

            // One strip per run of visible points.
            bool open = false;
            for (int k = 0; k <= samples; ++k) {
                const Vertex& v = line[k];
                if (v.hidden) {
                    if (open)
                        glEnd();
                    open = false;
                    continue;
                }
                if (!open)
                    glBegin(GL_LINE_STRIP);
                open = true;
                glColor4f(v.r, v.g, v.b, v.a);
                glVertex3f(v.x, v.y, v.z);
            }
            if (open)
                glEnd();
        }
    }

    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
}

// A black horizon, stretched along its axis if it's still ringing.
void draw_horizon(const Body& b, float y) {
    static GLUquadric* quadric = gluNewQuadric();
    float squash = 1.0f / sqrtf(1.0f + b.stretch);
    glColor3f(0.0f, 0.0f, 0.0f);
    glPushMatrix();
    glTranslatef(b.x, y, b.z);
    glRotatef(b.angle * 180.0f / kPi, 0.0f, 1.0f, 0.0f);
    glScalef(1.0f + b.stretch, squash, squash);
    gluSphere(quadric, b.radius, 48, 24);
    glPopMatrix();
}

// Fading trail of the last ~1.5 orbits behind each hole (the file samples
// 60 points per orbit), draped on the sheet. After merger the trails end
// where the holes met and fade away.
void draw_trails(const Trajectory& traj, double t, const Scene& sc) {
    const int length = 90;
    int last = std::min(traj.index_at(t), traj.merge_index);
    int first = std::max(0, last - length);
    float fade = sc.merged ? 1.0f - smoothstep01(sc.since_merger / 20.0f) : 1.0f;
    if (fade <= 0.0f || last <= first)
        return;
    const Vec3 colors[2] = {{1.0f, 0.72f, 0.38f}, {0.45f, 0.78f, 1.0f}};

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(2.0f);
    glDepthMask(GL_FALSE);
    for (int b = 0; b < 2; ++b) {
        Vec3 col = colors[b];
        glBegin(GL_LINE_STRIP);
        for (int i = first; i <= last; ++i) {
            const TrajectorySample& s = traj.samples[i];
            float x = b == 0 ? s.x1 : s.x2;
            float z = -(b == 0 ? s.y1 : s.y2);
            glColor4f(col.x, col.y, col.z, 0.9f * fade * (float)(i - first) / (float)(last - first));
            glVertex3f(x, sc.height(x, z) + 0.08f, z);
        }
        if (!sc.merged) { // on up to the hole itself
            const Body& body = sc.bodies[b];
            glColor4f(col.x, col.y, col.z, 0.9f);
            glVertex3f(body.x, sc.height(body.x, body.z) + 0.08f, body.z);
        }
        glEnd();
    }
    glDepthMask(GL_TRUE);
    glLineWidth(1.0f);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
}

// Faint camera-facing glow peaking at the photon sphere (1.5x the horizon
// radius), where gravity bends light into circular orbits, plus a soft haze
// hugging the horizon -- the spacetime screen's black hole look, so the
// black horizons read against the dark sky.
void draw_photon_glow(Vec3 center, float radius, Vec3 eye, float strength) {
    if (strength <= 0.0f)
        return;
    Vec3 view = normalize(center - eye);
    Vec3 right = normalize(cross(view, {0.0f, 1.0f, 0.0f}));
    Vec3 up = cross(right, view);
    const int rings = 24, segments = 96;
    const float inner = 1.0f, outer = 2.2f; // in units of the horizon radius

    auto glow = [](float rho) {
        float ring = expf(-powf((rho - 1.5f) / 0.045f, 2.0f));
        float haze = expf(-(rho - 1.0f) / 0.3f) * 0.22f;
        return ring * 0.22f + haze;
    };

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);
    for (int i = 0; i < rings; ++i) {
        float rho0 = lerp(inner, outer, (float)i / (float)rings);
        float rho1 = lerp(inner, outer, (float)(i + 1) / (float)rings);
        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= segments; ++j) {
            float a = 2.0f * kPi * (float)j / (float)segments;
            Vec3 dir = right * cosf(a) + up * sinf(a);
            float rhos[2] = {rho1, rho0};
            for (float rho : rhos) {
                float k = glow(rho) * strength;
                Vec3 p = center + dir * (rho * radius);
                glColor3f(1.0f * k, 0.78f * k, 0.52f * k);
                glVertex3f(p.x, p.y, p.z);
            }
        }
        glEnd();
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

} // namespace

void render_merger_scene(float screen_w, float screen_h, const Trajectory& trajectory, double t_m,
                         float cam_azimuth, float cam_elevation, float cam_distance) {
    if (!trajectory.loaded())
        return;
    Scene sc = build_scene(trajectory, t_m);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50.0, (double)screen_w / (double)screen_h, 0.1, 400.0);

    // Orbit camera around a point just below the plane, among the wells.
    Vec3 target = {0.0f, -1.5f, 0.0f};
    Vec3 eye = {target.x + cam_distance * cosf(cam_elevation) * sinf(cam_azimuth),
                target.y + cam_distance * sinf(cam_elevation),
                target.z + cam_distance * cosf(cam_elevation) * cosf(cam_azimuth)};
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, 0.0, 1.0, 0.0);

    glDisable(GL_LIGHTING);
    draw_stars(eye);
    draw_grid(sc);

    float heights[3];
    for (int i = 0; i < sc.count; ++i) {
        const Body& b = sc.bodies[i];
        heights[i] = sc.height(b.x, b.z);
        bool swallowed = i < 2 && sc.merged && b.glow <= 0.0f; // fully inside the remnant
        if (!swallowed)
            draw_horizon(b, heights[i]);
    }
    draw_trails(trajectory, t_m, sc);
    for (int i = 0; i < sc.count; ++i) {
        const Body& b = sc.bodies[i];
        draw_photon_glow({b.x, heights[i], b.z}, b.radius, eye, b.glow);
    }
}

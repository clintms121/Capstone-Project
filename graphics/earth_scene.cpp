// The 3D view for the Space Time Simulation: Earth or a black hole sitting in
// a grid curved the way general relativity says it should be (Flamm's
// paraboloid), with a star field, Earth's texture and atmosphere, and the
// black hole's glowing accretion disk. Drawn with old-style OpenGL calls.
#include "earth_scene.h"
#include "image_texture.h"
#ifdef _WIN32
#include <windows.h> // must precede GL/glu.h so CALLBACK etc. are defined
#endif
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <algorithm>
#include <cmath>
#include <vector>

// GL 1.2 lighting enums; Windows' gl.h stops at 1.1.
#ifndef GL_LIGHT_MODEL_COLOR_CONTROL
#define GL_LIGHT_MODEL_COLOR_CONTROL 0x81F8
#endif
#ifndef GL_SEPARATE_SPECULAR_COLOR
#define GL_SEPARATE_SPECULAR_COLOR 0x81FA
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kGridExtent = 14.0f;  // radius of the round grid, world units
constexpr float kEarthTiltDeg = 23.44f;
const char* kEarthTexturePath = "graphics/Earth_diff.jpg";

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

// Embedding-diagram height of the equatorial slice of the space around a
// static, spherically symmetric mass: the curved 2D space is drawn as a
// surface in flat 3D whose in-surface distances match the real ones, so the
// shape itself is exact. Outside the body that's the exterior Schwarzschild
// solution, Flamm's paraboloid:
//   w(r) = 2 sqrt(r_s (r - r_s))
// It turns vertical only at r = r_s. A black hole's surface reaches that
// point -- the funnel ends at the horizon. An ordinary body is far bigger
// than its r_s, so the surface meets it while still gently sloped, and
// carries on inside as a smooth bowl (the interior Schwarzschild solution)
// that the planet itself hides -- so the grid simply stops at its surface.
float embedding_w(float r, float r_s) {
    return r_s > 0.0f ? 2.0f * sqrtf(r_s * std::max(0.0f, r - r_s)) : 0.0f;
}

// The grid surface for one body: height 0 at the rim, dipping toward the
// mass at a true 1:1 vertical scale, from the body's surface outward.
struct Sheet {
    float body_radius;
    float r_s;
    float w_rim;

    Sheet(float body_radius, float r_s)
        : body_radius(body_radius), r_s(r_s), w_rim(embedding_w(kGridExtent, r_s)) {}

    float r_min() const { return body_radius; }
    float height(float r) const { return embedding_w(std::max(r, body_radius), r_s) - w_rim; }
};

// Deterministic generator so the star field doesn't depend on (or disturb)
// rand(), which the title screen's warp stars use.
float next_random(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) / 16777216.0f;
}

// Fixed background stars on a sphere around the camera, so they show no
// parallax as the camera orbits -- like real, effectively infinitely far
// stars.
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

// Square grid of lines draped over the sheet, clipped to a round rim (so
// the corners don't curl up past the edge) and to the body's surface.
// Lines brighten as they sink and fade out toward the rim. They also fade
// out just before reaching the body: they end exactly on its equator, where
// line and surface are at the same depth and would otherwise flicker as
// bright specks against the spinning sphere's faces.
void draw_grid(const Sheet& sheet, float spacing) {
    const float r_min = sheet.r_min();
    const float depth = std::max(0.001f, -sheet.height(r_min));

    auto emit = [&](int axis, float c, float t) {
        float x = axis == 0 ? c : t;
        float z = axis == 0 ? t : c;
        float r = sqrtf(x * x + z * z);
        float y = sheet.height(r);
        float sink = clamp01(-y / depth);
        float rim_fade = clamp01((kGridExtent - r) / (kGridExtent * 0.3f));
        float surface_fade = clamp01((r - r_min) / 0.5f);
        glColor4f(lerp(0.30f, 0.80f, sink), lerp(0.50f, 0.90f, sink), 1.0f,
                  rim_fade * surface_fade * lerp(0.45f, 0.9f, sink));
        glVertex3f(x, y, z);
    };

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);

    std::vector<float> ts;
    int steps = (int)(kGridExtent / spacing);
    glBegin(GL_LINES);
    for (int axis = 0; axis < 2; ++axis) {
        for (int i = -steps; i <= steps; ++i) {
            float c = i * spacing;
            if (fabsf(c) >= kGridExtent)
                continue;
            float half = sqrtf(kGridExtent * kGridExtent - c * c);

            ts.clear();
            const int samples = 240;
            for (int k = 0; k <= samples; ++k)
                ts.push_back(-half + 2.0f * half * (float)k / (float)samples);
            if (fabsf(c) < r_min) {
                // The line crosses the hole. Add its exact edge points plus
                // extra samples just outside, where the surface turns
                // vertical and even spacing would leave a jagged step.
                float edge = sqrtf(r_min * r_min - c * c);
                for (int k = 0; k <= 24; ++k) {
                    float d = (float)k / 24.0f;
                    d = d * d * 1.5f;
                    if (edge + d < half) {
                        ts.push_back(edge + d);
                        ts.push_back(-edge - d);
                    }
                }
            }
            std::sort(ts.begin(), ts.end());

            for (size_t k = 1; k < ts.size(); ++k) {
                float t0 = ts[k - 1], t1 = ts[k];
                if (t1 - t0 < 1e-5f)
                    continue;
                float mid = 0.5f * (t0 + t1);
                if (sqrtf(c * c + mid * mid) < r_min ||
                    sqrtf(c * c + t0 * t0) < r_min - 1e-3f ||
                    sqrtf(c * c + t1 * t1) < r_min - 1e-3f)
                    continue;
                emit(axis, c, t0);
                emit(axis, c, t1);
            }
        }
    }
    glEnd();

    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
}

// Fallback continent pattern for when graphics/Earth_diff.jpg can't be
// loaded: blends ocean blue with green/brown land using a handful of
// overlapping sine-wave blobs. Not real geography.
void earth_fallback_color(float lat_deg, float lon_deg) {
    float land = sinf(lon_deg * 0.05f + 1.0f) * cosf(lat_deg * 0.07f) +
                 sinf(lon_deg * 0.13f - 2.0f) * cosf(lat_deg * 0.11f + 1.0f) * 0.6f +
                 sinf(lon_deg * 0.03f + 4.0f) * sinf(lat_deg * 0.05f) * 0.5f;
    if (fabsf(lat_deg) > 65.0f)
        glColor3f(0.92f, 0.94f, 0.98f); // polar ice
    else if (land > 0.15f)
        glColor3f(0.35f, 0.55f, 0.25f); // land
    else
        glColor3f(0.10f, 0.30f, 0.65f); // ocean
}

enum class SphereShading {
    Textured,      // equirectangular texture coordinates
    EarthFallback, // per-vertex procedural continents
    CurrentColor,  // whatever glColor is already set
};

// UV sphere with normals. Longitude increases counterclockwise seen from
// above the north pole (+y), so with the texture applied, east is the
// direction Earth spins.
void draw_sphere(float radius, int lat_segments, int lon_segments, SphereShading shading) {
    for (int i = 0; i < lat_segments; ++i) {
        float theta0 = kPi * (float)i / (float)lat_segments - kPi * 0.5f;
        float theta1 = kPi * (float)(i + 1) / (float)lat_segments - kPi * 0.5f;

        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= lon_segments; ++j) {
            float u = (float)j / (float)lon_segments;
            // The closing column reuses phi = 0 exactly: sinf(2 pi) isn't
            // quite 0, and that sliver of a gap shows as pinholes along the
            // seam meridian.
            float phi = j == lon_segments ? 0.0f : u * 2.0f * kPi;
            float thetas[2] = {theta1, theta0};
            for (float theta : thetas) {
                float x = cosf(theta) * cosf(phi);
                float y = sinf(theta);
                float z = -cosf(theta) * sinf(phi);
                if (shading == SphereShading::Textured)
                    glTexCoord2f(u, 0.5f - theta / kPi);
                else if (shading == SphereShading::EarthFallback)
                    earth_fallback_color(theta * 180.0f / kPi, u * 360.0f - 180.0f);
                glNormal3f(x, y, z);
                glVertex3f(x * radius, y * radius, z * radius);
            }
        }
        glEnd();
    }
}

// Thin shell just above the surface, glowing blue toward the limb (where
// the line of sight passes through the most air) and on the sunlit side.
// Per-vertex, computed on the CPU since the fixed-function pipeline has no
// view-dependent shading of its own.
void draw_atmosphere(Vec3 center, float radius, Vec3 eye, Vec3 sun_dir) {
    const float shell = radius * 1.035f;
    const int lat_segments = 48, lon_segments = 96;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);
    for (int i = 0; i < lat_segments; ++i) {
        float theta0 = kPi * (float)i / (float)lat_segments - kPi * 0.5f;
        float theta1 = kPi * (float)(i + 1) / (float)lat_segments - kPi * 0.5f;
        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= lon_segments; ++j) {
            float phi = j == lon_segments ? 0.0f : 2.0f * kPi * (float)j / (float)lon_segments;
            float thetas[2] = {theta1, theta0};
            for (float theta : thetas) {
                Vec3 n = {cosf(theta) * cosf(phi), sinf(theta), cosf(theta) * sinf(phi)};
                Vec3 p = center + n * shell;
                float facing = fabsf(dot(n, normalize(eye - p)));
                float rim = powf(1.0f - facing, 3.0f);
                float lit = clamp01(dot(n, sun_dir) * 0.7f + 0.45f);
                glColor4f(0.35f, 0.62f, 1.0f, rim * lit);
                glVertex3f(p.x, p.y, p.z);
            }
        }
        glEnd();
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void draw_earth(Vec3 center, float radius, float time_seconds, Vec3 eye, Vec3 sun_dir) {
    const ImageTexture& texture = load_image_texture(kEarthTexturePath);
    bool textured = texture.texture != 0;

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    // Directional sunlight, set in world space (before the spin below) so
    // the day/night line stays put while the planet turns under it.
    GLfloat light_dir[] = {sun_dir.x, sun_dir.y, sun_dir.z, 0.0f};
    GLfloat light_diffuse[] = {1.0f, 0.98f, 0.94f, 1.0f};
    GLfloat light_specular[] = {0.25f, 0.25f, 0.25f, 1.0f};
    GLfloat no_ambient[] = {0.0f, 0.0f, 0.0f, 1.0f};
    GLfloat scene_ambient[] = {0.06f, 0.06f, 0.08f, 1.0f}; // faint night side
    glLightfv(GL_LIGHT0, GL_POSITION, light_dir);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_AMBIENT, no_ambient);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, scene_ambient);
    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);

    GLfloat material_specular[] = {0.35f, 0.35f, 0.35f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 40.0f);

    if (textured) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture.texture);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glColor3f(1.0f, 1.0f, 1.0f);
    }

    glPushMatrix();
    glTranslatef(center.x, center.y, center.z);
    glRotatef(kEarthTiltDeg, 0.0f, 0.0f, 1.0f);
    glRotatef(time_seconds * 12.0f, 0.0f, 1.0f, 0.0f); // slow spin, west to east
    draw_sphere(radius, 64, 128, textured ? SphereShading::Textured : SphereShading::EarthFallback);
    glPopMatrix();

    if (textured) {
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    }
    GLfloat no_specular[] = {0.0f, 0.0f, 0.0f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, no_specular);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHTING);

    draw_atmosphere(center, radius, eye, sun_dir);
}

// Rough blackbody color for a temperature relative to the disk's hottest
// ring (1.0): deep red when cool, through orange and gold, to white and
// then bluish-white when Doppler-boosted well past the peak.
Vec3 blackbody_color(float t) {
    static const float stops[][4] = {
        {0.00f, 0.00f, 0.00f, 0.00f},
        {0.25f, 0.35f, 0.03f, 0.00f},
        {0.50f, 0.85f, 0.22f, 0.03f},
        {0.75f, 1.00f, 0.48f, 0.12f},
        {1.00f, 1.00f, 0.72f, 0.38f},
        {1.35f, 1.00f, 0.95f, 0.85f},
        {1.70f, 0.85f, 0.90f, 1.00f},
    };
    const int count = (int)(sizeof(stops) / sizeof(stops[0]));
    if (t >= stops[count - 1][0])
        return {stops[count - 1][1], stops[count - 1][2], stops[count - 1][3]};
    for (int i = 1; i < count; ++i) {
        if (t <= stops[i][0]) {
            float f = (t - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
            return {lerp(stops[i - 1][1], stops[i][1], f), lerp(stops[i - 1][2], stops[i][2], f),
                    lerp(stops[i - 1][3], stops[i][3], f)};
        }
    }
    return {0.0f, 0.0f, 0.0f};
}

// Accretion disk lying in the equatorial plane -- which on an embedding
// diagram means draped over the funnel itself. Physics behind the look:
//  - Inner edge at the innermost stable circular orbit, r = 3 r_s; gas
//    closer than that can't orbit and plunges in.
//  - Brightness follows the thin-disk (Novikov-Thorne) profile,
//    flux ~ r^-3 (1 - sqrt(r_in / r)), peaking just outside r_in, with
//    temperature ~ flux^(1/4).
//  - Each ring orbits at its relativistic speed, beta = sqrt(r_s / (2(r - r_s)))
//    (c/2 at the inner edge), shifting the light by
//    g = sqrt(1 - 1.5 r_s / r) / (1 - beta cos(angle to camera)) -- the
//    side swinging toward the camera is brighter and bluer, the receding
//    side dimmer and redder (intensity ~ g^4). Light bending isn't modeled.
//  - Streaks drift at the Keplerian rate (inner rings faster), so they
//    shear out as the disk turns.
void draw_accretion_disk(const Sheet& sheet, float r_s, Vec3 eye, float time_seconds) {
    const float r_in = 3.0f * r_s;
    const float r_out = std::min(6.0f * r_s, kGridExtent * 0.95f);
    if (r_out <= r_in)
        return;
    const int rings = 48, segments = 288;

    auto flux = [&](float r) { return powf(r, -3.0f) * (1.0f - sqrtf(r_in / r)); };
    const float peak_flux = flux(r_in * 49.0f / 36.0f);

    auto ring_radius = [&](int i) {
        float f = (float)i / (float)rings;
        return r_in + (r_out - r_in) * f * f; // denser rings near the bright inner edge
    };

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE); // emitted light adds up
    glDepthMask(GL_FALSE);
    for (int i = 0; i < rings; ++i) {
        float radii[2] = {ring_radius(i + 1), ring_radius(i)};
        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= segments; ++j) {
            float angle = 2.0f * kPi * (float)j / (float)segments;
            Vec3 tangent = {-sinf(angle), 0.0f, cosf(angle)}; // direction of orbit
            for (float r : radii) {
                Vec3 p = {r * cosf(angle), sheet.height(r) + 0.02f, r * sinf(angle)};

                float emitted = flux(r) / peak_flux;
                float temperature = powf(std::max(emitted, 0.0f), 0.25f);

                float beta = sqrtf(r_s / (2.0f * (r - r_s)));
                float cos_view = dot(tangent, normalize(eye - p));
                float g = sqrtf(1.0f - 1.5f * r_s / r) / (1.0f - beta * cos_view);
                float intensity = emitted * g * g * g * g;

                float spin = 1.2f * powf(r_in / r, 1.5f) * time_seconds;
                float a = angle - spin;
                float lr = logf(r);
                float streaks = 0.6f + 0.4f * (0.5f * sinf(9.0f * a + 14.0f * lr) +
                                               0.3f * sinf(23.0f * a - 21.0f * lr + 1.3f) +
                                               0.2f * sinf(41.0f * a + 35.0f * lr + 2.1f));
                float edge = clamp01((r_out - r) / (0.6f * (r_out - r_in)));
                float edge_fade = edge * edge * (3.0f - 2.0f * edge); // smoothstep

                Vec3 color = blackbody_color(temperature * g) * (intensity * streaks * edge_fade * 1.1f);
                // Soft tone curve so the bright approaching side saturates
                // gracefully instead of clipping flat.
                glColor3f(1.0f - expf(-color.x), 1.0f - expf(-color.y), 1.0f - expf(-color.z));
                glVertex3f(p.x, p.y, p.z);
            }
        }
        glEnd();
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

// Faint camera-facing glow peaking at the photon sphere (r = 1.5 r_s), the
// radius where gravity bends light into circular orbits, plus a soft haze
// hugging the horizon -- so the black sphere reads against the dark sky.
void draw_photon_glow(Vec3 center, float r_s, Vec3 eye) {
    Vec3 view = normalize(center - eye);
    Vec3 right = normalize(cross(view, {0.0f, 1.0f, 0.0f}));
    Vec3 up = cross(right, view);
    const int rings = 24, segments = 96;
    const float inner = 1.0f, outer = 2.2f; // in units of r_s

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
                float k = glow(rho);
                Vec3 p = center + dir * (rho * r_s);
                glColor3f(1.0f * k, 0.78f * k, 0.52f * k);
                glVertex3f(p.x, p.y, p.z);
            }
        }
        glEnd();
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void draw_unlit_black_sphere(Vec3 center, float radius) {
    glColor3f(0.0f, 0.0f, 0.0f);
    glPushMatrix();
    glTranslatef(center.x, center.y, center.z);
    draw_sphere(radius, 32, 64, SphereShading::CurrentColor);
    glPopMatrix();
}

} // namespace

void render_earth_scene(float screen_w, float screen_h, float time_seconds,
                        float body_radius, float schwarzschild_radius,
                        float cam_azimuth, float cam_elevation, float cam_distance,
                        CelestialBody body) {
    const bool black_hole = body == CelestialBody::BlackHole;
    Sheet sheet(body_radius, black_hole ? body_radius : schwarzschild_radius);

    // The sphere's equator sits where the sheet meets its surface.
    Vec3 center = {0.0f, sheet.height(body_radius), 0.0f};

    // Aim partway down toward the body, easing when the depth changes
    // (switching bodies or dragging the size slider) instead of jumping.
    static float target_y = center.y * 0.55f;
    static float last_time = time_seconds;
    float dt = std::min(0.1f, std::max(0.0f, time_seconds - last_time));
    last_time = time_seconds;
    target_y += (center.y * 0.55f - target_y) * std::min(1.0f, dt * 4.0f);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50.0, (double)screen_w / (double)screen_h, 0.1, 200.0);

    // Orbit camera: eye sits on a sphere of `cam_distance` around the
    // target, positioned by azimuth (around Y) and elevation (above XZ).
    Vec3 target = {0.0f, target_y, 0.0f};
    Vec3 eye = {target.x + cam_distance * cosf(cam_elevation) * sinf(cam_azimuth),
                target.y + cam_distance * sinf(cam_elevation),
                target.z + cam_distance * cosf(cam_elevation) * cosf(cam_azimuth)};

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, 0.0, 1.0, 0.0);

    glDisable(GL_LIGHTING);
    draw_stars(eye);
    draw_grid(sheet, 1.0f);

    if (black_hole) {
        draw_unlit_black_sphere(center, body_radius);
        draw_accretion_disk(sheet, body_radius, eye, time_seconds);
        draw_photon_glow(center, body_radius, eye);
    } else {
        Vec3 sun_dir = normalize({1.0f, 0.35f, 0.25f});
        draw_earth(center, body_radius, time_seconds, eye, sun_dir);
    }
}

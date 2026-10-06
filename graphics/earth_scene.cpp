#include "earth_scene.h"
#ifdef _WIN32
#include <windows.h> // must precede GL/glu.h so CALLBACK etc. are defined
#endif
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Height of the warped grid at radial distance `r` from the sphere's
// center, using Flamm's paraboloid -- the standard embedding-diagram
// formula for how the Schwarzschild spatial metric curves around a mass:
//   z(r) = 2 * sqrt(sphere_radius * (r - sphere_radius))
// Flipped and shifted here (z(r) - z(extent)) so the grid sits flat (0) at
// its outer edge and dips toward the sphere -- the familiar "ball on a
// rubber sheet" picture. sphere_radius plays the role of the Schwarzschild
// radius; r is clamped to it so the formula stays defined under the sphere
// itself (that region is occluded by the sphere anyway).
float warp_height(float r, float sphere_radius, float extent, float warp_scale) {
    float r_eff = r < sphere_radius ? sphere_radius : r;
    float z = 2.0f * sqrtf(sphere_radius * (r_eff - sphere_radius));
    float z_far = 2.0f * sqrtf(sphere_radius * (extent - sphere_radius));
    return (z - z_far) * warp_scale;
}

void draw_warped_grid(float sphere_radius, float extent, float warp_scale, float spacing) {
    auto vertex_at = [&](float x, float z) {
        float r = sqrtf(x * x + z * z);
        float y = warp_height(r, sphere_radius, extent, warp_scale);
        glVertex3f(x, y, z);
    };

    const int segs_per_line = 24;
    int steps = (int)(extent / spacing);

    glColor3f(0.55f, 0.70f, 1.0f);
    glBegin(GL_LINES);
    for (int i = -steps; i <= steps; ++i) {
        float x = i * spacing;
        float prev_z = -extent;
        for (int s = 1; s <= segs_per_line; ++s) {
            float z = -extent + (2.0f * extent) * (float)s / (float)segs_per_line;
            vertex_at(x, prev_z);
            vertex_at(x, z);
            prev_z = z;
        }
    }
    for (int i = -steps; i <= steps; ++i) {
        float z = i * spacing;
        float prev_x = -extent;
        for (int s = 1; s <= segs_per_line; ++s) {
            float x = -extent + (2.0f * extent) * (float)s / (float)segs_per_line;
            vertex_at(prev_x, z);
            vertex_at(x, z);
            prev_x = x;
        }
    }
    glEnd();
}

// Simple deterministic "Earth-like" continent pattern: blends ocean blue
// with green/brown land using a handful of overlapping sine-wave blobs in
// (longitude, latitude) space. Not real geography -- just enough visual
// variation to read as a planet instead of a flat-colored ball, without
// needing an external texture image.
void earth_color(float lat_deg, float lon_deg, float& r, float& g, float& b) {
    float land = sinf(lon_deg * 0.05f + 1.0f) * cosf(lat_deg * 0.07f)
                + sinf(lon_deg * 0.13f - 2.0f) * cosf(lat_deg * 0.11f + 1.0f) * 0.6f
                + sinf(lon_deg * 0.03f + 4.0f) * sinf(lat_deg * 0.05f) * 0.5f;

    if (fabsf(lat_deg) > 65.0f) {
        r = 0.92f; g = 0.94f; b = 0.98f; // polar ice
    } else if (land > 0.15f) {
        r = 0.35f; g = 0.55f; b = 0.25f; // land
    } else {
        r = 0.10f; g = 0.30f; b = 0.65f; // ocean
    }
}

// Pure black -- the event horizon casts no light of its own and reflects
// none, so unlike earth_color() this ignores lat/lon entirely.
void black_hole_color(float /*lat_deg*/, float /*lon_deg*/, float& r, float& g, float& b) {
    r = g = b = 0.0f;
}

typedef void (*SphereColorFn)(float lat_deg, float lon_deg, float& r, float& g, float& b);

void draw_sphere(float radius, int lat_segments, int lon_segments, SphereColorFn color_fn) {
    for (int i = 0; i < lat_segments; ++i) {
        float lat0 = 180.0f * (float)i / (float)lat_segments - 90.0f;
        float lat1 = 180.0f * (float)(i + 1) / (float)lat_segments - 90.0f;
        float theta0 = lat0 * kPi / 180.0f;
        float theta1 = lat1 * kPi / 180.0f;

        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= lon_segments; ++j) {
            float lon = 360.0f * (float)j / (float)lon_segments;
            float phi = lon * kPi / 180.0f;

            float thetas[2] = { theta1, theta0 };
            for (float theta : thetas) {
                float lat_deg = theta * 180.0f / kPi;
                float x = cosf(theta) * cosf(phi);
                float y = sinf(theta);
                float z = cosf(theta) * sinf(phi);

                float r, g, b;
                color_fn(lat_deg, lon, r, g, b);

                glNormal3f(x, y, z);
                glColor3f(r, g, b);
                glVertex3f(x * radius, y * radius, z * radius);
            }
        }
        glEnd();
    }
}

// Flat glowing ring in the local x-z plane (matching the disk orientation
// convention in the black_hole reference repo). Unlit -- an accretion disk
// is emitting its own light, not reflecting a scene light -- with a
// per-vertex radial gradient from a hot yellow-white inner edge to a
// cooler orange-red outer edge; GL_TRIANGLE_STRIP interpolates the rest.
void draw_accretion_disk(float inner_radius, float outer_radius, int segments) {
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * kPi * (float)i / (float)segments;
        float cx = cosf(angle);
        float sx = sinf(angle);

        glColor3f(0.85f, 0.35f, 0.05f);
        glVertex3f(cx * outer_radius, 0.0f, sx * outer_radius);

        glColor3f(1.0f, 0.95f, 0.75f);
        glVertex3f(cx * inner_radius, 0.0f, sx * inner_radius);
    }
    glEnd();
}

} // namespace

void render_earth_scene(float screen_w, float screen_h, float time_seconds, float sphere_radius,
                         float cam_azimuth, float cam_elevation, float cam_distance,
                         CelestialBody body) {
    double aspect = (double)screen_w / (double)screen_h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50.0, aspect, 0.1, 100.0);

    // Orbit camera: eye sits on a sphere of `cam_distance` around `target`,
    // positioned by azimuth (around Y) and elevation (above the XZ plane).
    const float target_x = 0.0f, target_y = -1.5f, target_z = 0.0f;
    float eye_x = target_x + cam_distance * cosf(cam_elevation) * sinf(cam_azimuth);
    float eye_y = target_y + cam_distance * sinf(cam_elevation);
    float eye_z = target_z + cam_distance * cosf(cam_elevation) * cosf(cam_azimuth);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye_x, eye_y, eye_z,
              target_x, target_y, target_z,
              0.0, 1.0, 0.0);  // up

    const float extent = 8.0f;
    const float warp_scale = 0.7f;

    glDisable(GL_LIGHTING);
    draw_warped_grid(sphere_radius, extent, warp_scale, 1.0f);

    float well_depth = warp_height(0.0f, sphere_radius, extent, warp_scale);

    glPushMatrix();
    glTranslatef(0.0f, well_depth, 0.0f);
    glRotatef(time_seconds * 12.0f, 0.0f, 1.0f, 0.0f); // slow spin

    if (body == CelestialBody::Earth) {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

        GLfloat light_pos[] = { 5.0f, 8.0f, 6.0f, 1.0f };
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
        GLfloat ambient[] = { 0.15f, 0.15f, 0.18f, 1.0f };
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);

        draw_sphere(sphere_radius, 24, 36, earth_color);

        glDisable(GL_LIGHTING);
        glDisable(GL_LIGHT0);
        glDisable(GL_COLOR_MATERIAL);
    } else {
        // Black hole: unlit event horizon (it emits/reflects nothing) plus
        // an unlit, self-glowing accretion disk. Sizes loosely follow the
        // reference repo's disk (outer ~5x the Schwarzschild radius); the
        // disk isn't rotated with the sphere so it stays a stable ring.
        glDisable(GL_LIGHTING);
        draw_sphere(sphere_radius, 24, 36, black_hole_color);
        glRotatef(-time_seconds * 12.0f, 0.0f, 1.0f, 0.0f); // undo the sphere spin above
        draw_accretion_disk(sphere_radius * 1.5f, sphere_radius * 5.0f, 64);
    }

    glPopMatrix();
}

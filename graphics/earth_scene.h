#pragma once

// Which object sits at the center of the warped grid.
enum class CelestialBody {
    Earth,
    BlackHole,
};

// Renders the spacetime-warping 3D scene: a body sitting in a grid that
// shows how it curves the space around it, drawn as an embedding diagram
// (see embedding_w() in earth_scene.cpp for the exact formulas).
//
// Earth renders as a textured, sunlit globe (graphics/Earth_diff.jpg) with
// its axial tilt and a thin atmosphere glow. BlackHole renders as a black
// event-horizon sphere at the bottom of its funnel, an accretion disk lying
// in the (embedded) equatorial plane from the innermost stable orbit
// outward, and a faint glow at the photon sphere. Both sit in a star field.
//
// Uses the classic OpenGL fixed-function pipeline (real depth-tested 3D
// geometry, a real perspective camera) -- see main.cpp for why the window
// is a compatibility-profile context. Sets up its own projection/modelview
// matrices, so call after clearing the color+depth buffers and before any
// 2D UI overlay draws.
//
// `body_radius` is the drawn sphere's radius and `schwarzschild_radius`
// (r_s = 2GM/c^2, in the same world units) sets how strongly it curves
// space. For a black hole they're equal; for an ordinary body r_s must be
// smaller than body_radius. Keep both well inside the grid (radius 14).
//
// `cam_azimuth`/`cam_elevation` (radians) and `cam_distance` (world units)
// place an orbit camera around the body -- the caller owns this state
// (e.g. persistent UI variables driven by mouse drag/scroll) and passes it
// in fresh each frame.
void render_earth_scene(float screen_w, float screen_h, float time_seconds,
                        float body_radius, float schwarzschild_radius,
                        float cam_azimuth, float cam_elevation, float cam_distance,
                        CelestialBody body);

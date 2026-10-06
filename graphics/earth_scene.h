#pragma once

// Which object sits at the center of the warped grid.
enum class CelestialBody {
    Earth,
    BlackHole,
};

// Renders the spacetime-warping 3D scene: a rotating body sitting at the
// bottom of a grid it warps via Flamm's paraboloid. Earth renders as a lit
// UV sphere with a procedural continent/ocean/ice texture; BlackHole
// renders as an unlit black event-horizon sphere plus a glowing accretion
// disk (loosely following the Schwarzschild-radius grid-warp and x-z-plane
// disk convention from github.com/kavan010/black_hole, the reference repo
// this was built against, adapted to this project's fixed-function
// pipeline rather than its ray-traced/compute-shader one). Uses the
// classic OpenGL fixed-function pipeline (real depth-tested 3D geometry, a
// real perspective camera) -- see main.cpp for why the window is a
// compatibility-profile context. Sets up its own projection/modelview
// matrices, so call after clearing the color+depth buffers and before any
// 2D UI overlay draws.
//
// `sphere_radius` doubles as the Schwarzschild radius in the paraboloid
// formula, so it drives both the sphere's visual size and how deep it dips
// the grid -- i.e. it's the "mass" slider on the caller's side. Keep it
// well under the grid's extent (8.0, hardcoded in earth_scene.cpp) so the
// paraboloid stays defined; UI callers should clamp to something like
// [0.5, 6.0].
//
// `cam_azimuth`/`cam_elevation` (radians) and `cam_distance` (world units)
// place an orbit camera around a fixed look-at target near the sphere --
// the caller owns this state (e.g. persistent UI variables driven by mouse
// drag/scroll) and passes it in fresh each frame.
void render_earth_scene(float screen_w, float screen_h, float time_seconds, float sphere_radius,
                         float cam_azimuth, float cam_elevation, float cam_distance,
                         CelestialBody body);

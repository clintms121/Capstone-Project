#pragma once

#include "trajectory.h"

// Renders a binary black hole merger at time `t_m` (units of the total mass
// M) along `trajectory`: the two horizons orbiting with fading trails, then
// fusing into one remnant whose horizon rings down to a sphere, all sitting
// in a grid that dips under each hole and ripples with the gravitational
// waves they send out. See merger_scene.cpp for what's to scale and what's
// exaggerated.
//
// Like render_earth_scene(), this uses the fixed-function pipeline and sets
// up its own projection/modelview matrices, so call it after clearing the
// color+depth buffers (with depth testing on) and before the 2D UI overlay.
// The caller owns the orbit camera state: `cam_azimuth`/`cam_elevation` in
// radians and `cam_distance` in units of M.
void render_merger_scene(float screen_w, float screen_h, const Trajectory& trajectory, double t_m,
                         float cam_azimuth, float cam_elevation, float cam_distance);

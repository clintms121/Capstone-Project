#pragma once

#include <string>
#include <vector>

// One row of a trajectory file written by physics/build_trajectories.py
// (format: physics/README.md). Lengths and times are in units of the
// binary's total mass M (G = c = 1), with the center of mass at the origin
// and the orbit in the x-y plane.
struct TrajectorySample {
    double t_m = 0.0;                      // time, units of M
    double t_s = 0.0;                      // time, seconds
    float x1 = 0.0f, y1 = 0.0f, z1 = 0.0f; // body 1 position
    float x2 = 0.0f, y2 = 0.0f, z2 = 0.0f; // body 2 position
    float r1 = 0.0f, r2 = 0.0f;            // horizon radii; the remnant's after merger
    float h = 0.0f;                        // plus-polarization strain, peak |h| = 1
    bool merged = false;                   // false in the inspiral, true in the ringdown

    float separation() const;
};

// A merger's trajectory, plus the emitted gravitational wave's envelope and
// phase, which the file doesn't store but which follow from it (see
// load_trajectory()). The simulation screen only plays this back; no
// physics is run at render time.
struct Trajectory {
    std::vector<TrajectorySample> samples;
    std::vector<float> wave_amplitude; // per sample: wave envelope, 1 at merger
    std::vector<float> wave_phase;     // per sample: l = m = 2 wave phase, radians
    int merge_index = 0;               // last inspiral sample: the horizons touch
    double seconds_per_m = 0.0;        // the total mass as a time, G M / c^3
    float ringdown_omega = 0.0f;       // remnant's ringing angular frequency, per M
    float ringdown_tau = 0.0f;         // and its damping time, units of M

    bool loaded() const { return merge_index > 0; }
    double start_time() const { return samples.front().t_m; }
    double end_time() const { return samples.back().t_m; }
    double merge_time() const { return samples[merge_index].t_m; }
    double total_mass_msun() const;

    // Index i of the samples bracketing t_m (samples[i] <= t_m < samples[i+1]),
    // clamped so that i + 1 is always valid.
    int index_at(double t_m) const;
    // State at t_m, interpolated linearly between samples (clamped at the ends).
    TrajectorySample at(double t_m) const;
    // Envelope and phase of the wave emitted at t_m. Before the file starts,
    // the first orbit's wave is run backward; after it ends, the ringing
    // keeps decaying.
    void wave_at(double t_m, float& amplitude, float& phase) const;
    // Rate of change of the wave phase at t_m, radians per M (twice the
    // orbital angular velocity before merger).
    float wave_omega(double t_m) const;
};

// Reads data/trajectories/<event_id>.csv. Returns a trajectory that isn't
// loaded() if the file is missing or malformed.
Trajectory load_trajectory(const std::string& csv_path);

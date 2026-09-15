#pragma once

#include <string>

// Parameters needed to kick off the merger visualization for one event.
// Produced offline by ml/mass_estimation + physics/inspiral.py/ringdown.py,
// listed in data/trajectories/index.csv, and read here at UI time. The
// renderer never computes physics itself -- it only plays back the
// trajectory file this points to.
struct SimulationParams {
    std::string event_id;
    double chirp_mass_msun = 0.0;
    double mass_ratio = 0.0;      // m2 / m1, in (0, 1]
    std::string trajectory_path;  // data/trajectories/<event_id>.npz

    bool is_selected() const { return !event_id.empty(); }
};

#pragma once

#include "simulation_params.h"
#include <string>
#include <vector>

// Reads data/trajectories/index.csv (event_id,chirp_mass_msun,mass_ratio,path),
// written by the Python pipeline once an event has been classified,
// mass-estimated, and turned into a trajectory file. Returns an empty list
// if the pipeline hasn't produced a manifest yet.
std::vector<SimulationParams> load_trajectory_manifest(const std::string& csv_path);

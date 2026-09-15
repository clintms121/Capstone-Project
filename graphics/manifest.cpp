#include "manifest.h"
#include <fstream>
#include <sstream>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
        fields.push_back(field);
    return fields;
}

} // namespace

std::vector<SimulationParams> load_trajectory_manifest(const std::string& csv_path) {
    std::vector<SimulationParams> events;

    std::ifstream file(csv_path);
    if (!file.is_open())
        return events; // pipeline hasn't produced a manifest yet

    std::string line;
    std::getline(file, line); // header: event_id,chirp_mass_msun,mass_ratio,path

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        auto fields = split_csv_line(line);
        if (fields.size() < 4)
            continue;

        SimulationParams p;
        p.event_id = fields[0];
        p.chirp_mass_msun = std::stod(fields[1]);
        p.mass_ratio = std::stod(fields[2]);
        p.trajectory_path = fields[3];
        events.push_back(p);
    }

    return events;
}

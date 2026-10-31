// Reads a merger's trajectory file (data/trajectories/<event>.csv) and looks
// up the black holes' positions and the emitted wave at any moment.
#include "trajectory.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSecondsPerSolarMass = 4.925491025543576e-6; // G M_sun / c^3, as in physics/inspiral.py

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
        fields.push_back(field);
    return fields;
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

// The ringdown's angular frequency and damping time, read off the strain
// column: zero crossings come every half cycle, and the peaks between them
// shrink as exp(-t / tau) (fit by least squares on ln|h|). Falls back to
// typical values for a remnant of ~0.95 M spinning at ~0.7 if there's too
// little ringdown to measure.
void measure_ringdown(const std::vector<TrajectorySample>& s, int merge, float& omega, float& tau) {
    omega = 0.53f;
    tau = 11.7f;

    const double t0 = s[merge].t_m;
    std::vector<double> crossings;
    double n = 0.0, sum_t = 0.0, sum_y = 0.0, sum_tt = 0.0, sum_ty = 0.0;
    for (int i = merge + 1; i + 1 < (int)s.size(); ++i) {
        float h0 = s[i].h, h1 = s[i + 1].h;
        if ((h0 < 0.0f) != (h1 < 0.0f))
            crossings.push_back(s[i].t_m + (s[i + 1].t_m - s[i].t_m) * h0 / (h0 - h1));
        bool peak = i > merge + 1 && fabsf(h0) > fabsf(s[i - 1].h) && fabsf(h0) >= fabsf(h1);
        if (peak) {
            double t = s[i].t_m - t0, y = log(fabs(h0));
            n += 1.0;
            sum_t += t;
            sum_y += y;
            sum_tt += t * t;
            sum_ty += t * y;
        }
    }

    if (crossings.size() >= 3)
        omega = (float)(kPi * (double)(crossings.size() - 1) / (crossings.back() - crossings.front()));
    if (n >= 3.0) {
        double slope = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t);
        if (slope < 0.0)
            tau = (float)(-1.0 / slope);
    }
}

} // namespace

float TrajectorySample::separation() const {
    float dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

double Trajectory::total_mass_msun() const {
    return seconds_per_m / kSecondsPerSolarMass;
}

int Trajectory::index_at(double t_m) const {
    auto after = std::upper_bound(samples.begin(), samples.end(), t_m,
                                  [](double t, const TrajectorySample& s) { return t < s.t_m; });
    int i = (int)(after - samples.begin()) - 1;
    return std::max(0, std::min(i, (int)samples.size() - 2));
}

TrajectorySample Trajectory::at(double t_m) const {
    if (t_m <= start_time())
        return samples.front();
    if (t_m >= end_time())
        return samples.back();

    int i = index_at(t_m);
    const TrajectorySample& a = samples[i];
    const TrajectorySample& b = samples[i + 1];
    double span = b.t_m - a.t_m;
    float u = span > 0.0 ? (float)((t_m - a.t_m) / span) : 1.0f;

    TrajectorySample s = a;
    s.t_m = t_m;
    s.t_s = a.t_s + (b.t_s - a.t_s) * u;
    s.x1 = lerp(a.x1, b.x1, u);
    s.y1 = lerp(a.y1, b.y1, u);
    s.z1 = lerp(a.z1, b.z1, u);
    s.x2 = lerp(a.x2, b.x2, u);
    s.y2 = lerp(a.y2, b.y2, u);
    s.z2 = lerp(a.z2, b.z2, u);
    s.r1 = lerp(a.r1, b.r1, u);
    s.r2 = lerp(a.r2, b.r2, u);
    s.h = lerp(a.h, b.h, u);
    return s;
}

void Trajectory::wave_at(double t_m, float& amplitude, float& phase) const {
    if (t_m <= start_time()) {
        // The orbit was barely wider just before the file starts, so keep
        // the first amplitude and run the phase back at the first frequency.
        amplitude = wave_amplitude.front();
        phase = wave_phase.front() + wave_omega(start_time()) * (float)(t_m - start_time());
        return;
    }
    if (t_m >= end_time()) {
        float since = (float)(t_m - end_time());
        amplitude = wave_amplitude.back() * expf(-since / ringdown_tau);
        phase = wave_phase.back() + ringdown_omega * since;
        return;
    }

    int i = index_at(t_m);
    double span = samples[i + 1].t_m - samples[i].t_m;
    float u = span > 0.0 ? (float)((t_m - samples[i].t_m) / span) : 1.0f;
    amplitude = lerp(wave_amplitude[i], wave_amplitude[i + 1], u);
    phase = lerp(wave_phase[i], wave_phase[i + 1], u);
}

float Trajectory::wave_omega(double t_m) const {
    int i = index_at(t_m);
    double span = samples[i + 1].t_m - samples[i].t_m;
    return span > 0.0 ? (float)((wave_phase[i + 1] - wave_phase[i]) / span) : 0.0f;
}

Trajectory load_trajectory(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open())
        return Trajectory{}; // pipeline hasn't produced this file

    // Columns are found by name, so their order in the file doesn't matter.
    std::string line;
    if (!std::getline(file, line))
        return Trajectory{};
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
    const char* names[12] = {"t_m", "t_s", "x1", "y1", "z1", "x2", "y2", "z2", "r1", "r2", "h", "merged"};
    int col[12];
    std::vector<std::string> header = split_csv_line(line);
    for (int k = 0; k < 12; ++k) {
        auto it = std::find(header.begin(), header.end(), names[k]);
        if (it == header.end())
            return Trajectory{};
        col[k] = (int)(it - header.begin());
    }

    Trajectory traj;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        std::vector<std::string> f = split_csv_line(line);
        if (f.size() < header.size())
            return Trajectory{};

        TrajectorySample s;
        try {
            s.t_m = std::stod(f[col[0]]);
            s.t_s = std::stod(f[col[1]]);
            s.x1 = std::stof(f[col[2]]);
            s.y1 = std::stof(f[col[3]]);
            s.z1 = std::stof(f[col[4]]);
            s.x2 = std::stof(f[col[5]]);
            s.y2 = std::stof(f[col[6]]);
            s.z2 = std::stof(f[col[7]]);
            s.r1 = std::stof(f[col[8]]);
            s.r2 = std::stof(f[col[9]]);
            s.h = std::stof(f[col[10]]);
            s.merged = std::stod(f[col[11]]) != 0.0;
        } catch (const std::exception&) {
            return Trajectory{};
        }
        if (!traj.samples.empty() && s.t_m < traj.samples.back().t_m)
            return Trajectory{}; // playback needs time to run forward
        traj.samples.push_back(s);
    }

    const int n = (int)traj.samples.size();
    int merge = -1;
    for (int i = 0; i < n; ++i)
        if (!traj.samples[i].merged)
            merge = i;
    if (n < 2 || merge < 1)
        return Trajectory{};

    traj.merge_index = merge;
    const TrajectorySample& last = traj.samples.back();
    traj.seconds_per_m = last.t_m > 0.0 ? last.t_s / last.t_m : 0.0;
    measure_ringdown(traj.samples, merge, traj.ringdown_omega, traj.ringdown_tau);

    // The wave the file's strain column comes from (physics/inspiral.py):
    // during the inspiral its phase is twice the orbital angle and its
    // envelope goes as 1 / separation; from merger on, it rings at the
    // measured frequency and dies away over the damping time.
    traj.wave_amplitude.resize(n);
    traj.wave_phase.resize(n);
    const float merge_separation = traj.samples[merge].separation();
    double orbit_angle = 0.0;
    for (int i = 0; i <= merge; ++i) {
        const TrajectorySample& s = traj.samples[i];
        double angle = atan2(s.y1, s.x1);
        if (i == 0) {
            orbit_angle = angle;
        } else {
            const TrajectorySample& p = traj.samples[i - 1];
            double step = angle - atan2(p.y1, p.x1);
            orbit_angle += step - 2.0 * kPi * floor((step + kPi) / (2.0 * kPi)); // unwrap
        }
        traj.wave_phase[i] = (float)(2.0 * orbit_angle);
        traj.wave_amplitude[i] = merge_separation / std::max(s.separation(), 1e-6f);
    }
    for (int i = merge + 1; i < n; ++i) {
        float since = (float)(traj.samples[i].t_m - traj.samples[merge].t_m);
        traj.wave_phase[i] = traj.wave_phase[merge] + traj.ringdown_omega * since;
        traj.wave_amplitude[i] = expf(-since / traj.ringdown_tau);
    }
    return traj;
}

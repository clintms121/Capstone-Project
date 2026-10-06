#pragma once

#include <map>
#include <string>
#include <vector>

// One input the classifier demo can show: a real merger recording, a Gravity
// Spy glitch, or a stretch of detector noise. Everything here was computed by
// the Python pipeline and exported by ml/export_demo.py -- the app replays
// results, it doesn't run the models.
struct DemoSample {
    int id = 0;
    std::string kind;        // "event", "glitch" or "noise"
    std::string name;        // event name, or glitch class + Gravity Spy id
    std::string detector;    // "H1" / "L1" for events, empty otherwise
    std::string true_label;  // classifier label it really is
    std::string split;       // "train", "val" or "test" (held out)
    std::string pred[3];     // classifier's top-3 labels, most likely first
    float prob[3] = {0.0f, 0.0f, 0.0f};
    float p_real = 0.0f;
    float snr = 0.0f;        // matched-filter SNR (events only, else NaN)
    float mc_est = 0.0f;     // matched-filter chirp mass, detector frame
    float mc_pub = 0.0f;     // LIGO's published detector-frame chirp mass
    bool measurable = false; // SNR cleared the noise threshold
    std::string image_path;  // 8-bit PGM Q-scan
    unsigned int texture = 0;
};

struct InjectionPoint {
    float snr;   // 0 = plain noise (the false-alarm row)
    int caught;
    int n;
};

struct DemoData {
    std::vector<DemoSample> samples;
    std::map<std::string, double> metrics;
    std::vector<InjectionPoint> injection;

    bool loaded() const { return !samples.empty(); }
    double metric(const std::string& key) const {
        auto it = metrics.find(key);
        return it == metrics.end() ? 0.0 : it->second;
    }
};

// Reads samples.csv, metrics.csv and injection.csv from `dir`. Returns an
// empty DemoData if the export hasn't been run.
DemoData load_demo_data(const std::string& dir);

// The sample's Q-scan as a viridis-coloured OpenGL texture, uploaded on first
// use (needs the GL context current). Returns 0 if the image can't be read.
unsigned int demo_texture(DemoSample& sample);

// "real_signal" -> "Real signal", "Koi_Fish" -> "Koi Fish", ...
std::string pretty_label(const std::string& label);

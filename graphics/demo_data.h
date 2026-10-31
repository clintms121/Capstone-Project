#pragma once

#include <string>
#include <vector>

// One recording from the Classifier screen's data. Everything here was
// computed by the Python pipeline and exported to data/demo/ by
// ml/export_demo.py -- the app shows results, it doesn't run the models.
struct DemoSample {
    int id = 0;
    std::string kind;            // "event" (a real merger), "false_positive" (noise it called a merger),
                                 // or "glitch"/"noise" (exported too, but not listed on screen)
    std::string name;            // event name like "GW150914", or what the recording is
    std::string detector;        // "H1" (Hanford) or "L1" (Livingston)
    std::string split;           // "train", "val" or "test" (held out from training)
    std::string predicted_label; // the classifier's top guess, e.g. "real_signal" or "Blip"
    float p_real = 0.0f;         // the classifier's probability that it's a real merger, 0..1
    float distance_ly = 0.0f;    // how far the waves traveled, in light-years (NaN if unpublished)
    std::string image_path;      // the Q-scan the classifier saw, as an 8-bit PGM
    unsigned int texture = 0;    // that image on the GPU, once demo_texture() has loaded it
};

struct DemoData {
    std::vector<DemoSample> samples;

    bool loaded() const { return !samples.empty(); }
};

// Reads data/demo/samples.csv. Returns an empty DemoData if the export
// hasn't been run.
DemoData load_demo_data(const std::string& dir);

// The sample's Q-scan as a viridis-colored OpenGL texture, uploaded on first
// use (needs the GL context current). Returns 0 if the image can't be read.
unsigned int demo_texture(DemoSample& sample);

// "real_signal" -> "Real signal", "Koi_Fish" -> "Koi Fish", ...
std::string pretty_label(const std::string& label);

// 4.0e9 -> "4.0 billion", 5.6e8 -> "561 million", 2.5e4 -> "25 thousand".
std::string format_count(double n);

// format_count(ly) + " light-years".
std::string format_light_years(double ly);

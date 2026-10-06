#include "demo_data.h"

#include <GLFW/glfw3.h>
#include <cmath>
#include <fstream>
#include <sstream>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F // GL 1.2; Windows' gl.h stops at 1.1
#endif

namespace {

using Row = std::map<std::string, std::string>;

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
        fields.push_back(field);
    return fields;
}

std::vector<Row> read_csv(const std::string& path) {
    std::vector<Row> rows;
    std::ifstream file(path);
    if (!file.is_open())
        return rows;

    std::string line;
    std::getline(file, line);
    std::vector<std::string> header = split_csv_line(line);
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        std::vector<std::string> fields = split_csv_line(line);
        Row row;
        for (size_t i = 0; i < header.size() && i < fields.size(); ++i)
            row[header[i]] = fields[i];
        rows.push_back(row);
    }
    return rows;
}

float to_float(const std::string& s) {
    try {
        return std::stof(s);
    } catch (...) {
        return NAN;
    }
}

// Five-stop approximation of matplotlib's viridis, the colormap Gravity Spy
// and LIGO's own Q-scan plots use.
void viridis(float t, unsigned char* rgb) {
    static const float stops[5][3] = {
        {0.267f, 0.005f, 0.329f}, {0.229f, 0.322f, 0.546f}, {0.128f, 0.567f, 0.551f},
        {0.369f, 0.789f, 0.383f}, {0.993f, 0.906f, 0.144f},
    };
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    float pos = t * 4.0f;
    int i = pos >= 4.0f ? 3 : (int)pos;
    float f = pos - (float)i;
    for (int c = 0; c < 3; ++c)
        rgb[c] = (unsigned char)std::lround(255.0f * (stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f));
}

} // namespace

DemoData load_demo_data(const std::string& dir) {
    DemoData data;

    for (const Row& r : read_csv(dir + "/samples.csv")) {
        DemoSample s;
        s.id = (int)to_float(r.at("id"));
        s.kind = r.at("kind");
        s.name = r.at("name");
        s.detector = r.at("detector");
        s.true_label = r.at("true_label");
        s.split = r.at("split");
        for (int k = 0; k < 3; ++k) {
            s.pred[k] = r.at("pred" + std::to_string(k + 1));
            s.prob[k] = to_float(r.at("p" + std::to_string(k + 1)));
        }
        s.p_real = to_float(r.at("p_real"));
        s.snr = to_float(r.at("snr"));
        s.mc_est = to_float(r.at("mc_est"));
        s.mc_pub = to_float(r.at("mc_pub"));
        s.measurable = r.at("measurable") == "1";
        s.image_path = r.at("image");
        data.samples.push_back(s);
    }

    for (const Row& r : read_csv(dir + "/metrics.csv"))
        data.metrics[r.at("key")] = to_float(r.at("value"));

    for (const Row& r : read_csv(dir + "/injection.csv"))
        data.injection.push_back({to_float(r.at("snr")), (int)to_float(r.at("caught")), (int)to_float(r.at("n"))});

    return data;
}

unsigned int demo_texture(DemoSample& sample) {
    if (sample.texture)
        return sample.texture;

    std::ifstream file(sample.image_path, std::ios::binary);
    std::string magic;
    int w = 0, h = 0, maxval = 0;
    file >> magic >> w >> h >> maxval;
    if (!file || magic != "P5" || w <= 0 || h <= 0)
        return 0;
    file.get(); // the single whitespace byte before the pixels

    std::vector<unsigned char> gray(w * h);
    file.read(reinterpret_cast<char*>(gray.data()), gray.size());
    if (!file)
        return 0;

    std::vector<unsigned char> rgb(w * h * 3);
    for (int i = 0; i < w * h; ++i)
        viridis(gray[i] / 255.0f, &rgb[i * 3]);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    sample.texture = tex;
    return tex;
}

std::string pretty_label(const std::string& label) {
    if (label == "real_signal")
        return "Real signal";
    if (label == "No_Glitch")
        return "No glitch (noise)";
    if (label == "None_of_the_Above")
        return "None of the above";
    std::string out = label;
    for (char& c : out)
        if (c == '_')
            c = ' ';
    return out;
}

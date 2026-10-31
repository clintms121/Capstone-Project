// Loads the Classifier screen's exported results (data/demo/) and turns
// each Q-scan image into a colored texture, plus small text helpers.
#include "demo_data.h"

#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F // GL 1.2; Windows' gl.h stops at 1.1
#endif

namespace {

using Row = std::map<std::string, std::string>; // column name -> value

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
        fields.push_back(field);
    return fields;
}

// Every row of a CSV file with a header line, keyed by column name.
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

// The number in `s`, or NaN if it isn't one (empty cells mean "unknown").
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

    for (const Row& row : read_csv(dir + "/samples.csv")) {
        DemoSample s;
        s.id = (int)to_float(row.at("id"));
        s.kind = row.at("kind");
        s.name = row.at("name");
        s.detector = row.at("detector");
        s.split = row.at("split");
        s.predicted_label = row.at("pred1");
        s.p_real = to_float(row.at("p_real"));
        s.distance_ly = row.count("distance_ly") ? to_float(row.at("distance_ly")) : NAN; // older exports lack it
        s.image_path = row.at("image");
        data.samples.push_back(s);
    }
    return data;
}

unsigned int demo_texture(DemoSample& sample) {
    if (sample.texture)
        return sample.texture;

    // A binary PGM is a short text header ("P5 width height maxval")
    // followed by one byte per pixel.
    std::ifstream file(sample.image_path, std::ios::binary);
    std::string format;
    int width = 0, height = 0, max_value = 0;
    file >> format >> width >> height >> max_value;
    if (!file || format != "P5" || width <= 0 || height <= 0)
        return 0;
    file.get(); // the single whitespace byte before the pixels

    std::vector<unsigned char> gray(width * height);
    file.read(reinterpret_cast<char*>(gray.data()), gray.size());
    if (!file)
        return 0;

    std::vector<unsigned char> rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i)
        viridis(gray[i] / 255.0f, &rgb[i * 3]);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
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

std::string format_count(double n) {
    char buf[32];
    if (n >= 1e9)
        std::snprintf(buf, sizeof(buf), "%.1f billion", n / 1e9);
    else if (n >= 1e6)
        std::snprintf(buf, sizeof(buf), "%.0f million", n / 1e6);
    else if (n >= 1e3)
        std::snprintf(buf, sizeof(buf), "%.0f thousand", n / 1e3);
    else
        std::snprintf(buf, sizeof(buf), "%.0f", n);
    return buf;
}

std::string format_light_years(double ly) { return format_count(ly) + " light-years"; }

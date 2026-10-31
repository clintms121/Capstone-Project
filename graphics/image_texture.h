#pragma once

// An image file uploaded to an OpenGL texture. texture == 0 means the file
// couldn't be loaded (missing, or a format stb_image can't read).
struct ImageTexture {
    unsigned int texture = 0;
    int width = 0;
    int height = 0;
};

// Loads a JPEG/PNG from `path` (relative to the working directory, i.e. the
// repo root) on first call and caches it; later calls return the cached
// result. Needs a current GL context.
const ImageTexture& load_image_texture(const char* path);

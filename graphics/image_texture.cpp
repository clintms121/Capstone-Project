// Loads image files (JPEG/PNG) into OpenGL textures the screens can draw,
// loading each file only once.
#include "image_texture.h"
#include <GLFW/glfw3.h>
#include <map>
#include <string>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F // GL 1.2; Windows' gl.h stops at 1.1
#endif
#ifndef GL_GENERATE_MIPMAP
#define GL_GENERATE_MIPMAP 0x8191 // GL 1.4, still in the compatibility profile
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

const ImageTexture& load_image_texture(const char* path) {
    static std::map<std::string, ImageTexture> cache;

    auto found = cache.find(path);
    if (found != cache.end())
        return found->second;

    ImageTexture& image = cache[path];
    int channels = 0;
    unsigned char* pixels = stbi_load(path, &image.width, &image.height, &channels, 4);
    if (!pixels)
        return image;

    glGenTextures(1, &image.texture);
    glBindTexture(GL_TEXTURE_2D, image.texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // Mipmaps keep the image smooth instead of shimmering when it's drawn
    // much smaller than its native size (slide thumbnails, a distant globe).
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(pixels);
    return image;
}

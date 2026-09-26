// Offline renderer: renders the Cornell Box progressively and writes a PNG.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "orbit_camera.h"
#include "renderer.h"

struct Options
{
    std::string output = "cornell_box.png";
    unsigned int width = 768;
    unsigned int height = 768;
    unsigned int samples = 1024;
    unsigned int maxDepth = 12;
    float yawDegrees = 0.0f;
    float pitchDegrees = 0.0f;
};

static void printUsage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " [options]\n"
              << "  -o <file>     output PNG (default cornell_box.png)\n"
              << "  -r <W>x<H>    resolution (default 768x768)\n"
              << "  -s <n>        samples per pixel (default 1024)\n"
              << "  -d <n>        max path depth (default 12)\n"
              << "  --orbit <yaw>,<pitch>\n"
              << "                orbit the camera around the box, in degrees (default 0,0)\n";
}

static Options parseArgs(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const bool hasValue   = i + 1 < argc;
        if (arg == "-o" && hasValue)
            opt.output = argv[++i];
        else if (arg == "-r" && hasValue)
        {
            if (std::sscanf(argv[++i], "%ux%u", &opt.width, &opt.height) != 2 || opt.width == 0 ||
                opt.height == 0)
                throw std::runtime_error("Bad resolution, expected e.g. 768x768");
        }
        else if (arg == "-s" && hasValue)
            opt.samples = std::max(1, std::atoi(argv[++i]));
        else if (arg == "-d" && hasValue)
            opt.maxDepth = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--orbit" && hasValue)
        {
            if (std::sscanf(argv[++i], "%f,%f", &opt.yawDegrees, &opt.pitchDegrees) != 2)
                throw std::runtime_error("Bad orbit, expected e.g. 30,15");
        }
        else
        {
            printUsage(argv[0]);
            std::exit(arg == "-h" || arg == "--help" ? 0 : 1);
        }
    }
    return opt;
}

int main(int argc, char** argv)
try
{
    const Options opt = parseArgs(argc, argv);

    Renderer renderer(opt.width, opt.height);
    renderer.setMaxDepth(opt.maxDepth);

    OrbitCamera orbit;
    orbit.rotate(opt.yawDegrees * kPi / 180.0f, opt.pitchDegrees * kPi / 180.0f);
    renderer.setCamera(orbit.camera());

    // Split the work into short launches so no single kernel runs long enough to trip a display watchdog.
    const unsigned int samplesPerLaunch = std::min(opt.samples, 16u);
    const unsigned int numLaunches      = (opt.samples + samplesPerLaunch - 1) / samplesPerLaunch;

    std::cout << "Rendering " << opt.width << "x" << opt.height << " at " << numLaunches * samplesPerLaunch
              << " spp, max depth " << opt.maxDepth << "\n";

    const auto start = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < numLaunches; ++i)
        renderer.render(samplesPerLaunch);
    std::vector<uchar4> pixels;
    renderer.downloadImage(pixels);  // waits for the launches
    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Rendered in " << elapsed.count() << " ms\n";

    if (!stbi_write_png(opt.output.c_str(), opt.width, opt.height, 4, pixels.data(), opt.width * 4))
        throw std::runtime_error("Failed to write " + opt.output);
    std::cout << "Wrote " << opt.output << "\n";
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}

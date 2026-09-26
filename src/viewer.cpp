// Interactive viewer: drag to rotate the Cornell Box, scroll to zoom. The image refines
// progressively while the view is still.
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbit_camera.h"
#include "renderer.h"

namespace
{

constexpr float kRadiansPerPixel = 0.005f;
constexpr float kRadiansPerKey   = 5.0f * kPi / 180.0f;
constexpr double kTargetFrameMs  = 12.0;  // GPU time budget per frame, leaving headroom under 60 Hz

struct Options
{
    unsigned int maxSamples = 16384;  // stop refining (and idle) after this many spp
    unsigned int maxDepth   = 12;
};

void printUsage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " [options]\n"
              << "  -s <n>   stop refining after n samples per pixel (default 16384)\n"
              << "  -d <n>   max path depth (default 12)\n";
}

Options parseArgs(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const bool hasValue   = i + 1 < argc;
        if (arg == "-s" && hasValue)
            opt.maxSamples = std::max(1, std::atoi(argv[++i]));
        else if (arg == "-d" && hasValue)
            opt.maxDepth = std::max(1, std::atoi(argv[++i]));
        else
        {
            printUsage(argv[0]);
            std::exit(arg == "-h" || arg == "--help" ? 0 : 1);
        }
    }
    return opt;
}

void sdlCheck(bool ok, const char* what)
{
    if (!ok)
        throw std::runtime_error(std::string(what) + " failed: " + SDL_GetError());
}

SDL_Texture* createTexture(SDL_Renderer* sdlRenderer, int width, int height)
{
    // The renderer writes bytes in R, G, B, A order.
    SDL_Texture* texture =
        SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);
    sdlCheck(texture != nullptr, "SDL_CreateTexture");
    return texture;
}

}  // namespace

int main(int argc, char** argv)
try
{
    const Options opt = parseArgs(argc, argv);

    sdlCheck(SDL_Init(SDL_INIT_VIDEO), "SDL_Init");
    SDL_Window* window = SDL_CreateWindow("Spectral", 768, 768, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    sdlCheck(window != nullptr, "SDL_CreateWindow");
    SDL_Renderer* sdlRenderer = SDL_CreateRenderer(window, nullptr);
    sdlCheck(sdlRenderer != nullptr, "SDL_CreateRenderer");
    SDL_SetRenderVSync(sdlRenderer, 1);

    int width = 0, height = 0;
    sdlCheck(SDL_GetWindowSizeInPixels(window, &width, &height), "SDL_GetWindowSizeInPixels");

    // Scoped so the renderer releases its GPU resources before SDL shuts down.
    {
        Renderer renderer(width, height);
        renderer.setMaxDepth(opt.maxDepth);
        SDL_Texture* texture = createTexture(sdlRenderer, width, height);

        std::cout << "Controls: drag to rotate, scroll or +/- to zoom, arrow keys to rotate,\n"
                  << "          R to reset the view, Esc or Q to quit.\n";

        OrbitCamera orbit;
        bool cameraChanged        = true;
        bool dragging             = false;
        bool running              = true;
        unsigned int samplesPerFrame = 1;
        std::vector<uchar4> pixels;

        auto lastTitle   = std::chrono::steady_clock::now();
        unsigned int framesSinceTitle = 0;

        auto handleEvent = [&](const SDL_Event& e) {
            switch (e.type)
            {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                switch (e.key.key)
                {
                case SDLK_ESCAPE:
                case SDLK_Q: running = false; break;
                case SDLK_R: orbit = OrbitCamera(); cameraChanged = true; break;
                case SDLK_LEFT: orbit.rotate(-kRadiansPerKey, 0.0f); cameraChanged = true; break;
                case SDLK_RIGHT: orbit.rotate(kRadiansPerKey, 0.0f); cameraChanged = true; break;
                case SDLK_UP: orbit.rotate(0.0f, -kRadiansPerKey); cameraChanged = true; break;
                case SDLK_DOWN: orbit.rotate(0.0f, kRadiansPerKey); cameraChanged = true; break;
                case SDLK_EQUALS:
                case SDLK_KP_PLUS: orbit.zoom(0.9f); cameraChanged = true; break;
                case SDLK_MINUS:
                case SDLK_KP_MINUS: orbit.zoom(1.0f / 0.9f); cameraChanged = true; break;
                default: break;
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (e.button.button == SDL_BUTTON_LEFT)
                    dragging = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (e.button.button == SDL_BUTTON_LEFT)
                    dragging = false;
                break;
            case SDL_EVENT_MOUSE_MOTION:
                // Trust the button state carried by the event, so a release that happened outside the
                // window (or during a focus change) can't leave the viewer stuck in drag mode.
                dragging = (e.motion.state & SDL_BUTTON_LMASK) != 0;
                // Turntable: dragging right spins the scene right, dragging down tips its top towards you.
                if (dragging)
                {
                    orbit.rotate(e.motion.xrel * kRadiansPerPixel, e.motion.yrel * kRadiansPerPixel);
                    cameraChanged = true;
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                orbit.zoom(std::pow(0.9f, e.wheel.y));
                cameraChanged = true;
                break;
            case SDL_EVENT_WINDOW_FOCUS_LOST:
                dragging = false;
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                if (e.window.data1 > 0 && e.window.data2 > 0)
                {
                    width  = e.window.data1;
                    height = e.window.data2;
                    renderer.resize(width, height);
                    SDL_DestroyTexture(texture);
                    texture = createTexture(sdlRenderer, width, height);
                }
                break;
            default:
                break;
            }
        };

        while (running)
        {
            // Once converged, sleep until something happens instead of spinning the GPU.
            const bool converged = renderer.accumulatedSamples() >= opt.maxSamples && !cameraChanged;
            SDL_Event e;
            if (converged && SDL_WaitEvent(&e))
                handleEvent(e);
            while (SDL_PollEvent(&e))
                handleEvent(e);
            if (!running)
                break;

            if (cameraChanged)
            {
                renderer.setCamera(orbit.camera());
                cameraChanged = false;
            }

            if (renderer.accumulatedSamples() < opt.maxSamples)
            {
                // While dragging, favour frame rate; otherwise fit as many samples as the budget allows.
                const unsigned int samples =
                    dragging ? 1u : std::min(samplesPerFrame, opt.maxSamples - renderer.accumulatedSamples());

                const auto start = std::chrono::steady_clock::now();
                renderer.render(samples);
                renderer.downloadImage(pixels);
                const std::chrono::duration<double, std::milli> gpuMs = std::chrono::steady_clock::now() - start;

                if (!dragging)
                {
                    const double scale = kTargetFrameMs / std::max(gpuMs.count(), 0.1);
                    samplesPerFrame = std::clamp(static_cast<unsigned int>(samples * std::min(scale, 2.0)), 1u, 64u);
                }
                sdlCheck(SDL_UpdateTexture(texture, nullptr, pixels.data(), width * 4), "SDL_UpdateTexture");
            }

            SDL_RenderClear(sdlRenderer);
            SDL_RenderTexture(sdlRenderer, texture, nullptr, nullptr);
            SDL_RenderPresent(sdlRenderer);

            ++framesSinceTitle;
            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> sinceTitle = now - lastTitle;
            if (sinceTitle.count() >= 0.25 || converged)
            {
                const std::string title = "Spectral - " + std::to_string(renderer.accumulatedSamples()) + " spp - " +
                                          std::to_string(static_cast<int>(framesSinceTitle / sinceTitle.count())) +
                                          " fps";
                SDL_SetWindowTitle(window, title.c_str());
                lastTitle        = now;
                framesSinceTitle = 0;
            }
        }

        SDL_DestroyTexture(texture);
    }

    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}

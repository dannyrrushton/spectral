// A progressive OptiX path tracer for the Cornell Box scene, shared by the offline CLI and the
// interactive viewer.
#pragma once

#include <optix_types.h>

#include <cuda_runtime.h>

#include <vector>

#include "shared.h"

struct Camera
{
    float3 eye;
    float3 lookAt;
    float3 up;
    float vfovDegrees;
};

class Renderer
{
public:
    Renderer(unsigned int width, unsigned int height);
    ~Renderer();
    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Changing any of these restarts accumulation.
    void resize(unsigned int width, unsigned int height);
    void setCamera(const Camera& camera);
    void setMaxDepth(unsigned int maxDepth);

    // Queues one launch that adds `samples` samples per pixel to the running average. Asynchronous;
    // downloadImage waits for it.
    void render(unsigned int samples);

    // Copies the current tonemapped sRGB image (RGBA8, top row first) into `pixels`.
    void downloadImage(std::vector<uchar4>& pixels);

    unsigned int width() const { return m_params.width; }
    unsigned int height() const { return m_params.height; }
    unsigned int accumulatedSamples() const { return m_params.accumulatedSamples; }

    // The camera from the Cornell data: 35 mm focal length, 25 mm x 25 mm film.
    static Camera defaultCamera();

private:
    void allocateFrameBuffers();
    void freeFrameBuffers();
    void updateCameraBasis();
    void resetAccumulation();

    OptixDeviceContext m_context = nullptr;
    OptixModule m_module         = nullptr;
    std::vector<OptixProgramGroup> m_programGroups;
    OptixPipeline m_pipeline     = nullptr;
    OptixShaderBindingTable m_sbt = {};       // camera path tracing
    OptixShaderBindingTable m_lightSbt = {};  // light tracing from the beam
    bool m_clearLightImage = true;
    std::vector<CUdeviceptr> m_sceneBuffers;  // geometry, acceleration structures and SBT records
    CUstream m_stream            = nullptr;
    LaunchParams* m_dParams      = nullptr;

    Camera m_camera = defaultCamera();
    LaunchParams m_params = {};
};

// OptiX setup and progressive rendering for the Cornell Box scene.
#include "renderer.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cornell_box.h"
#include "vec_math.h"

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                              \
            std::ostringstream ss;                                                              \
            ss << "CUDA error: " << #call << " failed: " << cudaGetErrorString(err_) << " ("   \
               << __FILE__ << ":" << __LINE__ << ")";                                           \
            throw std::runtime_error(ss.str());                                                 \
        }                                                                                       \
    } while (0)

#define OPTIX_CHECK(call)                                                                       \
    do {                                                                                        \
        OptixResult res_ = (call);                                                              \
        if (res_ != OPTIX_SUCCESS) {                                                            \
            std::ostringstream ss;                                                              \
            ss << "OptiX error: " << #call << " failed: " << optixGetErrorName(res_) << " ("   \
               << __FILE__ << ":" << __LINE__ << ")";                                           \
            throw std::runtime_error(ss.str());                                                 \
        }                                                                                       \
    } while (0)

// For calls that also produce a compile/link log.
#define OPTIX_CHECK_LOG(call)                                                                   \
    do {                                                                                        \
        char log[2048];                                                                         \
        size_t sizeof_log = sizeof(log);                                                        \
        OptixResult res_  = (call);                                                             \
        if (res_ != OPTIX_SUCCESS) {                                                            \
            std::ostringstream ss;                                                              \
            ss << "OptiX error: " << #call << " failed: " << optixGetErrorName(res_) << " ("   \
               << __FILE__ << ":" << __LINE__ << ")\nLog:\n" << log;                            \
            throw std::runtime_error(ss.str());                                                 \
        }                                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};

template <typename T>
static CUdeviceptr uploadBuffer(const T* data, size_t count)
{
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * count));
    CUDA_CHECK(cudaMemcpy(ptr, data, sizeof(T) * count, cudaMemcpyHostToDevice));
    return reinterpret_cast<CUdeviceptr>(ptr);
}

static void freeBuffer(CUdeviceptr ptr)
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
}

// Builds an acceleration structure and returns its handle; the backing buffer goes in `output`.
static OptixTraversableHandle buildAccel(OptixDeviceContext context, const OptixBuildInput* inputs,
                                         unsigned int numInputs, CUdeviceptr& output)
{
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, inputs, numInputs, &sizes));

    void* temp = nullptr;
    void* out  = nullptr;
    CUDA_CHECK(cudaMalloc(&temp, sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(&out, sizes.outputSizeInBytes));
    output = reinterpret_cast<CUdeviceptr>(out);

    OptixTraversableHandle handle = 0;
    OPTIX_CHECK(optixAccelBuild(context, nullptr, &accelOptions, inputs, numInputs,
                                reinterpret_cast<CUdeviceptr>(temp), sizes.tempSizeInBytes, output,
                                sizes.outputSizeInBytes, &handle, nullptr, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(temp));
    return handle;
}

static std::string readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open " + path);
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static void contextLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    std::cerr << "[OptiX " << level << "][" << tag << "] " << message << "\n";
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

Camera Renderer::defaultCamera()
{
    const float vfov = 2.0f * std::atan(0.0125f / 0.035f);
    return {{278.0f, 273.0f, -800.0f}, {278.0f, 273.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, vfov * 180.0f / kPi};
}

Renderer::Renderer(unsigned int width, unsigned int height)
{
    // --- Context ---------------------------------------------------------
    CUDA_CHECK(cudaFree(nullptr));  // initialise the CUDA primary context
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions contextOptions = {};
    contextOptions.logCallbackFunction       = &contextLogCallback;
    contextOptions.logCallbackLevel          = 3;  // errors and warnings
    OptixDeviceContext context               = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &contextOptions, &context));

    // --- Scene geometry --------------------------------------------------
    std::vector<float3> vertices;
    std::vector<unsigned int> triangleMaterialIds;
    for (const Quad& q : cornellBoxQuads())
    {
        for (int i : {0, 1, 2, 0, 2, 3})
            vertices.push_back(q.v[i]);
        triangleMaterialIds.push_back(q.material);
        triangleMaterialIds.push_back(q.material);
    }

    std::vector<float3> sphereCenters;
    std::vector<float> sphereRadii;
    std::vector<unsigned int> sphereMaterialIds;
    for (const Sphere& sp : cornellBoxSpheres())
    {
        sphereCenters.push_back(sp.center);
        sphereRadii.push_back(sp.radius);
        sphereMaterialIds.push_back(sp.material);
    }
    const auto materials   = cornellBoxMaterials();
    const PrismSetup prism = cornellBoxPrism(materials[MAT_GLASS]);
    for (const Triangle& t : prism.triangles)
    {
        vertices.insert(vertices.end(), t.v.begin(), t.v.end());
        triangleMaterialIds.push_back(t.material);
    }

    CUdeviceptr dVertices            = uploadBuffer(vertices.data(), vertices.size());
    CUdeviceptr dTriangleMaterialIds = uploadBuffer(triangleMaterialIds.data(), triangleMaterialIds.size());
    CUdeviceptr dSphereCenters       = uploadBuffer(sphereCenters.data(), sphereCenters.size());
    CUdeviceptr dSphereRadii         = uploadBuffer(sphereRadii.data(), sphereRadii.size());
    CUdeviceptr dSphereMaterialIds   = uploadBuffer(sphereMaterialIds.data(), sphereMaterialIds.size());
    CUdeviceptr dMaterials           = uploadBuffer(materials.data(), materials.size());

    // --- Acceleration structures -----------------------------------------
    // A GAS holds one primitive type, so triangles and spheres get one each, joined by an IAS.
    const unsigned int geometryFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    CUdeviceptr dTriangleGas = 0;
    OptixTraversableHandle triangleGas = 0;
    {
        OptixBuildInput input                   = {};
        input.type                              = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = sizeof(float3);
        input.triangleArray.numVertices         = static_cast<unsigned int>(vertices.size());
        input.triangleArray.vertexBuffers       = &dVertices;
        input.triangleArray.flags               = &geometryFlags;
        input.triangleArray.numSbtRecords       = 1;
        triangleGas = buildAccel(context, &input, 1, dTriangleGas);
    }

    CUdeviceptr dSphereGas = 0;
    OptixTraversableHandle sphereGas = 0;
    {
        OptixBuildInput input                 = {};
        input.type                            = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        input.sphereArray.vertexBuffers       = &dSphereCenters;
        input.sphereArray.vertexStrideInBytes = sizeof(float3);
        input.sphereArray.numVertices         = static_cast<unsigned int>(sphereCenters.size());
        input.sphereArray.radiusBuffers       = &dSphereRadii;
        input.sphereArray.radiusStrideInBytes = sizeof(float);
        input.sphereArray.flags               = &geometryFlags;
        input.sphereArray.numSbtRecords       = 1;
        sphereGas = buildAccel(context, &input, 1, dSphereGas);
    }

    CUdeviceptr dIas = 0;
    OptixTraversableHandle iasHandle = 0;
    {
        OptixInstance instances[GEOMETRY_COUNT] = {};
        const OptixTraversableHandle gases[GEOMETRY_COUNT] = {triangleGas, sphereGas};
        for (unsigned int i = 0; i < GEOMETRY_COUNT; ++i)
        {
            const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
            std::memcpy(instances[i].transform, identity, sizeof(identity));
            instances[i].instanceId        = i;
            instances[i].sbtOffset         = i * RAY_TYPE_COUNT;
            instances[i].visibilityMask    = 1;
            instances[i].flags             = OPTIX_INSTANCE_FLAG_NONE;
            instances[i].traversableHandle = gases[i];
        }
        CUdeviceptr dInstances = uploadBuffer(instances, GEOMETRY_COUNT);

        OptixBuildInput input             = {};
        input.type                        = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        input.instanceArray.instances     = dInstances;
        input.instanceArray.numInstances  = GEOMETRY_COUNT;
        iasHandle = buildAccel(context, &input, 1, dIas);
        freeBuffer(dInstances);
    }

    // --- Module ----------------------------------------------------------
    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineOptions      = {};
    pipelineOptions.usesMotionBlur                   = 0;
    pipelineOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues                 = 2;
    pipelineOptions.numAttributeValues               = 2;
    pipelineOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    const std::string ir = readFile(SPECTRAL_DEVICE_CODE_PATH);
    OptixModule module   = nullptr;
    OPTIX_CHECK_LOG(optixModuleCreate(context, &moduleOptions, &pipelineOptions, ir.data(), ir.size(), log,
                                      &sizeof_log, &module));

    // Spheres use OptiX's built-in intersection program.
    OptixModule sphereModule          = nullptr;
    OptixBuiltinISOptions sphereIS    = {};
    sphereIS.builtinISModuleType      = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK(optixBuiltinISModuleGet(context, &moduleOptions, &pipelineOptions, &sphereIS, &sphereModule));

    // --- Program groups --------------------------------------------------
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc descs[6]     = {};

    descs[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descs[0].raygen.module            = module;
    descs[0].raygen.entryFunctionName = "__raygen__pathtrace";

    descs[1].kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descs[1].miss.module            = module;
    descs[1].miss.entryFunctionName = "__miss__radiance";

    descs[2].kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descs[2].miss.module            = module;
    descs[2].miss.entryFunctionName = "__miss__shadow";

    descs[3].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descs[3].hitgroup.moduleCH            = module;
    descs[3].hitgroup.entryFunctionNameCH = "__closesthit__triangle";

    descs[4].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descs[4].hitgroup.moduleCH            = module;
    descs[4].hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    descs[4].hitgroup.moduleIS            = sphereModule;

    descs[5].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descs[5].raygen.module            = module;
    descs[5].raygen.entryFunctionName = "__raygen__lighttrace";

    OptixProgramGroup groups[6] = {};
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, descs, 6, &pgOptions, log, &sizeof_log, groups));
    OptixProgramGroup raygenPG       = groups[0];
    OptixProgramGroup missRadiancePG = groups[1];
    OptixProgramGroup missShadowPG   = groups[2];
    OptixProgramGroup triangleHitPG  = groups[3];
    OptixProgramGroup sphereHitPG    = groups[4];
    OptixProgramGroup lightRaygenPG  = groups[5];

    // --- Pipeline --------------------------------------------------------
    const unsigned int maxTraceDepth = 1;  // all rays are traced from raygen
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth            = maxTraceDepth;

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipelineOptions, &linkOptions, groups, 6, log, &sizeof_log,
                                        &pipeline));

    OptixStackSizes stackSizes = {};
    for (OptixProgramGroup pg : groups)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, pipeline));
    unsigned int dcFromTraversal = 0, dcFromState = 0, continuation = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, maxTraceDepth, 0, 0, &dcFromTraversal, &dcFromState,
                                           &continuation));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, dcFromTraversal, dcFromState, continuation,
                                          2 /* IAS -> GAS */));

    // --- Shader binding table --------------------------------------------
    SbtRecord<EmptyData> raygenRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &raygenRecord));
    SbtRecord<EmptyData> lightRaygenRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(lightRaygenPG, &lightRaygenRecord));

    SbtRecord<EmptyData> missRecords[RAY_TYPE_COUNT] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missRadiancePG, &missRecords[RAY_TYPE_RADIANCE]));
    OPTIX_CHECK(optixSbtRecordPackHeader(missShadowPG, &missRecords[RAY_TYPE_SHADOW]));

    // One record per (geometry, ray type), matching each instance's sbtOffset. Shadow rays disable
    // closest-hit, but still run the sphere intersection program, so they use the same groups.
    SbtRecord<HitGroupData> hitRecords[GEOMETRY_COUNT * RAY_TYPE_COUNT] = {};
    const OptixProgramGroup hitPGs[GEOMETRY_COUNT] = {triangleHitPG, sphereHitPG};
    const HitGroupData hitData[GEOMETRY_COUNT]     = {
        {reinterpret_cast<const float3*>(dVertices), reinterpret_cast<const unsigned int*>(dTriangleMaterialIds)},
        {nullptr, reinterpret_cast<const unsigned int*>(dSphereMaterialIds)},
    };
    for (unsigned int g = 0; g < GEOMETRY_COUNT; ++g)
    {
        for (unsigned int r = 0; r < RAY_TYPE_COUNT; ++r)
        {
            SbtRecord<HitGroupData>& rec = hitRecords[g * RAY_TYPE_COUNT + r];
            OPTIX_CHECK(optixSbtRecordPackHeader(hitPGs[g], &rec));
            rec.data = hitData[g];
        }
    }

    OptixShaderBindingTable sbt       = {};
    sbt.raygenRecord                  = uploadBuffer(&raygenRecord, 1);
    sbt.missRecordBase                = uploadBuffer(missRecords, RAY_TYPE_COUNT);
    sbt.missRecordStrideInBytes       = sizeof(SbtRecord<EmptyData>);
    sbt.missRecordCount               = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase            = uploadBuffer(hitRecords, GEOMETRY_COUNT * RAY_TYPE_COUNT);
    sbt.hitgroupRecordStrideInBytes   = sizeof(SbtRecord<HitGroupData>);
    sbt.hitgroupRecordCount           = GEOMETRY_COUNT * RAY_TYPE_COUNT;


    m_context       = context;
    m_module        = module;
    m_programGroups = std::vector<OptixProgramGroup>(std::begin(groups), std::end(groups));
    m_pipeline      = pipeline;
    m_sbt           = sbt;
    // The light-tracing pass shares everything but the raygen program.
    m_lightSbt              = sbt;
    m_lightSbt.raygenRecord = uploadBuffer(&lightRaygenRecord, 1);
    m_sceneBuffers  = {dVertices,          dTriangleMaterialIds, dSphereCenters,    dSphereRadii,
                       dSphereMaterialIds, dMaterials,           dTriangleGas,      dSphereGas,
                       dIas,               sbt.raygenRecord,     sbt.missRecordBase, sbt.hitgroupRecordBase,
                       m_lightSbt.raygenRecord};

    // --- Launch parameters -----------------------------------------------
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_dParams), sizeof(LaunchParams)));

    m_params.maxDepth  = 12;
    m_params.light     = cornellBoxLight();
    m_params.beam      = prism.beam;
    m_params.handle    = iasHandle;
    m_params.materials = reinterpret_cast<const Material*>(dMaterials);

    // Integrate y-bar over the rendered range with the same fit the device uses (midpoint rule).
    {
        const int steps = 3000;
        const float dl  = (kLambdaMax - kLambdaMin) / steps;
        double sum      = 0.0;
        for (int i = 0; i < steps; ++i)
            sum += cieXYZ(kLambdaMin + (i + 0.5f) * dl).y * dl;
        m_params.cieYIntegral = static_cast<float>(sum);
    }

    m_params.width  = width;
    m_params.height = height;
    allocateFrameBuffers();
    updateCameraBasis();
}

Renderer::~Renderer()
{
    // Best-effort cleanup: destructors must not throw, so errors are ignored.
    cudaStreamSynchronize(m_stream);
    freeFrameBuffers();
    cudaFree(m_dParams);
    cudaStreamDestroy(m_stream);
    for (CUdeviceptr buffer : m_sceneBuffers)
        cudaFree(reinterpret_cast<void*>(buffer));
    optixPipelineDestroy(m_pipeline);
    for (OptixProgramGroup pg : m_programGroups)
        optixProgramGroupDestroy(pg);
    optixModuleDestroy(m_module);  // the built-in sphere module is owned by the context
    optixDeviceContextDestroy(m_context);
}

void Renderer::allocateFrameBuffers()
{
    const size_t numPixels = static_cast<size_t>(m_params.width) * m_params.height;
    CUDA_CHECK(cudaMalloc(&m_params.accum, numPixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&m_params.image, numPixels * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&m_params.lightImage, numPixels * sizeof(float4)));
    resetAccumulation();
}

void Renderer::freeFrameBuffers()
{
    cudaFree(m_params.accum);
    cudaFree(m_params.image);
    cudaFree(m_params.lightImage);
    m_params.accum      = nullptr;
    m_params.image      = nullptr;
    m_params.lightImage = nullptr;
}

void Renderer::resize(unsigned int width, unsigned int height)
{
    if (width == 0 || height == 0 || (width == m_params.width && height == m_params.height))
        return;
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    freeFrameBuffers();
    m_params.width  = width;
    m_params.height = height;
    allocateFrameBuffers();
    updateCameraBasis();  // the aspect ratio changed
}

void Renderer::setCamera(const Camera& camera)
{
    m_camera = camera;
    updateCameraBasis();
}

void Renderer::setMaxDepth(unsigned int maxDepth)
{
    m_params.maxDepth = maxDepth;
    resetAccumulation();
}

void Renderer::updateCameraBasis()
{
    const float aspect = static_cast<float>(m_params.width) / static_cast<float>(m_params.height);
    const float vfov   = m_camera.vfovDegrees * kPi / 180.0f;

    m_params.eye     = m_camera.eye;
    m_params.W       = m_camera.lookAt - m_camera.eye;
    const float vlen = length(m_params.W) * std::tan(0.5f * vfov);
    m_params.U       = normalize(cross(m_params.W, m_camera.up)) * (vlen * aspect);
    m_params.V       = normalize(cross(m_params.U, m_params.W)) * vlen;

    resetAccumulation();  // light-path splats depend on the camera too
}

void Renderer::resetAccumulation()
{
    m_params.accumulatedSamples = 0;
    m_params.lightPathCount     = 0.0f;
    m_clearLightImage           = true;
}

void Renderer::render(unsigned int samples)
{
    const size_t numPixels = static_cast<size_t>(m_params.width) * m_params.height;
    if (m_clearLightImage)
    {
        CUDA_CHECK(cudaMemsetAsync(m_params.lightImage, 0, numPixels * sizeof(float4), m_stream));
        m_clearLightImage = false;
    }
    m_params.samplesPerLaunch = samples;

    // Pageable host-to-device copies return once the source is staged, so m_params can change after,
    // and stream order makes each launch see its own copy.
    auto launch = [&](const OptixShaderBindingTable& sbt) {
        CUDA_CHECK(cudaMemcpyAsync(m_dParams, &m_params, sizeof(LaunchParams), cudaMemcpyHostToDevice, m_stream));
        OPTIX_CHECK(optixLaunch(m_pipeline, m_stream, reinterpret_cast<CUdeviceptr>(m_dParams),
                                sizeof(LaunchParams), &sbt, m_params.width, m_params.height, 1));
    };

    // Light paths first, so the camera pass composites this frame's splats: one light path per camera path.
    if (m_params.beam.enabled)
    {
        launch(m_lightSbt);
        m_params.lightPathCount += static_cast<float>(numPixels) * samples;
    }
    launch(m_sbt);

    m_params.accumulatedSamples += samples;
    m_params.frameIndex++;
}

void Renderer::downloadImage(std::vector<uchar4>& pixels)
{
    pixels.resize(static_cast<size_t>(m_params.width) * m_params.height);
    CUDA_CHECK(cudaMemcpyAsync(pixels.data(), m_params.image, pixels.size() * sizeof(uchar4),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

// Types shared between the host and the OptiX device programs.
#pragma once

#include <optix.h>
#include <cuda_runtime.h>

#include "spectrum.h"

enum MaterialType : unsigned int
{
    MATERIAL_DIFFUSE = 0,  // Lambertian, uses `reflectance`
    MATERIAL_DIELECTRIC,   // smooth glass, uses the Sellmeier coefficients
};

struct Material
{
    MaterialType type;
    Spectrum reflectance;
    Spectrum emission;  // spectral radiance; all zero for non-emitters

    // Sellmeier equation, n^2 = 1 + sum(B_i * l^2 / (l^2 - C_i)) with l in micrometres.
    // All C_i = 0 gives a constant, non-dispersive index.
    float sellmeierB[3];
    float sellmeierC[3];
};

HD float refractiveIndex(const Material& m, float lambdaNm)
{
    const float l2 = (lambdaNm * 1e-3f) * (lambdaNm * 1e-3f);
    float n2       = 1.0f;
    for (int i = 0; i < 3; ++i)
        n2 += m.sellmeierB[i] * l2 / (l2 - m.sellmeierC[i]);
    return sqrtf(n2);
}

HD bool isDispersive(const Material& m)
{
    return m.sellmeierC[0] != 0.0f || m.sellmeierC[1] != 0.0f || m.sellmeierC[2] != 0.0f;
}

// Per-geometry data stored in each hit group SBT record.
struct HitGroupData
{
    const float3* vertices;           // triangles only: 3 per triangle, non-indexed
    const unsigned int* materialIds;  // 1 per primitive
};

// A parallelogram area light: points are corner + u*v1 + v*v2 for u, v in [0, 1].
struct ParallelogramLight
{
    float3 corner;
    float3 v1;
    float3 v2;
    float3 normal;             // emitting side
    unsigned int materialId;   // supplies the emission spectrum
};

// A perfectly collimated beam leaving a rectangular aperture, like sunlight through a slit.
// Its emission spectrum is the irradiance on a plane perpendicular to the beam. A camera path can
// never hit a beam, so it is rendered only by the light-tracing pass.
struct BeamLight
{
    float3 center;     // centre of the aperture
    float3 direction;  // unit
    float3 halfWidth;  // aperture half-extents, perpendicular to `direction`
    float3 halfHeight;
    unsigned int materialId;  // supplies the emission (irradiance) spectrum
    int enabled;
};

struct LaunchParams
{
    float4* accum;       // running average of CIE XYZ from camera paths
    float4* lightImage;  // sum of CIE XYZ splatted by light paths; divided by lightPathCount
    float lightPathCount;  // light paths traced so far, including the current launch
    uchar4* image;  // tonemapped sRGB output
    unsigned int width;
    unsigned int height;
    unsigned int frameIndex;          // increments every launch; seeds the random numbers
    unsigned int accumulatedSamples;  // samples per pixel already in `accum`; 0 restarts it
    unsigned int samplesPerLaunch;
    unsigned int maxDepth;
    float cieYIntegral;  // integral of y-bar over [kLambdaMin, kLambdaMax], normalises XYZ

    // Pinhole camera: ray direction = x*U + y*V + W for x, y in [-1, 1].
    float3 eye;
    float3 U;
    float3 V;
    float3 W;

    ParallelogramLight light;
    BeamLight beam;

    OptixTraversableHandle handle;  // instance AS over the triangle and sphere GASes
    const Material* materials;
};

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW   = 1,
    RAY_TYPE_COUNT
};

// Hit group SBT layout: RAY_TYPE_COUNT records per geometry kind, in this order.
enum GeometryKind
{
    GEOMETRY_TRIANGLES = 0,
    GEOMETRY_SPHERES   = 1,
    GEOMETRY_COUNT
};

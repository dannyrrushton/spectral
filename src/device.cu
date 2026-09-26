// OptiX device programs: an iterative spectral path tracer for diffuse and glass surfaces, with
// next-event estimation on diffuse surfaces.
#include <optix.h>

#include "shared.h"
#include "vec_math.h"

extern "C" __constant__ LaunchParams params;

constexpr float kRayEpsilon = 0.01f;  // scene units are millimetres

// ---------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------

// TEA hash, used to decorrelate per-pixel seeds.
template <unsigned int N>
static __forceinline__ __device__ unsigned int tea(unsigned int v0, unsigned int v1)
{
    unsigned int s0 = 0;
    for (unsigned int n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

// Uniform float in [0, 1).
static __forceinline__ __device__ float rnd(unsigned int& state)
{
    state = 1664525u * state + 1013904223u;
    return static_cast<float>(state & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

// Stratified wavelengths for one path: a random offset plus evenly spaced rotations of it,
// as in hero wavelength sampling. Each wavelength is uniformly distributed over the range.
static __forceinline__ __device__ SampledSpectrum sampleWavelengths(float u)
{
    const float range = kLambdaMax - kLambdaMin;
    SampledSpectrum lambdas;
    for (int i = 0; i < kWavelengthsPerPath; ++i)
    {
        const float v = u + static_cast<float>(i) / kWavelengthsPerPath;
        lambdas.v[i]  = kLambdaMin + range * (v - floorf(v));
    }
    return lambdas;
}

// Cosine-weighted direction around n; pdf = cos(theta) / pi.
static __forceinline__ __device__ float3 sampleCosineHemisphere(float3 n, float u1, float u2)
{
    const float r   = sqrtf(u1);
    const float phi = 2.0f * kPi * u2;
    const float x   = r * cosf(phi);
    const float y   = r * sinf(phi);
    const float z   = sqrtf(fmaxf(0.0f, 1.0f - u1));

    // Orthonormal basis (Duff et al. 2017).
    const float sign = copysignf(1.0f, n.z);
    const float a    = -1.0f / (sign + n.z);
    const float b    = n.x * n.y * a;
    const float3 t   = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    const float3 bt  = make_float3(b, sign + n.y * n.y * a, -n.y);
    return x * t + y * bt + z * n;
}

// Fresnel reflectance for unpolarised light at a smooth dielectric boundary.
// eta = n_incident / n_transmitted; cosT must come from a non-TIR refraction.
static __forceinline__ __device__ float fresnelDielectric(float cosI, float cosT, float eta)
{
    const float rs = (eta * cosI - cosT) / (eta * cosI + cosT);
    const float rp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rs * rs + rp * rp);
}

// Scatters `direction` off a smooth dielectric with refractive index n. N faces the incoming ray.
// Reflection vs refraction is chosen with probability equal to the Fresnel term, which cancels it
// from the path weight; total internal reflection always reflects. (The 1/eta^2 radiance scaling at
// each interface cancels on entering and leaving a closed object.)
static __forceinline__ __device__ float3 sampleDielectric(float3 direction, float3 N, bool frontFace, float n,
                                                          float u)
{
    const float eta   = frontFace ? 1.0f / n : n;
    const float cosI  = -dot(direction, N);
    const float sin2T = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    const float cosT  = sqrtf(fmaxf(0.0f, 1.0f - sin2T));
    const float F     = sin2T >= 1.0f ? 1.0f : fresnelDielectric(cosI, cosT, eta);
    if (u < F)
        return direction + 2.0f * cosI * N;
    return normalize(eta * direction + (eta * cosI - cosT) * N);
}

// ---------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------

// Closest-hit only reports geometry; spectra are evaluated in raygen, where the wavelengths live.
struct RadiancePRD
{
    float3 position;
    float3 normal;           // faces the incoming ray
    float3 geometricNormal;  // as wound (triangles) or outward (spheres); used for camera culling
    unsigned int materialId;
    bool frontFace;  // ray arrived from outside the surface (meaningful for closed shapes)
    bool missed;
};

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long p = reinterpret_cast<unsigned long long>(ptr);
    i0 = static_cast<unsigned int>(p >> 32);
    i1 = static_cast<unsigned int>(p & 0xFFFFFFFFull);
}

static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned long long p =
        (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>(p);
}

static __forceinline__ __device__ void traceRadiance(float3 origin, float3 direction, unsigned int rayFlags,
                                                     RadiancePRD& prd)
{
    unsigned int p0, p1;
    packPointer(&prd, p0, p1);
    optixTrace(params.handle, origin, direction, kRayEpsilon, 1e16f, 0.0f, OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | rayFlags, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0,
               p1);
}

static __forceinline__ __device__ bool traceOcclusion(float3 origin, float3 direction, float tmax,
                                                      unsigned int rayFlags = 0)
{
    // Only the shadow miss program runs; it clears the flag when nothing is in the way.
    unsigned int occluded = 1;
    optixTrace(params.handle, origin, direction, kRayEpsilon, tmax, 0.0f, OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | rayFlags,
               RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW, occluded);
    return occluded != 0;
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

static __forceinline__ __device__ float linearToSrgb(float c)
{
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

extern "C" __global__ void __raygen__pathtrace()
{
    const uint3 idx         = optixGetLaunchIndex();
    const unsigned int w    = params.width;
    const unsigned int h    = params.height;
    const unsigned int pix  = idx.y * w + idx.x;
    const ParallelogramLight& light = params.light;
    const float lightArea   = length(cross(light.v1, light.v2));
    const Spectrum& lightLe = params.materials[light.materialId].emission;

    unsigned int seed = tea<4>(pix, params.frameIndex);
    float3 xyz        = make_float3(0.0f, 0.0f, 0.0f);

    for (unsigned int s = 0; s < params.samplesPerLaunch; ++s)
    {
        // Jittered pixel position in [-1, 1]^2, with +y at the top of the image.
        const float px = 2.0f * (idx.x + rnd(seed)) / w - 1.0f;
        const float py = 1.0f - 2.0f * (idx.y + rnd(seed)) / h;

        const SampledSpectrum lambdas = sampleWavelengths(rnd(seed));

        float3 origin     = params.eye;
        float3 direction  = normalize(px * params.U + py * params.V + params.W);
        SampledSpectrum throughput = SampledSpectrum::constant(1.0f);
        SampledSpectrum radiance   = SampledSpectrum::constant(0.0f);
        bool specularBounce = false;  // last bounce was off glass, so NEE could not see the light
        bool heroOnly       = false;  // secondary wavelengths dropped after dispersion

        for (unsigned int depth = 0; depth < params.maxDepth; ++depth)
        {
            // Camera rays ignore triangles seen from behind: every Cornell quad faces into the room
            // (or out of a block), so an orbiting camera sees through the walls into the box, like a
            // dollhouse cutaway. Light transport still sees both sides.
            RadiancePRD prd;
            traceRadiance(origin, direction, depth == 0 ? OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES : 0u, prd);
            if (prd.missed)
                break;

            const Material& mat = params.materials[prd.materialId];

            // Light reached by a diffuse bounce was already counted by next-event estimation, so
            // only count emission seen directly or through glass (this is what makes caustics).
            // The light only emits from its underside.
            if ((depth == 0 || specularBounce) && dot(direction, light.normal) < 0.0f)
                radiance += throughput * sampleSpectrum(mat.emission, lambdas);

            const float3 P = prd.position;
            const float3 N = prd.normal;

            if (mat.type == MATERIAL_DIELECTRIC)
            {
                // With dispersion each wavelength refracts differently, so the path can only follow
                // the hero wavelength. Scaling by the wavelength count keeps the estimate unbiased.
                if (isDispersive(mat) && !heroOnly)
                {
                    const float hero = throughput.v[0] * kWavelengthsPerPath;
                    throughput       = SampledSpectrum::constant(0.0f);
                    throughput.v[0]  = hero;
                    heroOnly   = true;
                }

                direction =
                    sampleDielectric(direction, N, prd.frontFace, refractiveIndex(mat, lambdas.v[0]), rnd(seed));
                origin         = P;
                specularBounce = true;
            }
            else
            {
                const SampledSpectrum albedo = sampleSpectrum(mat.reflectance, lambdas);

                // Next-event estimation: sample one point on the area light.
                const float3 lightPos = light.corner + rnd(seed) * light.v1 + rnd(seed) * light.v2;
                float3 L              = lightPos - P;
                const float dist      = length(L);
                L                     = L / dist;
                const float cosSurf   = dot(N, L);
                const float cosLight  = -dot(light.normal, L);
                if (cosSurf > 0.0f && cosLight > 0.0f && !traceOcclusion(P, L, dist - kRayEpsilon))
                {
                    // (albedo / pi) * Le * cosSurf * cosLight / dist^2 / pdf, with pdf = 1 / area.
                    const float g = cosSurf * cosLight * lightArea / (kPi * dist * dist);
                    radiance += throughput * albedo * sampleSpectrum(lightLe, lambdas) * g;
                }

                // Continue the path; the cosine pdf cancels the Lambertian BRDF's cos / pi.
                throughput *= albedo;
                origin         = P;
                direction      = sampleCosineHemisphere(N, rnd(seed), rnd(seed));
                specularBounce = false;
            }

            // Russian roulette after a few bounces.
            if (depth >= 3)
            {
                const float q = fminf(maxComponent(throughput), 0.95f);
                if (rnd(seed) >= q)
                    break;
                throughput = throughput / q;
            }
        }

        // Project onto the CIE observer. Each wavelength has pdf 1 / (lambda range), and XYZ is
        // normalised so a constant spectrum of 1 has Y = 1.
        const float weight =
            (kLambdaMax - kLambdaMin) / (kWavelengthsPerPath * params.cieYIntegral);
        for (int i = 0; i < kWavelengthsPerPath; ++i)
            xyz += cieXYZ(lambdas.v[i]) * (radiance.v[i] * weight);
    }
    xyz = xyz / static_cast<float>(params.samplesPerLaunch);

    // Progressive running average across launches, weighted by sample count.
    float3 accum = xyz;
    if (params.accumulatedSamples > 0)
    {
        const float4 prev = params.accum[pix];
        const float a     = static_cast<float>(params.samplesPerLaunch) /
                        static_cast<float>(params.accumulatedSamples + params.samplesPerLaunch);
        accum = make_float3(prev.x, prev.y, prev.z) * (1.0f - a) + xyz * a;
    }
    params.accum[pix] = make_float4(accum.x, accum.y, accum.z, 1.0f);

    // Add the light-traced beam contribution, averaged over all light paths so far.
    float3 total = accum;
    if (params.lightPathCount > 0.0f)
    {
        const float4 splat = params.lightImage[pix];
        total += make_float3(splat.x, splat.y, splat.z) / params.lightPathCount;
    }

    const float3 rgb  = xyzToLinearSrgb(total);
    params.image[pix] = make_uchar4(static_cast<unsigned char>(linearToSrgb(rgb.x) * 255.0f + 0.5f),
                                    static_cast<unsigned char>(linearToSrgb(rgb.y) * 255.0f + 0.5f),
                                    static_cast<unsigned char>(linearToSrgb(rgb.z) * 255.0f + 0.5f), 255);
}

// ---------------------------------------------------------------------------
// Light tracing (for the beam)
// ---------------------------------------------------------------------------

// Splats the light a diffuse vertex reflects towards the pinhole camera, if the camera sees it.
// `weight` is the light path's flux weight at P for wavelength lambda.
static __forceinline__ __device__ void splatToCamera(const RadiancePRD& hit, float albedo, float weight, float lambda)
{
    float3 toEye     = params.eye - hit.position;
    const float dist = length(toEye);
    toEye            = toEye / dist;

    // Reflect only on the side the light arrived from, and only where camera rays would not cull
    // the surface for the cutaway view.
    const float cosSurface = dot(hit.normal, toEye);
    if (cosSurface <= 0.0f || dot(hit.geometricNormal, toEye) <= 0.0f)
        return;

    // Project onto the image plane: eye->P is parallel to px*U + py*V + W.
    const float3 d = hit.position - params.eye;
    const float t  = dot(d, params.W) / dot(params.W, params.W);
    if (t <= 0.0f)
        return;
    const float px = dot(d, params.U) / (dot(params.U, params.U) * t);
    const float py = dot(d, params.V) / (dot(params.V, params.V) * t);
    if (fabsf(px) >= 1.0f || fabsf(py) >= 1.0f)
        return;

    // Camera rays skip back-facing triangles, so seen from P those are the front-facing ones.
    if (traceOcclusion(hit.position, toEye, dist - kRayEpsilon, OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES))
        return;

    // A pixel stores the average radiance over its area on the image plane. With that plane at unit
    // distance, a flux weight at P maps to (albedo/pi) * weight * cos(surface) / (dist^2 cos^3(camera) A_pixel).
    const float lenW      = length(params.W);
    const float cosCamera = t * lenW / dist;
    const float pixelArea = 4.0f * length(params.U) * length(params.V) /
                            (static_cast<float>(params.width) * params.height * lenW * lenW);
    const float value     = albedo / kPi * weight * cosSurface /
                        (dist * dist * cosCamera * cosCamera * cosCamera * pixelArea);

    const float3 xyz = cieXYZ(lambda) * (value / params.cieYIntegral);
    const unsigned int ix = min(static_cast<unsigned int>((px + 1.0f) * 0.5f * params.width), params.width - 1);
    const unsigned int iy = min(static_cast<unsigned int>((1.0f - py) * 0.5f * params.height), params.height - 1);
    float4* dst = &params.lightImage[iy * params.width + ix];
    atomicAdd(&dst->x, xyz.x);
    atomicAdd(&dst->y, xyz.y);
    atomicAdd(&dst->z, xyz.z);
}

// Traces particles from the beam through the scene, connecting every diffuse vertex to the camera.
// Glass is sampled exactly as in the camera pass, so dispersion splits the beam into a spectrum.
extern "C" __global__ void __raygen__lighttrace()
{
    const uint3 idx        = optixGetLaunchIndex();
    const unsigned int pix = idx.y * params.width + idx.x;
    const BeamLight& beam  = params.beam;
    const Spectrum& beamE  = params.materials[beam.materialId].emission;
    const float range      = kLambdaMax - kLambdaMin;
    const float apertureArea = 4.0f * length(beam.halfWidth) * length(beam.halfHeight);

    // Offset the seed so light paths are independent of the camera paths for the same pixel.
    unsigned int seed = tea<4>(pix, params.frameIndex ^ 0x80000000u);

    for (unsigned int s = 0; s < params.samplesPerLaunch; ++s)
    {
        // One wavelength per light path: it will almost always meet dispersive glass.
        const float lambda = kLambdaMin + range * rnd(seed);
        float3 origin = beam.center + (2.0f * rnd(seed) - 1.0f) * beam.halfWidth +
                        (2.0f * rnd(seed) - 1.0f) * beam.halfHeight;
        float3 direction = beam.direction;

        // Flux weight: irradiance * aperture area, over the wavelength pdf 1 / range.
        const float emitted = sampleSpectrum(beamE, lambda) * apertureArea * range;
        float throughput    = 1.0f;

        for (unsigned int depth = 0; depth < params.maxDepth; ++depth)
        {
            RadiancePRD prd;
            traceRadiance(origin, direction, 0u, prd);
            if (prd.missed)
                break;

            const Material& mat = params.materials[prd.materialId];
            if (mat.type == MATERIAL_DIELECTRIC)
            {
                direction = sampleDielectric(direction, prd.normal, prd.frontFace, refractiveIndex(mat, lambda),
                                             rnd(seed));
            }
            else
            {
                const float albedo = sampleSpectrum(mat.reflectance, lambda);
                splatToCamera(prd, albedo, emitted * throughput, lambda);

                throughput *= albedo;
                direction = sampleCosineHemisphere(prd.normal, rnd(seed), rnd(seed));
            }
            origin = prd.position;

            // Russian roulette after a few bounces.
            if (depth >= 3)
            {
                const float q = fminf(throughput, 0.95f);
                if (rnd(seed) >= q)
                    break;
                throughput /= q;
            }
        }
    }
}

extern "C" __global__ void __miss__radiance()
{
    getPRD()->missed = true;
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__triangle()
{
    const HitGroupData& data = *reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const unsigned int prim  = optixGetPrimitiveIndex();
    const float3 v0          = data.vertices[3 * prim + 0];
    const float3 v1          = data.vertices[3 * prim + 1];
    const float3 v2          = data.vertices[3 * prim + 2];

    const float3 rayDir = optixGetWorldRayDirection();
    const float3 Ng     = normalize(cross(v1 - v0, v2 - v0));
    const bool front    = dot(Ng, rayDir) < 0.0f;  // closed glass meshes are wound outward

    RadiancePRD* prd     = getPRD();
    prd->position        = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
    prd->normal          = front ? Ng : -Ng;  // treat every surface as two-sided
    prd->geometricNormal = Ng;
    prd->materialId      = data.materialIds[prim];
    prd->frontFace       = front;
    prd->missed          = false;
}

extern "C" __global__ void __closesthit__sphere()
{
    const HitGroupData& data = *reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    float4 sphere[1];
    optixGetSphereData(sphere);  // xyz = centre, w = radius; instances use identity transforms
    const float3 center = make_float3(sphere[0].x, sphere[0].y, sphere[0].z);

    const float3 rayDir = optixGetWorldRayDirection();
    const float3 hit    = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
    const float3 Nout   = normalize(hit - center);
    const bool front    = dot(rayDir, Nout) < 0.0f;

    RadiancePRD* prd = getPRD();
    prd->position    = center + Nout * sphere[0].w;  // project onto the surface to reduce error
    prd->normal      = front ? Nout : -Nout;
    prd->geometricNormal = Nout;
    prd->materialId  = data.materialIds[optixGetPrimitiveIndex()];
    prd->frontFace   = front;
    prd->missed      = false;
}

// The Cornell Box, using the measured data from
// https://www.graphics.cornell.edu/online/box/data.html (units are millimetres).
// The tall block is left out, to keep the back wall clear for the prism's spectrum.
#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <cmath>
#include <utility>
#include <vector>

#include "cornell_spectra.h"
#include "shared.h"
#include "vec_math.h"

enum CornellMaterial : unsigned int
{
    MAT_WHITE = 0,
    MAT_RED,
    MAT_GREEN,
    MAT_LIGHT,
    MAT_GLASS,
    MAT_BEAM,  // no geometry: holds the beam light's spectrum
    MAT_COUNT
};

struct Quad
{
    std::array<float3, 4> v;
    CornellMaterial material;
};

struct Sphere
{
    float3 center;
    float radius;
    CornellMaterial material;
};

inline std::vector<Quad> cornellBoxQuads()
{
    return {
        // Floor
        {{{{552.8f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 559.2f}, {549.6f, 0.0f, 559.2f}}}, MAT_WHITE},
        // Ceiling
        {{{{556.0f, 548.8f, 0.0f}, {556.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 0.0f}}}, MAT_WHITE},
        // Back wall
        {{{{549.6f, 0.0f, 559.2f}, {0.0f, 0.0f, 559.2f}, {0.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 559.2f}}}, MAT_WHITE},
        // Right wall (green)
        {{{{0.0f, 0.0f, 559.2f}, {0.0f, 0.0f, 0.0f}, {0.0f, 548.8f, 0.0f}, {0.0f, 548.8f, 559.2f}}}, MAT_GREEN},
        // Left wall (red)
        {{{{552.8f, 0.0f, 0.0f}, {549.6f, 0.0f, 559.2f}, {556.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 0.0f}}}, MAT_RED},

        // Short block
        {{{{130.0f, 165.0f, 65.0f}, {82.0f, 165.0f, 225.0f}, {240.0f, 165.0f, 272.0f}, {290.0f, 165.0f, 114.0f}}}, MAT_WHITE},
        {{{{290.0f, 0.0f, 114.0f}, {290.0f, 165.0f, 114.0f}, {240.0f, 165.0f, 272.0f}, {240.0f, 0.0f, 272.0f}}}, MAT_WHITE},
        {{{{130.0f, 0.0f, 65.0f}, {130.0f, 165.0f, 65.0f}, {290.0f, 165.0f, 114.0f}, {290.0f, 0.0f, 114.0f}}}, MAT_WHITE},
        {{{{82.0f, 0.0f, 225.0f}, {82.0f, 165.0f, 225.0f}, {130.0f, 165.0f, 65.0f}, {130.0f, 0.0f, 65.0f}}}, MAT_WHITE},
        {{{{240.0f, 0.0f, 272.0f}, {240.0f, 165.0f, 272.0f}, {82.0f, 165.0f, 225.0f}, {82.0f, 0.0f, 225.0f}}}, MAT_WHITE},

        // Light, nudged just below the ceiling so the two don't z-fight.
        {{{{343.0f, 548.7f, 227.0f}, {343.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 227.0f}}}, MAT_LIGHT},
    };
}

// Not part of the original box: a glass ball resting on the floor at the front left, placed so its
// caustic lands where the camera can see it.
inline std::vector<Sphere> cornellBoxSpheres()
{
    return {
        {{410.0f, 80.0f, 150.0f}, 80.0f, MAT_GLASS},
    };
}

constexpr float kBeamIrradiance = 40.0f;

inline std::array<Material, MAT_COUNT> cornellBoxMaterials()
{
    auto fromTable = [](const float (&table)[kSpectrumSamples]) {
        Spectrum s;
        std::copy(std::begin(table), std::end(table), s.values);
        return s;
    };
    auto constant = [](float v) {
        Spectrum s;
        std::fill(std::begin(s.values), std::end(s.values), v);
        return s;
    };

    auto diffuse = [](Spectrum reflectance, Spectrum emission) {
        Material m    = {};
        m.type        = MATERIAL_DIFFUSE;
        m.reflectance = reflectance;
        m.emission    = emission;
        return m;
    };

    std::array<Material, MAT_COUNT> m;
    m[MAT_WHITE] = diffuse(fromTable(kCornellWhiteReflectance), constant(0.0f));
    m[MAT_RED]   = diffuse(fromTable(kCornellRedReflectance), constant(0.0f));
    m[MAT_GREEN] = diffuse(fromTable(kCornellGreenReflectance), constant(0.0f));
    m[MAT_LIGHT] = diffuse(constant(kCornellLightReflectance), fromTable(kCornellLightEmission));

    // Schott N-SF11, a dense flint glass (n_d = 1.785, Abbe number 25.7), chosen for strong
    // dispersion. For a milder crown glass use N-BK7: B = {1.03961212, 0.231792344, 1.01046945},
    // C = {0.00600069867, 0.0200179144, 103.560653}.
    Material glass = {};
    glass.type     = MATERIAL_DIELECTRIC;
    glass.reflectance = constant(0.0f);
    glass.emission    = constant(0.0f);
    const float glassB[3] = {1.73759695f, 0.313747346f, 1.89878101f};
    const float glassC[3] = {0.013188707f, 0.0623068142f, 155.23629f};
    std::copy(std::begin(glassB), std::end(glassB), glass.sellmeierB);
    std::copy(std::begin(glassC), std::end(glassC), glass.sellmeierC);
    m[MAT_GLASS] = glass;

    // Equal-energy white; irradiance across the beam, tuned so the rainbow is bright but not clipped.
    m[MAT_BEAM] = diffuse(constant(0.0f), constant(kBeamIrradiance));
    return m;
}

inline ParallelogramLight cornellBoxLight()
{
    ParallelogramLight light;
    light.corner     = {343.0f, 548.6f, 227.0f};
    light.v1         = {0.0f, 0.0f, 105.0f};
    light.v2         = {-130.0f, 0.0f, 0.0f};
    light.normal     = {0.0f, -1.0f, 0.0f};
    light.materialId = MAT_LIGHT;
    return light;
}

// ---------------------------------------------------------------------------
// Prism and beam (not part of the original box)
// ---------------------------------------------------------------------------

struct Triangle
{
    std::array<float3, 3> v;
    CornellMaterial material;
};

struct PrismSetup
{
    std::vector<Triangle> triangles;  // wound outward, as glass requires
    BeamLight beam;
};

// A standing 60-degree glass prism on the short block, lit by a slit-shaped white beam that enters
// through the open front of the box. Its spectrum fans out horizontally across the back wall.
//
// The prism is built around its centroid, starting from minimum deviation at 550 nm (the most
// symmetric but least dispersive orientation) with the 550 nm ray leaving along +z. It is then turned
// against the beam so light leaves the exit face more steeply, which widens the fan from 10.6 to
// 15.4 degrees; past a 6.1 degree turn violet is lost to total internal reflection. Finally the
// whole setup is mirrored, rotated and placed in the box.
//
// The placement came from a 2D search over position (floor or short-block top), orientation and
// handedness that maximised the spectrum's width on the back wall (153 mm) subject to: 400-700 nm
// passing cleanly through both beam edges, the spectrum landing 30 mm clear of the wall corners at
// no more than 60 degrees from the wall normal, the beam entering through the open front, the beam
// and fan clearing the ball and block, the prism standing on the block with 8 mm to spare, and the
// camera seeing the whole spectrum. The fan's angle is fixed by the glass, so the box's size is what
// limits the width; the larger prism mostly buys a taller spectrum.
inline PrismSetup cornellBoxPrism(const Material& glass)
{
    const float3 up        = {0.0f, 1.0f, 0.0f};
    const float apex       = 60.0f * kPi / 180.0f;
    const float side       = 120.0f;  // triangle edge, mm
    const float height     = 160.0f;  // extrusion along y
    const float baseY      = 165.5f;  // just above the short block, avoiding coplanar faces
    const float3 centroid  = {182.0f, baseY, 158.0f};
    const float orientation = 30.5f * kPi / 180.0f;
    const bool mirrored    = true;    // fan bends towards +x, across the open back wall
    const float prismTurn  = 5.0f * kPi / 180.0f;  // prism relative to the beam, towards grazing exit
    const float beamHalfHeight = 45.0f;

    // Rotation about the vertical axis through `pivot`; positive turns +z towards +x.
    auto rotateY = [](float3 v, float angle, float3 pivot) {
        const float c = std::cos(angle), s = std::sin(angle);
        const float x = v.x - pivot.x, z = v.z - pivot.z;
        return make_float3(pivot.x + x * c + z * s, v.y, pivot.z - x * s + z * c);
    };
    const float3 zero = {0.0f, 0.0f, 0.0f};

    // Minimum deviation: the ray crosses the prism parallel to its base, bent by `deviation` overall.
    const float n         = refractiveIndex(glass, 550.0f);
    const float deviation = 2.0f * std::asin(n * std::sin(0.5f * apex)) - apex;
    const float3 dOut     = {0.0f, 0.0f, 1.0f};
    const float3 dIn      = rotateY(dOut, deviation, zero);

    // Prism frame, centroid at the origin: ex runs along the base, ey points to the apex.
    const float3 ex     = normalize(dIn + dOut);
    const float3 ey     = normalize(dIn - dOut);
    const float h       = side * std::sqrt(3.0f) / 2.0f;
    const float3 origin = ey * (-h / 3.0f);

    // Inside the prism the ray runs parallel to the base at the centroid's height, so outside it must
    // be aimed at the point where that line meets the entry face, not at the centroid.
    const float beamY  = h / 3.0f;
    float3 entry       = origin + ex * (-0.5f * side + beamY / std::tan(apex)) + ey * beamY;
    float3 beamDir     = dIn;
    float3 flat[3]     = {origin - ex * (0.5f * side), origin + ex * (0.5f * side), origin + ey * h};
    for (float3& v : flat)
        v = rotateY(v, prismTurn, zero);

    // Mirror (x -> -x), then orient and place everything in the box.
    auto place = [&](float3 v, bool isPoint) {
        if (mirrored)
            v.x = -v.x;
        v = rotateY(v, orientation, zero);
        return isPoint ? make_float3(v.x + centroid.x, v.y, v.z + centroid.z) : v;
    };
    for (float3& v : flat)
        v = place(v, true);
    entry   = place(entry, true);
    beamDir = place(beamDir, false);

    auto at = [&](int i, float y) { return make_float3(flat[i].x, y, flat[i].z); };
    const float y0 = baseY, y1 = baseY + height;
    const float beamCenterY = baseY + 0.5f * height;
    std::vector<std::array<float3, 3>> tris = {
        {at(0, y0), at(1, y0), at(2, y0)},  // bottom cap
        {at(0, y1), at(1, y1), at(2, y1)},  // top cap
    };
    for (int i = 0; i < 3; ++i)
    {
        const int j = (i + 1) % 3;
        tris.push_back({at(i, y0), at(j, y0), at(j, y1)});
        tris.push_back({at(i, y0), at(j, y1), at(i, y1)});
    }

    PrismSetup setup;
    const float3 middle = make_float3(centroid.x, 0.5f * (y0 + y1), centroid.z);
    for (auto& t : tris)
    {
        const float3 faceCenter = (t[0] + t[1] + t[2]) / 3.0f;
        if (dot(cross(t[1] - t[0], t[2] - t[0]), faceCenter - middle) < 0.0f)
            std::swap(t[1], t[2]);  // wind outward
        setup.triangles.push_back({t, MAT_GLASS});
    }

    // A 4 mm slit, centred on the prism's height, placed just outside the open front of the box.
    entry.y = beamCenterY;
    BeamLight& beam = setup.beam;
    beam.direction  = beamDir;
    beam.center     = entry - beamDir * (entry.z / beamDir.z + 20.0f);
    beam.halfWidth  = normalize(cross(beamDir, up)) * 2.0f;
    beam.halfHeight = up * beamHalfHeight;
    beam.materialId = MAT_BEAM;
    beam.enabled    = 1;
    return setup;
}

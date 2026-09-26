// Spectral quantities: sampled spectra, the CIE 1931 observer, and conversion to sRGB.
#pragma once

#include "vec_math.h"

// All spectra cover the range of the Cornell measurements, sampled every 4 nm.
constexpr float kLambdaMin       = 400.0f;
constexpr float kLambdaMax       = 700.0f;
constexpr int kSpectrumSamples   = 76;
constexpr float kSpectrumSpacing = (kLambdaMax - kLambdaMin) / (kSpectrumSamples - 1);

// Wavelengths traced together along one path (see sampleWavelengths in device.cu).
constexpr int kWavelengthsPerPath = 8;

struct Spectrum
{
    float values[kSpectrumSamples];
};

// Piecewise-linear lookup; clamps outside [kLambdaMin, kLambdaMax].
HD float sampleSpectrum(const Spectrum& s, float lambda)
{
    const float x = fminf(fmaxf((lambda - kLambdaMin) / kSpectrumSpacing, 0.0f),
                          static_cast<float>(kSpectrumSamples - 1));
    const int i   = x < kSpectrumSamples - 2 ? static_cast<int>(x) : kSpectrumSamples - 2;
    const float t = x - static_cast<float>(i);
    return s.values[i] * (1.0f - t) + s.values[i + 1] * t;
}

// A quantity carried along a path at each of its kWavelengthsPerPath wavelengths.
struct SampledSpectrum
{
    float v[kWavelengthsPerPath];

    static HD SampledSpectrum constant(float c)
    {
        SampledSpectrum r;
        for (int i = 0; i < kWavelengthsPerPath; ++i)
            r.v[i] = c;
        return r;
    }
};

HD SampledSpectrum operator+(const SampledSpectrum& a, const SampledSpectrum& b)
{
    SampledSpectrum r;
    for (int i = 0; i < kWavelengthsPerPath; ++i)
        r.v[i] = a.v[i] + b.v[i];
    return r;
}
HD SampledSpectrum operator*(const SampledSpectrum& a, const SampledSpectrum& b)
{
    SampledSpectrum r;
    for (int i = 0; i < kWavelengthsPerPath; ++i)
        r.v[i] = a.v[i] * b.v[i];
    return r;
}
HD SampledSpectrum operator*(const SampledSpectrum& a, float s)
{
    SampledSpectrum r;
    for (int i = 0; i < kWavelengthsPerPath; ++i)
        r.v[i] = a.v[i] * s;
    return r;
}
HD SampledSpectrum operator/(const SampledSpectrum& a, float s) { return a * (1.0f / s); }
HD SampledSpectrum& operator+=(SampledSpectrum& a, const SampledSpectrum& b) { a = a + b; return a; }
HD SampledSpectrum& operator*=(SampledSpectrum& a, const SampledSpectrum& b) { a = a * b; return a; }
HD float maxComponent(const SampledSpectrum& a)
{
    float m = a.v[0];
    for (int i = 1; i < kWavelengthsPerPath; ++i)
        m = fmaxf(m, a.v[i]);
    return m;
}

// The path's wavelengths use the same type; lambdas.v[0] is the hero wavelength.
HD SampledSpectrum sampleSpectrum(const Spectrum& s, const SampledSpectrum& lambdas)
{
    SampledSpectrum r;
    for (int i = 0; i < kWavelengthsPerPath; ++i)
        r.v[i] = sampleSpectrum(s, lambdas.v[i]);
    return r;
}

// CIE 1931 2-degree colour matching functions, using the multi-lobe Gaussian fit from
// Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching
// Functions", JCGT 2013.
HD float cieLobe(float lambda, float mu, float sigmaLow, float sigmaHigh)
{
    const float t = (lambda - mu) / (lambda < mu ? sigmaLow : sigmaHigh);
    return expf(-0.5f * t * t);
}

HD float3 cieXYZ(float lambda)
{
    const float x = 1.056f * cieLobe(lambda, 599.8f, 37.9f, 31.0f) + 0.362f * cieLobe(lambda, 442.0f, 16.0f, 26.7f) -
                    0.065f * cieLobe(lambda, 501.1f, 20.4f, 26.2f);
    const float y = 0.821f * cieLobe(lambda, 568.8f, 46.9f, 40.5f) + 0.286f * cieLobe(lambda, 530.9f, 16.3f, 31.1f);
    const float z = 1.217f * cieLobe(lambda, 437.0f, 11.8f, 36.0f) + 0.681f * cieLobe(lambda, 459.0f, 26.0f, 13.8f);
    return make_float3(x, y, z);
}

// Linear sRGB (D65 primaries) from CIE XYZ. No chromatic adaptation is applied.
HD float3 xyzToLinearSrgb(float3 xyz)
{
    return make_float3(3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
                       -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
                       0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

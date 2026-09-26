// Minimal float3 math shared by host and device code.
#pragma once

#include <cuda_runtime.h>
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

constexpr float kPi = 3.14159265358979323846f;

HD float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
HD float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
HD float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
HD float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
HD float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
HD float3 operator*(float s, float3 a) { return a * s; }
HD float3 operator/(float3 a, float s) { return a * (1.0f / s); }
HD float3& operator+=(float3& a, float3 b) { a = a + b; return a; }
HD float3& operator*=(float3& a, float3 b) { a = a * b; return a; }

HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
HD float length(float3 a) { return sqrtf(dot(a, a)); }
HD float3 normalize(float3 a) { return a / length(a); }
HD float maxComponent(float3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }

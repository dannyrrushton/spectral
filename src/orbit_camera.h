// A turntable camera orbiting the centre of the Cornell Box. Moving the camera around the scene
// looks the same as rotating the scene, and needs no acceleration-structure rebuild.
#pragma once

#include <algorithm>
#include <cmath>

#include "renderer.h"
#include "vec_math.h"

struct OrbitCamera
{
    // Yaw 0, pitch 0 at the default distance reproduces the Cornell camera.
    static constexpr float3 kTarget       = {278.0f, 273.0f, 279.6f};
    static constexpr float kDefaultDistance = 1079.6f;
    static constexpr float kMaxPitch      = 85.0f * kPi / 180.0f;  // stay clear of the poles

    float yaw      = 0.0f;  // radians, positive moves the camera towards +x (screen left)
    float pitch    = 0.0f;  // radians, positive moves the camera up
    float distance = kDefaultDistance;

    void rotate(float dYaw, float dPitch)
    {
        yaw += dYaw;
        pitch = std::clamp(pitch + dPitch, -kMaxPitch, kMaxPitch);
    }

    void zoom(float factor) { distance = std::clamp(distance * factor, 200.0f, 5000.0f); }

    Camera camera() const
    {
        const float3 offset = make_float3(std::sin(yaw) * std::cos(pitch), std::sin(pitch),
                                          -std::cos(yaw) * std::cos(pitch));
        Camera c  = Renderer::defaultCamera();
        c.eye     = kTarget + offset * distance;
        c.lookAt  = kTarget;
        return c;
    }
};

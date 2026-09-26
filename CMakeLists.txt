cmake_minimum_required(VERSION 3.27)
project(spectral LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

set(OPTIX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/optix/include"
    CACHE PATH "Path to the OptiX SDK include directory")

# Device programs, compiled to OptiX IR and loaded by the host at runtime.
add_library(spectral_device OBJECT src/device.cu)
set_target_properties(spectral_device PROPERTIES CUDA_OPTIX_COMPILATION ON)
target_include_directories(spectral_device PRIVATE ${OPTIX_INCLUDE_DIR})
target_compile_options(spectral_device PRIVATE --use_fast_math -lineinfo
    -diag-suppress 20044)  # "extern __constant__ params treated as static" is expected for OptiX

# The renderer, shared by the offline CLI and the interactive viewer.
add_library(spectral_renderer STATIC src/renderer.cpp)
add_dependencies(spectral_renderer spectral_device)
target_include_directories(spectral_renderer PUBLIC ${OPTIX_INCLUDE_DIR} src)
target_link_libraries(spectral_renderer PUBLIC CUDA::cudart ${CMAKE_DL_LIBS})
target_compile_definitions(spectral_renderer PRIVATE
    SPECTRAL_DEVICE_CODE_PATH="$<TARGET_OBJECTS:spectral_device>")

# Offline renderer: writes a PNG.
add_executable(spectral src/main.cpp)
target_include_directories(spectral PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
target_link_libraries(spectral PRIVATE spectral_renderer)

# Interactive viewer.
find_package(SDL3 REQUIRED CONFIG)
add_executable(spectral_viewer src/viewer.cpp)
target_link_libraries(spectral_viewer PRIVATE spectral_renderer SDL3::SDL3)

# spectral

A GPU spectral path tracer built on NVIDIA OptiX 9.1 and CUDA 13. It renders a modified Cornell Box
from measured physical data, and simulates light by wavelength rather than as red, green and blue.
That lets it show things an RGB renderer can't, such as glass splitting white light into a rainbow.

It comes as two programs sharing one rendering core: `spectral` writes PNG images, and
`spectral_viewer` is an interactive window you can rotate with the mouse.

## Building

Requirements: an NVIDIA RTX GPU and driver (the driver provides the OptiX runtime), the CUDA
toolkit, CMake 3.27+, a C++17 compiler and SDL3 (for the viewer). The OptiX headers and
stb_image_write are vendored in `third_party/`.

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

The executables load the compiled GPU code from the build tree at runtime, so run them from this
machine's build.

## Usage

```bash
./build/spectral                              # 768x768, 1024 samples per pixel -> cornell_box.png
./build/spectral -s 8192 -r 1024x1024 -o out.png
./build/spectral --orbit 25,5                 # camera orbited 25 degrees sideways, 5 up
./build/spectral_viewer                       # interactive
```

`spectral` options: `-s` samples per pixel, `-r` resolution, `-d` maximum number of bounces,
`-o` output file, `--orbit yaw,pitch` camera angle in degrees.

`spectral_viewer` controls: drag to rotate, scroll or `+`/`-` to zoom, arrow keys to rotate,
`R` to reset the view, `Esc` or `Q` to quit. Options: `-s` stops refining after that many samples
per pixel, `-d` sets the maximum number of bounces.

## The scene

- **Base scene:** Cornell University's published Cornell Box data (wall and block geometry, the
  camera, and measured color spectra for the white, red and green paint and the ceiling light),
  with the tall block removed.
- **Glass ball:** on the floor at front left. It focuses the ceiling light into a bright spot on
  the floor (a caustic).
- **Glass prism:** a 60° prism on the short block, lit by a narrow white beam that enters through
  the open front of the box. It spreads the beam into a rainbow on the back wall. Its position was
  found by a search for the widest visible rainbow; the rules are documented in
  `cornellBoxPrism()` in `src/cornell_box.h`.
- **Glass:** both glass objects use Schott N-SF11, a dense flint glass. Its refractive index is
  computed per wavelength from the manufacturer's published formula, so it separates colors
  strongly.

## How light is simulated

**Colors are wavelengths.** Every material is described by how much light it reflects or emits at
each wavelength from 400 to 700 nm, in 4 nm steps. Each light path carries several wavelengths at
once, evenly spread from a random starting point; the count is `kWavelengthsPerPath` in
`src/spectrum.h`. Only at the end is the light converted to what the eye sees (CIE XYZ), then to
the screen's sRGB colors.

**Two passes each frame:**

1. **Camera pass.** Rays go out from the camera and bounce around the room (standard path tracing).
   - **Matte surfaces:** at each bounce the renderer also checks directly whether the ceiling light
     is visible, which greatly reduces noise.
   - **Glass:** a ray either reflects or refracts, with the choice weighted by how much light real
     glass reflects at that angle. When a path meets glass, its wavelengths would bend differently,
     so it keeps just one and weights it up so the overall average stays correct.
   - **Caustics:** when a path reaches the ceiling light straight after bouncing off or through
     glass, that light is counted. This produces the bright spot under the ball.
2. **Light pass.** A perfectly parallel beam can never be found by rays leaving the camera, so this
   pass works the other way round. It fires light particles from the beam, one wavelength each,
   through the prism. Wherever a particle lands on a matte surface, its contribution goes straight
   to the camera pixel that sees that spot. This pass draws the rainbow. It handles only the beam,
   and the camera pass handles only the ceiling light, so nothing is counted twice.

The two passes are blended per pixel, and the image keeps improving as long as the view stays
still.

**On the GPU:** the box is a triangle mesh and the ball uses OptiX's built-in sphere shape. They're
grouped under a single top-level structure that the GPU searches for ray hits. The camera pass and
light pass run as separate OptiX programs in the same pipeline.

**The viewer** renders just enough per frame to stay near 60 fps, and stops once the image is clean
enough. From the sides or behind, the walls between you and the room are left out (a dollhouse
cutaway) so you can see inside; lighting is unaffected.

## Code layout

| File | Role |
|---|---|
| `src/device.cu` | GPU programs: camera pass, light pass, glass handling, what happens when a ray hits something |
| `src/renderer.cpp/.h` | OptiX setup, geometry structures, frame buffers, running the two passes |
| `src/cornell_box.h` | Scene: geometry, materials, glass, prism and beam placement |
| `src/cornell_spectra.h` | Cornell's measured spectra, generated from the published data table |
| `src/spectrum.h` | Spectrum types, the eye's color response, XYZ→sRGB conversion |
| `src/main.cpp`, `src/viewer.cpp`, `src/orbit_camera.h` | The two programs and the shared orbit camera |
| `third_party/` | OptiX 9.1 headers ([NVIDIA/optix-dev](https://github.com/NVIDIA/optix-dev)), [stb_image_write](https://github.com/nothings/stb) |

## How it was checked

- **Spectra and glass data:** the color data came from Cornell's
  [data page](https://www.graphics.cornell.edu/online/box/data.html) (via an archived copy), and
  the glass formula reproduces Schott's catalog values exactly.
- **Glass color handling:** the ball's caustic matches, within 0.1%, a render with non-dispersive
  glass of the same strength. So dropping to one wavelength at glass doesn't change the overall
  brightness.
- **Prism placement:** the rainbow's position matches the prism simulation to within a few
  millimetres.

## Limitations

- **Noise:** there's grain in the caustics and faint color speckle wherever light has passed through
  glass. It clears with more samples; there's no denoiser yet.
- **Missing reflections:** the rainbow doesn't show up when seen through or reflected in glass,
  because the light pass only reaches the camera directly.
- **Unverified brightness:** the light pass's brightness hasn't been checked against an
  independent reference.
- **Wavelength count:** `kWavelengthsPerPath` is currently 8. That measured about 50% slower per
  sample than 4, with no measurable quality gain in this scene.
- **Display:** there's no tone mapping, so very bright colors clip.
- **Supported scene:** only this scene is supported, with no scene-file loading, and only matte,
  glass and light-emitting materials.

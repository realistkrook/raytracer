# raytracker

A multithreaded path tracer in Rust. Spheres, matte/metal/glass materials,
antialiasing, defocus blur, a positionable camera, and a BVH to keep it fast.

```
cargo run --release -- --scene final --width 1200 --samples 500 --out render.png
```

That renders 1200x675 at 500 samples per pixel in about 30 seconds on a 10-core
M-series Mac. For a preview while iterating, drop the numbers:

```
cargo run --release -- --scene three --width 400 --samples 50 --out preview.png
```

Always build with `--release`. A debug build is roughly 30x slower.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--scene <name>` | `three` | `three`, `glass`, or `final` |
| `--width <px>` | 400 | Image width; height follows the scene's aspect ratio |
| `--samples <n>` | 100 | Rays per pixel. Noise falls as the square root, so 4x the samples halves it |
| `--depth <n>` | 50 | Maximum bounces before a path is abandoned |
| `--seed <n>` | 0 | Changes the noise pattern, nothing else |
| `--out <path>` | `render.png` | Output PNG |
| `--no-bvh` | off | Linear intersection instead of the BVH. For timing comparisons |

## Scenes

- **three** — matte, glass, and fuzzy metal spheres over a yellow ground.
- **glass** — a solid glass ball beside a hollow one, with markers behind them.
  The solid ball inverts what is behind it; the hollow shell does not. If either
  of those is wrong, `refract` has a sign error.
- **final** — 485 randomly placed spheres around three large ones.

## Layout

| File | Contents |
|---|---|
| `main.rs` | CLI, scene setup, timing, PNG output |
| `vec3.rs` | Vector/color math and the random-sampling helpers |
| `ray.rs` | Ray origin, direction, and `at(t)` |
| `interval.rs` | Scalar ranges for ray `t` windows and clamping |
| `hittable.rs` | `HitRecord`, the `Hittable` trait, and `HittableList` |
| `sphere.rs` | Ray-sphere intersection |
| `material.rs` | `Lambertian`, `Metal`, `Dielectric` |
| `camera.rs` | Viewport basis, ray generation, and the parallel render loop |
| `bvh.rs` | Axis-aligned bounding boxes and the bounding volume hierarchy |
| `scene.rs` | Scene builders |

## Notes on the implementation

**Renders are deterministic.** Each scanline seeds its own RNG from its row
index, so output is bit-identical regardless of how rayon schedules threads:

```
cargo run --release -- --scene final --width 200 --samples 20 --out a.png
RAYON_NUM_THREADS=1 cargo run --release -- --scene final --width 200 --samples 20 --out b.png
cmp a.png b.png
```

**The BVH is worth about 4.5x** on the 485-object final scene, and the gap
widens with object count. `--no-bvh` renders a pixel-identical image the slow
way, which is a useful check after touching intersection code.

**Ray `t` starts at 0.001, not 0.** Otherwise a bounce immediately re-hits the
surface it just left and the image fills with black speckle.

**Color is accumulated linearly and gamma-corrected on write.** Skipping that
step makes everything look too dark.

## Tests

```
cargo test
```

47 tests, covering the vector math (including `reflect` and `refract` against
Snell's law), interval endpoints, ray-sphere intersection for hit/miss/tangent
and rays starting inside a sphere, material scattering behavior, AABB slab
tests including rays parallel to a slab, and CLI parsing.

## Extending it

The module boundaries are the extension points. Triangle meshes are another
`Hittable`; emissive materials are a method on `Material`; textures replace the
`albedo` field with a trait. None of those require touching the ray loop.

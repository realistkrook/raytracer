use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::hittable::Hittable;
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3, vec3};

/// Everything the caller chooses. `Camera::new` turns this into the derived
/// viewport basis once, up front.
#[derive(Clone, Copy, Debug)]
pub struct CameraConfig {
    pub image_width: u32,
    pub aspect_ratio: f64,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
    /// Vertical field of view, in degrees.
    pub vfov: f64,
    pub lookfrom: Point3,
    pub lookat: Point3,
    pub vup: Vec3,
    /// Cone angle of the lens aperture, in degrees. Zero gives a pinhole
    /// camera with everything in focus.
    pub defocus_angle: f64,
    /// Distance from the camera to the plane that is perfectly in focus.
    pub focus_dist: f64,
    pub seed: u64,
}

impl Default for CameraConfig {
    fn default() -> CameraConfig {
        CameraConfig {
            image_width: 400,
            aspect_ratio: 16.0 / 9.0,
            samples_per_pixel: 100,
            max_depth: 50,
            vfov: 20.0,
            lookfrom: vec3(0.0, 0.0, 0.0),
            lookat: vec3(0.0, 0.0, -1.0),
            vup: vec3(0.0, 1.0, 0.0),
            defocus_angle: 0.0,
            focus_dist: 10.0,
            seed: 0,
        }
    }
}

pub struct Camera {
    config: CameraConfig,
    image_height: u32,
    center: Point3,
    /// World-space location of the center of the top-left pixel.
    pixel00_loc: Point3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
}

impl Camera {
    pub fn new(config: CameraConfig) -> Camera {
        let image_height = ((config.image_width as f64 / config.aspect_ratio) as u32).max(1);

        // Viewport dimensions come from the focus distance and field of view.
        let theta = config.vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * config.focus_dist;
        let viewport_width =
            viewport_height * (config.image_width as f64 / image_height as f64);

        // Orthonormal basis: w points back from the target, u right, v up.
        let w = (config.lookfrom - config.lookat).unit_vector();
        let u = config.vup.cross(w).unit_vector();
        let v = w.cross(u);

        // Viewport edge vectors. v is negated because image rows run downward
        // while the camera's up axis runs upward.
        let viewport_u = u * viewport_width;
        let viewport_v = -v * viewport_height;

        let pixel_delta_u = viewport_u / config.image_width as f64;
        let pixel_delta_v = viewport_v / image_height as f64;

        let viewport_upper_left = config.lookfrom
            - (w * config.focus_dist)
            - viewport_u / 2.0
            - viewport_v / 2.0;
        let pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

        let defocus_radius = config.focus_dist * (config.defocus_angle / 2.0).to_radians().tan();

        Camera {
            image_height,
            center: config.lookfrom,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
            defocus_disk_u: u * defocus_radius,
            defocus_disk_v: v * defocus_radius,
            config,
        }
    }

    pub fn image_width(&self) -> u32 {
        self.config.image_width
    }

    pub fn image_height(&self) -> u32 {
        self.image_height
    }

    /// Render the scene to an 8-bit RGB buffer, one row per rayon task.
    ///
    /// Each row seeds its own RNG from its row index, so the output is
    /// identical no matter how the rows get scheduled across threads.
    pub fn render(&self, world: &dyn Hittable) -> Vec<u8> {
        let width = self.config.image_width as usize;
        let height = self.image_height as usize;
        let rows_done = AtomicUsize::new(0);

        let mut buffer = vec![0u8; width * height * 3];

        buffer
            .par_chunks_mut(width * 3)
            .enumerate()
            .for_each(|(j, row)| {
                let mut rng = SmallRng::seed_from_u64(
                    self.config.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64 + 1),
                );

                for i in 0..width {
                    let mut color = Color::ZERO;
                    for _ in 0..self.config.samples_per_pixel {
                        let r = self.get_ray(i as u32, j as u32, &mut rng);
                        color += self.ray_color(r, world, &mut rng);
                    }
                    color /= self.config.samples_per_pixel as f64;

                    let px = &mut row[i * 3..i * 3 + 3];
                    px[0] = to_byte(color.x);
                    px[1] = to_byte(color.y);
                    px[2] = to_byte(color.z);
                }

                let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(16) || done == height {
                    eprint!("\r  rendering: {done}/{height} rows");
                }
            });

        eprintln!();
        buffer
    }

    /// A ray from the lens toward a randomly jittered point inside pixel (i, j).
    /// The jitter is what antialiases edges; the lens offset is what blurs
    /// anything off the focus plane.
    fn get_ray(&self, i: u32, j: u32, rng: &mut SmallRng) -> Ray {
        let offset = vec3(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5, 0.0);
        let pixel_sample = self.pixel00_loc
            + self.pixel_delta_u * (i as f64 + offset.x)
            + self.pixel_delta_v * (j as f64 + offset.y);

        let origin = if self.config.defocus_angle <= 0.0 {
            self.center
        } else {
            let p = Vec3::random_in_unit_disk(rng);
            self.center + self.defocus_disk_u * p.x + self.defocus_disk_v * p.y
        };

        Ray::new(origin, pixel_sample - origin)
    }

    /// Follow a ray through the scene, multiplying in each surface's
    /// attenuation until it escapes to the sky or runs out of bounces.
    ///
    /// Written as a loop rather than recursion: the throughput product is the
    /// same either way, and this keeps deep paths off the call stack.
    fn ray_color(&self, ray: Ray, world: &dyn Hittable, rng: &mut SmallRng) -> Color {
        let mut throughput = Color::ONE;
        let mut ray = ray;

        for _ in 0..self.config.max_depth {
            // t starts slightly above zero so a bounce cannot re-hit the very
            // surface it left — otherwise the image speckles with black dots.
            let Some(rec) = world.hit(&ray, Interval::new(0.001, f64::INFINITY)) else {
                return throughput.mul_elem(sky_color(ray.direction));
            };

            match rec.material.scatter(&ray, &rec, rng) {
                Some(scatter) => {
                    throughput = throughput.mul_elem(scatter.attenuation);
                    ray = scatter.scattered;
                }
                None => return Color::ZERO,
            }
        }

        // Out of bounces: treat the path as having collected no light.
        Color::ZERO
    }
}

/// Vertical gradient standing in for a sky dome.
fn sky_color(direction: Vec3) -> Color {
    let unit = direction.unit_vector();
    let a = 0.5 * (unit.y + 1.0);
    Color::ONE * (1.0 - a) + vec3(0.5, 0.7, 1.0) * a
}

/// Linear light to sRGB-ish display value (gamma 2).
fn linear_to_gamma(linear: f64) -> f64 {
    if linear > 0.0 { linear.sqrt() } else { 0.0 }
}

fn to_byte(linear_component: f64) -> u8 {
    const RANGE: Interval = Interval::new(0.0, 0.999);
    (256.0 * RANGE.clamp(linear_to_gamma(linear_component))) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera(defocus_angle: f64) -> Camera {
        Camera::new(CameraConfig {
            image_width: 16,
            aspect_ratio: 1.0,
            vfov: 90.0,
            lookfrom: vec3(0.0, 0.0, 0.0),
            lookat: vec3(0.0, 0.0, -1.0),
            focus_dist: 1.0,
            defocus_angle,
            ..CameraConfig::default()
        })
    }

    #[test]
    fn gamma_and_clamping() {
        assert_eq!(to_byte(0.0), 0);
        assert_eq!(to_byte(-1.0), 0);
        assert_eq!(to_byte(1.0), 255);
        assert_eq!(to_byte(5.0), 255);
        // 0.25 linear -> 0.5 after gamma 2 -> mid grey.
        assert_eq!(to_byte(0.25), 128);
    }

    #[test]
    fn rays_leave_the_pinhole_origin_and_point_into_the_scene() {
        let cam = test_camera(0.0);
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..100 {
            let r = cam.get_ray(8, 8, &mut rng);
            assert_eq!(r.origin, Vec3::ZERO);
            // Looking down -z, so every ray must travel that way.
            assert!(r.direction.z < 0.0);
        }
    }

    #[test]
    fn defocus_blur_moves_the_origin_onto_the_lens_disk() {
        let cam = test_camera(10.0);
        let mut rng = SmallRng::seed_from_u64(2);
        let mut moved = 0;
        for _ in 0..100 {
            let r = cam.get_ray(8, 8, &mut rng);
            if (r.origin - Vec3::ZERO).length() > 1e-12 {
                moved += 1;
            }
            // The lens radius for a 10 degree cone at distance 1.
            let radius = (10.0f64 / 2.0).to_radians().tan();
            assert!(r.origin.length() <= radius + 1e-12);
        }
        assert_eq!(moved, 100);
    }

    #[test]
    fn image_height_follows_the_aspect_ratio() {
        let cam = Camera::new(CameraConfig {
            image_width: 400,
            aspect_ratio: 16.0 / 9.0,
            ..CameraConfig::default()
        });
        assert_eq!(cam.image_height(), 225);

        // A very wide aspect must still leave at least one row.
        let squashed = Camera::new(CameraConfig {
            image_width: 10,
            aspect_ratio: 1000.0,
            ..CameraConfig::default()
        });
        assert_eq!(squashed.image_height(), 1);
    }

    #[test]
    fn corner_rays_straddle_the_view_axis() {
        let cam = test_camera(0.0);
        let mut rng = SmallRng::seed_from_u64(3);
        // With a 90 degree vfov the top-left pixel looks up and to the left,
        // the bottom-right down and to the right.
        let tl = cam.get_ray(0, 0, &mut rng).direction;
        let br = cam.get_ray(15, 15, &mut rng).direction;
        assert!(tl.x < 0.0 && tl.y > 0.0);
        assert!(br.x > 0.0 && br.y < 0.0);
    }
}

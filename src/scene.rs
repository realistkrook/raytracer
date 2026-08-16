use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::camera::CameraConfig;
use crate::hittable::HittableList;
use crate::material::{Dielectric, Lambertian, Material, Metal};
use crate::sphere::Sphere;
use crate::vec3::{Color, Vec3, vec3};

/// A scene pairs geometry with the camera that frames it, since the two are
/// chosen together.
pub struct Scene {
    pub world: HittableList,
    pub camera: CameraConfig,
}

pub const SCENE_NAMES: [&str; 3] = ["three", "glass", "final"];

pub fn build(name: &str, base: CameraConfig) -> Option<Scene> {
    match name {
        "three" => Some(three_spheres(base)),
        "glass" => Some(glass_test(base)),
        "final" => Some(final_scene(base)),
        _ => None,
    }
}

/// The classic three-ball lineup: matte centre, glass left, metal right.
fn three_spheres(base: CameraConfig) -> Scene {
    let mut world = HittableList::new();

    let ground = Arc::new(Lambertian::new(vec3(0.8, 0.8, 0.0)));
    let center = Arc::new(Lambertian::new(vec3(0.1, 0.2, 0.5)));
    let left = Arc::new(Dielectric::new(1.5));
    // Air bubble inside the glass ball: same refractive index inverted.
    let bubble = Arc::new(Dielectric::new(1.0 / 1.5));
    let right = Arc::new(Metal::new(vec3(0.8, 0.6, 0.2), 0.3));

    world.add(sphere(vec3(0.0, -100.5, -1.0), 100.0, ground));
    world.add(sphere(vec3(0.0, 0.0, -1.2), 0.5, center));
    world.add(sphere(vec3(-1.0, 0.0, -1.0), 0.5, left));
    world.add(sphere(vec3(-1.0, 0.0, -1.0), 0.4, bubble));
    world.add(sphere(vec3(1.0, 0.0, -1.0), 0.5, right));

    Scene {
        world,
        camera: CameraConfig {
            // Pulled back and widened until all three balls clear the frame
            // edges, with a modest aperture so the depth of field reads
            // without smearing the subject.
            vfov: 30.0,
            lookfrom: vec3(-1.6, 1.4, 2.2),
            lookat: vec3(0.0, 0.0, -1.0),
            vup: vec3(0.0, 1.0, 0.0),
            defocus_angle: 2.0,
            focus_dist: 3.84,
            ..base
        },
    }
}

/// Two glass balls side by side against coloured markers, the quickest way to
/// tell whether the dielectric maths is right.
///
/// The solid ball on the left acts as a strong lens and flips what is behind
/// it. The hollow shell on the right is glass-air-glass, which very nearly
/// cancels out, so the view through it stays upright. Getting one of those two
/// wrong is the classic sign of a sign error in `refract`.
fn glass_test(base: CameraConfig) -> Scene {
    let mut world = HittableList::new();

    world.add(sphere(
        vec3(0.0, -100.5, -1.0),
        100.0,
        Arc::new(Lambertian::new(vec3(0.5, 0.5, 0.5))),
    ));

    // Solid.
    world.add(sphere(
        vec3(-0.6, 0.0, -1.0),
        0.5,
        Arc::new(Dielectric::new(1.5)),
    ));

    // Hollow: shell plus an inverted-index bubble.
    world.add(sphere(
        vec3(0.6, 0.0, -1.0),
        0.5,
        Arc::new(Dielectric::new(1.5)),
    ));
    world.add(sphere(
        vec3(0.6, 0.0, -1.0),
        0.4,
        Arc::new(Dielectric::new(1.0 / 1.5)),
    ));

    // Markers behind the glass. The green one sits high so the flip through the
    // solid ball is unmistakable.
    world.add(sphere(
        vec3(-0.6, 0.5, -2.4),
        0.25,
        Arc::new(Lambertian::new(vec3(0.1, 0.85, 0.15))),
    ));
    world.add(sphere(
        vec3(-1.4, -0.2, -2.4),
        0.25,
        Arc::new(Lambertian::new(vec3(0.9, 0.1, 0.1))),
    ));
    world.add(sphere(
        vec3(0.6, 0.5, -2.4),
        0.25,
        Arc::new(Lambertian::new(vec3(0.95, 0.75, 0.1))),
    ));
    world.add(sphere(
        vec3(1.5, -0.1, -2.2),
        0.35,
        Arc::new(Metal::new(vec3(0.8, 0.8, 0.9), 0.0)),
    ));

    Scene {
        world,
        camera: CameraConfig {
            vfov: 45.0,
            lookfrom: vec3(0.0, 0.35, 1.6),
            lookat: vec3(0.0, 0.0, -1.0),
            vup: vec3(0.0, 1.0, 0.0),
            defocus_angle: 0.0,
            focus_dist: 2.6,
            ..base
        },
    }
}

/// The cover image: a field of small random spheres around three large ones.
/// The RNG is seeded fixed, so the layout is the same on every run.
fn final_scene(base: CameraConfig) -> Scene {
    let mut world = HittableList::new();
    let mut rng = SmallRng::seed_from_u64(1984);

    world.add(sphere(
        vec3(0.0, -1000.0, 0.0),
        1000.0,
        Arc::new(Lambertian::new(vec3(0.5, 0.5, 0.5))),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let center = vec3(
                a as f64 + 0.9 * rng.random::<f64>(),
                0.2,
                b as f64 + 0.9 * rng.random::<f64>(),
            );

            // Keep clear of the three feature spheres.
            if (center - vec3(4.0, 0.2, 0.0)).length() <= 0.9 {
                continue;
            }

            let choose_mat = rng.random::<f64>();
            let material: Arc<dyn Material> = if choose_mat < 0.8 {
                let albedo = random_color(&mut rng).mul_elem(random_color(&mut rng));
                Arc::new(Lambertian::new(albedo))
            } else if choose_mat < 0.95 {
                let albedo = Vec3::random(&mut rng, 0.5, 1.0);
                Arc::new(Metal::new(albedo, rng.random_range(0.0..0.5)))
            } else {
                Arc::new(Dielectric::new(1.5))
            };

            world.add(sphere(center, 0.2, material));
        }
    }

    world.add(sphere(
        vec3(0.0, 1.0, 0.0),
        1.0,
        Arc::new(Dielectric::new(1.5)),
    ));
    world.add(sphere(
        vec3(-4.0, 1.0, 0.0),
        1.0,
        Arc::new(Lambertian::new(vec3(0.4, 0.2, 0.1))),
    ));
    world.add(sphere(
        vec3(4.0, 1.0, 0.0),
        1.0,
        Arc::new(Metal::new(vec3(0.7, 0.6, 0.5), 0.0)),
    ));

    Scene {
        world,
        camera: CameraConfig {
            vfov: 20.0,
            lookfrom: vec3(13.0, 2.0, 3.0),
            lookat: vec3(0.0, 0.0, 0.0),
            vup: vec3(0.0, 1.0, 0.0),
            defocus_angle: 0.6,
            focus_dist: 10.0,
            ..base
        },
    }
}

fn sphere(center: Vec3, radius: f64, material: Arc<dyn Material>) -> Arc<Sphere> {
    Arc::new(Sphere::new(center, radius, material))
}

fn random_color(rng: &mut SmallRng) -> Color {
    vec3(rng.random(), rng.random(), rng.random())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_advertised_scene_builds() {
        for name in SCENE_NAMES {
            let scene = build(name, CameraConfig::default())
                .unwrap_or_else(|| panic!("scene {name} failed to build"));
            assert!(!scene.world.is_empty());
        }
    }

    #[test]
    fn unknown_scene_is_rejected() {
        assert!(build("nope", CameraConfig::default()).is_none());
    }

    #[test]
    fn final_scene_layout_is_reproducible() {
        let a = final_scene(CameraConfig::default());
        let b = final_scene(CameraConfig::default());
        assert_eq!(a.world.len(), b.world.len());
        // Enough spheres to be worth a BVH, but bounded by the 22x22 grid.
        assert!(a.world.len() > 400, "only {} objects", a.world.len());
    }
}

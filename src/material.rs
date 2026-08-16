use rand::Rng;

use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::vec3::{Color, Vec3};

/// The outcome of a ray meeting a surface: how much light survives the bounce,
/// and where the bounced ray goes.
pub struct Scatter {
    pub attenuation: Color,
    pub scattered: Ray,
}

/// Re-exported so callers can name the RNG handle. `Material` takes
/// `&mut dyn RngCore` rather than a generic parameter to stay object-safe;
/// `rand`'s blanket impl still hands every implementation the full `Rng` API.
pub use rand::RngCore;

pub trait Material: Send + Sync {
    /// `None` means the surface swallowed the ray.
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut dyn RngCore) -> Option<Scatter>;
}

/// Matte surface: bounce in a random direction biased toward the normal.
pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Lambertian {
        Lambertian { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord, rng: &mut dyn RngCore) -> Option<Scatter> {
        let mut direction = rec.normal + Vec3::random_unit_vector(&mut *rng);

        // If the random vector lands exactly opposite the normal the sum cancels
        // out, leaving a zero direction that would produce NaNs downstream.
        if direction.near_zero() {
            direction = rec.normal;
        }

        Some(Scatter {
            attenuation: self.albedo,
            scattered: Ray::new(rec.p, direction),
        })
    }
}

/// Polished or brushed metal: mirror reflection, optionally roughened.
pub struct Metal {
    albedo: Color,
    fuzz: f64,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f64) -> Metal {
        Metal {
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
        }
    }
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut dyn RngCore) -> Option<Scatter> {
        let reflected = r_in.direction.unit_vector().reflect(rec.normal);
        let direction = reflected + Vec3::random_unit_vector(&mut *rng) * self.fuzz;

        // Heavy fuzz can kick the ray below the surface; count that as absorbed
        // rather than letting it tunnel through the object.
        if direction.dot(rec.normal) <= 0.0 {
            return None;
        }

        Some(Scatter {
            attenuation: self.albedo,
            scattered: Ray::new(rec.p, direction),
        })
    }
}

/// Clear material such as glass or water: refracts when it can, reflects when
/// it cannot.
pub struct Dielectric {
    /// Refractive index relative to the surrounding medium.
    refraction_index: f64,
}

impl Dielectric {
    pub fn new(refraction_index: f64) -> Dielectric {
        Dielectric { refraction_index }
    }

    /// Schlick's cheap approximation to Fresnel reflectance — the reason glass
    /// goes mirror-like at glancing angles.
    fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
        let r0 = ((1.0 - refraction_index) / (1.0 + refraction_index)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut dyn RngCore) -> Option<Scatter> {
        let ri = if rec.front_face {
            1.0 / self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = r_in.direction.unit_vector();
        let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        // Past the critical angle Snell's law has no solution, so the surface
        // acts as a perfect mirror.
        let cannot_refract = ri * sin_theta > 1.0;
        let reflect_probability = Dielectric::reflectance(cos_theta, ri);
        let direction = if cannot_refract || reflect_probability > rng.random::<f64>() {
            unit_direction.reflect(rec.normal)
        } else {
            unit_direction.refract(rec.normal, ri)
        };

        Some(Scatter {
            // Glass tints nothing; all the color comes from what it transmits.
            attenuation: Color::ONE,
            scattered: Ray::new(rec.p, direction),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn hit_facing_up<'a>(r: &Ray, material: &'a dyn Material) -> HitRecord<'a> {
        HitRecord::new(r, Vec3::ZERO, vec3(0.0, 1.0, 0.0), 1.0, material)
    }

    #[test]
    fn lambertian_always_scatters_above_the_surface() {
        let mut rng = SmallRng::seed_from_u64(1);
        let m = Lambertian::new(vec3(0.8, 0.3, 0.3));
        let r = Ray::new(vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0));
        for _ in 0..500 {
            let rec = hit_facing_up(&r, &m);
            let s = m.scatter(&r, &rec, &mut rng).expect("matte never absorbs");
            assert!(s.scattered.direction.dot(rec.normal) >= -1e-12);
            assert_eq!(s.attenuation, vec3(0.8, 0.3, 0.3));
        }
    }

    #[test]
    fn polished_metal_reflects_exactly() {
        let mut rng = SmallRng::seed_from_u64(2);
        let m = Metal::new(vec3(0.8, 0.8, 0.8), 0.0);
        let r = Ray::new(vec3(-1.0, 1.0, 0.0), vec3(1.0, -1.0, 0.0));
        let rec = hit_facing_up(&r, &m);
        let s = m.scatter(&r, &rec, &mut rng).unwrap();
        let expected = vec3(1.0, -1.0, 0.0).unit_vector().reflect(rec.normal);
        assert!((s.scattered.direction - expected).length() < 1e-12);
    }

    #[test]
    fn glass_at_grazing_incidence_reflects_instead_of_refracting() {
        let mut rng = SmallRng::seed_from_u64(3);
        // Exiting glass (front_face false gives ri = 1.5) at a shallow angle is
        // past the critical angle, so it must reflect.
        let m = Dielectric::new(1.5);
        let direction = vec3(1.0, -0.05, 0.0).unit_vector();
        let r = Ray::new(vec3(-1.0, 0.05, 0.0), direction);
        let rec = HitRecord::new(&r, Vec3::ZERO, vec3(0.0, -1.0, 0.0), 1.0, &m);
        assert!(!rec.front_face);
        let s = m.scatter(&r, &rec, &mut rng).unwrap();
        // A reflection off a horizontal surface flips y and keeps x.
        assert!(s.scattered.direction.y > 0.0);
        assert!((s.scattered.direction.x - direction.x).abs() < 1e-12);
    }

    #[test]
    fn glass_head_on_mostly_transmits() {
        let mut rng = SmallRng::seed_from_u64(4);
        let m = Dielectric::new(1.5);
        let r = Ray::new(vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0));
        let mut transmitted = 0;
        for _ in 0..1000 {
            let rec = hit_facing_up(&r, &m);
            let s = m.scatter(&r, &rec, &mut rng).unwrap();
            if s.scattered.direction.y < 0.0 {
                transmitted += 1;
            }
        }
        // Schlick at normal incidence for n=1.5 is ~4% reflectance.
        assert!(transmitted > 940, "transmitted only {transmitted}/1000");
    }

    #[test]
    fn heavy_fuzz_can_absorb() {
        let mut rng = SmallRng::seed_from_u64(5);
        let m = Metal::new(Color::ONE, 1.0);
        // Grazing hit plus max fuzz: some bounces get pushed below the surface.
        let r = Ray::new(vec3(-1.0, 0.02, 0.0), vec3(1.0, -0.02, 0.0));
        let absorbed = (0..2000)
            .filter(|_| {
                let rec = hit_facing_up(&r, &m);
                m.scatter(&r, &rec, &mut rng).is_none()
            })
            .count();
        assert!(absorbed > 0, "expected some rays scattered below the surface");
    }
}

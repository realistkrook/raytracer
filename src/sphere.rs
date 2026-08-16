use std::sync::Arc;

use crate::bvh::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct Sphere {
    center: Point3,
    radius: f64,
    material: Arc<dyn Material>,
    bbox: Aabb,
}

impl Sphere {
    /// A negative `radius` keeps the geometry but flips the outward normal,
    /// which is how a hollow glass shell is built.
    pub fn new(center: Point3, radius: f64, material: Arc<dyn Material>) -> Sphere {
        let r = radius.abs();
        let rvec = Vec3::splat(r);
        Sphere {
            center,
            radius,
            material,
            bbox: Aabb::from_points(center - rvec, center + rvec),
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        // |o + td - c|^2 = r^2 expands to a quadratic in t. Writing the linear
        // coefficient as -2h lets the 2s cancel out of the quadratic formula.
        let oc = self.center - r.origin;
        let a = r.direction.length_squared();
        let h = r.direction.dot(oc);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // Nearest root in range; fall back to the far one when the near root is
        // behind us (which happens when the ray starts inside the sphere).
        let mut root = (h - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrtd) / a;
            if !ray_t.surrounds(root) {
                return None;
            }
        }

        let p = r.at(root);
        let outward_normal = (p - self.center) / self.radius;
        Some(HitRecord::new(
            r,
            p,
            outward_normal,
            root,
            self.material.as_ref(),
        ))
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Lambertian;
    use crate::vec3::vec3;

    fn unit_sphere_at_origin() -> Sphere {
        Sphere::new(
            Vec3::ZERO,
            1.0,
            Arc::new(Lambertian::new(vec3(0.5, 0.5, 0.5))),
        )
    }

    #[test]
    fn head_on_ray_hits_the_near_face() {
        let s = unit_sphere_at_origin();
        // Starting 3 units out along -z, aimed at the origin.
        let r = Ray::new(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0));
        let rec = s.hit(&r, Interval::new(0.001, f64::INFINITY)).unwrap();
        assert!((rec.t - 2.0).abs() < 1e-12);
        assert_eq!(rec.p, vec3(0.0, 0.0, -1.0));
        assert!(rec.front_face);
        // Normal points back at the ray.
        assert_eq!(rec.normal, vec3(0.0, 0.0, -1.0));
    }

    #[test]
    fn a_ray_pointing_away_misses() {
        let s = unit_sphere_at_origin();
        let r = Ray::new(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, -1.0));
        assert!(s.hit(&r, Interval::new(0.001, f64::INFINITY)).is_none());
    }

    #[test]
    fn a_ray_that_passes_beside_the_sphere_misses() {
        let s = unit_sphere_at_origin();
        let r = Ray::new(vec3(2.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0));
        assert!(s.hit(&r, Interval::new(0.001, f64::INFINITY)).is_none());
    }

    #[test]
    fn tangent_ray_grazes_at_a_single_point() {
        let s = unit_sphere_at_origin();
        let r = Ray::new(vec3(1.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0));
        let rec = s.hit(&r, Interval::new(0.001, f64::INFINITY)).unwrap();
        assert!((rec.t - 3.0).abs() < 1e-9);
        assert!((rec.p.x - 1.0).abs() < 1e-9 && rec.p.z.abs() < 1e-9);
    }

    #[test]
    fn a_ray_starting_inside_hits_the_far_wall_from_behind() {
        let s = unit_sphere_at_origin();
        let r = Ray::new(Vec3::ZERO, vec3(0.0, 0.0, 1.0));
        let rec = s.hit(&r, Interval::new(0.001, f64::INFINITY)).unwrap();
        assert!((rec.t - 1.0).abs() < 1e-12);
        assert!(!rec.front_face);
        // Flipped to face the ray, so it points back toward -z.
        assert_eq!(rec.normal, vec3(0.0, 0.0, -1.0));
    }

    #[test]
    fn hits_outside_the_t_window_are_rejected() {
        let s = unit_sphere_at_origin();
        let r = Ray::new(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0));
        // The near hit is at t=2, the far at t=4.
        assert!(s.hit(&r, Interval::new(0.001, 1.5)).is_none());
        let rec = s.hit(&r, Interval::new(2.5, f64::INFINITY)).unwrap();
        assert!((rec.t - 4.0).abs() < 1e-12);
    }

    #[test]
    fn negative_radius_flips_the_normal_inward() {
        let s = Sphere::new(
            Vec3::ZERO,
            -1.0,
            Arc::new(Lambertian::new(vec3(0.5, 0.5, 0.5))),
        );
        let r = Ray::new(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0));
        let rec = s.hit(&r, Interval::new(0.001, f64::INFINITY)).unwrap();
        assert!((rec.t - 2.0).abs() < 1e-12);
        // Geometrically the same surface, but now counted as an inside face.
        assert!(!rec.front_face);
        assert_eq!(rec.normal, vec3(0.0, 0.0, -1.0));
    }

    #[test]
    fn bounding_box_wraps_the_sphere() {
        let s = Sphere::new(
            vec3(1.0, 2.0, 3.0),
            2.0,
            Arc::new(Lambertian::new(Vec3::ONE)),
        );
        let bbox = s.bounding_box();
        assert_eq!(bbox.axis(0).min, -1.0);
        assert_eq!(bbox.axis(0).max, 3.0);
        assert_eq!(bbox.axis(1).min, 0.0);
        assert_eq!(bbox.axis(2).max, 5.0);
    }
}

use std::sync::Arc;

use crate::bvh::Aabb;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// What a ray found where it struck a surface.
pub struct HitRecord<'a> {
    pub p: Point3,
    /// Always points against the incoming ray, so shading never has to ask
    /// which side it is on. `front_face` records which side that was.
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub material: &'a dyn Material,
}

impl<'a> HitRecord<'a> {
    /// `outward_normal` must be unit length and point out of the shape.
    pub fn new(
        r: &Ray,
        p: Point3,
        outward_normal: Vec3,
        t: f64,
        material: &'a dyn Material,
    ) -> HitRecord<'a> {
        let front_face = r.direction.dot(outward_normal) < 0.0;
        HitRecord {
            p,
            normal: if front_face {
                outward_normal
            } else {
                -outward_normal
            },
            t,
            front_face,
            material,
        }
    }
}

pub trait Hittable: Send + Sync {
    /// Nearest intersection with `t` strictly inside `ray_t`, if any.
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord<'_>>;

    fn bounding_box(&self) -> Aabb;
}

#[derive(Clone, Default)]
pub struct HittableList {
    pub objects: Vec<Arc<dyn Hittable>>,
    bbox: Aabb,
}

impl HittableList {
    pub fn new() -> HittableList {
        HittableList {
            objects: Vec::new(),
            bbox: Aabb::EMPTY,
        }
    }

    pub fn add(&mut self, object: Arc<dyn Hittable>) {
        self.bbox = Aabb::enclosing(self.bbox, object.bounding_box());
        self.objects.push(object);
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut closest = ray_t.max;
        let mut result = None;

        for object in &self.objects {
            // Shrink the window as we go, so later objects only register when
            // they are genuinely in front of the best hit so far.
            if let Some(rec) = object.hit(r, Interval::new(ray_t.min, closest)) {
                closest = rec.t;
                result = Some(rec);
            }
        }

        result
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

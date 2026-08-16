use std::cmp::Ordering;
use std::sync::Arc;

use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::Point3;

/// An axis-aligned bounding box, stored as one interval per axis.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl Default for Aabb {
    fn default() -> Aabb {
        Aabb::EMPTY
    }
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        x: Interval::EMPTY,
        y: Interval::EMPTY,
        z: Interval::EMPTY,
    };

    /// Build from two opposite corners, in either order.
    pub fn from_points(a: Point3, b: Point3) -> Aabb {
        Aabb {
            x: Interval::new(a.x.min(b.x), a.x.max(b.x)),
            y: Interval::new(a.y.min(b.y), a.y.max(b.y)),
            z: Interval::new(a.z.min(b.z), a.z.max(b.z)),
        }
        .padded()
    }

    pub fn enclosing(a: Aabb, b: Aabb) -> Aabb {
        Aabb {
            x: Interval::enclosing(a.x, b.x),
            y: Interval::enclosing(a.y, b.y),
            z: Interval::enclosing(a.z, b.z),
        }
    }

    pub fn axis(&self, n: usize) -> Interval {
        match n {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    /// A perfectly flat box would have zero thickness on one axis, and a ray
    /// travelling in that plane could slip through it. Give every axis a
    /// minimum width.
    fn padded(self) -> Aabb {
        const DELTA: f64 = 1e-4;
        Aabb {
            x: if self.x.size() < DELTA {
                self.x.expand(DELTA)
            } else {
                self.x
            },
            y: if self.y.size() < DELTA {
                self.y.expand(DELTA)
            } else {
                self.y
            },
            z: if self.z.size() < DELTA {
                self.z.expand(DELTA)
            } else {
                self.z
            },
        }
    }

    /// The longest axis, used to pick a split plane that keeps children compact.
    pub fn longest_axis(&self) -> usize {
        if self.x.size() > self.y.size() {
            if self.x.size() > self.z.size() { 0 } else { 2 }
        } else if self.y.size() > self.z.size() {
            1
        } else {
            2
        }
    }

    /// Slab method: clip the ray's `t` window against each pair of parallel
    /// planes and see whether anything is left.
    pub fn hit(&self, r: &Ray, mut ray_t: Interval) -> bool {
        for axis in 0..3 {
            let ax = self.axis(axis);
            let inv_d = 1.0 / r.direction[axis];
            let orig = r.origin[axis];

            let mut t0 = (ax.min - orig) * inv_d;
            let mut t1 = (ax.max - orig) * inv_d;
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }

            if t0 > ray_t.min {
                ray_t.min = t0;
            }
            if t1 < ray_t.max {
                ray_t.max = t1;
            }

            // NaN from a zero-component direction compares false here, which
            // correctly leaves a ray parallel to the slab unclipped unless it
            // is outside the slab entirely.
            if ray_t.max <= ray_t.min {
                return false;
            }
        }
        true
    }
}

/// A node in a bounding volume hierarchy. Ray/scene cost drops from linear in
/// the object count to roughly logarithmic.
pub struct BvhNode {
    left: Arc<dyn Hittable>,
    right: Arc<dyn Hittable>,
    bbox: Aabb,
}

impl BvhNode {
    pub fn from_list(list: HittableList) -> BvhNode {
        let mut objects = list.objects;
        let len = objects.len();
        BvhNode::build(&mut objects, 0, len)
    }

    fn build(objects: &mut Vec<Arc<dyn Hittable>>, start: usize, end: usize) -> BvhNode {
        // Split along whichever axis the enclosing box is widest on.
        let mut bbox = Aabb::EMPTY;
        for object in objects[start..end].iter() {
            bbox = Aabb::enclosing(bbox, object.bounding_box());
        }
        let axis = bbox.longest_axis();

        let span = end - start;
        let (left, right): (Arc<dyn Hittable>, Arc<dyn Hittable>) = match span {
            1 => (objects[start].clone(), objects[start].clone()),
            2 => (objects[start].clone(), objects[start + 1].clone()),
            _ => {
                objects[start..end].sort_by(|a, b| box_compare(a.as_ref(), b.as_ref(), axis));
                let mid = start + span / 2;
                (
                    Arc::new(BvhNode::build(objects, start, mid)),
                    Arc::new(BvhNode::build(objects, mid, end)),
                )
            }
        };

        BvhNode { left, right, bbox }
    }
}

fn box_compare(a: &dyn Hittable, b: &dyn Hittable, axis: usize) -> Ordering {
    let a_min = a.bounding_box().axis(axis).min;
    let b_min = b.bounding_box().axis(axis).min;
    a_min.partial_cmp(&b_min).unwrap_or(Ordering::Equal)
}

impl Hittable for BvhNode {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        if !self.bbox.hit(r, ray_t) {
            return None;
        }

        let hit_left = self.left.hit(r, ray_t);
        // Narrow the window with the left hit so the right subtree only reports
        // something closer.
        let right_max = hit_left.as_ref().map_or(ray_t.max, |rec| rec.t);
        let hit_right = self.right.hit(r, Interval::new(ray_t.min, right_max));

        hit_right.or(hit_left)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    fn unit_box() -> Aabb {
        Aabb::from_points(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0))
    }

    #[test]
    fn ray_through_the_middle_hits() {
        let b = unit_box();
        let r = Ray::new(vec3(0.0, 0.0, -5.0), vec3(0.0, 0.0, 1.0));
        assert!(b.hit(&r, Interval::new(0.001, f64::INFINITY)));
    }

    #[test]
    fn ray_passing_beside_the_box_misses() {
        let b = unit_box();
        let r = Ray::new(vec3(3.0, 0.0, -5.0), vec3(0.0, 0.0, 1.0));
        assert!(!b.hit(&r, Interval::new(0.001, f64::INFINITY)));
    }

    #[test]
    fn box_behind_the_ray_is_outside_the_t_window() {
        let b = unit_box();
        let r = Ray::new(vec3(0.0, 0.0, 5.0), vec3(0.0, 0.0, 1.0));
        assert!(!b.hit(&r, Interval::new(0.001, f64::INFINITY)));
    }

    #[test]
    fn ray_parallel_to_a_slab_hits_when_inside_it_and_misses_when_outside() {
        let b = unit_box();
        // Travelling along x with zero y and z components: parallel to both the
        // y and z slabs. Inside them, so it should hit.
        let inside = Ray::new(vec3(-5.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0));
        assert!(b.hit(&inside, Interval::new(0.001, f64::INFINITY)));

        // Same direction, but riding above the box entirely.
        let outside = Ray::new(vec3(-5.0, 2.0, 0.0), vec3(1.0, 0.0, 0.0));
        assert!(!b.hit(&outside, Interval::new(0.001, f64::INFINITY)));
    }

    #[test]
    fn enclosing_spans_both_boxes() {
        let a = Aabb::from_points(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
        let b = Aabb::from_points(vec3(2.0, -3.0, 0.5), vec3(3.0, -2.0, 0.6));
        let c = Aabb::enclosing(a, b);
        assert_eq!(c.x.min, 0.0);
        assert_eq!(c.x.max, 3.0);
        assert_eq!(c.y.min, -3.0);
        assert_eq!(c.y.max, 1.0);
    }

    #[test]
    fn longest_axis_picks_the_widest_side() {
        let b = Aabb::from_points(vec3(0.0, 0.0, 0.0), vec3(1.0, 5.0, 2.0));
        assert_eq!(b.longest_axis(), 1);
        let flat = Aabb::from_points(vec3(0.0, 0.0, 0.0), vec3(9.0, 0.0, 2.0));
        assert_eq!(flat.longest_axis(), 0);
    }

    #[test]
    fn bvh_finds_the_nearest_of_many_spheres() {
        use crate::material::Lambertian;
        use crate::sphere::Sphere;

        // Spheres strung out along z; the ray should come back with the closest.
        let mut list = HittableList::new();
        for i in 0..32 {
            let z = i as f64 * 3.0;
            list.add(Arc::new(Sphere::new(
                vec3(0.0, 0.0, z),
                1.0,
                Arc::new(Lambertian::new(vec3(0.5, 0.5, 0.5))),
            )));
        }
        let reference = list.clone();
        let bvh = BvhNode::from_list(list);

        let r = Ray::new(vec3(0.0, 0.0, -10.0), vec3(0.0, 0.0, 1.0));
        let window = Interval::new(0.001, f64::INFINITY);
        let via_bvh = bvh.hit(&r, window).unwrap();
        let via_list = reference.hit(&r, window).unwrap();

        assert!((via_bvh.t - 9.0).abs() < 1e-12);
        // The acceleration structure must not change the answer.
        assert!((via_bvh.t - via_list.t).abs() < 1e-12);
    }
}

use crate::vec3::{Point3, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    #[test]
    fn at_walks_along_the_direction() {
        let r = Ray::new(vec3(1.0, 2.0, 3.0), vec3(0.0, 1.0, 0.0));
        assert_eq!(r.at(0.0), vec3(1.0, 2.0, 3.0));
        assert_eq!(r.at(2.5), vec3(1.0, 4.5, 3.0));
        assert_eq!(r.at(-1.0), vec3(1.0, 1.0, 3.0));
    }
}

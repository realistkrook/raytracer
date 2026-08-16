use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub};

use rand::Rng;

/// A 3-component vector, doubling as a point in space and as a linear RGB color.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub type Point3 = Vec3;
pub type Color = Vec3;

pub const fn vec3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = vec3(0.0, 0.0, 0.0);
    pub const ONE: Vec3 = vec3(1.0, 1.0, 1.0);

    pub const fn splat(v: f64) -> Vec3 {
        vec3(v, v, v)
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn dot(self, rhs: Vec3) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Vec3) -> Vec3 {
        vec3(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn unit_vector(self) -> Vec3 {
        self / self.length()
    }

    /// True when the vector is close enough to zero that normalizing it would
    /// produce garbage. Used to catch degenerate Lambertian scatter directions.
    pub fn near_zero(self) -> bool {
        const EPS: f64 = 1e-8;
        self.x.abs() < EPS && self.y.abs() < EPS && self.z.abs() < EPS
    }

    /// Componentwise product — only meaningful when both operands are colors.
    pub fn mul_elem(self, rhs: Vec3) -> Vec3 {
        vec3(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    pub fn reflect(self, normal: Vec3) -> Vec3 {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Refract an incoming *unit* vector across `normal`, where `etai_over_etat`
    /// is the ratio of the incident to the transmitted refractive index.
    /// Callers must rule out total internal reflection first.
    pub fn refract(self, normal: Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = (-self).dot(normal).min(1.0);
        let r_out_perp = (self + normal * cos_theta) * etai_over_etat;
        let r_out_parallel = normal * -(1.0 - r_out_perp.length_squared()).abs().sqrt();
        r_out_perp + r_out_parallel
    }

    pub fn random(rng: &mut (impl Rng + ?Sized), min: f64, max: f64) -> Vec3 {
        vec3(
            rng.random_range(min..max),
            rng.random_range(min..max),
            rng.random_range(min..max),
        )
    }

    /// A uniformly distributed direction on the unit sphere, by rejection
    /// sampling. The lower bound guards against normalizing a near-zero vector,
    /// which would blow up to infinity.
    pub fn random_unit_vector(rng: &mut (impl Rng + ?Sized)) -> Vec3 {
        loop {
            let p = Vec3::random(rng, -1.0, 1.0);
            let len_sq = p.length_squared();
            if (1e-160..=1.0).contains(&len_sq) {
                return p / len_sq.sqrt();
            }
        }
    }

    /// A unit-sphere direction restricted to the hemisphere facing `normal`.
    pub fn random_on_hemisphere(rng: &mut (impl Rng + ?Sized), normal: Vec3) -> Vec3 {
        let v = Vec3::random_unit_vector(rng);
        if v.dot(normal) > 0.0 { v } else { -v }
    }

    /// A point in the unit disk on the xy plane — the lens sample for defocus blur.
    pub fn random_in_unit_disk(rng: &mut (impl Rng + ?Sized)) -> Vec3 {
        loop {
            let p = vec3(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        vec3(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        vec3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f64) -> Vec3 {
        vec3(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f64) -> Vec3 {
        self * (1.0 / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        vec3(-self.x, -self.y, -self.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        *self = *self + rhs;
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of range: {i}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn dot_of_perpendicular_axes_is_zero() {
        assert_eq!(vec3(1.0, 0.0, 0.0).dot(vec3(0.0, 1.0, 0.0)), 0.0);
        assert_eq!(vec3(1.0, 2.0, 3.0).dot(vec3(1.0, 2.0, 3.0)), 14.0);
    }

    #[test]
    fn cross_follows_right_hand_rule() {
        assert_eq!(
            vec3(1.0, 0.0, 0.0).cross(vec3(0.0, 1.0, 0.0)),
            vec3(0.0, 0.0, 1.0)
        );
        // A vector crossed with itself is the zero vector.
        assert_eq!(vec3(3.0, -2.0, 7.0).cross(vec3(3.0, -2.0, 7.0)), Vec3::ZERO);
    }

    #[test]
    fn unit_vector_has_unit_length() {
        let v = vec3(3.0, -4.0, 12.0);
        assert!(close(v.length(), 13.0));
        assert!(close(v.unit_vector().length(), 1.0));
    }

    #[test]
    fn reflect_off_the_ground_flips_only_y() {
        // Travelling down and to the right, bouncing off a floor facing +y.
        let incoming = vec3(1.0, -1.0, 0.0);
        let reflected = incoming.reflect(vec3(0.0, 1.0, 0.0));
        assert_eq!(reflected, vec3(1.0, 1.0, 0.0));
    }

    #[test]
    fn refract_at_normal_incidence_is_undeviated() {
        // Straight down the normal: direction survives refraction unchanged.
        let incoming = vec3(0.0, -1.0, 0.0);
        let out = incoming.refract(vec3(0.0, 1.0, 0.0), 1.0 / 1.5);
        assert!(close(out.x, 0.0) && close(out.z, 0.0));
        assert!(close(out.y, -1.0));
    }

    #[test]
    fn refract_bends_toward_the_normal_entering_denser_medium() {
        let incoming = vec3(1.0, -1.0, 0.0).unit_vector();
        let normal = vec3(0.0, 1.0, 0.0);
        let out = incoming.refract(normal, 1.0 / 1.5);
        let sin_in = incoming.x.abs();
        let sin_out = out.x.abs();
        // Snell: sin(out) = (n1/n2) * sin(in), so the ray straightens up.
        assert!(sin_out < sin_in);
        assert!(close(sin_out, sin_in / 1.5));
        assert!(close(out.length(), 1.0));
    }

    #[test]
    fn near_zero_boundary() {
        assert!(vec3(0.0, 0.0, 0.0).near_zero());
        assert!(vec3(1e-9, -1e-9, 1e-9).near_zero());
        assert!(!vec3(1e-7, 0.0, 0.0).near_zero());
    }

    #[test]
    fn random_unit_vector_is_normalized() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..1000 {
            let v = Vec3::random_unit_vector(&mut rng);
            assert!((v.length() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn random_in_unit_disk_stays_flat_and_inside() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        for _ in 0..1000 {
            let p = Vec3::random_in_unit_disk(&mut rng);
            assert_eq!(p.z, 0.0);
            assert!(p.length_squared() < 1.0);
        }
    }
}

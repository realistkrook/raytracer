/// A closed range of scalars, used for the valid `t` window along a ray and
/// for clamping color components.
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub min: f64,
    pub max: f64,
}

impl Interval {
    /// The empty interval — nothing is inside it.
    pub const EMPTY: Interval = Interval {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };

    pub const fn new(min: f64, max: f64) -> Interval {
        Interval { min, max }
    }

    /// The tightest interval enclosing both inputs.
    pub fn enclosing(a: Interval, b: Interval) -> Interval {
        Interval::new(a.min.min(b.min), a.max.max(b.max))
    }

    pub fn size(&self) -> f64 {
        self.max - self.min
    }

    /// Exclusive of the endpoints — what ray intersection wants, so a surface
    /// exactly at `t_min` doesn't re-hit itself.
    pub fn surrounds(&self, x: f64) -> bool {
        self.min < x && x < self.max
    }

    pub fn clamp(&self, x: f64) -> f64 {
        x.clamp(self.min, self.max)
    }

    /// Widen by `delta` total (half on each side). Keeps degenerate bounding
    /// box slabs from being infinitely thin.
    pub fn expand(&self, delta: f64) -> Interval {
        let padding = delta / 2.0;
        Interval::new(self.min - padding, self.max + padding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrounds_excludes_the_endpoints() {
        let i = Interval::new(1.0, 3.0);
        assert!(i.surrounds(2.0));
        // Endpoints are out: a bounce leaving a surface at exactly t_min must
        // not count as hitting that same surface again.
        assert!(!i.surrounds(1.0));
        assert!(!i.surrounds(3.0));
        assert!(!i.surrounds(0.999));
        assert!(!i.surrounds(3.001));
    }

    #[test]
    fn empty_surrounds_nothing() {
        assert!(!Interval::EMPTY.surrounds(0.0));
        assert!(!Interval::EMPTY.surrounds(f64::INFINITY));
    }

    #[test]
    fn clamp_and_expand() {
        let i = Interval::new(0.0, 1.0);
        assert_eq!(i.clamp(-5.0), 0.0);
        assert_eq!(i.clamp(0.5), 0.5);
        assert_eq!(i.clamp(5.0), 1.0);

        let e = i.expand(0.2);
        assert!((e.min - -0.1).abs() < 1e-12);
        assert!((e.max - 1.1).abs() < 1e-12);
    }

    #[test]
    fn enclosing_spans_both() {
        let a = Interval::new(-1.0, 2.0);
        let b = Interval::new(1.0, 5.0);
        let c = Interval::enclosing(a, b);
        assert_eq!(c.min, -1.0);
        assert_eq!(c.max, 5.0);
    }
}

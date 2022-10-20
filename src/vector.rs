use float_eq::float_eq;

/// The type of a real-valued scalar
pub type Scalar = f64;

/// A mathy vector to do linear and geometric algebra with
#[derive(Debug)]
pub struct Vector<const DIMENSION: usize> {
    coordinates: [Scalar; DIMENSION],
}

/// A convenience type for 2D vectors
///
/// ```
/// use galg::vector::Vector2;
///
/// let my_vector2: Vector2 = Vector2::new([1.0, 2.0]);
/// ```
pub type Vector2 = Vector<2>;
/// A convenience type for 3D vectors
///
/// ```
/// use galg::vector::Vector3;
///
/// let my_vector3: Vector3 = Vector3::new([1.0, 2.0, 3.0]);
/// ```
pub type Vector3 = Vector<3>;
/// A convenience type for 4D vectors
///
/// ```
/// use galg::vector::Vector4;
///
/// let my_vector4: Vector4 = Vector4::new([1.0, 2.0, 3.0, 4.0]);
/// ```
pub type Vector4 = Vector<4>;

impl<const DIMENSION: usize> Vector<DIMENSION> {
    /// Make a vector whose elements are given
    pub fn new(coordinates: [Scalar; DIMENSION]) -> Self {
        Vector::<DIMENSION> { coordinates }
    }

    /// Get a vector whose elements are all `0.0`
    pub fn zero() -> Self {
        Vector::<DIMENSION> {
            coordinates: [0.0; DIMENSION],
        }
    }
}

impl<const DIMENSION: usize> PartialEq for Vector<DIMENSION> {
    /// This method tests for self and other values to be equal, and is used by ==.
    /// [Read more](https://doc.rust-lang.org/1.63.0/core/cmp/trait.PartialEq.html#tymethod.eq)
    ///
    /// This computes approximate equality with within a couple
    /// [ULP](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    fn eq(&self, other: &Self) -> bool {
        // an ULP tolerance is how many representable floats away another float can be
        const ULP_TOLERANCE: u64 = 2;

        self.coordinates
            .iter()
            .enumerate()
            .all(|(index, &coordinate)| {
                float_eq!(coordinate, other.coordinates[index], ulps <= ULP_TOLERANCE)
            })
    }
}

impl<const DIMENSION: usize> std::ops::Mul<Vector<DIMENSION>> for Scalar {
    type Output = Vector<DIMENSION>;

    fn mul(self, other: Self::Output) -> Self::Output {
        Vector::<DIMENSION>::new(
            other
                .coordinates
                .into_iter()
                .map(|coordinate| self * coordinate)
                .collect::<Vec<Scalar>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const DIMENSION: usize> std::ops::Add for Vector<DIMENSION> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Vector::<DIMENSION>::new(
            self.coordinates
                .iter()
                .enumerate()
                .map(|(index, coordinate)| coordinate + other.coordinates[index])
                .collect::<Vec<Scalar>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const DIMENSION: usize> std::ops::Sub for Vector<DIMENSION> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Vector::<DIMENSION>::new(
            self.coordinates
                .iter()
                .enumerate()
                .map(|(index, coordinate)| coordinate - other.coordinates[index])
                .collect::<Vec<Scalar>>()
                .try_into()
                .unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_under_scalar_multiplication() {
        // Macdonald Definition 2.1 V0 part 1
        assert_eq!(
            1.0 * Vector::<2>::new([1.0, 1.0]),
            Vector::<2>::new([1.0, 1.0])
        );
        assert_eq!(
            -1.0 * Vector::<2>::new([1.0, 1.0]),
            Vector::<2>::new([-1.0, -1.0])
        );
        assert_eq!(
            0.5 * Vector::<2>::new([2.0, -1.0]),
            Vector::<2>::new([1.0, -0.5])
        );
        assert_eq!(
            -2.0 * Vector::<2>::new([2.0, -1.0]),
            Vector::<2>::new([-4.0, 2.0])
        );
    }

    #[test]
    fn closed_under_vector_addition() {
        // Macdonald Definition 2.1 V0 part 2
        assert_eq!(
            Vector::<2>::new([1.0, 1.0]) + Vector::<2>::new([1.0, 1.0]),
            Vector::<2>::new([2.0, 2.0])
        );
        assert_eq!(
            Vector::<2>::new([-1.0, -1.0]) + Vector::<2>::new([-1.0, -1.0]),
            Vector::<2>::new([-2.0, -2.0])
        );
        assert_eq!(
            Vector::<2>::new([0.5, -8.0]) + Vector::<2>::new([-16.2, 0.8]),
            Vector::<2>::new([-15.7, -7.2])
        );
    }

    #[test]
    fn vector_addition_is_commutative() {
        // Macdonald Definition 2.1 V1
        assert_eq!(
            Vector::<2>::new([1.0, 1.0]) + Vector::<2>::new([2.0, 1.0]),
            Vector::<2>::new([2.0, 1.0]) + Vector::<2>::new([1.0, 1.0])
        );
        assert_eq!(
            Vector::<2>::new([-1.0, 10.0]) + Vector::<2>::new([666.6, 42.8]),
            Vector::<2>::new([666.6, 42.8]) + Vector::<2>::new([-1.0, 10.0])
        );
    }

    #[test]
    fn vector_addition_is_associative() {
        // Macdonald Definition 2.1 V2
        assert_eq!(
            (Vector::<2>::new([1.0, 1.0]) + Vector::<2>::new([2.0, 1.0]))
                + Vector::<2>::new([-10.1, 0.0]),
            Vector::<2>::new([1.0, 1.0])
                + (Vector::<2>::new([2.0, 1.0]) + Vector::<2>::new([-10.1, 0.0]))
        );
        assert_eq!(
            (Vector::<2>::new([-1.0, 10.0]) + Vector::<2>::new([666.6, 42.8]))
                + Vector::<2>::new([98.7, -273.15]),
            Vector::<2>::new([-1.0, 10.0])
                + (Vector::<2>::new([666.6, 42.8]) + Vector::<2>::new([98.7, -273.15]))
        );
    }

    #[test]
    fn zero_vector_is_the_additive_identity() {
        // Macdonald Definition 2.1 V3
        assert_eq!(
            Vector::<2>::new([1.0, 1.0]) + Vector::<2>::zero(),
            Vector::<2>::new([1.0, 1.0])
        );
        assert_eq!(
            Vector::<2>::new([-1.0, -1.0]) + Vector::<2>::zero(),
            Vector::<2>::new([-1.0, -1.0])
        );
        assert_eq!(
            Vector::<2>::new([98.7, -273.15]) + Vector::<2>::zero(),
            Vector::<2>::new([98.7, -273.15])
        );
    }

    #[test]
    fn scalar_multiplication_by_0_yields_the_zero_vector() {
        // Macdonald Definition 2.1 V4
        assert_eq!(0.0 * Vector::<2>::new([1.0, 1.0]), Vector::<2>::zero());
        assert_eq!(0.0 * Vector::<2>::new([-1.0, -1.0]), Vector::<2>::zero());
        assert_eq!(0.0 * Vector::<2>::new([98.7, -273.15]), Vector::<2>::zero());
    }

    #[test]
    fn scalar_1_is_the_multiplicative_identity() {
        // Macdonald Definition 2.1 V5
        assert_eq!(
            1.0 * Vector::<2>::new([1.0, 1.0]),
            Vector::<2>::new([1.0, 1.0])
        );
        assert_eq!(
            1.0 * Vector::<2>::new([-1.0, -1.0]),
            Vector::<2>::new([-1.0, -1.0])
        );
        assert_eq!(
            1.0 * Vector::<2>::new([98.7, -273.15]),
            Vector::<2>::new([98.7, -273.15])
        );
    }

    #[test]
    fn scalar_multiplication_is_associative() {
        // Macdonald Definition 2.1 V6
        assert_eq!(
            2.0 * (3.0 * Vector::<2>::new([1.0, 1.0])),
            (2.0 * 3.0) * Vector::<2>::new([1.0, 1.0])
        );
        assert_eq!(
            -3.8 * (22.2 * Vector::<2>::new([-1.0, -1.0])),
            (-3.8 * 22.2) * Vector::<2>::new([-1.0, -1.0])
        );
        assert_eq!(
            -98.7 * (-273.15 * Vector::<2>::new([98.7, -273.15])),
            (-98.7 * -273.15) * Vector::<2>::new([98.7, -273.15])
        );
    }

    #[test]
    fn scalar_multiplication_distributes_over_vector_addition() {
        // Macdonald Definition 2.1 V7
        assert_eq!(
            9.0 * (Vector::<2>::new([1.0, 1.0]) + Vector::<2>::new([2.0, 1.0])),
            9.0 * Vector::<2>::new([1.0, 1.0]) + 9.0 * Vector::<2>::new([2.0, 1.0])
        );
        assert_eq!(
            -12.7 * (Vector::<2>::new([-1.0, 10.0]) + Vector::<2>::new([666.6, 42.8])),
            -12.7 * Vector::<2>::new([-1.0, 10.0]) + (-12.7) * Vector::<2>::new([666.6, 42.8])
        );
    }

    #[test]
    fn scalar_multiplication_distributes_over_scalar_addition() {
        // Macdonald Definition 2.1 V8
        assert_eq!(
            (2.0 + 3.0) * Vector::<2>::new([1.0, 1.0]),
            2.0 * Vector::<2>::new([1.0, 1.0]) + 3.0 * Vector::<2>::new([1.0, 1.0])
        );
        assert_eq!(
            (-3.8 + 22.2) * Vector::<2>::new([-1.0, -1.0]),
            -3.8 * Vector::<2>::new([-1.0, -1.0]) + 22.2 * Vector::<2>::new([-1.0, -1.0])
        );
        assert_eq!(
            (-98.7 + -273.15) * Vector::<2>::new([98.7, -273.15]),
            -98.7 * Vector::<2>::new([98.7, -273.15])
                + (-273.15) * Vector::<2>::new([98.7, -273.15])
        );
    }

    #[test]
    fn vector_subtraction_is_like_vector_addition() {
        assert_eq!(
            Vector::<2>::new([2.0, 2.0]) - Vector::<2>::new([1.0, 3.0]),
            Vector::<2>::new([2.0, 2.0]) + (-1.0) * Vector::<2>::new([1.0, 3.0])
        )
    }

    #[test]
    fn other_dimension_vectors() {
        Vector::<0>::new([]); // it doesn't do much, but I see no harm in having it
        Vector::<1>::new([89.3]);
        Vector::<3>::new([3.0, 2.0, 1.0]);
        Vector::<256>::new([-78.2; 256]);
    }

    #[test]
    fn a_vector_plus_its_additive_inverse_is_zero() {
        assert_eq!(
            Vector::<2>::new([1.0, 1.0]) + Vector::<2>::new([-1.0, -1.0]),
            Vector::<2>::zero()
        );
    }
}

use float_eq::float_eq;

/// The type of a real-valued scalar
///
/// Multiplication with very large values is hairy, so I only guarantee
/// things to work for `abs(Scalar) < 10^150`.
pub type Scalar = f64;

/// A mathy vector to do linear and geometric algebra with
#[derive(Debug, Clone)]
pub struct Vector<const N: usize> {
    coordinates: [Scalar; N],
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

impl<const N: usize> Vector<N> {
    /// Make a vector whose elements are given
    pub fn new(coordinates: [Scalar; N]) -> Self {
        Vector::<N> { coordinates }
    }

    /// Get a vector whose elements are all `0.0`
    pub fn zero() -> Self {
        Vector::<N> {
            coordinates: [0.0; N],
        }
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    fn eq(&self, other: &Self) -> bool {
        const TOLERANCE: f64 = 1.0e-7;

        self.coordinates
            .iter()
            .enumerate()
            .all(|(index, &coordinate)| {
                float_eq!(coordinate, other.coordinates[index], rmax <= TOLERANCE)
            })
    }
}

impl<const N: usize> std::ops::Mul<Vector<N>> for Scalar {
    type Output = Vector<N>;

    fn mul(self, other: Self::Output) -> Self::Output {
        Vector::<N>::new(other.coordinates.map(|coordinate| self * coordinate))
    }
}

impl<const N: usize> std::ops::Add for Vector<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut sum = [0.0; N];
        for i in 0..N {
            sum[i] = self.coordinates[i] + other.coordinates[i];
        }
        Vector::<N>::new(sum)
    }
}

impl<const N: usize> std::ops::Sub for Vector<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut difference = [0.0; N];
        for i in 0..N {
            difference[i] = self.coordinates[i] - other.coordinates[i];
        }
        Vector::<N>::new(difference)
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
    fn other_dimension_vectors() {
        Vector::<0>::new([]); // it doesn't do much, but I see no harm in having it
        Vector::<1>::new([89.3]);
        Vector::<3>::new([3.0, 2.0, 1.0]);
        Vector::<256>::new([-78.2; 256]);
    }

    use proptest::prelude::*;

    fn arbitrary_scalar() -> impl Strategy<Value = Scalar> {
        // any larger a range and you start to run into annoying
        // floating-point issues
        -1.0e+150_f64..1.0e+150_f64
    }

    fn arbitrary_vector<const N: usize>() -> BoxedStrategy<Vector<N>> {
        proptest::array::uniform::<_, N>(arbitrary_scalar())
            .prop_map(|array| Vector::<N>::new(array))
            .boxed()
    }

    proptest! {
        #[test]
        fn vector_addition_is_commutative(
            vector0 in arbitrary_vector::<3>(),
            vector1 in arbitrary_vector::<3>(),
        ) {
            // Macdonald Definition 2.1 V1
            prop_assert_eq!(
                vector0.clone() + vector1.clone(),
                vector1 + vector0
            );
        }

      #[test]
        fn vector_addition_is_associative(
            vector0 in arbitrary_vector::<3>(),
            vector1 in arbitrary_vector::<3>(),
            vector2 in arbitrary_vector::<3>()
        ) {
            // Macdonald Definition 2.1 V2
            prop_assert_eq!(
                (vector0.clone() + vector1.clone()) + vector2.clone(),
                vector0 + (vector1 + vector2)
            );
        }

        #[test]
        fn zero_vector_is_the_additive_identity(vector in arbitrary_vector::<3>()) {
            // Macdonald Definition 2.1 V3
            prop_assert_eq!(
                vector.clone() + Vector::<3>::zero(),
                vector
            );
        }

        #[test]
        fn scalar_multiplication_by_0_yields_the_zero_vector(
            vector in arbitrary_vector::<3>()
        ) {
            // Macdonald Definition 2.1 V4
            prop_assert_eq!(0.0 * vector, Vector::<3>::zero());
        }

        #[test]
        fn scalar_1_is_the_multiplicative_identity(vector in arbitrary_vector::<3>()) {
            // Macdonald Definition 2.1 V5
            prop_assert_eq!(
                1.0 * vector.clone(),
                vector
            );
        }

        #[test]
        fn scalar_multiplication_is_associative(
            a in arbitrary_scalar(),
            b in arbitrary_scalar(),
            vector in arbitrary_vector::<3>()
        ) {
            // Macdonald Definition 2.1 V6
            prop_assert_eq!(
                a * (b * vector.clone()),
                (a * b) * vector
            );
        }

        #[test]
        fn scalar_multiplication_distributes_over_vector_addition(
            vector0 in arbitrary_vector::<3>(),
            vector1 in arbitrary_vector::<3>(),
            a in arbitrary_scalar()
        ) {
            // Macdonald Definition 2.1 V7
            prop_assert_eq!(
                a * (vector0.clone() + vector1.clone()),
                a * vector0 + a * vector1
            );
        }

        #[test]
        fn scalar_multiplication_distributes_over_scalar_addition(
            a in arbitrary_scalar(),
            b in arbitrary_scalar(),
            vector in arbitrary_vector::<3>()
        ) {
            // Macdonald Definition 2.1 V8
            prop_assert_eq!(
                (a + b) * vector.clone(),
                a * vector.clone() + b * vector
            );
        }

        #[test]
        fn vector_subtraction_is_like_vector_addition(
            vector0 in arbitrary_vector::<3>(),
            vector1 in arbitrary_vector::<3>()
        ) {
            prop_assert_eq!(
                vector0.clone() - vector1.clone(),
                vector0 + (-1.0) * vector1
            )
        }

        #[test]
        fn a_vector_plus_its_additive_inverse_is_zero(vector in arbitrary_vector::<3>()) {
            prop_assert_eq!(
                vector.clone() + (-1.0) * vector,
                Vector::<3>::zero()
            );
        }
    }
}

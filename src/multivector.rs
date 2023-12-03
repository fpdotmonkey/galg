// pub struct Multivector<const N: usize> {
//     a: [f64; N],
//     b: [f64; 2 * N],
// }

const fn factorial(n: usize) -> Option<usize> {
    if n > 12 {
        // integer overflow
        return None;
    }
    let mut number = n;
    let mut result = 1;
    // for loops aren't allowed in a const context
    while number > 0 {
        result *= number;
        number -= 1;
    }
    Some(result)
}

const fn binomial_coefficient(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let mut result: usize = 1;
    let mut i = 1;
    while i <= k {
        result *= (n + 1 - i) / i;
        i += 1;
    }
    Some(result)
}

const fn multivector_shape<const N: usize>() -> (usize, [usize; N]) {
    (2usize.pow(N as u32), [N; N])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // m = Multivector::<2> {a: [1., 1.], b: [1., 1., 1., 1.]};
        multivector_shape::<3>();
    }

    #[test]
    fn factorial_is_correct() {
        assert_eq!(factorial(0), Some(1));
        assert_eq!(factorial(1), Some(1));
        assert_eq!(factorial(2), Some(2));
        assert_eq!(factorial(3), Some(6));
        assert_eq!(factorial(4), Some(24));
        assert_eq!(factorial(5), Some(120));
        assert_eq!(factorial(6), Some(720));
        assert_eq!(factorial(7), Some(5040));
        assert_eq!(factorial(8), Some(40320));
        assert_eq!(factorial(9), Some(362880));
        assert_eq!(factorial(10), Some(3628800));
        assert_eq!(factorial(11), Some(39916800));
        assert_eq!(factorial(12), Some(479001600));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn factorial_is_last_factorial_times_n(n in 1_usize..) {
            if let (Some(a), Some(b)) = (factorial(n), factorial(n - 1)) {
                prop_assert_eq!(a, n * b)
            }
        }

        #[test]
        fn binomial_coefficient_related_to_nearby(n in 1_usize.., k in 1_usize..) {
            prop_assert_eq!(
                binomial_coefficient(n, k).unwrap(),
                n / k * binomial_coefficient(n - 1, k - 1).unwrap()
            );
        }

        #[test]
        fn pascals_triangle_rows_add_to_power_of_2(n in 0_usize..=32) {
            prop_assert_eq!(
                (0..=n).map(|k| binomial_coefficient(n, k).unwrap()).sum::<usize>(),
                2_usize.pow(n as u32)
            );
        }
    }
}

use neural_core::{Semiring, Vector};

#[derive(Debug, Clone, Default)]
pub struct BooleanSemiring;

impl Semiring<bool> for BooleanSemiring {
    fn add(a: bool, b: bool) -> bool { a || b }
    fn mul(a: bool, b: bool) -> bool { a && b }
    fn zero() -> bool { false }
    fn one() -> bool { true }
}

#[derive(Debug, Clone)]
pub struct DenseVector<T> {
    data: Vec<T>,
    default_value: T,
}

impl<T: Default + Clone> Default for DenseVector<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            default_value: T::default(),
        }
    }
}

impl<T: Clone + PartialEq> DenseVector<T> {
    pub fn new(size: usize, default_value: T) -> Self {
        Self {
            data: vec![default_value.clone(); size],
            default_value,
        }
    }
}

impl<T: Clone + Copy + PartialEq> Vector<T> for DenseVector<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<T> {
        self.data.get(index).cloned()
    }

    fn set(&mut self, index: usize, value: T) {
        if index < self.data.len() {
            self.data[index] = value;
        }
    }

    fn iter_active(&self) -> impl Iterator<Item = (usize, T)> {
        self.data.iter().enumerate()
            .filter(|(_, v)| *v != &self.default_value)
            .map(|(i, v)| (i, *v))
    }
}

// Default impl for Matrix trait for any type that implements necessary logic?
// No, Matrix is usually implemented by CsrMatrix or AdjacencyMatrix.
// DenseMatrix could be here.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_vector() {
        let mut v = DenseVector::new(5, 0.0);
        v.set(0, 1.0);
        v.set(2, 3.5);

        assert_eq!(v.get(0), Some(1.0));
        assert_eq!(v.get(1), Some(0.0));
        assert_eq!(v.get(2), Some(3.5));

        let active: Vec<_> = v.iter_active().collect();
        assert_eq!(active, vec![(0, 1.0), (2, 3.5)]);
    }

    #[test]
    fn test_semiring_boolean() {
        assert_eq!(BooleanSemiring::add(true, false), true);
        assert_eq!(BooleanSemiring::mul(true, false), false);
        assert_eq!(BooleanSemiring::zero(), false);
        assert_eq!(BooleanSemiring::one(), true);
    }
}

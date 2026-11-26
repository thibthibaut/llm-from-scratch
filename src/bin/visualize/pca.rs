pub const PCA_COMPONENTS: usize = 16;
use tch::{Kind, Tensor};

/// Mean and top-k principal directions (largest variance first).
pub struct Pca {
    mean: Vec<f32>,
    basis: Vec<Vec<f32>>,
}

impl Pca {
    /// `data` is row-major [n_rows, dim].
    pub fn fit(data: &[f32], n_rows: usize, dim: usize, k: usize) -> Pca {
        // f64 internally so the eigenvectors stay orthonormal to 1e-4.
        let x = Tensor::from_slice(data)
            .view([n_rows as i64, dim as i64])
            .to_kind(Kind::Double);
        let mean = x.mean_dim(0, false, Kind::Double);
        let c = &x - &mean;
        let cov = c.tr().matmul(&c) / (n_rows as f64 - 1.0);
        let (_, vecs) = cov.linalg_eigh("L"); // eigenvalues ascending, vectors as columns
        let top = vecs
            .narrow(1, dim as i64 - k as i64, k as i64)
            .flip([1])
            .tr();
        let flat =
            |t: &Tensor| Vec::<f32>::try_from(t.to_kind(Kind::Float).flatten(0, -1)).unwrap();
        Pca {
            mean: flat(&mean),
            basis: flat(&top).chunks(dim).map(<[f32]>::to_vec).collect(),
        }
    }

    /// basis · v, with no mean subtraction: for vectors that are not positions.
    pub fn project_direction(&self, v: &[f32]) -> Vec<f32> {
        self.basis
            .iter()
            .map(|b| b.iter().zip(v).map(|(b, v)| b * v).sum())
            .collect()
    }

    /// (v − mean) · basisᵀ → k values.
    pub fn project(&self, v: &[f32]) -> Vec<f32> {
        self.basis
            .iter()
            .map(|b| {
                b.iter()
                    .zip(v)
                    .zip(&self.mean)
                    .map(|((b, v), m)| b * (v - m))
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn basis_is_orthonormal() {
        let mut rng = StdRng::seed_from_u64(1);
        let data: Vec<f32> = (0..200 * 20).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let p = Pca::fit(&data, 200, 20, 16);
        for a in 0..16 {
            for b in 0..16 {
                let d: f32 = p.basis[a].iter().zip(&p.basis[b]).map(|(x, y)| x * y).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-4, "{a},{b}: {d}");
            }
        }
    }

    #[test]
    fn first_component_finds_dominant_axis() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut data = Vec::new();
        for _ in 0..500 {
            let t: f32 = rng.gen_range(-1.0..1.0);
            for j in 0..8 {
                let noise: f32 = rng.gen_range(-1.0..1.0);
                data.push(0.1 * noise + if j == 3 { 10.0 * t } else { 0.0 });
            }
        }
        let p = Pca::fit(&data, 500, 8, 2);
        assert!(p.basis[0][3].abs() > 0.99);
    }

    #[test]
    fn mean_projects_to_zero() {
        let mut rng = StdRng::seed_from_u64(3);
        let (n, dim) = (100, 6);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(0.0..5.0)).collect();
        let means: Vec<f32> = (0..dim)
            .map(|j| (0..n).map(|i| data[i * dim + j]).sum::<f32>() / n as f32)
            .collect();
        let p = Pca::fit(&data, n, dim, 4);
        let out = p.project(&means);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|x| x.abs() < 1e-4));
    }

    #[test]
    fn project_direction_skips_mean() {
        let data: Vec<f32> = (0..40).map(|i| 5.0 + ((i * 7) % 11) as f32).collect();
        let p = Pca::fit(&data, 10, 4, 2);
        let v = [1.0, -2.0, 0.5, 3.0];
        let (d, c) = (p.project_direction(&v), p.project(&v));
        for (k, b) in p.basis.iter().enumerate() {
            let want: f32 = b.iter().zip(&v).map(|(b, v)| b * v).sum();
            let mean_part: f32 = b.iter().zip(&p.mean).map(|(b, m)| b * m).sum();
            assert!((d[k] - want).abs() < 1e-5);
            assert!(mean_part.abs() > 0.1);
            assert!((d[k] - c[k] - mean_part).abs() < 1e-4);
        }
    }
}

//! Next-token selection for `generate-text`: greedy argmax, or sampling from the model's
//! distribution reshaped by temperature, top-k and top-p.

use rand::Rng;
use rand::distributions::{Distribution, WeightedIndex};

/// How `generate-text` picks each next token.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum DecodingMode {
    /// Always take the most likely token (argmax).
    #[default]
    Greedy,
    /// Draw from the model's distribution, shaped by --temperature, --top-k and --top-p.
    Sample,
}

/// Sampling settings, built by chaining `with_*` calls:
///
/// ```ignore
/// let sampler = Sampler::default().with_temperature(0.8).with_top_k(50).with_top_p(0.9);
/// let id = sampler.sample(&logits, &mut rng);
/// ```
///
/// `Sampler::default()` samples the model's distribution as is (temperature 1, no filtering).
/// `sample` applies the settings in a fixed order, the same as HF `transformers` and llama.cpp:
/// divide the logits by the temperature, softmax, keep the `top_k` most likely tokens, keep the
/// smallest prefix of those whose probability mass reaches `top_p`, then draw from what is left.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sampler {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
}

const INVALID: &str = "cannot sample: logits are not a valid distribution (NaN or empty)";

impl Sampler {
    /// Below 1 sharpens the distribution towards the argmax, above 1 flattens it.
    pub fn with_temperature(self, t: f32) -> Self {
        assert!(
            t.is_finite() && t > 0.0,
            "temperature must be a finite number > 0, got {t}"
        );
        Self {
            temperature: t,
            ..self
        }
    }

    /// Only the `k` most likely tokens can be drawn. `with_top_k(1)` is greedy.
    pub fn with_top_k(self, k: usize) -> Self {
        assert!(k >= 1, "top_k must be >= 1");
        Self {
            top_k: Some(k),
            ..self
        }
    }

    /// Only the smallest set of most likely tokens whose total probability reaches `p` can be
    /// drawn (nucleus sampling). The most likely token is always kept.
    pub fn with_top_p(self, p: f32) -> Self {
        assert!(p > 0.0 && p <= 1.0, "top_p must be in (0, 1], got {p}");
        Self {
            top_p: Some(p),
            ..self
        }
    }

    /// The tokens `sample` can draw, with their renormalized probabilities, most likely first.
    /// Ties keep the lower id first.
    pub fn distribution(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        assert!(
            !logits.is_empty() && !logits.iter().any(|l| l.is_nan()),
            "{INVALID}"
        );

        // Subtracting the max keeps exp() from overflowing at low temperatures.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut ranked: Vec<(usize, f32)> = logits
            .iter()
            .map(|&l| ((l - max) / self.temperature).exp())
            .enumerate()
            .collect();
        // Stable sort: tied tokens keep the lower id first.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        if let Some(k) = self.top_k {
            ranked.truncate(k);
        }
        if let Some(p) = self.top_p {
            let total: f32 = ranked.iter().map(|&(_, w)| w).sum();
            let mut cumulative = 0.0;
            let keep = ranked
                .iter()
                .position(|&(_, w)| {
                    cumulative += w / total;
                    cumulative >= p
                })
                // f32 rounding can leave the full sum a hair under p = 1.0: keep everything.
                .map_or(ranked.len(), |i| i + 1);
            ranked.truncate(keep);
        }

        let total: f32 = ranked.iter().map(|&(_, w)| w).sum();
        for r in &mut ranked {
            r.1 /= total;
        }
        ranked
    }

    /// Draw one token id from the last position's raw `logits` (length = vocab size).
    pub fn sample<R: Rng + ?Sized>(&self, logits: &[f32], rng: &mut R) -> usize {
        let ranked = self.distribution(logits);
        let dist = WeightedIndex::new(ranked.iter().map(|&(_, p)| p)).expect(INVALID);
        ranked[dist.sample(rng)].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    /// Logits whose softmax is exactly `p`, so expected frequencies stay readable.
    fn logits(p: &[f32]) -> Vec<f32> {
        p.iter().map(|x| x.ln()).collect()
    }

    fn freq(s: Sampler, l: &[f32], n: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut c = vec![0usize; l.len()];
        for _ in 0..n {
            c[s.sample(l, &mut rng)] += 1;
        }
        c.iter().map(|&x| x as f32 / n as f32).collect()
    }

    #[test]
    fn distribution_sums_to_one() {
        let d = Sampler::default().distribution(&logits(&[0.5, 0.3, 0.2]));
        assert_eq!(d.iter().map(|&(i, _)| i).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!((d.iter().map(|&(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((d[0].1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn distribution_applies_top_k_and_top_p() {
        assert_eq!(
            Sampler::default()
                .with_top_k(2)
                .distribution(&logits(&[0.5, 0.3, 0.2]))
                .len(),
            2
        );
        let d = Sampler::default()
            .with_top_p(0.85)
            .distribution(&logits(&[0.6, 0.3, 0.1]));
        assert_eq!(d.len(), 2);
        assert!((d[0].1 - 0.6 / 0.9).abs() < 1e-5);
    }

    #[test]
    fn sample_draws_from_distribution() {
        let s = Sampler::default().with_top_k(3).with_temperature(1.5);
        let l = logits(&[0.4, 0.3, 0.2, 0.1]);
        let kept: Vec<usize> = s.distribution(&l).iter().map(|&(i, _)| i).collect();
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..500 {
            assert!(kept.contains(&s.sample(&l, &mut rng)));
        }
    }

    #[test]
    fn default_matches_distribution() {
        let f = freq(Sampler::default(), &logits(&[0.7, 0.2, 0.1]), 20_000, 42);
        assert!(
            (f[0] - 0.7).abs() < 0.02 && (f[2] - 0.1).abs() < 0.02,
            "{f:?}"
        );
    }

    #[test]
    fn top_k_1_is_argmax() {
        let f = freq(
            Sampler::default().with_top_k(1),
            &logits(&[0.2, 0.5, 0.3]),
            500,
            0,
        );
        assert_eq!(f, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn top_k_2_never_picks_third() {
        let f = freq(
            Sampler::default().with_top_k(2),
            &logits(&[0.5, 0.3, 0.2]),
            5_000,
            1,
        );
        assert_eq!(f[2], 0.0);
        assert!((f[0] - 0.625).abs() < 0.03, "{f:?}"); // 0.5 / 0.8 after renormalizing
    }

    #[test]
    fn top_p_cuts_tail() {
        let f = freq(
            Sampler::default().with_top_p(0.85),
            &logits(&[0.6, 0.3, 0.1]),
            5_000,
            2,
        );
        assert_eq!(f[2], 0.0); // 0.6 + 0.3 = 0.9 >= 0.85
        assert!(f[1] > 0.0);
    }

    #[test]
    fn top_p_keeps_at_least_one() {
        let f = freq(
            Sampler::default().with_top_p(0.1),
            &logits(&[0.6, 0.3, 0.1]),
            500,
            3,
        );
        assert_eq!(f[0], 1.0);
    }

    #[test]
    fn low_temperature_is_stable() {
        let f = freq(
            Sampler::default().with_temperature(0.1),
            &[80.0, 79.0, 0.0],
            500,
            4,
        );
        assert_eq!(f[0], 1.0); // (80-79)/0.1 = 10 nats apart, and no NaN
    }

    #[test]
    fn high_temperature_flattens() {
        let f = freq(
            Sampler::default().with_temperature(100.0),
            &logits(&[0.9, 0.05, 0.05]),
            20_000,
            5,
        );
        assert!(f.iter().all(|&x| (x - 1.0 / 3.0).abs() < 0.03), "{f:?}");
    }

    #[test]
    fn top_k_then_top_p_order() {
        // top_k(2) leaves [0.4, 0.35], renormalized to [0.533, 0.467]; top_p(0.5) then keeps only
        // index 0. Doing top-p first would keep {0, 1} (0.4 < 0.5 <= 0.75), so this pins the order.
        let s = Sampler::default().with_top_k(2).with_top_p(0.5);
        assert_eq!(
            freq(s, &logits(&[0.4, 0.35, 0.25]), 500, 6),
            vec![1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn same_seed_same_draws() {
        let s = Sampler::default().with_temperature(0.8).with_top_p(0.9);
        let l = logits(&[0.25; 4]);
        let draw = |seed| {
            let mut r = StdRng::seed_from_u64(seed);
            (0..50).map(|_| s.sample(&l, &mut r)).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
    }

    #[test]
    #[should_panic(expected = "temperature must be")]
    fn rejects_zero_temperature() {
        Sampler::default().with_temperature(0.0);
    }

    #[test]
    #[should_panic(expected = "top_k must be")]
    fn rejects_zero_top_k() {
        Sampler::default().with_top_k(0);
    }

    #[test]
    #[should_panic(expected = "top_p must be")]
    fn rejects_top_p_above_one() {
        Sampler::default().with_top_p(1.5);
    }

    #[test]
    #[should_panic(expected = "cannot sample")]
    fn rejects_nan_logits() {
        Sampler::default().sample(&[f32::NAN, 0.0], &mut StdRng::seed_from_u64(0));
    }
}

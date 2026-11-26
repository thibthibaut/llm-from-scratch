//! KV-cache comparison: tag each position's K/V as read, written or recomputed
//! by comparing this pass with the previous one.
use llm_from_scratch::model::Trace;
use llm_from_scratch::tokenizer::Token;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheTag {
    Read,
    Write,
    Recomputed,
}

type Heads = Vec<Vec<Vec<f32>>>; // [heads][seq][head_dim]

/// Tokens plus every layer's K and V from one forward pass.
#[derive(Clone, Debug, Default)]
pub struct KvSnapshot {
    tokens: Vec<Token>,
    k: Vec<Heads>,
    v: Vec<Heads>,
}

impl KvSnapshot {
    pub fn new(tokens: &[Token], trace: &Trace) -> KvSnapshot {
        KvSnapshot {
            tokens: tokens.to_vec(),
            k: trace.blocks.iter().map(|b| b.k.clone()).collect(),
            v: trace.blocks.iter().map(|b| b.v.clone()).collect(),
        }
    }
}

/// True when every head's row at `pos` matches within 1e-4 * max(1, max|row|).
fn rows_match(a: &Heads, b: &Heads, pos: usize) -> bool {
    a.iter().zip(b).all(|(ha, hb)| {
        let (ra, rb) = (&ha[pos], &hb[pos]);
        let scale = ra.iter().fold(1f32, |m, x| m.max(x.abs()));
        ra.iter()
            .zip(rb)
            .all(|(x, y)| (x - y).abs() <= 1e-4 * scale)
    })
}

/// tags[layer][position] for the new pass. Layer index is 0-based.
pub fn cache_tags(prev: Option<&KvSnapshot>, now: &KvSnapshot) -> Vec<Vec<CacheTag>> {
    let p = prev.map_or(0, |pr| {
        pr.tokens
            .iter()
            .zip(&now.tokens)
            .take_while(|(a, b)| a == b)
            .count()
    });
    (0..now.k.len())
        .map(|l| {
            (0..now.tokens.len())
                .map(|i| match prev {
                    Some(pr) if i < p => {
                        if rows_match(&pr.k[l], &now.k[l], i) && rows_match(&pr.v[l], &now.v[l], i)
                        {
                            CacheTag::Read
                        } else {
                            CacheTag::Recomputed
                        }
                    }
                    _ => CacheTag::Write,
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::CacheTag::{Read as R, Recomputed as Rc, Write as W};
    use super::*;

    fn toks(t: &[u32]) -> Vec<Token> {
        t.iter().map(|&x| Token(x)).collect()
    }

    /// 2 layers, 2 heads, head_dim 3; values depend only on (layer, head, pos, dim).
    fn snap_with(t: &[u32], f: impl Fn(usize, usize, usize, usize) -> f32) -> KvSnapshot {
        let mk = |off: f32| -> Vec<Heads> {
            (0..2)
                .map(|l| {
                    (0..2)
                        .map(|h| {
                            (0..t.len())
                                .map(|p| (0..3).map(|d| f(l, h, p, d) + off).collect())
                                .collect()
                        })
                        .collect()
                })
                .collect()
        };
        KvSnapshot {
            tokens: toks(t),
            k: mk(0.0),
            v: mk(0.25),
        }
    }

    fn snap(t: &[u32]) -> KvSnapshot {
        snap_with(t, |l, h, p, d| {
            (l * 100 + h * 10 + p) as f32 + d as f32 * 0.1
        })
    }

    #[test]
    fn first_pass_is_all_write() {
        assert_eq!(cache_tags(None, &snap(&[1, 2])), vec![vec![W, W]; 2]);
    }

    #[test]
    fn append_reads_prefix_writes_new() {
        let t = cache_tags(Some(&snap(&[1, 2])), &snap(&[1, 2, 3]));
        assert_eq!(t, vec![vec![R, R, W]; 2]);
    }

    #[test]
    fn slid_window_is_all_write() {
        let t = cache_tags(Some(&snap(&[1, 2, 3])), &snap(&[2, 3, 4]));
        assert_eq!(t, vec![vec![W, W, W]; 2]);
    }

    #[test]
    fn edit_keeps_prefix_reads() {
        let t = cache_tags(Some(&snap(&[1, 2, 3])), &snap(&[1, 2, 9, 4]));
        assert_eq!(t, vec![vec![R, R, W, W]; 2]);
    }

    #[test]
    fn mismatch_is_recomputed() {
        let prev = snap(&[1, 2]);
        let mut now = snap(&[1, 2, 3]);
        now.v[1][1][0][0] += 0.5;
        let t = cache_tags(Some(&prev), &now);
        assert_eq!(t[0], vec![R, R, W]);
        assert_eq!(t[1], vec![Rc, R, W]);
    }

    #[test]
    fn tolerance_is_relative() {
        let big = |l: usize, h: usize, p: usize, d: usize| 1000.0 + (l + h + p + d) as f32;
        let prev = snap_with(&[1, 2], big);
        let mut now = snap_with(&[1, 2], big);
        now.k[0][0][0][0] += 0.05;
        assert_eq!(cache_tags(Some(&prev), &now)[0], vec![R, R]);
        now.k[0][0][0][0] += 0.15; // total 0.2 > 1e-4 * 1000
        assert_eq!(cache_tags(Some(&prev), &now)[0], vec![Rc, R]);
    }

    #[test]
    fn unchanged_rerun_is_all_read() {
        let t = cache_tags(Some(&snap(&[1, 2])), &snap(&[1, 2]));
        assert_eq!(t, vec![vec![R, R]; 2]);
    }
}

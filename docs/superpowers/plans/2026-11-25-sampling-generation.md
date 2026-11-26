# Configurable Sampler for `generate-text` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `generate-text` draw each next token from the model's distribution (`--decoding sample`), with temperature, top-k and top-p settings. Today's argmax stays the default (`--decoding greedy`).

**Architecture:** A new `src/sampling.rs` holds a `DecodingMode` enum and a `Sampler` value type with a consuming builder: `Sampler::default().with_temperature(0.8).with_top_k(50).with_top_p(0.9)`. `Sampler::sample(&self, logits, rng)` works on the raw last-position logits as a `&[f32]` in plain Rust. Vocabulary is 32k, so no tensor ops are needed. It draws with `rand::distributions::WeightedIndex` (rand 0.8, already a dependency). `run_generate_text` in `src/main.rs` builds one `Sampler` from the CLI flags. Each step it passes the logits and offers the drawn token as the Enter default. The top-5 display (model probabilities at temperature 1) and the manual 1-5 / `q` picks stay unchanged in both modes.

**Tech Stack:** Rust, burn 0.21 (`LibTorch<f32>`), clap 4 derive (`ValueEnum`), rand 0.8 (`StdRng`, `WeightedIndex`).

**Spec:** User requests (2026-11-25): "a new method for a true sampling of the output distribution. Keep the greedy one." Then: "include temperature, top-k and top-p as settings for the sampler … builder pattern … `Sampler.with_temperature(0.8).with_topp(…)`".

## Global Constraints

- Greedy stays the default. `generate-text` with no new flags must print exactly what it prints today.
- `Sampler::default()` is true sampling: temperature `1.0`, no top-k, no top-p.
- The builder methods are `with_temperature(f32)`, `with_top_k(usize)` and `with_top_p(f32)`. Each takes `self` and returns `Self`.
- Filter order in `sample` is fixed, as in HF `transformers` and llama.cpp:
  1. divide the logits by the temperature;
  2. apply softmax;
  3. apply top-k;
  4. apply top-p;
  5. draw from the renormalized remainder.
- Add no new dependency.
- The tokenizer still comes from `config.json`. Add no tokenizer flag.

## Review Focus

- A sampler flag passed with `--decoding greedy` (or with no `--decoding`) must be rejected with an error, not ignored. Pinned by Task 2 Step 4.
- Non-interactive stdin (piped, or EOF): Enter/EOF must accept the *sampled* token in sample mode. Today EOF does `break 1`, which means argmax. Pinned by Task 2 Step 4.
- The sampled token is not in the top 5 (likely at temperature > 1): still print it, and do not index `top5` with it. Pinned by Task 2 Step 4.
- A top-p value smaller than the top token's probability must still keep that one token, never an empty set. Pinned by `top_p_keeps_at_least_one`.
- Large logits (e.g. 80.0) at a low temperature (0.1) must not overflow to `inf`/NaN. Subtract the max before `exp`. Pinned by `low_temperature_is_stable`.

---

### Task 1: `Sampler` and `DecodingMode`

**Files:**
- Create: `src/sampling.rs`
- Modify: `src/main.rs:52-57` (add `mod sampling;`)

**Interfaces:**
- Produces:
  - `#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, clap::ValueEnum)] pub enum DecodingMode { #[default] Greedy, Sample }`. Document each variant with a `///` line, because clap shows these lines as help text.
  - `#[derive(Copy, Clone, Debug, PartialEq)] pub struct Sampler { temperature: f32, top_k: Option<usize>, top_p: Option<f32> }`. Its fields are private. `impl Default` gives `1.0, None, None`.
  - `pub fn with_temperature(self, t: f32) -> Self`. Panics unless `t.is_finite() && t > 0.0`, with message `"temperature must be a finite number > 0, got {t}"`.
  - `pub fn with_top_k(self, k: usize) -> Self`. Panics if `k == 0`, with message `"top_k must be >= 1"`.
  - `pub fn with_top_p(self, p: f32) -> Self`. Panics unless `p > 0.0 && p <= 1.0`, with message `"top_p must be in (0, 1], got {p}"`.
  - `pub fn sample<R: rand::Rng + ?Sized>(&self, logits: &[f32], rng: &mut R) -> usize`. Returns a vocabulary index. Panics with `"cannot sample: logits are not a valid distribution (NaN or empty)"` when the logits contain NaN or are empty.

Algorithm for `sample` (the signature does not fix it):
1. Compute `p_i ∝ exp((l_i - max_l) / T)`.
2. Sort the indices by `p` in descending order. Use a stable sort, so ties keep the lower index.
3. If `top_k` is set, truncate to the first `k`.
4. If `top_p` is set, normalize the remaining weights. Then keep the shortest prefix whose cumulative sum is `>= p`; that prefix is always at least 1 entry.
5. Use `WeightedIndex` over the kept weights, then map back to the original index. `WeightedIndex` normalizes, so do not renormalize by hand.

- [ ] **Step 1: Write the failing tests** in `#[cfg(test)] mod tests` inside `src/sampling.rs`. Write the logits as `ln(prob)`, so the expected probabilities are readable.

```rust
use rand::{SeedableRng, rngs::StdRng};
fn logits(p: &[f32]) -> Vec<f32> { p.iter().map(|x| x.ln()).collect() }
fn freq(s: Sampler, l: &[f32], n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = vec![0usize; l.len()];
    for _ in 0..n { c[s.sample(l, &mut rng)] += 1; }
    c.iter().map(|&x| x as f32 / n as f32).collect()
}

#[test] fn default_matches_distribution() {
    let f = freq(Sampler::default(), &logits(&[0.7, 0.2, 0.1]), 20_000, 42);
    assert!((f[0] - 0.7).abs() < 0.02 && (f[2] - 0.1).abs() < 0.02, "{f:?}");
}
#[test] fn top_k_1_is_argmax() {
    assert_eq!(freq(Sampler::default().with_top_k(1), &logits(&[0.2, 0.5, 0.3]), 500, 0), vec![0.0, 1.0, 0.0]);
}
#[test] fn top_k_2_never_picks_third() {
    let f = freq(Sampler::default().with_top_k(2), &logits(&[0.5, 0.3, 0.2]), 5_000, 1);
    assert_eq!(f[2], 0.0);
    assert!((f[0] - 0.625).abs() < 0.03, "{f:?}"); // 0.5 / 0.8 after renormalizing
}
#[test] fn top_p_cuts_tail() {
    let f = freq(Sampler::default().with_top_p(0.85), &logits(&[0.6, 0.3, 0.1]), 5_000, 2);
    assert_eq!(f[2], 0.0); assert!(f[1] > 0.0);         // 0.6 + 0.3 = 0.9 >= 0.85
}
#[test] fn top_p_keeps_at_least_one() {
    assert_eq!(freq(Sampler::default().with_top_p(0.1), &logits(&[0.6, 0.3, 0.1]), 500, 3)[0], 1.0);
}
#[test] fn low_temperature_is_stable() {
    let f = freq(Sampler::default().with_temperature(0.1), &[80.0, 79.0, 0.0], 500, 4);
    assert_eq!(f[0], 1.0);                               // (80-79)/0.1 = 10 nats apart, and no NaN
}
#[test] fn high_temperature_flattens() {
    let f = freq(Sampler::default().with_temperature(100.0), &logits(&[0.9, 0.05, 0.05]), 20_000, 5);
    assert!(f.iter().all(|&x| (x - 1.0 / 3.0).abs() < 0.03), "{f:?}");
}
#[test] fn top_k_then_top_p_order() {
    // top_k(2) leaves [0.4, 0.35], renormalized to [0.533, 0.467]; then top_p(0.5) keeps only index 0.
    // Doing top-p first would keep {0, 1} (0.4 < 0.5 <= 0.75), so this pins the order.
    let f = freq(Sampler::default().with_top_k(2).with_top_p(0.5), &logits(&[0.4, 0.35, 0.25]), 500, 6);
    assert_eq!(f, vec![1.0, 0.0, 0.0]);
}
#[test] fn same_seed_same_draws() {
    let s = Sampler::default().with_temperature(0.8).with_top_p(0.9);
    let l = logits(&[0.25; 4]);
    let draw = |seed| { let mut r = StdRng::seed_from_u64(seed); (0..50).map(|_| s.sample(&l, &mut r)).collect::<Vec<_>>() };
    assert_eq!(draw(7), draw(7));
}
#[test] #[should_panic(expected = "temperature must be")] fn rejects_zero_temperature() { Sampler::default().with_temperature(0.0); }
#[test] #[should_panic(expected = "top_k must be")]       fn rejects_zero_top_k() { Sampler::default().with_top_k(0); }
#[test] #[should_panic(expected = "top_p must be")]       fn rejects_top_p_above_one() { Sampler::default().with_top_p(1.5); }
#[test] #[should_panic(expected = "cannot sample")]       fn rejects_nan_logits() {
    Sampler::default().sample(&[f32::NAN, 0.0], &mut StdRng::seed_from_u64(0));
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release sampling::`
Expected: compile error, `cannot find type Sampler`.

- [ ] **Step 3: Implement `DecodingMode` and `Sampler` in `src/sampling.rs`, and add `mod sampling;` to `src/main.rs`**

Give `Sampler` a `///` doc comment that shows the builder chain and states the filter order.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `cargo test --release sampling::`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/sampling.rs src/main.rs
git commit -m "Add next-token Sampler with temperature, top-k and top-p builder"
```

### Task 2: Wire the sampler into `generate-text`

**Files:**
- Modify: `src/main.rs:216-234` (`Commands::GenerateText` fields)
- Modify: `src/main.rs:475-640` (`run_generate_text`)
- Modify: `src/main.rs:706-712` (dispatch)
- Modify: `src/main.rs:17-19` (module doc), `README.md` (one paragraph near the other `generate-text` notes)

**Interfaces:**
- Consumes: `sampling::{DecodingMode, Sampler}` from Task 1.
- Produces, as CLI flags on `generate-text`:
  - `--decoding greedy|sample`, default `greedy`: `#[arg(long, value_enum, default_value_t = DecodingMode::Greedy)]`.
  - `--temperature <f32>`, `--top-k <usize>`, `--top-p <f32>`: each is an `Option`, valid only with `sample`.
  - `--seed <u64>`: an `Option`, valid only with `sample`. When it is absent, use `StdRng::from_entropy()`.
- New signature: `fn run_generate_text(prompt: String, max_new_tokens: usize, config_path: String, model_path: String, device: LibTorchDevice, decoding: DecodingMode, sampler: Sampler, seed: Option<u64>)`.

- [ ] **Step 1: Add the flags. In the `Commands::GenerateText` dispatch arm, validate them and build the `Sampler`, then pass it in**

If `decoding == Greedy` and any of `temperature`/`top_k`/`top_p`/`seed` is `Some`, exit through clap. Use `Cli::command().error(clap::error::ErrorKind::ArgumentConflict, "--temperature, --top-k, --top-p and --seed require --decoding sample").exit()`. Otherwise, start from `Sampler::default()` and chain `with_*` only for the flags that are `Some`. The builder's own panics cover out-of-range values.

- [ ] **Step 2: Change the per-step choice logic in `run_generate_text`**

Behavior to implement:
- Build the RNG once, before the loop.
- Each step, keep `last_logits` as a `Vec<f32>` (`.into_data().to_vec()`), and still compute the softmax for the top-5 display as today.
- Set `default_id` for this step:
  - `Greedy`: `top5[0].0`.
  - `Sample`: `sampler.sample(&logits_vec, &mut rng)`.
- In `Sample` mode, print `Sampled: "<piece>" (id=<id>, prob=<p:.4>)` after the top-5 list, where `p` is the model probability at temperature 1, `probs_data[id]`.
- Prompt text: in `Greedy` keep `default 1`. In `Sample`, use `Enter = sampled`.
- Enter and stdin EOF accept `default_id`. A number 1-5 picks `top5[n-1].0`. `q` quits. Make the pick an id (`usize`), not a `top5` rank, so a sampled token outside the top 5 works.
- The `Chosen:` label is `default` for an Enter/EOF pick in greedy mode, `sampled` for one in sample mode, and `user pick` for a number.
- Startup line: print `(argmax, ...)` in greedy mode. In sample mode, print `(sampling: {sampler:?}, ...)`.
- Update the doc comment on `run_generate_text` and the module doc to describe both modes.

- [ ] **Step 3: Build and run all the tests**

Run: `cargo build --release && cargo test --release`
Expected: the build succeeds and every test passes, including the 13 from Task 1.

- [ ] **Step 4: Check manually against the downloaded checkpoint** (`artifacts/hf/`, already set up)

```bash
B=./target/release/llm-from-scratch
A="generate-text --config-path artifacts/hf/config.json --model-path artifacts/hf/model --max-new-tokens 40 --prompt Photosynthesis"
$B $A < /dev/null | tail -1                                              # 1: same greedy text as before
$B $A --decoding sample --seed 7 < /dev/null | tail -1 > /tmp/a.txt
$B $A --decoding sample --seed 7 < /dev/null | tail -1 > /tmp/b.txt
$B $A --decoding sample --seed 8 < /dev/null | tail -1 > /tmp/c.txt
diff /tmp/a.txt /tmp/b.txt && ! diff -q /tmp/a.txt /tmp/c.txt            # 2: seeded runs are reproducible
$B $A --decoding sample --temperature 1.5 --seed 1 < /dev/null | grep Sampled | head  # 3: some Sampled ids not in top 5
$B $A --decoding sample --top-k 1 < /dev/null | tail -1                  # 4: equals the greedy text from 1
$B $A --temperature 0.8; echo "exit=$?"                                  # 5: clap error, exit=2
```

Expected: each numbered comment holds.

- [ ] **Step 5: Add a README paragraph** that covers:
  - `--decoding sample` with `--temperature` / `--top-k` / `--top-p` / `--seed`;
  - the filter order;
  - that `--top-k 1` equals greedy.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs README.md
git commit -m "generate-text: add --decoding sample with temperature/top-k/top-p"
```

# Inference Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `visualize` binary: a ratatui app that lets a viewer type a prompt and step through the real GPT forward pass stage by stage. It shows the embeddings (PCA to 16 values), the attention per layer and head, the output probabilities and a live-tunable sampler.

**Architecture:** The crate becomes a library plus two binaries:
- `src/lib.rs` holds the existing modules. `src/main.rs` stays the CLI.
- `src/bin/visualize/` is the new app: `main.rs` with the terminal loop, plus `pca.rs`, `app.rs` (a pure state machine) and `ui.rs` (rendering).
- The model records a `Trace` through the *same* forward code path, via private `…_impl(…, Option<&mut …>)` functions. Attention weights are recomputed explicitly from the Q/K that feed the fused kernel, but only when tracing.
- The app runs one traced forward per pass. The stages only reveal parts of that trace.

**Tech Stack:** Rust, burn 0.21 (`LibTorch<f32>`), `tch` 0.22 (PCA `linalg_eigh`), ratatui 0.30 (crossterm via `ratatui::crossterm`), clap 4, rand 0.8.

**Spec:** `docs/superpowers/specs/2026-11-25-inference-visualizer-design.md`

## Global Constraints

- One forward code path: training, `generate-text` and the visualizer all run the same `forward_*` code. Training's `forward_with` keeps its signature and does no extra work when not tracing.
- `src/main.rs` CLI behavior is unchanged. Greedy `generate-text` output must be byte-identical before and after this change.
- New dependency: only `ratatui = "0.30"`. Use crossterm through `ratatui::crossterm`, and do not add it separately.
- `LibTorch<f32>` backend. Tracing runs at `Precision::F32`, batch 1.
- Model input cap: the last 256 tokens (`MAX_MODEL_TOKENS = 256`). The full token list is kept.
- Display window: the last 8 tokens (`WINDOW = 8`) in both the embeddings and attention panels.
- PCA: 16 components (`PCA_COMPONENTS = 16`), fitted once on the token embedding table.
- Two modes. In edit mode every key is text. In inference mode the prompt is locked and keys are commands. The mode badges are `EDIT` / `INFERENCE`.
- No explanatory text in the UI. Labels and key hints are allowed.

## Review Focus

- A panic while the terminal is in raw mode (for example a builder assert) must restore the terminal. Use `ratatui::init()` / `ratatui::restore()`; `init` installs a restoring panic hook. Pinned by the Task 6 manual check.
- A tiny terminal (40×12) must not panic. Pinned by `renders_without_panic` (Task 6).
- A 1-token prompt gives a 1×1 attention and one embedding row, with no index panics. Pinned by `one_token_trace` (Task 2) and `renders_one_token` (Task 6).
- A prompt longer than 256 tokens must feed only the last 256 to the model. Pinned by `model_input_caps_at_256` (Task 5).
- Non-ASCII typing (`é`): Backspace removes one char, not one byte. Pinned by `backspace_removes_a_char` (Task 5).

---

### Task 1: Split the crate into a library and the CLI binary

**Files:**
- Create: `src/lib.rs`, `src/inference.rs`
- Modify: `src/main.rs` (drop the `mod` lines at 52-58, move `torch_rng_lock` at 67-71, `parse_device` at 260, `load_tokenizer` at 361; use the library), `src/tokenizer.rs` (receives `load_tokenizer`)

**Interfaces:**
- Produces:
  - `src/lib.rs`: `pub mod dataset; pub mod inference; pub mod loader; pub mod model; pub mod ops; pub mod sampling; pub mod tokenizer; pub mod training;`, `#[cfg(test)] pub fn torch_rng_lock()` (body unchanged; tests keep calling `crate::torch_rng_lock`), and `pub fn parse_device(s: &str) -> Result<LibTorchDevice, String>` (body unchanged).
  - `llm_from_scratch::tokenizer::load_tokenizer(kind: TokenizerKind, path: Option<String>) -> (AnyTokenizer, String)` (body unchanged).
  - `llm_from_scratch::inference::load_trained_model(config_path: &str, model_path: &str, device: &LibTorchDevice) -> (TrainingConfig, GPTModel<LibTorch<f32>>, AnyTokenizer)`. This is the load code now at the top of `run_generate_text`, including the vocab-size assert and the `.expect` messages. `run_generate_text` calls it.

- [ ] **Step 1: Record the greedy baseline before touching anything**

Run: `./target/release/llm-from-scratch generate-text --config-path artifacts/hf/config.json --model-path artifacts/hf/model --max-new-tokens 30 --prompt Photosynthesis < /dev/null > /tmp/claude-1000/viz-greedy-before.txt`
Expected: the file ends with a `Final text:` line.

- [ ] **Step 2: Create `src/lib.rs` and `src/inference.rs`, move the three functions, and switch `src/main.rs` to `use llm_from_scratch::…`**

Make items `pub` only where the compiler asks for it across the crate boundary.

- [ ] **Step 3: Build and run the full suite**

Run: `cargo build --release 2>&1 | grep -E "^(warning|error)"; cargo test --release 2>&1 | grep "test result"`
Expected: no new warnings or errors. The test result shows 46 passed in total across the lib and bin targets.

- [ ] **Step 4: Confirm the CLI output is unchanged**

Run: the Step 1 command with output to `/tmp/claude-1000/viz-greedy-after.txt`, then `diff /tmp/claude-1000/viz-greedy-before.txt /tmp/claude-1000/viz-greedy-after.txt`
Expected: no diff. (The `Tokenizer:` line printed by `load_tokenizer` stays as well.)

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/inference.rs src/main.rs src/tokenizer.rs
git commit -m "Split crate into a library and the CLI binary"
```

### Task 2: Record a `Trace` through the forward pass

**Files:**
- Modify: `src/model.rs` (`GPTModel` 131-155, `MultiHeadAttention::forward_with` 277-317, `TransformerBlock::forward_with` 366-394, tests)
- Modify: `src/inference.rs`

**Interfaces:**
- Consumes: `inference.rs` from Task 1.
- Produces (in `src/model.rs`):

```rust
#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub embedding: Vec<Vec<f32>>,     // [seq][d_model], token + position embedding
    pub blocks: Vec<BlockTrace>,      // one per transformer block, in order
    pub last_logits: Vec<f32>,        // [vocab], logits at the last position
}
#[derive(Debug, Clone, Default)]
pub struct BlockTrace {
    pub attention: Vec<Vec<Vec<f32>>>, // [heads][seq][seq], softmax(QKᵀ/√head_dim + causal mask)
    pub residual: Vec<Vec<f32>>,       // [seq][d_model], residual stream after the block
}
impl<B: FusedOps> GPTModel<B> {
    pub fn forward_traced(&self, indices: Tensor<B, 2, Int>) -> Trace;  // panics unless batch == 1
    pub fn token_embedding_weights(&self) -> Tensor<B, 2>;             // [vocab, d_model]
}
```

- Also in `src/inference.rs`: `pub fn trace_tokens(model: &GPTModel<LibTorch<f32>>, tokens: &[Token], device: &LibTorchDevice) -> Trace`. It builds the `[1, len]` index tensor and calls `forward_traced`. The caller has already capped the tokens.
- Private single path:
  - `GPTModel::forward_impl(&self, indices, precision, trace: Option<&mut Trace>) -> Tensor<B, 3>`
  - `TransformerBlock::forward_impl(&self, x, precision, trace: Option<&mut BlockTrace>) -> Tensor<B, 3>`
  - `MultiHeadAttention::forward_impl(&self, seq, precision, attention: Option<&mut Vec<Vec<Vec<f32>>>>) -> Tensor<B, 3>`

  The existing `forward_with` functions call these with `None`.
- How attention is recomputed: when the attention slot is `Some`, take the same split-head `queries`/`keys` (upcast to f32), compute `softmax(q.matmul(kᵀ) / √head_dim + mask, dim 3)`, where `mask` is `-inf` above the diagonal, then batch 0 → nested `Vec`. The block's output still comes from `causal_attention`.

- [ ] **Step 1: Write the failing tests** in `src/model.rs` `mod tests`, using `test_gpt_config` (d_model 8, 2 heads, 1 layer) and taking `crate::torch_rng_lock()` first:

```rust
#[test] fn traced_logits_match_forward() {
    // indices [0,1,2,3] with context 4: forward(x) at the last position == trace.last_logits within 1e-5
}
#[test] fn traced_attention_is_causal_softmax() {
    // trace.blocks.len() == 1; attention dims [2][4][4]
    // each row sums to 1 within 1e-4; attention[h][i][j] == 0.0 for j > i
}
#[test] fn traced_attention_reproduces_mha_output() {
    // for a random [1,4,8] input x: rebuild the MHA output from the recorded weights:
    //   V = value_weights(x) split into heads; ctx = weights @ V, heads merged; out = out_projection(ctx)
    // assert out ≈ mha.forward(x) within 1e-4
}
#[test] fn traced_residual_shapes() {
    // trace.embedding is [4][8]; blocks[0].residual is [4][8]; last_logits.len() == 16
}
#[test] fn one_token_trace() {
    // indices [5] -> attention [2][1][1] == [[[1.0]], [[1.0]]] within 1e-6
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --lib model::tests::traced 2>&1 | grep -E "^error" | head -3`
Expected: `error[E0599]: no method named forward_traced`.

- [ ] **Step 3: Implement `Trace`, `BlockTrace`, the `…_impl` path, `forward_traced`, `token_embedding_weights`, and `inference::trace_tokens`**

- [ ] **Step 4: Run the tests, the full suite and the CLI diff**

Run: `cargo test --release 2>&1 | grep "test result"`, then repeat Task 1 Step 4.
Expected: 51 passed in total. No diff in the greedy output.

- [ ] **Step 5: Commit**

```bash
git add src/model.rs src/inference.rs
git commit -m "Record embeddings, attention weights and residuals through the forward pass"
```

### Task 3: `Sampler::distribution`

**Files:**
- Modify: `src/sampling.rs`

**Interfaces:**
- Produces: `pub fn distribution(&self, logits: &[f32]) -> Vec<(usize, f32)>`. It returns the kept token ids with renormalized probabilities, sorted by descending probability (stable, so ties keep the lower id), and panics with the same `INVALID` message as `sample`.
- `sample` becomes a `WeightedIndex` draw over `distribution`'s probabilities, mapped back to the ids.

- [ ] **Step 1: Write the failing tests** in `src/sampling.rs` `mod tests`:

```rust
#[test] fn distribution_sums_to_one() {
    let d = Sampler::default().distribution(&logits(&[0.5, 0.3, 0.2]));
    assert_eq!(d.iter().map(|&(i, _)| i).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert!((d.iter().map(|&(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-6);
    assert!((d[0].1 - 0.5).abs() < 1e-5);
}
#[test] fn distribution_applies_top_k_and_top_p() {
    assert_eq!(Sampler::default().with_top_k(2).distribution(&logits(&[0.5, 0.3, 0.2])).len(), 2);
    let d = Sampler::default().with_top_p(0.85).distribution(&logits(&[0.6, 0.3, 0.1]));
    assert_eq!(d.len(), 2);
    assert!((d[0].1 - 0.6 / 0.9).abs() < 1e-5);
}
#[test] fn sample_draws_from_distribution() {
    let s = Sampler::default().with_top_k(3).with_temperature(1.5);
    let l = logits(&[0.4, 0.3, 0.2, 0.1]);
    let kept: Vec<usize> = s.distribution(&l).iter().map(|&(i, _)| i).collect();
    let mut rng = StdRng::seed_from_u64(9);
    for _ in 0..500 { assert!(kept.contains(&s.sample(&l, &mut rng))); }
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --lib sampling:: 2>&1 | grep -E "^error" | head -2`
Expected: `no method named distribution`.

- [ ] **Step 3: Implement `distribution`, and rewrite `sample` on top of it**

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --lib sampling:: 2>&1 | grep "test result"`
Expected: `16 passed` (the 13 existing tests unchanged, plus 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/sampling.rs
git commit -m "Sampler: expose the filtered distribution, sample from it"
```

### Task 4: PCA basis

**Files:**
- Create: `src/bin/visualize/main.rs` (stub: `mod pca; fn main() {}`), `src/bin/visualize/pca.rs`

**Interfaces:**
- Produces (in `pca.rs`):

```rust
pub const PCA_COMPONENTS: usize = 16;
pub struct Pca { /* mean: Vec<f32> [dim], basis: Vec<Vec<f32>> [k][dim], largest variance first */ }
impl Pca {
    /// `data` is row-major [n_rows, dim].
    pub fn fit(data: &[f32], n_rows: usize, dim: usize, k: usize) -> Pca;
    /// (v − mean) · basisᵀ → k values.
    pub fn project(&self, v: &[f32]) -> Vec<f32>;
}
```

- How `fit` works, with `tch`: `x = Tensor::from_slice(data).view([n, dim])`, centered on `mean_dim(0)`, then `cov = cᵀc / (n − 1)` and `cov.linalg_eigh("L")`. The eigenvalues come back ascending, so take the last `k` eigenvector columns in reverse order.

- [ ] **Step 1: Write the failing tests** in `pca.rs` `#[cfg(test)] mod tests`:

```rust
#[test] fn basis_is_orthonormal() {
    // 200×20 pseudo-random data (rand StdRng seed 1), k = 16: for all a, b, |basis[a]·basis[b] − δ_ab| < 1e-4
}
#[test] fn first_component_finds_dominant_axis() {
    // 500 rows in dim 8: x = 10·t·e3 + 0.1·noise, k = 2 → |basis[0][3]| > 0.99
}
#[test] fn mean_projects_to_zero() {
    // project(column means) is all within 1e-4 of 0, and its len == k
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize 2>&1 | grep -E "^error" | head -2`
Expected: `cannot find type Pca` / `fit`.

- [ ] **Step 3: Implement `Pca`**

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --bin visualize 2>&1 | grep "test result"`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/bin/visualize
git commit -m "visualize: PCA basis over the token embedding table"
```

### Task 5: App state machine

**Files:**
- Create: `src/bin/visualize/app.rs`
- Modify: `src/bin/visualize/main.rs` (`mod app;`), `Cargo.toml` (`ratatui = "0.30"`, needed here for `KeyEvent`)

**Interfaces:**
- Consumes: `model::Trace` (Task 2), `sampling::Sampler` (`distribution`/`sample`, Task 3), `tokenizer::{Token, Tokenizer}`.
- Produces (in `app.rs`):

```rust
pub const MAX_MODEL_TOKENS: usize = 256;
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stage { Editing, Embedding, Block(usize) /* 1-based */, Output, Sampler, Drawn }
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Effect { RunForward, Quit }
pub struct App<T: Tokenizer> {
    pub text: String, pub tokens: Vec<Token>, pub stage: Stage,
    pub layer: usize,           // 0 = input embedding, n = after block n
    pub head: usize,            // 0-based
    pub temperature: f32, pub top_k: Option<usize>, pub top_p: Option<f32>,
    pub trace: Option<Trace>, pub drawn: Option<usize>,
    pub tokenizer: T, // plus private: rng: StdRng, num_layers, num_heads
}
impl<T: Tokenizer> App<T> {
    pub fn new(tokenizer: T, num_layers: usize, num_heads: usize, rng: StdRng) -> Self; // T=1.0, k/p None
    pub fn on_key(&mut self, key: KeyEvent) -> Option<Effect>;   // ratatui::crossterm::event::KeyEvent
    pub fn set_trace(&mut self, trace: Trace);                    // stage = Embedding, layer = 0, drawn = None
    pub fn model_input(&self) -> &[Token];                        // last MAX_MODEL_TOKENS of tokens
    pub fn sampler(&self) -> Sampler;                             // built with the with_* chain from the fields
    pub fn reached_layer(&self) -> usize;                         // Embedding → 0, Block(n) → n, Output/Sampler/Drawn → num_layers
    pub fn is_editing(&self) -> bool;
}
```

Key rules. The spec's "Modes" and "Stages and keys" sections are binding. Concretely:
- **`Ctrl+C`:** `Quit`, in any stage.
- **`Editing`:**
  - A char (including space) is pushed onto `text`, and `tokens = encode(text)`.
  - Backspace pops one `char` and re-encodes.
  - Enter on a non-empty `tokens` returns `RunForward`. Enter otherwise returns `None` and the stage stays `Editing`.
- **Enter in inference mode:**
  - `Embedding` → `Block(1)`, … `Block(num_layers)` → `Output` → `Sampler`.
  - From `Sampler`: `drawn = Some(sampler().sample(last_logits, rng))`, then go to `Drawn`.
  - From `Drawn`: push `Token(drawn)`, set `text = decode(tokens)`, and return `RunForward`.
- **Space in inference mode:** go to `Sampler`.
- **After any stage advance:** `layer = reached_layer()`.
- **Arrow keys:**
  - ↑ increases `layer` and ↓ decreases it, clamped to `0..=reached_layer()`.
  - ← and → change `head`, clamped to `0..num_heads`.
- **`Esc`** (inference mode): `stage = Editing`, `trace = None`, `drawn = None`. `text` and `tokens` are kept.
- **Sampler keys** (only in `Output` and `Sampler`):
  - `t`/`T`: temperature ∓0.1, rounded to 1 decimal, clamped to [0.1, 3.0].
  - `k`/`K`: move left/right in the list `None, 1, 5, 10, 15, …`. `k` stops at `None`.
  - `p`/`P`: top-p ∓0.05, rounded to 2 decimals, in [0.05, 1.0]. 1.0 is stored as `None`. From `None`, `p` gives 0.95.
- Any other key in inference mode does nothing. In particular, letters never change `text`.

- [ ] **Step 1: Write the failing tests** in `app.rs` `mod tests`. The tests use:
  - a stub `CharTokenizer` implementing `Tokenizer` (one token per char: `encode` maps a char to `Token(c as u32 % 64)`, `decode` maps back, vocab 64, `token_to_piece` = the char);
  - `fake_trace(seq)`: 2 blocks, 2 heads, d = 4, 64 logits, with logit 7 the largest;
  - `key(code)` / `ctrl(c)` helpers;
  - `App::new(CharTokenizer, 2, 2, StdRng::seed_from_u64(0))`.

```rust
#[test] fn typing_updates_tokens() { /* type "ab" → text "ab", tokens.len() == 2, stage Editing */ }
#[test] fn backspace_removes_a_char() { /* type "aé", Backspace → text "a" */ }
#[test] fn enter_on_empty_prompt_does_nothing() { /* Enter → None, Editing */ }
#[test] fn full_walk() {
    // type "ab"; Enter → Some(RunForward); set_trace(fake_trace(2)) → Embedding, layer 0
    // Enter → Block(1) (layer 1), Enter → Block(2) (layer 2), Enter → Output, Enter → Sampler,
    // Enter → Drawn with drawn.is_some(), Enter → Some(RunForward) and tokens.len() == 3
}
#[test] fn letters_do_not_edit_in_inference() { /* in Embedding: 'x', ' ' (Space jumps to Sampler) → text unchanged */ }
#[test] fn esc_returns_to_editing_keeping_text() { /* trace None, text "ab" */ }
#[test] fn layer_clamped_to_reached() { /* at Block(1): Up twice → layer 1; Down twice → 0 */ }
#[test] fn head_clamped() { /* Right ×5 → head 1; Left ×5 → 0 */ }
#[test] fn sampler_keys_clamp() {
    // at Sampler: 't' ×20 → temperature 0.1; 'T' ×40 → 3.0
    // 'K' → top_k Some(1), 'K' → Some(5), 'k' → Some(1), 'k' → None, 'k' → None
    // 'p' → Some(0.95); 'P' → None
}
#[test] fn sampler_keys_inactive_in_embedding() { /* 't' at Embedding → temperature stays 1.0 */ }
#[test] fn greedy_draw_with_top_k_1() { /* 'K' at Sampler → k = 1; Enter → drawn == Some(7) */ }
#[test] fn model_input_caps_at_256() { /* type 300 chars → model_input().len() == 256, == tokens[44..] */ }
#[test] fn ctrl_c_quits_in_both_modes() { /* Editing and Sampler → Some(Quit) */ }
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize 2>&1 | grep -E "^error" | head -2`
Expected: `cannot find type App`.

- [ ] **Step 3: Implement `App`**

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --bin visualize 2>&1 | grep "test result"`
Expected: `16 passed` (3 PCA + 13 app).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/bin/visualize
git commit -m "visualize: edit/inference state machine with live sampler settings"
```

### Task 6: Rendering, terminal loop, README

**Files:**
- Create: `src/bin/visualize/ui.rs`
- Modify: `src/bin/visualize/main.rs` (the real `main`), `README.md` (a "Visualizer" section after "Generating text")

**Interfaces:**
- Consumes: `App`, `Stage`, `Effect`, `MAX_MODEL_TOKENS` (Task 5), `Pca`/`PCA_COMPONENTS` (Task 4), `inference::{load_trained_model, trace_tokens}` (Tasks 1-2), `parse_device` (Task 1), `Sampler::distribution` (Task 3).
- Produces (in `ui.rs`):
  - `pub const WINDOW: usize = 8;`
  - `pub fn draw<T: Tokenizer>(frame: &mut Frame, app: &App<T>, pca: &Pca)`
  - `pub fn display_piece(piece: &str) -> String` (`Ġ`→`␣`, `Ċ`→`↵`)
  - `pub fn embedding_grey(v: f32, row_max_abs: f32) -> u8` (`128 + 127·v/max`; returns 128 when `max == 0`)
  - `pub fn attention_grey(w: f32) -> u8` (`255·w`, clamped)

Panel content follows the spec's "Panels" sketch:
- **Title bar:** the mode badge `EDIT` / `INFERENCE`.
- **Prompt:** the border is highlighted and the cursor is shown only in edit mode.
- **Tokens:** `[piece id]` chips.
- **Embeddings `layer N`:** the last `WINDOW` tokens, each as `PCA_COMPONENTS` cells. Layer 0 uses `trace.embedding`, layer n uses `blocks[n-1].residual`. Each cell is `Color::Rgb(g, g, g)`, with `g` from `embedding_grey`.
- **Attention `layer N · head H`:** the last-`WINDOW` sub-matrix, with greys from `attention_grey`. It stays empty at layer 0.
- **Next token:** the top 10 of `Sampler::default().distribution(last_logits)` as horizontal bars. Shown only from `Output` on.
- **Sampler `T=… top-k … top-p …`:** for the top 10 model tokens, the model probability and the filtered probability from `app.sampler().distribution`. Tokens missing from the filtered distribution show `✗`. In `Drawn`, the drawn token gets `◀ drawn` and is appended as an extra row if it is outside the top 10. Shown only from `Sampler` on.
- **Footer:**
  - Edit mode: `type to edit · Enter run · Ctrl+C quit`.
  - Inference mode: `<stage> · Enter next · Space sampler · ↑↓ layer · ←→ head · t/T temp · k/K top-k · p/P top-p · Esc edit · Ctrl+C quit`, where `<stage>` is `embedding`, `block n/12`, `output`, `sampler` or `drawn`.
- Panels not yet reached stay empty, with their titles still visible.

`main.rs`:
- clap args: `--config-path` (default `artifacts/config.json`), `--model-path` (default `artifacts/model`), `--device` (default `cpu`, via `parse_device`).
- Call `load_trained_model`, then fit `Pca::fit` on `token_embedding_weights()` with k = 16, then `ratatui::init()`.
- The loop is `draw`, then a blocking `event::read()` (key presses only), then `on_key`. On `RunForward`, call `app.set_trace(trace_tokens(&model, app.model_input(), &device))`. On `Quit`, break.
- Finish with `ratatui::restore()`.
- `App::new` gets `StdRng::from_entropy()`, with the layer and head counts from the config.

- [ ] **Step 1: Write the failing tests** in `ui.rs` `mod tests`, reusing Task 5's `CharTokenizer` and `fake_trace`. Make them `pub(crate)` in `app.rs` under `#[cfg(test)]`. Use `ratatui::backend::TestBackend` and a `Pca` fitted on the fake 4-dim data with k = 4.

```rust
#[test] fn display_piece_maps_bpe_markers() { assert_eq!(display_piece("Ġcat"), "␣cat"); assert_eq!(display_piece("Ċ"), "↵"); }
#[test] fn greys() {
    assert_eq!(embedding_grey(0.0, 2.0), 128); assert_eq!(embedding_grey(2.0, 2.0), 255);
    assert_eq!(embedding_grey(-2.0, 2.0), 1); assert_eq!(embedding_grey(1.0, 0.0), 128);
    assert_eq!(attention_grey(0.0), 0); assert_eq!(attention_grey(1.0), 255);
}
#[test] fn renders_without_panic() {
    // for each size in [(40,12), (80,24), (200,60)] and each stage of a full walk on fake_trace(3),
    // terminal.draw(|f| draw(f, &app, &pca)) is Ok
}
#[test] fn renders_one_token() { /* fake_trace(1), stage Block(2), 80×24: draw is Ok */ }
#[test] fn mode_badge_and_footer() {
    // 120×30 buffer text: Editing contains "EDIT" and "type to edit"; after Enter + set_trace it contains "INFERENCE" and "Esc edit"
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize 2>&1 | grep -E "^error" | head -2`
Expected: `cannot find function draw`.

- [ ] **Step 3: Implement `ui.rs` and the real `main.rs`**

- [ ] **Step 4: Run all the tests**

Run: `cargo build --release 2>&1 | grep -E "^(warning|error)"; cargo test --release 2>&1 | grep "test result"`
Expected: no warnings or errors, and every target passes (54 lib, plus 21 visualize).

- [ ] **Step 5: Check manually on `artifacts/hf`**

Run: `./target/release/visualize --config-path artifacts/hf/config.json --model-path artifacts/hf/model` in a real terminal. Expected, checked by a human:
1. The `EDIT` badge shows, and typing `The cat sat on the` updates the token chips live.
2. Enter switches to `INFERENCE`; the embedding rows show.
3. Each Enter goes one block further: the embedding rows change and the attention for the current block shows.
4. ↑↓ and ←→ change the layer and head.
5. The output bars show at the `output` stage. `T`, `k` and `p` redraw the sampler panel at once.
6. Enter shows `◀ drawn`. Enter again appends the token and starts a new pass.
7. `Esc` returns to `EDIT` with the text kept. `Ctrl+C` exits, and the shell prompt is normal (no raw mode left).

- [ ] **Step 6: Write the README section**

It covers the launch command, the two modes, and the key list from the footer.

- [ ] **Step 7: Commit**

```bash
git add src/bin/visualize README.md
git commit -m "visualize: ratatui panels, terminal loop and README"
```

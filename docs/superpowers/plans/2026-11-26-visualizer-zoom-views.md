# Visualizer Zoom Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a head zoom (`z`) that walks one attention row through q⊙k → score → softmax → weighted v, with KV-cache read/write tags and a cache map. Also add a raw-score toggle (`s`), an MLP zoom (`m`) showing the in vector, the 3,072 GELU neurons and the out vector, and a compact Next-token panel for a 120×40 layout.

**Architecture:** `BlockTrace` records q/k/v, the pre-softmax scores and the MLP in/hidden/out through the existing single `forward_impl` path in `src/model.rs`. A new `src/bin/visualize/cache.rs` compares each pass's K/V against the previous pass's to derive per-layer, per-position `read`/`write`/`recomputed` tags. `App` gains zoom, query-position, score-mode and cache state. `ui.rs` gains the compact output panel, the score toggle and the two zoom panels.

**Tech Stack:** Rust, burn 0.21 (`LibTorch<f32>`), ratatui 0.30, rand 0.8.

**Spec:** `docs/superpowers/specs/2026-11-26-visualizer-zoom-views-design.md`, on top of `docs/superpowers/specs/2026-11-25-inference-visualizer-design.md`

## Global Constraints

- One forward code path. Training's `forward_with` keeps its signature, and the `None` trace path does no extra tensor work. Greedy `generate-text` output must stay byte-identical. The baseline is `/tmp/claude-1000/viz-greedy-before.txt`, recorded with `./target/release/llm-from-scratch generate-text --config-path artifacts/hf/config.json --model-path artifacts/hf/model --max-new-tokens 30 --prompt Photosynthesis < /dev/null`.
- Every value shown comes from the recorded trace. The only arithmetic the app does itself is `out = Σ weight × v` and the q ⊙ k products, both computed from recorded values.
- Cache tags are checked, never assumed:
  - `read` requires `i < p` (p = longest common prefix with the previous pass's model input) **and** k and v match within max |Δ| ≤ 1e-4 · max(1, max |row|);
  - `write` means `i ≥ p`;
  - `recomputed` means `i < p` and a mismatch.
- The previous pass's K/V survive `Esc` and edits, and "previous" means the immediately preceding forward pass. Keep only the previous pass's **K and V**, never the whole trace.
- Minimum terminal 120×40. No panic at 40×12, 80×24, 120×40 or 200×60, in any stage and zoom.
- Display window: the last 8 tokens (`WINDOW = 8`, moved to `app.rs`). Model input cap: unchanged.
- No new dependency. No explanatory sentences in the UI; labels and key hints only.
- Do not run `cargo fmt` on the whole crate. Format only the files you change.

## Review Focus

- A zoom open while Enter commits a draw: the new pass at layer 0 must render the zoom frame with an empty body, then fill again from `Block(1)`, and the query must re-default to the new last token. Pinned by `zoom_survives_new_pass` (Task 3) and `zooms_render_in_every_stage` (Task 5).
- An edit that changes only the end of the prompt must give `read` on the shared prefix, not all `write`. Pinned by `edit_keeps_prefix_reads` (Task 2).
- A real model's K/V recomputed through a longer sequence must stay within the tolerance, so no spurious `recomputed`. Pinned by the Task 2 Step 5 measurement on `artifacts/hf`.
- Prompts shorter than 8 tokens, including 1 token: the zooms show only the existing key/value rows, with no index panic. Pinned by `zooms_render_one_token` (Task 5).
- `s`, `z`, `m`, `[` and `]` typed in edit mode must be text, not commands. Pinned by `zoom_keys_are_text_in_edit_mode` (Task 3).

---

### Task 1: Record q/k/v, scores and the MLP in the trace

**Files:**
- Modify: `src/model.rs` (`BlockTrace` about line 217; `MultiHeadAttention::forward_impl` 361-420; `TransformerBlock::forward_impl` 478-512; tests)
- Modify: `src/bin/visualize/app.rs` `test_support::fake_trace`. It builds `BlockTrace` with a struct literal, so add `..Default::default()` there so the bin still compiles. Task 3 fills the new fields properly.

**Interfaces:**
- Produces: these new `pub` fields on `BlockTrace`, all f32 and batch 0:

```rust
pub q: Vec<Vec<Vec<f32>>>,      // [heads][seq][head_dim]
pub k: Vec<Vec<Vec<f32>>>,      // [heads][seq][head_dim]
pub v: Vec<Vec<Vec<f32>>>,      // [heads][seq][head_dim]
pub scores: Vec<Vec<Vec<f32>>>, // [heads][seq][seq], q·kᵀ/√head_dim, masked cells = f32::NEG_INFINITY
pub mlp_in: Vec<Vec<f32>>,      // [seq][d_model], layer_norm2 output
pub mlp_hidden: Vec<Vec<f32>>,  // [seq][ff_dim], activation output
pub mlp_out: Vec<Vec<f32>>,     // [seq][d_model], linear2 output (added to the residual)
```

- `MultiHeadAttention::forward_impl`'s last parameter becomes `trace: Option<&mut BlockTrace>`. It fills `attention`, `scores`, `q`, `k` and `v` from the same split-head tensors that feed `causal_attention`. `TransformerBlock::forward_impl` passes `trace.as_deref_mut()` to it and fills the three `mlp_*` fields under `if let Some`.
- Convert each `[heads, seq, x]` tensor with **one** `into_data` per tensor, then split it on the host. This also clears the deferred minor about 144 per-head syncs.

- [ ] **Step 1: Write the failing tests** in `src/model.rs` `mod tests`, using `test_gpt_config` (d 8, 2 heads, head_dim 4, ff 32) and taking `crate::torch_rng_lock()` first:

```rust
#[test] fn traced_qkv_and_mlp_shapes() {
    // indices [0,1,2] → blocks[0]: q/k/v [2][3][4]; scores [2][3][3]; mlp_in [3][8]; mlp_hidden [3][32]; mlp_out [3][8]
    // scores[h][i][j] == f32::NEG_INFINITY for j > i
}
#[test] fn softmax_of_scores_is_attention() {
    // for every h, i: softmax over j ≤ i of scores[h][i][j] == attention[h][i][j] within 1e-5
}
#[test] fn scores_are_scaled_q_dot_k() {
    // scores[h][i][j] == dot(q[h][i], k[h][j]) / sqrt(4) within 1e-5, for j ≤ i
}
#[test] fn mlp_fields_are_what_the_block_used() {
    // block = &model.transformer_layers[0]; mid = residual − mlp_out (row-wise)
    // block.layer_norm2.forward(mid) ≈ mlp_in within 1e-5
    // block.activation.forward(block.linear1.forward(mlp_in)) ≈ mlp_hidden within 1e-5
    // block.linear2.forward(mlp_hidden) ≈ mlp_out within 1e-5
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --lib model::tests 2>&1 | grep -E "^error" | head -3`
Expected: `no field q` / `scores` on `BlockTrace`.

- [ ] **Step 3: Implement the fields and the recording**

- [ ] **Step 4: Run the tests, the full lib suite, and the greedy diff**

Run: `cargo test --release --lib 2>&1 | grep "test result"`, then `cargo build --release` and the Global Constraints greedy command into `/tmp/claude-1000/viz-greedy-zoom.txt`, then `diff /tmp/claude-1000/viz-greedy-before.txt /tmp/claude-1000/viz-greedy-zoom.txt`.
Expected: `58 passed` (54 + 4). No diff. Known flake: `training::tests::test_resumed_run_matches_uninterrupted_run` may fail once in parallel runs; rerun once.

- [ ] **Step 5: Commit**

```bash
git add src/model.rs src/bin/visualize/app.rs
git commit -m "Trace q/k/v, attention scores and the MLP in/hidden/out"
```

### Task 2: KV-cache comparison

**Files:**
- Create: `src/bin/visualize/cache.rs`
- Modify: `src/bin/visualize/main.rs` (`mod cache;`)

**Interfaces:**
- Consumes: `BlockTrace::{k, v}` (Task 1), `llm_from_scratch::tokenizer::Token` (`Copy`, `PartialEq`).
- Produces:

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheTag { Read, Write, Recomputed }
#[derive(Clone, Debug, Default)]
pub struct KvSnapshot { /* tokens: Vec<Token>, k: Vec<[heads][seq][hd]> per layer, v: same */ }
impl KvSnapshot {
    pub fn new(tokens: &[Token], trace: &Trace) -> KvSnapshot; // copies only tokens + every block's k and v
}
/// tags[layer][position] for the new pass. Layer index is 0-based (block 1 = index 0).
pub fn cache_tags(prev: Option<&KvSnapshot>, now: &KvSnapshot) -> Vec<Vec<CacheTag>>;
```

- The rule is copied from Global Constraints. A position is `Read` only if every head's k **and** v rows match. Any mismatch makes it `Recomputed`. With `prev == None`, every tag is `Write`.

- [ ] **Step 1: Write the failing tests** in `cache.rs` `mod tests`. Build snapshots by hand: 2 layers, 2 heads, head_dim 3. `snap(tokens, f)` fills k/v with `f(layer, head, pos, dim)`, so that the same positions give the same values across passes.

```rust
#[test] fn first_pass_is_all_write() { /* cache_tags(None, &snap([1,2])) == [[W,W],[W,W]] */ }
#[test] fn append_reads_prefix_writes_new() { /* prev [1,2], now [1,2,3] → each layer [R,R,W] */ }
#[test] fn slid_window_is_all_write() { /* prev [1,2,3], now [2,3,4] → all W (p = 0) */ }
#[test] fn edit_keeps_prefix_reads() { /* prev [1,2,3], now [1,2,9,4] → each layer [R,R,W,W] */ }
#[test] fn mismatch_is_recomputed() { /* prev [1,2]; now [1,2,3] but layer 1, head 1, pos 0, v[0] += 0.5 → layer 1 = [Rc,R,W], layer 0 = [R,R,W] */ }
#[test] fn tolerance_is_relative() { /* row values ~1000, now = prev + 0.05 (≤ 1e-4·1000) → R; + 0.2 → Rc */ }
#[test] fn unchanged_rerun_is_all_read() { /* prev [1,2], now [1,2] → [R,R] per layer */ }
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize cache:: 2>&1 | grep -E "^error" | head -2`
Expected: `cannot find function cache_tags`.

- [ ] **Step 3: Implement `CacheTag`, `KvSnapshot` and `cache_tags`**

Add `#[allow(dead_code)]` on `mod cache;` only until Task 3 uses it, with a one-line comment saying so.

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --bin visualize cache:: 2>&1 | grep "test result"`
Expected: `7 passed`.

- [ ] **Step 5: Measure the tolerance on the real checkpoint** (throwaway, do not commit)

Add a temporary `#[test] #[ignore]` in `cache.rs`, or a temporary example. It loads `artifacts/hf` via `llm_from_scratch::inference::{load_trained_model, trace_tokens}`, traces "The cat sat on the", then appends 3 greedy tokens one at a time. For each pass it prints the max over layers, heads and prefix positions of `max|Δ| / max(1, max|row|)` for k and v, and asserts `cache_tags` gives no `Recomputed`.
Expected: the printed ratio is well under 1e-4 (aim for ≤ 1e-5), with 0 `Recomputed`. Put the numbers in the report. If the ratio exceeds 1e-5, stop and report it as a concern; do not raise the tolerance. Then delete the throwaway.

- [ ] **Step 6: Commit**

```bash
git add src/bin/visualize/cache.rs src/bin/visualize/main.rs
git commit -m "visualize: derive KV-cache read/write tags by comparing passes"
```

### Task 3: App state for zooms, query, score toggle and cache

**Files:**
- Modify: `src/bin/visualize/app.rs`, `src/bin/visualize/ui.rs` (import `WINDOW` from `app`), `src/bin/visualize/main.rs` (remove the `cache` dead-code allow)

**Interfaces:**
- Consumes: `cache::{CacheTag, KvSnapshot, cache_tags}` (Task 2).
- Produces (in `app.rs`):

```rust
pub const WINDOW: usize = 8; // moved from ui.rs
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Zoom { Off, Head, Mlp }
// new pub fields on App:
pub zoom: Zoom,
pub zoom_pos: usize,                    // absolute index into model_input() of the selected query/MLP token
pub show_scores: bool,
pub cache: Option<Vec<Vec<CacheTag>>>,  // tags[layer][position] for the current trace
// private: prev_kv: Option<KvSnapshot>
pub fn window_start(&self) -> usize;    // model_input().len().saturating_sub(WINDOW)
```

- `set_trace(trace)`:
  - builds `now = KvSnapshot::new(model_input(), &trace)`, sets `cache = Some(cache_tags(prev_kv.as_ref(), &now))` and `prev_kv = Some(now)`;
  - sets `zoom_pos = model_input().len() - 1`;
  - then does what it does today. `zoom` and `head` are **not** reset.
- `Esc` and edits never clear `prev_kv`.
- Keys in inference mode. The new arms come **before** the sampler-key arm, so they win in `Output`/`Sampler`.
  - **`z`:** if `zoom == Head`, set `Off`. Else, if `layer ≥ 1`, set `Head`. Else nothing.
  - **`m`:** the same, with `Mlp`.
  - **`s`:** toggles `show_scores`. Only when `zoom == Off`.
  - **`Esc`:** if `zoom != Off`, set `zoom = Off` only. Else, today's behavior.
  - **↑/↓ when `zoom != Off`:** change `zoom_pos`, clamped to `window_start()..=model_input().len() - 1`. ↑ goes to the previous token, ↓ to the next. When `zoom == Off`, they change the layer as today.
  - **`[` / `]` when `zoom != Off`:** layer −1 / +1, clamped to `1..=reached_layer()`. When `zoom == Off`, they do nothing.
  - **Enter/Space:** unchanged, and `go()` still sets `layer = reached_layer()`, so the zoom follows the newest layer.
- Edit mode: unchanged. Every char is text.

- [ ] **Step 1: Write the failing tests** in `app.rs` `mod tests`, using the existing helpers. `fake_trace` must gain the Task 1 fields with consistent shapes (`k`/`v` filled deterministically by position, so two passes over the same prefix match).

```rust
#[test] fn z_and_m_toggle_and_switch() { /* at Block(1): z → Head; m → Mlp; m → Off; z → Head; z → Off */ }
#[test] fn zoom_needs_a_block() { /* at Embedding (layer 0): z, m → Off */ }
#[test] fn esc_closes_zoom_only() { /* Head; Esc → Off and stage unchanged; Esc → Editing */ }
#[test] fn arrows_pick_query_in_zoom() { /* tokens "abc", Head: zoom_pos 2; Up → 1; Up×5 → 0; Down×5 → 2; layer unchanged */ }
#[test] fn brackets_change_layer_in_zoom() { /* at Output (reached 2), Head: layer 2; '[' → 1; '[' → 1; ']' → 2; ']' → 2; with zoom Off '[' → layer unchanged */ }
#[test] fn s_toggles_scores_and_persists() { /* s → true; set_trace(fake_trace(..)) → still true; s in Head zoom → unchanged */ }
#[test] fn zoom_keys_beat_sampler_keys() { /* at Sampler: z → Head, top_k / temperature unchanged */ }
#[test] fn zoom_keys_are_text_in_edit_mode() { /* type "zms[]" → text "zms[]", zoom Off, show_scores false */ }
#[test] fn zoom_survives_new_pass() { /* Head at Drawn; Enter → RunForward; set_trace(fake_trace(3)) → zoom Head, zoom_pos 2, stage Embedding */ }
#[test] fn cache_tags_after_append() { /* "ab" pass then draw + set_trace(fake_trace(3)) → every layer [Read, Read, Write] */ }
#[test] fn prev_kv_survives_esc() { /* "ab" pass; Esc; type "c"; Enter; set_trace(fake_trace(3)) → [Read, Read, Write] */ }
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize app:: 2>&1 | grep -E "^error" | head -2`
Expected: `no field zoom`.

- [ ] **Step 3: Implement the state, the keys and `set_trace`**

Also remove the `cache` dead-code allow in `main.rs`.

- [ ] **Step 4: Run all the visualize tests**

Run: `cargo test --release --bin visualize 2>&1 | grep "test result"`
Expected: `50 passed` (32 + 7 cache + 11 app).

- [ ] **Step 5: Commit**

```bash
git add src/bin/visualize
git commit -m "visualize: zoom, query, score-toggle and KV-cache state"
```

### Task 4: Compact output panel, score toggle, 120×40 layout, footer

**Files:**
- Modify: `src/bin/visualize/ui.rs`

**Interfaces:**
- Consumes: `App::{show_scores, zoom}` (Task 3), `BlockTrace::scores` (Task 1).
- Produces:
  - `fn output_panel<T: Tokenizer>(frame, app, area)`, replacing `next_token_panel` and `sampler_panel`;
  - `pub fn score_grey(s: f32, max_abs: f32) -> u8` (diverging: `128 + 127·s/max_abs`, clamped; returns 128 when `max_abs == 0`; masked cells are not passed in).
  - Task 6 reuses `score_grey` for every diverging grey. Its shape matches `embedding_grey`.

Spec "Compact output panel" and "Score toggle" are binding:
- **Output panel:**
  - Title: `Next token · T=… top-k … top-p …`.
  - Rows: label │ model bar + value │ after-T/k/p bar + value, or `✗ cut` │ `◀ drawn` (in `Drawn`).
  - Both columns appear from `Output` on.
  - Height: header + 10 + drawn row + borders = 14 rows.
- **Layout:** title 1, prompt 3, tokens 4, middle `Fill(1)` (at least `WINDOW + 3`), output `Length(14)`, footer 2.
- **Score toggle:** the attention panel title ends in `weights` or `scores`. In scores mode the cells use `score_grey`, scaled by the max |finite value| of the visible window rows. Masked cells are black in both modes.
- **Footer**, verbatim. `{stage}` is as today.
  - Main inference view: `{stage} · Enter next · Space sampler · ↑↓ layer · ←→ head · z head · m mlp · s scores · t/T temp · k/K top-k · p/P top-p · Esc edit · Ctrl+C quit`
  - Head zoom: `{stage} · Enter next · ↑↓ query · ←→ head · [ ] layer · m mlp · z/Esc close · t/T temp · k/K top-k · p/P top-p · Ctrl+C quit`
  - MLP zoom: `{stage} · Enter next · ↑↓ token · [ ] layer · z head · m/Esc close · t/T temp · k/K top-k · p/P top-p · Ctrl+C quit`
  - Edit mode: unchanged.
- The 80×24 content tests (`core_panels_visible_at_80x24`, `attention_last_column_visible_at_80_cols`) move to 120×40 with the same assertions, since 80×24 is no longer a goal. `output_panels_visible_at_120x40` is rewritten for the merged panel.

- [ ] **Step 1: Write the failing tests** in `ui.rs` `mod tests`:

```rust
#[test] fn score_grey_diverges() { assert_eq!(score_grey(0.0, 2.0), 128); assert_eq!(score_grey(2.0, 2.0), 255); assert_eq!(score_grey(-2.0, 2.0), 1); assert_eq!(score_grey(1.0, 0.0), 128); }
#[test] fn output_panel_merged() { /* at Drawn, 120×40: contains "Next token · T=1.0", "◀ drawn", and does not contain " Sampler " */ }
#[test] fn output_columns_from_output_stage() { /* at Output with top_k 1: contains "✗ cut"; at Block(1): no "█" in the output panel */ }
#[test] fn attention_title_shows_mode() { /* Block(1): contains "weights"; after 's': contains "scores" */ }
#[test] fn footer_lists_zoom_keys() { /* 120×40 main view contains "z head · m mlp · s scores"; Head zoom contains "z/Esc close"; Mlp zoom contains "m/Esc close" */ }
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize ui:: 2>&1 | grep -E "^error|panicked" | head -3`
Expected: `cannot find function score_grey` (or the new assertions fail).

- [ ] **Step 3: Implement the panel, the toggle, the layout and the footer.** Move the two 80×24 tests to 120×40, and rewrite `output_panels_visible_at_120x40`.

- [ ] **Step 4: Run all the visualize tests**

Run: `cargo test --release --bin visualize 2>&1 | grep "test result"`
Expected: `55 passed`, with no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/bin/visualize/ui.rs
git commit -m "visualize: merged output panel, score toggle, 120x40 layout"
```

### Task 5: Head zoom panel and KV-cache map

**Files:**
- Modify: `src/bin/visualize/ui.rs`

**Interfaces:**
- Consumes: `App::{zoom, zoom_pos, window_start(), cache, layer, head}` (Task 3), `BlockTrace::{q, k, v, scores, attention}` (Task 1), `CacheTag` (Task 2), `score_grey` (Task 4).
- Produces: `fn head_zoom_panel<T: Tokenizer>(frame, app, area)`. It is drawn when `zoom == Head`, and it replaces the middle band **and** the output panel's rows.

Spec "Head zoom" is binding:
- **Title:** `Head {layer}·{head+1} · query "{label}"`.
- **q row:** 64 cells (`head_dim`), diverging grey, scaled per row.
- **Key rows** for window positions `window_start()..=zoom_pos`. Each row has:
  - the label;
  - q ⊙ k cells, scaled by the max |value| over all key rows of this query;
  - the score `{:+.1}`;
  - the weight bar with its value;
  - the tag `read` / `write` / `recomputed`, from `cache[layer-1][pos]`.
- **Value rows**, the same positions. Each row has: the label, the weight bar in a fixed-width column, `×`, the v cells (scaled per row), and the tag.
- **`= out` row:** `Σ weight × v`, computed here from the recorded values, labelled `head output (before W_O)`.
- **KV-cache map:**
  - a header row with the window labels, then one row per reached layer (`layer  1` …);
  - each cell is dim grey for `Read`, bright for `Write`, red for `Recomputed`;
  - `← write` after the row of the selected layer when its newest position is `Write`.
- If height is short, the map shows the rows nearest the selected layer. The selected layer's row is always visible.
- At layer 0 or with no trace, only the frame shows.
- Cell width: 1 column per element. If the inner width is under the content width, the cells clip at the right; they do not wrap.

- [ ] **Step 1: Write the failing tests** in `ui.rs` `mod tests`. Use a `zoomed(text, zoom)` helper that runs the app to `Block(2)` on `fake_trace` and opens the zoom.

```rust
#[test] fn head_zoom_shows_pipeline() { /* 120×40: contains "head output (before W_O)", "Head 2·1", a score like "+" followed by a digit, and "write" */ }
#[test] fn head_zoom_reads_after_append() { /* pass "ab", draw, set_trace(fake_trace(3)), Block(1), z: contains "read", "write", and "← write" */ }
#[test] fn head_zoom_rows_follow_query() { /* tokens "abc", Up once (query = pos 1): key rows = 2 (labels "a", "b" present after "k ", "c" absent from key rows) */ }
#[test] fn zooms_render_one_token() { /* "a", Block(1), z then m: draw Ok at 120×40 and 40×12 */ }
#[test] fn zooms_render_in_every_stage() { /* Head zoom open through a full walk + commit + new pass, at 40×12, 80×24, 120×40, 200×60: draw Ok; at Embedding the body is empty but the title "Head" is present */ }
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize ui:: 2>&1 | grep -E "panicked|^error" | head -3`
Expected: the assertions on `head output (before W_O)` fail.

- [ ] **Step 3: Implement `head_zoom_panel` and the cache map**

- [ ] **Step 4: Run all the visualize tests**

Run: `cargo test --release --bin visualize 2>&1 | grep "test result"`
Expected: `60 passed`, with no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/bin/visualize/ui.rs
git commit -m "visualize: head zoom (q·k, softmax, weighted v) with KV-cache map"
```

### Task 6: MLP zoom panel, README, manual check

**Files:**
- Modify: `src/bin/visualize/pca.rs`, `src/bin/visualize/ui.rs`, `README.md` (the "Visualizer" section)

**Interfaces:**
- Consumes: `BlockTrace::{mlp_in, mlp_hidden, mlp_out}` (Task 1), `App::{zoom, zoom_pos, layer}` (Task 3), `score_grey` (Task 4).
- Produces:
  - `pub fn Pca::project_direction(&self, v: &[f32]) -> Vec<f32>` (basis · v, no mean subtraction);
  - `fn mlp_zoom_panel<T: Tokenizer>(frame, app, area)`, which takes the same area as the head zoom;
  - `pub fn neuron_grey(a: f32, max: f32) -> u8` (`a ≤ 0` → 0; `255·a/max` clamped; returns 0 when `max ≤ 0`).

Spec "MLP zoom" is binding:
- **Title:** `MLP · layer {layer} · token "{label}"`.
- **`in` row:** `project_direction(mlp_in[zoom_pos])`, 16 cells, diverging per row, labelled `after LayerNorm`.
- **Hidden grid:** 64 columns × 24 rows of `▀`. The top neuron is the foreground colour and the bottom one the background colour; the neuron at grid row r, column c, half h is index `(2r + h)·64 + c`. Greys come from `neuron_grey`, scaled by this token's max value.
- **`out` row:** `project_direction(mlp_out[zoom_pos])`, labelled `added to residual`.
- At layer 0 or with no trace, only the frame shows.

- [ ] **Step 1: Write the failing tests**

```rust
// pca.rs
#[test] fn project_direction_skips_mean() { /* data with mean ≠ 0: project_direction(v) == basis·v, and differs from project(v) by basis·mean */ }
// ui.rs
#[test] fn neuron_grey_maps() { assert_eq!(neuron_grey(-0.1, 2.0), 0); assert_eq!(neuron_grey(2.0, 2.0), 255); assert_eq!(neuron_grey(1.0, 0.0), 0); }
#[test] fn mlp_zoom_shows_in_hidden_out() { /* zoomed("abc", Mlp) at 120×40: contains "after LayerNorm", "added to residual", "MLP · layer 2", and at least 24 '▀' characters */ }
```

(`fake_trace` uses `ff_dim` = 8 or similar. The test only checks that the grid renders, so size the grid from `mlp_hidden[pos].len()` with 64 columns: `ceil(len / 128)` rows.)

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test --release --bin visualize 2>&1 | grep -E "^error|panicked" | head -3`
Expected: `no method named project_direction` / `cannot find function neuron_grey`.

- [ ] **Step 3: Implement `project_direction`, `neuron_grey` and `mlp_zoom_panel`**

Then update the README's "Visualizer" section:
- the minimum terminal is 120×40;
- the key table gains `z` head zoom, `m` MLP zoom, `s` scores/weights, and, inside a zoom, ↑↓ token, `[ ]` layer and `Esc` close;
- one short paragraph on the cache tags: `read` / `write` / `recomputed` come from comparing each pass's K/V with the previous one, since the model itself always recomputes everything.

- [ ] **Step 4: Run all the tests and the build**

Run: `cargo build --release 2>&1 | grep -E "^(warning|error)"; cargo test --release --no-fail-fast 2>&1 | grep "test result"`
Expected: no warnings. The lib has 58 passed and visualize has 63 passed.

- [ ] **Step 5: Check what you can on `artifacts/hf`, without a real terminal**

Write a throwaway, uncommitted `TestBackend` render at 120×40. It traces "The cat sat on the", draws one greedy token and re-traces, opens the head zoom at layer 6 and then the MLP zoom. Print the buffer into the report.
Expected:
- the key rows show `read` for the first 5 positions and `write` for the new token;
- the softmax column sums to at most 1;
- the neuron grid shows mostly dark cells with some bright ones.

List the interactive checks a human must still do: the colours, the `[ ]`/↑↓/←→ feel, and the cache map's newest column moving over several draws.

- [ ] **Step 6: Commit**

```bash
git add src/bin/visualize README.md
git commit -m "visualize: MLP zoom (in, 3072 neurons, out) and README"
```

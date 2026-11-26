# Visualizer Zoom Views — Design

Status: approved by the user · 2026-11-26
Builds on: `docs/superpowers/specs/2026-11-25-inference-visualizer-design.md` (the "base spec"). Everything there still holds unless this document changes it.

## Intent

The user tested the `visualize` app and wants to look deeper inside one transformer block. Viewers should be able to see three things:

1. **How one attention head computes its row.** This is the Q·K dot product, the scaling, the softmax, and the weighted sum of V. The user is especially keen to *see the softmax*.
2. **What the MLP does to one token.** That means its input vector, which of the 3,072 neurons fire, and its output vector.
3. **The KV cache idea:** after a token is appended, every earlier token's K and V come out unchanged. That is why a real inference engine caches them. The code has no KV cache, and the app must show this honestly, without building one.

The tool is still a teaching tool: viewers drive it themselves, and it shows pure visualization with labels and key hints, no explanatory sentences. The base spec's anti-goal holds: every value shown must come from the real forward code path.

## Goals

1. **Head zoom** (`z`): for the selected layer and head and a selectable query token, show:
   - q;
   - for each visible key: q ⊙ k, the raw score `q·k/√head_dim` and the softmax weight;
   - for each key: v with its weight;
   - the weighted sum `out`;
   - read/write cache tags and a KV-cache map.
2. **Score toggle** (`s`): the main attention panel switches between softmax weights and raw scaled scores.
3. **MLP zoom** (`m`): for the selected layer and a selectable token, show the MLP input (after LayerNorm), the 3,072 GELU activations as a grid, and the MLP output.
4. **Compact output:** merge the Next token and Sampler panels into one panel.
5. **Minimum terminal size: 120×40.** Below that the app must not panic, but it need not be pretty. Supporting 80×24 is no longer a goal.

## Non-goals

- A real KV cache or incremental decoding in the model.
- Animating individual cache reads, or per-head read traffic.
- A per-token `Δ` difference readout. The user found it confusing.
- A 2D key/query map, which is a lossy projection.
- A "top neurons" list, or an MLP in/out ratio. The user found both confusing.
- Showing the result after `W_O` per head. The `out` row is the head's weighted sum *before* the output projection, and it is labelled that way.

## Anti-goals

- A `read` tag on a key or value that the real model actually recomputed to a different value. The cache tags must be checked, not assumed.
- Any value computed outside the model's forward code. For example, re-running attention with a modified softmax is not allowed.

## Constraints

- Same trace mechanism as the base spec: the private `forward_impl(…, Option<&mut …>)` path in `src/model.rs`. The `None` path, used by training, does no extra work.
- Model input cap: unchanged (`min(256, context_size)`). Display window: unchanged, the last 8 tokens.
- Memory, at a 256-token input and 12 layers:
  - The extra per-pass trace is about 125 MB in f32: q/k/v 28 MB, scores 37 MB, MLP in/out 19 MB, hidden 38 MB.
  - Keeping the previous pass's **K and V only** for the cache comparison adds about 19 MB.
  - With the existing 37 MB of attention weights, the total is about 180 MB. That is acceptable. Do not keep the whole previous trace.
- No new dependency.

## Approaches considered

1. **Show the cache by comparing passes (chosen).** Keep the previous pass's K/V, and compare each layer, head and token with the new pass. The model stays a full recompute, and the tags are verified facts.
2. **A real KV cache in the model.** That means a second, incremental forward path, which breaks "one code path". Rejected.
3. **Show tags with no comparison** (assume earlier tokens are cached). This could claim a read that the model didn't do (window slide, edited prompt). It hits the first anti-goal, so it is rejected.

For the zoom views, the one sensible way is to record the extra tensors on the existing trace path. A separate computation would hit the second anti-goal.

## Design

### Trace additions (`src/model.rs`)

`BlockTrace` gains these fields. All are batch 0, f32, recorded only when tracing:

```rust
pub q: Vec<Vec<Vec<f32>>>,        // [heads][seq][head_dim], the split-head queries fed to the attention
pub k: Vec<Vec<Vec<f32>>>,        // [heads][seq][head_dim]
pub v: Vec<Vec<Vec<f32>>>,        // [heads][seq][head_dim]
pub scores: Vec<Vec<Vec<f32>>>,   // [heads][seq][seq], q·kᵀ/√head_dim, with the causal mask applied as -inf
pub mlp_in: Vec<Vec<f32>>,        // [seq][d_model], output of layer_norm2 (what linear1 receives)
pub mlp_hidden: Vec<Vec<f32>>,    // [seq][ff_dim = 3072], output of the activation (GELU)
pub mlp_out: Vec<Vec<f32>>,       // [seq][d_model], output of linear2 (what is added to the residual)
```

- Take `q`/`k`/`v` from the same split-head tensors that feed `causal_attention`, cast to f32. `scores` is the pre-softmax tensor the existing attention recompute already builds. `attention` stays the softmax of it.
- The MLP fields come from the tensors of the single block code path, recorded under `if let Some(trace)`, like `residual`.
- Existing tests must still pass. The builder may refactor how the attention slot is passed, for example to a `&mut BlockTrace`, as long as the `None` path does no extra tensor work.

### Cache comparison (app)

The app keeps, from the previous pass, its model input tokens and every layer's `k` and `v`. After a new pass:

- Let `p` be the length of the longest common prefix of the previous and new model input token lists.
- For each layer, head and token position `i`:
  - **`read`:** `i < p`, and both the k and v rows match the previous pass within a tolerance *relative* to the row: max |Δ| ≤ 1e-4 · max(1, max |row|). Builder's call, recorded here: K/V are recomputed through a different matmul shape each pass, and an absolute 1e-4 could misfire on large deep-layer rows. The plan must measure the real max relative Δ on `artifacts/hf` over a few appended tokens and confirm that it sits well under the tolerance.
  - **`write`:** `i ≥ p`, so the token is new at this position.
  - **`recomputed`:** `i < p` but the values differ. Not expected; shown honestly if it happens.
- "Window slid" means the **model input** (capped at `min(256, context_size)`) dropping its oldest token. It does *not* mean the 8-token display window moving. When the model input slides, every position shifts, `p = 0`, and every row is `write`. The same happens on the first pass, and after an edit that changes the first token.
- An edit that only appends text gives a long common prefix, so those rows are correctly `read`. That matches how real prefix caching behaves.
- "Previous pass" means the immediately preceding forward pass, whatever triggered it. The stored previous K/V **survive `Esc` and prompt edits** (builder's call): typing more text and pressing Enter compares against the last pass, so a shared prefix is correctly `read`. Re-running an unchanged prompt therefore shows every row `read`, which is accurate.
- A `recomputed` tag that genuinely appears is shown, never hidden. Since it is unexpected, the plan should treat a reproducible one as a finding to investigate.
- The comparison runs once per pass. Its results, a per-layer per-position tag, are stored for the head zoom and the cache map. Since the tag covers both k and v, store it per layer and position, plus the head if the builder prefers. Any head that differs makes the row `recomputed`.

### Head zoom (`z`)

- Available in inference mode once the selected layer is ≥ 1, meaning a block has been reached. It replaces the middle band (the Embeddings and Attention panels). `z` or `Esc` returns to the normal view. `Esc` inside a zoom closes the zoom only, it does not go back to edit mode.
- **Keys inside the zoom:**
  - ↑↓ select the query token among the window's tokens, defaulting to the last token.
  - ←→ change the head.
  - `[` / `]` change the layer (user's choice), clamped to 1..=the deepest block reached. The zoom title shows the layer.
  - Enter and Space still advance the stage. The zoom stays open and follows the newest layer, like the main view.
  - The sampler keys behave as in the main view.
- Layout, top to bottom (see the sketch):
  1. **q row:** the query token's label, then 64 cells (`head_dim`). Diverging grey: 0 is mid-grey, scaled per row by max |value|.
  2. **Key rows** (one per window token up to and including the query, because of causality). Each row has:
     - the token label;
     - **q ⊙ k** as 64 cells, the element-wise product with the same diverging grey, scaled by the max |value| over all key rows of this query so rows can be compared;
     - the raw score (`+4.1` format, from `scores`);
     - the softmax weight as a bar with its value (from `attention`);
     - the cache tag (`read` / `write` / `recomputed`).
  3. **Value rows**, one per key row:
     - the token label;
     - the weight bar with its value and `×`;
     - v as 64 cells (diverging, scaled per row);
     - the cache tag.

     The weight bars sit in a fixed-width column so the vectors line up.
  4. **`= out` row:** the weighted sum `Σ weight × v`, computed in the app from the recorded weights and v. Label: `head output (before W_O)`. Diverging grey, scaled per row. The app computes this sum itself, but only from recorded values. The existing test, which rebuilds the MHA output from the weights and V, already shows those values are the model's.
  5. **KV-cache map:** one row per layer **reached so far** (stage-gated like the rest of the screen, builder's call; rows for unreached layers are blank) and one column per window token. Each cell shows that layer and position's tag: dim for `read`, bright for `write`, red for `recomputed`, with a `← write` marker on the newest column. Its header row holds the token labels.
- **Zoom across passes (builder's call):** an open zoom stays open when Enter commits a draw and the new pass starts. At layer 0 (`Embedding`) it shows its frame with an empty body, and it fills again from `Block(1)`. The selected query or MLP token re-defaults to the **new last token** on every new pass, since the new token is the interesting one. The selected head persists.
- Keys outside the window (tokens before the last 8) do not appear. Weights in the zoom can therefore sum to less than 1, as already accepted in the base spec.

```
┌ Head 3·7 · query "mat" ─────────────────────────────────────────────────────────────────────────────┐
│ q  mat    ▓▓░▒▒█░▒░▓▒▒░█▓▓░▒▒█░▒░▓▒▒░█▓▓░▒▒█░▒░▓▒▒░█▓▓░▒▒█░▒░▓▒▒░█▓▓░▒▒█░▒░                          │
│           q ⊙ k                                                             score  softmax          │
│ k  The    ░░▒▓▒░░▒▒░▒░▓░░░▒▓▒░░▒▒░▒░▓░░░▒▓▒░░▒▒░▒░▓░░░▒▓▒░░▒▒░▒░▓░░░▒▓▒░░▒   +0.4   ▏      0.03   read  │
│ k  cat    ███▓█▓██▒██▓█▓███▓█▓██▒██▓█▓███▓█▓██▒██▓█▓███▓█▓██▒██▓█▓███▓█▓██   +4.1   ██████ 0.81   read  │
│ k  mat    ▓░▒░░▓▒░▒░▓░░▓░▒░░▓▒░▒░▓░░▓░▒░░▓▒░▒░▓░░▓░▒░░▓▒░▒░▓░░▓░▒░░▓▒░▒░▓   +0.8   ▎      0.07   write │
│ v  The    ▏      0.03 ×  ▓▒░░▒▓█░░▒▓▒░░▓▒░░▒▓█░░▒▓▒░░▓▒░░▒▓█░░▒▓▒░░▓▒░░▒▓█░░▒▓▒░░                read  │
│ v  cat    ██████ 0.81 ×  ░▒▓▒█░▒░▓▒▒░█░░▒▓▒█░▒░▓▒▒░█░░▒▓▒█░▒░▓▒▒░█░░▒▓▒█░▒░▓▒▒░█░                read  │
│ v  mat    ▎      0.07 ×  ░▓▒▒░█▒░░▓░▒▒░░▓▒▒░█▒░░▓░▒▒░░▓▒▒░█▒░░▓░▒▒░░▓▒▒░█▒░░▓░▒▒░                write │
│ = out                    ░▒▓▒█░▒░▓▒▒░▓░░▒▓▒█░▒░▓▒▒░▓░░▒▓▒█░▒░▓▒▒░▓░░▒▓▒█░▒░▓▒▒░▓░  head output (before W_O) │
│ KV cache   The  cat  sat  on   the  mat                                                             │
│ layer  1   ░░░  ░░░  ░░░  ░░░  ░░░  ███  ← write                                                    │
│ …                                                                                                   │
│ layer 12   ░░░  ░░░  ░░░  ░░░  ░░░  ███                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- Fitting the height is the builder's call. With up to 8 key rows, 8 value rows and 12 cache-map rows, the zoom needs about 34 rows. If it doesn't fit, for example at 120×40 with a 3-row prompt, a 4-row token panel and a 2-row footer, the builder may compact the cache map. One option is to show only the selected layer plus the newest-column summary. Another is to let the zoom take the token panel's rows too. Record the choice in the plan.

### Score toggle (`s`)

- Active in inference mode, in the main view. In the head zoom, both scores and weights are always visible.
- The attention panel title shows `weights` or `scores`.
- In scores mode, each cell uses the same diverging grey, scaled by the max |finite value| of the visible window rows. Masked cells (above the diagonal) stay black, the same as in weights mode.
- The toggle persists across stages and passes until it is pressed again.

### MLP zoom (`m`)

- Available under the same conditions as the head zoom. It replaces the middle band. `m` or `Esc` returns.
- **Keys:** ↑↓ select the token (default: the last). `[` / `]` change the layer, as in the head zoom. ←→ do nothing. Enter, Space and the sampler keys behave as in the main view.
- `z` and `m` switch between the two zooms directly.
- **Layout:**
  1. **`in` row:** `mlp_in[token]` projected on the 16 PCA directions, using the same projection and per-row diverging grey as the embeddings panel. Label: `after LayerNorm`.
  2. **`hidden` grid:** 3,072 values drawn with half-block characters (`▀`). Each cell is two neurons, the top one in the foreground colour and the bottom one in the background colour. That makes 64 columns × 24 rows; the neuron at row r, column c is index `r × 64 + c`. Grey: `≤ 0` is black and the token's max value is white, linear in between. (GELU's negative outputs are small, at least −0.17, and draw as black.)
  3. **`out` row:** `mlp_out[token]` on the PCA directions, with per-row diverging grey. Label: `added to residual`.
- Both rows are projected onto the PCA basis **without subtracting the embedding mean** (builder's call): `mlp_out` is a change added to the residual, not a position, and `mlp_in` is LayerNorm-normalized, so the embedding table's mean means nothing for either. Residual rows elsewhere keep the mean subtraction. `Pca` gains a `project_direction(&self, v) -> Vec<f32>` (basis · v, no centering).
- The PCA basis was fitted on token embeddings, so these rows show the same 16 directions as the embeddings panel. The base spec already accepts that drift.

### Compact output panel

- One panel, `Next token · T=… top-k … top-p …`, replaces Next token and Sampler. One row per token, for the top 10 at T=1:
  - the token label;
  - the model-probability bar with its value;
  - the after-T/k/p bar with its value, or `✗ cut`;
  - `◀ drawn` in `Drawn`.
- As before, the drawn token gets an extra row when it is outside the top 10.
- **Visibility:**
  - The model-probability column shows from `Output` on.
  - The after-T/k/p column and the settings in the title show from `Output` on, since the sampler keys already work in `Output`. The deferred minor "tuning blind in `Output`" is resolved this way.
  - The `Sampler` stage itself stays, the stage where Enter draws.
- **Height:** 10 rows, plus 1 header, 1 drawn row and 2 borders, for 14 rows. This frees one panel's worth of rows for the middle band.

### Footer

In inference mode the footer adds `z head · m mlp · s scores`. Inside a zoom, it shows the keys valid there, for example `↑↓ query · ←→ head · [ ] layer · z/Esc close`. It stays 2 rows with wrapping. All other strings are unchanged.

## Testing

- **Trace:**
  - `q`/`k`/`v`/`scores`/MLP fields have the stated shapes on `test_gpt_config`.
  - `softmax(scores) == attention` within 1e-5.
  - `mlp_out[i] + (residual after attention)[i] == residual[i]` within 1e-5. This proves `mlp_out` is what the model added.
  - `mlp_in == layer_norm2(residual after attention)` by the same identity, or by an equivalent check that the builder justifies.
  - Rebuilding the head output from the recorded `attention` and `v` matches the existing rebuild test.
- **Cache comparison:**
  - With prompt "ab", then a pass after appending "c", rows 0–1 are `read` and row 2 is `write`, on every layer.
  - A slid window (with a small `context_size`) gives all `write`.
  - A first pass gives all `write`.
  - Doctoring one k value in the stored previous pass gives `recomputed` for that row.
- **App state:**
  - `z` and `m` open and close, are ignored at layer 0 and in edit mode, and switch between each other.
  - `Esc` inside a zoom closes only the zoom.
  - ↑↓ inside a zoom select the token, clamped to the window. Outside a zoom they still change the layer.
  - `[` / `]` inside a zoom change the layer, clamped to 1..=reached. Outside a zoom they do nothing.
  - `s` toggles and persists.
- **Render** (`TestBackend`):
  - The head zoom at 120×40 shows `head output (before W_O)`, `read`, `write`, and a score value.
  - The MLP zoom shows `after LayerNorm` and `added to residual`.
  - The compact panel shows `✗ cut` and `◀ drawn`.
  - Nothing panics at 40×12, 80×24, 120×40 or 200×60, in any stage and zoom.
- **Manually, on `artifacts/hf`:** step to a block, open both zooms, change the query and head, toggle scores, draw two tokens, and watch the cache map's newest column move.

## Left to the builder

- Exact column widths, the colours for `read`/`write`/`recomputed`, and the bar widths.
- How the cache tags are stored.
- How the zoom fits at 120×40 (see "Head zoom").
- Whether `scores` keeps `-inf` or uses a sentinel for masked cells, as long as the display shows masked cells as black.

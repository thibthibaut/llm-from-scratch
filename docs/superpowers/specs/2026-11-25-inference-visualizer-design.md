# Inference Visualizer — Design

Status: reviewed by a builder check, awaiting user approval · 2026-11-25

## Intent

A terminal app (ratatui) that shows how this repo's GPT-2-small model works inside, one inference stage at a time. **Viewers drive it themselves.** It shows **pure visualization**: numbers, shading, bars and token labels, with no explanatory sentences ("this token looks at *cat*" is out). A viewer types a prompt, watches it split into tokens, then presses Enter to walk through the embeddings, each of the 12 transformer blocks, the output distribution, and the sampler drawing the next token.

## Goals

1. Live tokenization: as the viewer types, the token pieces and their ids update. Typing never runs the model.
2. Step-by-step inference: each Enter reveals one more stage of the current forward pass.
3. Embeddings: one row per token, 16 values shown as grey cells. The rows update after each block, so the viewer sees the residual stream change.
4. Attention: one matrix for the selected layer and head, picked with the arrow keys, over a window of up to 8 tokens.
5. Output distribution: a horizontal bar chart of the top probabilities.
6. Sampler: model probabilities next to the probabilities after temperature, top-k and top-p, with tokens cut by k/p marked and the drawn token highlighted. **Temperature, top-k and top-p can be tuned live**, and the bars redraw instantly.
7. The app shows the real model: every value displayed comes from the same forward code path as training and `generate-text`.

## Non-goals

- Explanatory text, tutorials, or tooltips.
- Mouse support. Arrow keys only; clicking could come later.
- Scrolling the 8-token window back through a longer prompt. The window always shows the last 8 tokens.
- MLP neuron views, or splitting a block into attention and MLP sub-steps.
- GPU performance work, batching, KV caching.
- Exporting, recording, or saving sessions.

## Anti-goals (a failure even if it runs)

- Showing values that the real model does not compute, for example from a re-implemented forward pass that can drift from `model.rs`.
- Changing training numerics or speed. The trace must cost nothing when it is off.
- Noticeable UI stalls: a forward pass must stay well under a second on CPU for a prompt within the cap.

## Constraints

- Same crate, same `LibTorch<f32>` backend and weights as `generate-text`. Load `config.json` and the model record exactly as `run_generate_text` does (`src/main.rs:502`), and take the tokenizer from the config, never from a flag.
- New dependencies, for the visualizer only: `ratatui` and crossterm (through ratatui's re-export `ratatui::crossterm`). Use the latest ratatui release.
- PCA uses the `tch` crate (already a dependency): `linalg_eigh` on a 768×768 covariance.
- Prompt cap: 256 tokens, so that the stored trace stays small. That is 12 layers × 12 heads × 256² × 4 B ≈ 38 MB, well within limits. Past the cap, the oldest tokens drop off, as `generate-text` does at the context size.

## Approaches considered

1. **Optional trace recorder inside the existing forward (chosen).** The forward pass threads an `Option<&mut Trace>` through. When it is `Some`, each block records its residual output, and attention also computes `softmax(QKᵀ/√d + causal mask)` explicitly, next to the fused kernel. There is one code path, and `None` costs nothing.
2. A separate `forward_traced` copy: training code untouched, but two forward passes that can silently diverge. Rejected because of the first anti-goal.
3. A background inference thread: unneeded, since one CPU pass takes about 80 ms, measured with `generate-text` on this laptop.

The user also chose **a separate binary** over a subcommand of the existing CLI.

## Design

### Crate layout

- `src/lib.rs` (new): `pub mod dataset, loader, model, ops, sampling, tokenizer, training;`, plus `torch_rng_lock`, moved from `main.rs`. Tests keep calling it as `crate::torch_rng_lock`. `load_tokenizer` and `parse_device` also move here (or into `tokenizer.rs` / a small `cli` helper), so both binaries share them.
- `src/main.rs`: unchanged CLI behavior. It imports from `llm_from_scratch::…` instead of declaring `mod`s.
- `src/bin/visualize.rs` (new) plus a `src/viz/` module under the lib, or `src/bin/visualize/` with submodules. That's the builder's call. Suggested split: `trace`/PCA, app state, and rendering.
- Launch: `cargo run --release --bin visualize -- --config-path artifacts/hf/config.json --model-path artifacts/hf/model [--device cpu]`. The defaults are `artifacts/config.json` / `artifacts/model`, like `generate-text`.

### Trace (in `src/model.rs`)

```rust
pub struct Trace {
    /// Token + position embedding, [seq, d_model].
    pub embedding: Vec<Vec<f32>>,
    /// One per transformer block, in order.
    pub blocks: Vec<BlockTrace>,
    /// Logits at the last position, [vocab].
    pub last_logits: Vec<f32>,
}
pub struct BlockTrace {
    /// softmax(QKᵀ/√head_dim + causal mask), [heads][seq][seq], f32.
    pub attention: Vec<Vec<Vec<f32>>>,
    /// Residual stream after the block, [seq, d_model].
    pub residual: Vec<Vec<f32>>,
}
```

The exact field types (nested `Vec`s or flat buffers with dims) are the builder's call. These are the contents and shapes.

- `GPTModel::forward_with`, `TransformerBlock::forward_with` and `MultiHeadAttention::forward_with` keep their signatures. Each delegates to a private `…_impl(…, trace: Option<&mut …>)`, which is the single code path.
- `pub fn GPTModel::forward_traced(&self, indices: Tensor<B, 2, Int>) -> Trace`. Batch must be 1, and it runs at `Precision::F32`.
- In the attention, when tracing, compute the weights from the same `queries`/`keys` tensors that feed `causal_attention`. The fused output stays what the block uses.
- `pub fn GPTModel::token_embedding_weights(&self) -> Tensor<B, 2>` (`[vocab, d_model]`), used for the PCA.

### PCA basis

At startup, take the token embedding table `[32000, 768]`, center it on the column means, form the 768×768 covariance, and run `linalg_eigh`. Keep the 16 eigenvectors with the largest eigenvalues, plus the mean.

Each displayed row is `(vector − mean) · basisᵀ` → 16 values. The same basis is used for the input embedding and for every block's residual, so the viewer sees the same 16 directions change block by block. The user accepted that deep blocks drift away from this space.

### Modes

The app has two modes, like vim. The viewer must always be able to tell which one they are in.

- **Edit mode** (the `Editing` stage): every printable key, Space included, plus Backspace, edits the prompt. Tokens update live. Nothing else runs, so input can never arrive in the middle of inference. Enter on a non-empty prompt switches to inference mode. Enter on an empty prompt does nothing.
- **Inference mode** (every other stage): the prompt is locked. Keys are commands: Enter, Space, the arrows, and the sampler keys. `Esc` discards the trace and returns to edit mode, keeping the text.
- `Ctrl+C` quits from either mode.
- **Showing the mode:** a mode badge (`EDIT` / `INFERENCE`) in the title bar. The prompt panel has a highlighted border and a visible cursor only in edit mode. A footer line lists the keys valid in the current mode (and, in inference mode, the stage). These are labels and key hints, not explanations, so they are within the "pure visualization" goal.

### Stages and keys

In inference mode the viewer is always at one stage:

| Stage | Enter does | What is visible |
|---|---|---|
| `Editing` (edit mode) | Runs `forward_traced` once on the current tokens, switches to inference mode at `Embedding` | prompt, tokens |
| `Embedding` | goes to `Block(1)` | + embeddings (input) |
| `Block(n)`, n = 1..=12 | goes to `Block(n+1)`, or `Output` after 12 | + embeddings after block n, attention for blocks ≤ n |
| `Output` | goes to `Sampler` | + top-probability bars |
| `Sampler` | draws a token and goes to `Drawn`. The settings are frozen from here | + sampler panel, live-tunable |
| `Drawn` | appends the drawn token, re-runs `forward_traced`, goes to `Embedding` of the new pass | + `◀ drawn` marker on the drawn token, listed even when it is outside the top 10 |

- The draw takes two steps (the user's decision): the first Enter shows which token was drawn, and the second Enter commits it.
- The model runs only once per pass, on leaving `Editing` or `Drawn`. The stages only reveal what the trace already holds.
- **Layer selection:** ↑/↓ select a layer from 0 (input embedding) up to the deepest block reached so far. Moving to a new stage sets the selection to the newest layer. The embeddings panel shows the selected layer's residual, and the attention panel the selected block's attention. At layer 0 the attention panel is empty.
- **Head selection:** ←/→ over heads 1..=12.
- Space jumps straight to `Sampler` (the whole pass revealed). Inference mode only.
- Generated tokens are appended as ids. The prompt box shows `decode(tokens)`. Going back to edit mode and typing re-encodes the full text, so the tokenization can change. That is accepted.
- The full token list is kept. Only the last 256 tokens go to the model, as `generate-text` truncates at the context size.
- Sampler keys are active in the `Sampler` stage (and may also be active in `Output`, the builder's call):
  - `t`/`T`: temperature ∓0.1, clamped to [0.1, 3.0].
  - `k`/`K`: top-k down/up through off → 1 → 5 → 10 → 15 → …, where 1 = greedy.
  - `p`/`P`: top-p ∓0.05, clamped to (0, 1], where 1.0 = off.
  - The panel redraws on every change. The draw happens only on Enter.
  - Start with the `Sampler::default()` settings: T = 1, no k, no p.

### Sampler support (in `src/sampling.rs`)

Add `pub fn Sampler::distribution(&self, logits: &[f32]) -> Vec<(usize, f32)>`: the kept token ids and their renormalized probabilities, in descending order. `sample` becomes a `WeightedIndex` draw over `distribution`'s result, so the panel and the draw share one code path. The existing 13 tests must still pass unchanged.

### Panels

The layout follows the sketch below; the builder decides sizes.

```
┌ Prompt ─────────────────────────────────────────────────┐
│ The cat sat on the▌                                     │
├ Tokens ─────────────────────────────────────────────────┤
│ [The 464] [␣cat 3797] [␣sat 3332] [␣on 319] [␣the 262]  │
├ Embeddings  layer 3 ──────┬ Attention  layer 3 · head 7 ┤
│ The  ▓▒░▒▓█░░▒▓▒░▒▓░▒     │       The cat sat  on the   │
│ cat  ░▒▓▓░▒█▒░░▓▒▒░▓▓     │ The    █                    │
│ sat  ▒░░▓█▒▒░▓░▒▓░▒▓▒     │ cat    ▓   █                │
│ …                         │ sat    ░   ▓   █            │
├ Next token ───────────────┴─────────────────────────────┤
│ ␣mat   ██████████████████ 0.31                          │
│ ␣floor ███████ 0.12                                     │
├ Sampler  T=0.8  top-k 50  top-p 0.90 ───────────────────┤
│            model (T=1)      after T, k, p               │
│ ␣mat    ██████████ 0.31    ██████████████ 0.44  ◀ drawn │
│ ␣table  ░░ 0.02            ✗ cut                        │
└─────────────────────────────────────────────────────────┘
```

- **Grey cells:** draw them as background-colored cells with `Color::Rgb(v, v, v)` (true colour). No dithering is needed.
- **Embedding cells:** 0 = mid-grey, negative darker, positive lighter, normalized per row by that row's max |value|, because the residual's size grows with depth.
- **Attention:** an 8×8 window of the last 8 tokens, queries as rows and keys as columns. Weight 0 = black, 1 = white, on an absolute scale with no normalization, since the rows are probabilities. Cells above the diagonal are masked by causality and stay black. Keys before the window are not shown, so a visible row can sum to less than 1. That is accepted.
- **Token labels:** `token_to_piece` returns raw byte-level BPE pieces. Map `Ġ`→`␣` and `Ċ`→`↵` for display.
- **Next-token bars:** the top 10 at T = 1, using ratatui `BarChart` (horizontal) or plain gauge rows.
- **Sampler panel:** for the top 10 model tokens, the model probability and the probability after filtering, with tokens cut by k/p marked `✗`. After a draw, the drawn token is marked, and it is listed even when it falls outside the top 10.

## Testing

- `forward_traced(x).last_logits` equals `forward(x)` at the last position, within 1e-5.
- Each attention row sums to 1 within 1e-4, and entries above the diagonal are 0.
- For one block, `attention @ V` reproduces the fused attention context, within 1e-4. This proves the recorded weights are the ones the model used.
- The PCA basis is orthonormal: `B Bᵀ ≈ I₁₆`.
- `Sampler::distribution`: its probabilities sum to 1, it obeys top-k and top-p, and `sample` stays consistent with it. Existing tests stay unchanged.
- App state machine, without a terminal:
  - Enter walks `Editing → Embedding → Block(1..12) → Output → Sampler → Drawn → Embedding`, and the token count goes up by 1.
  - In inference mode, letters and Space never change the prompt.
  - `Esc` returns to `Editing` with the text kept.
  - Enter on an empty prompt stays in `Editing`.
  - Layer selection is clamped to the deepest block reached.
  - The sampler keys clamp, and top-k steps off → 1 → 5.
- One render smoke test with ratatui's `TestBackend`: the full layout draws without panicking at 80×24 and at 200×60.
- Manually, on `artifacts/hf`: type a prompt, step through every stage, tune T/k/p, and draw several tokens.

## Left to the builder

The module split for the visualizer, the exact `Trace` storage types, panel sizes, colours other than the grey scales, the ratatui widget choice per panel, and whether the sampler keys also work in `Output`.

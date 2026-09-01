# LLM from scratch

Work in progress.

A small GPT model written from scratch in Rust using [burn](https://burn.dev), with the
`tch` (LibTorch) backend for CUDA training.

## Tokenizer

Two implementations live behind the `Tokenizer` trait in `src/tokenizer.rs`:

- `SimpleTokenizer` — the original word-level tokenizer, built by `create-vocab` into
  `vocab.json`. It is **lossy**: it lowercases, drops all whitespace, drops every non-ASCII
  byte, splits numbers into single digits, has no token for symbols outside
  `[.,!?;:'"()]`, and maps everything past the top-32k words to `<UNK>`.
- `BpeTokenizer` — a byte-level BPE tokenizer backed by the `tokenizers` crate, built by
  `train-tokenizer` into `tokenizer.json`. It is **lossless**: `decode(encode(x)) == x` for
  any input, and it has no `<UNK>` at all, because the trainer is seeded with all 256
  byte-level characters so every possible byte has a token.

Train one:

```bash
cargo run --release -- train-tokenizer \
  --vocab-size 32768 \
  --num-docs 100000 \
  --min-frequency 2 \
  --output tokenizer.json
```

The merges are fitted on a strided sample of the **train split only** — letting the tokenizer
see validation or test documents would leak them into every downstream eval. The command
prints held-out bytes-per-token at the end; that number is how you compare two `--vocab-size`
settings, or compare against a pretrained `tokenizer.json` from the HuggingFace Hub (which
`BpeTokenizer::from_file` loads just as happily).

`--num-docs` is the memory knob, not `--vocab-size`: the trainer holds the word-frequency
table for the whole sample in RAM before it starts merging.

### Choosing one

`train` and `inspect-batch` take `--tokenizer simple|bpe` (default **`bpe`**) plus an optional
`--tokenizer-path`. Both dispatch through `AnyTokenizer`, an enum implementing the same
`Tokenizer` trait — an enum rather than `Box<dyn Tokenizer>` because the dataloader clones the
batcher once per worker and a boxed trait object is not `Clone`.

`generate-text` takes **no** tokenizer flag. It reads the choice back out of the saved
`config.json`, because decoding a checkpoint with the wrong tokenizer does not fail loudly —
the two id spaces overlap, so you just get confident nonsense. `train` records
`tokenizer_kind` and `tokenizer_path` there for exactly this reason, and both commands assert
that the tokenizer's vocab size matches the model's embedding size.

A `config.json` written before those fields existed still loads: they are `Option`, and `None`
resolves to the word-level tokenizer, which is what such a run necessarily used.

The two tokenizers are not interchangeable for an already-trained model — switching changes
`vocab_size`, which changes the shape of the embedding and output head, so it invalidates
existing checkpoints under `artifacts/`. (`train` clears `artifacts/` on every run anyway.)

Eyeball the training data before committing GPU hours to it:

```bash
cargo run --release -- inspect-batch --batch-size 2 --context-size 64 --tokenizer bpe
```

## GPU training setup (RunPod, or any bare CUDA box)

Notes from getting this running on a rented RunPod pod, kept here so a Dockerfile can be
written from them later.

### System packages

```bash
apt-get update
apt-get install -y zstd unzip pkg-config llvm build-essential
```

- `zstd` — decompresses the dataset transfer, and is also required by the git smudge/clean
  filter that keeps `vocab.json` and `tokenizer.json` compressed in the repo (see
  `.gitattributes`, `compress-vocab`, `decompress-vocab`).
- `unzip`, `pkg-config` — precautionary; `torch-sys`'s `download-libtorch` build script and
  its dependencies may shell out to these.
- `llvm` — pulled in after a `rust-lld` crash (`Bus error`, signal 7) while linking
  `equator-macro`. Not confirmed as the actual fix — see "Known issue" below — but it's part
  of the working setup so it's listed here until we understand it better.
- `build-essential` — usually already present on GPU-cloud base images (needed for `cc`/`ld`),
  listed for completeness.

### Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
```

### Build environment variables (managed via direnv)

The repo expects `CARGO_TARGET_DIR`, `TORCH_CUDA_VERSION`, and `RUSTFLAGS` to be set — see
"Known issue" below for why `CARGO_TARGET_DIR` in particular matters here. Rather than
re-exporting these by hand in every new shell/tmux window, use
[direnv](https://direnv.net/) so they're loaded automatically on `cd` into the repo:

```bash
apt-get install -y direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc   # or the hook for your shell
source ~/.bashrc                                 # or open a new shell
```

Then, in the repo root, create `.envrc` (this file is machine-specific — gitignored, not
committed — since the target dir path below only makes sense on the box you're building on):

```bash
cat > .envrc <<'EOF'
export CARGO_TARGET_DIR=/root/build/llm-from-scratch-target
export TORCH_CUDA_VERSION=cu128
export RUSTFLAGS="-C target-cpu=native"
EOF
direnv allow .envrc
```

From then on, any shell that `cd`s into the repo (new SSH session, new tmux pane, etc.) picks
these up automatically — direnv prints `direnv: loading .envrc` / `direnv: export +...` when
it does. No manual `export` needed, and nothing is silently stale from a shell opened before
the vars were set.

```bash
cargo build --release
```

- `CARGO_TARGET_DIR=/root/build/llm-from-scratch-target` — puts build output on the **host**
  disk, not `/workspace`. See "Known issue" below; this isn't just a style choice.
- `TORCH_CUDA_VERSION=cu128` — **required** on recent (Blackwell, `sm_120`) GPUs. The `tch`
  crate's `download-libtorch` fetches libtorch to match; without this it grabs whatever
  `torch-sys` defaults to, which does not include Blackwell kernels. Confirmed by testing
  `torch==2.9.0+cu128`'s `arch_list` directly — it includes `sm_120` and runs a real CUDA
  matmul on an RTX PRO 4000 Blackwell. Older/other GPU generations may need a different
  `TORCH_CUDA_VERSION` (check `nvidia-smi`'s reported compute capability and cross-reference
  against the libtorch build you're pulling).
- `RUSTFLAGS="-C target-cpu=native"` — architecture-specific codegen for our own Rust code.
  Does not affect libtorch's own kernels (that's a prebuilt binary).
- `Cargo.toml` also carries a `[profile.release]` with `lto = "fat"` and `codegen-units = 1`
  for maximum optimization — already committed, nothing extra to set for that.

### Dataset

The training/vocab code loads the fineweb-edu dataset directly from
`~/.cache/burn-dataset/HuggingFaceFWfineweb-edu-sample-10BT.db` (see
`default_fineweb_dataset_path()` in `src/dataset.rs`) — this path is currently hardcoded, not
configurable via a flag.

To move the ~56GB file to a pod: compress locally first (zstd -3 gets ~3x, cutting transfer
time roughly 3x too), transfer, decompress remotely:

```bash
# locally
zstd -3 -T0 ~/.cache/burn-dataset/HuggingFaceFWfineweb-edu-sample-10BT.db

# transfer (adjust host/port/key)
scp -P <port> -i ~/.ssh/id_ed25519 \
  ~/.cache/burn-dataset/HuggingFaceFWfineweb-edu-sample-10BT.db.zst \
  root@<host>:/workspace/

# on the pod — decompress onto the persistent network volume, then point the
# hardcoded cache path at it
zstd -d /workspace/HuggingFaceFWfineweb-edu-sample-10BT.db.zst \
  -o /workspace/HuggingFaceFWfineweb-edu-sample-10BT.db
mkdir -p ~/.cache/burn-dataset
ln -s /workspace/HuggingFaceFWfineweb-edu-sample-10BT.db ~/.cache/burn-dataset/
```

### Disk layout gotcha

RunPod (and likely similar providers) gives a small **container/root disk** (20GB in our
case) plus a much larger **network volume** (`/workspace`) that's persistent across pod
restarts (but not across deleting the pod entirely). Anything that needs to survive a pod
restart — the dataset, the cloned repo — has to live on the network volume.

**The network volume's real capacity is not what `df -h` reports.** `df -h /workspace` shows
the size of the whole shared MooseFS cluster backing it (we saw `2.0P` total, `987T` free) —
that is not your quota. Check what you actually rented via the RunPod API instead:

```bash
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { myself { networkVolumes { id name size dataCenterId } } }"}'
```

In our case that returned a **60GB** volume, not petabytes. This matters a lot for a 56GB
dataset: it barely fits on its own, and doesn't fit at all alongside a cloned repo plus a
10GB+ `target/` directory. `target/` needs to live elsewhere — see below.

The root disk fills up fast too: a throwaway Python venv used just to sanity-check GPU/CUDA
compatibility (`torch==2.9.0+cu128`, see above) alone used 6.7GB. Clean up scratch venvs /
temp downloads from the root disk promptly — there's very little room to spare (20GB total).

### Known issue: `Bus error` (signal 7) linking `equator-macro`

Hit once, partway through `cargo build --release`, while linking a small proc-macro crate,
with `target/` on `/workspace` (the network volume). Leading theory: `rust-lld` mmaps object
files for speed, and network/FUSE-mounted filesystems (MooseFS here) are known to have flaky
mmap semantics — that lines up with a crash on a small, otherwise unremarkable link step.

Installing `llvm` (`apt-get install -y llvm`) was tried first and the subsequent build got
past the same point, but that was never confirmed as the actual fix — `rustc`/`rust-lld`
bundle their own LLVM and don't normally depend on the system `llvm` package, and the crash's
mention of `llvm-symbolizer` is just the crash handler's generic "how to get a readable
backtrace" message, not a diagnosis. It's equally possible that retry just got past a
transient network-filesystem hiccup.

**Resolution:** moved `target/` off `/workspace` entirely, onto host disk, via
`CARGO_TARGET_DIR` in `.envrc` (see above). This also happens to be required regardless of
the Bus error, purely for the disk-space reasons above — a `target/` directory doesn't fit on
the network volume next to the dataset. If a Bus error shows up again with `target/` on host
disk, that would be strong evidence it's genuinely an `llvm` dependency issue rather than the
network filesystem — worth revisiting then.

### Why a Dockerfile would help here

A prebuilt image would bake in the compiled binary (or at least the toolchain + dependency
build cache), so a fresh pod wouldn't need to redo the ~network-filesystem-flaky compile step
described above at all — only the dataset lookup at runtime would touch the network volume,
which is just a file read, not a heavy mmap-based linker workload. Worth revisiting once
training runs stop being one-off experiments.


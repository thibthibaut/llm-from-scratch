//! The training loop.
//!
//! This used to be burn-train's `SupervisedTraining`/`Learner`. That only validates and
//! checkpoints at epoch boundaries, and an epoch over fineweb-edu 10BT is ~150k optimizer steps,
//! so the only way to get a checkpoint was to shrink "epochs" to a few hundred steps by training
//! on a capped slice of the corpus. The loop below walks the *whole* train split every epoch and
//! instead validates and checkpoints every `--eval-every` steps, so a run can be watched,
//! evaluated and resumed (`--resume`) part-way through an epoch.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::dataset::{
    PackedBatch, TextBatcher, TextItem, documents_per_batch, load_default_fineweb_dataset,
    split_dataset,
};
use crate::loader::OrderedLoader;
use crate::model::{GPTModel, GPTModelConfig};
use crate::tokenizer::{AnyTokenizer, Tokenizer, TokenizerKind};
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::ShuffledDataset;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::ElementConversion;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

/// Seed for packing the validation batches, so every evaluation scores the exact same tokens.
const VALID_SEED: u64 = 0x5EED_0F_7A11D;

impl<B: Backend> GPTModel<B> {
    /// Mean next-token cross-entropy over a batch.
    ///
    /// Only the loss: the old burn-train step also returned the full `[batch, seq, vocab]`
    /// logits plus an argmax over them for its metrics, keeping a multi-GB tensor alive per
    /// step for nothing the loop reads.
    pub fn loss(&self, inputs: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [batch_size, seq_len] = inputs.dims();
        let logits = self.forward(inputs);
        let [_, _, vocab_size] = logits.dims();
        let logits = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets = targets.reshape([batch_size * seq_len]);
        CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: GPTModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 256)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 123456)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    /// Which tokenizer produced the token ids this model was trained on, and where it lives.
    ///
    /// Saved so `generate-text` can rebuild the *same* tokenizer instead of guessing. Decoding a
    /// BPE-trained model with the word-level vocab (or vice versa) does not fail loudly — the id
    /// spaces overlap — it just produces confident nonsense.
    ///
    /// `Option` rather than a defaulted field because burn's `#[config(default)]` only feeds the
    /// generated builder, not deserialization; a plain field would make every `config.json`
    /// written before this existed fail to load. `None` means exactly that case, and such a run
    /// was necessarily word-level — see `tokenizer_spec`.
    pub tokenizer_kind: Option<TokenizerKind>,
    pub tokenizer_path: Option<String>,
}

impl TrainingConfig {
    /// The tokenizer this config describes, resolving the pre-`tokenizer_kind` case.
    pub fn tokenizer_spec(&self) -> (TokenizerKind, String) {
        let kind = self.tokenizer_kind.unwrap_or_default();
        let path = self
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| kind.default_path().to_string());
        (kind, path)
    }
}

/// How the loop is driven. Unlike `TrainingConfig` this is not part of what a checkpoint *is*,
/// so it is not saved in `config.json` and can change between a run and its resumption.
#[derive(Clone, Debug)]
pub struct RunOptions {
    /// Validate and checkpoint every this many optimizer steps (and at the end).
    pub eval_every: usize,
    /// Number of batches from the start of the validation split scored at each evaluation.
    pub valid_batches: usize,
    /// Print and record the training loss every this many steps.
    pub log_every: usize,
    /// Stop (after a final checkpoint) once this many steps have run in total.
    pub max_steps: Option<usize>,
    /// Continue from `<artifact_dir>/checkpoint` instead of starting fresh.
    pub resume: bool,
    /// Allow a fresh run to wipe an artifact dir that holds a checkpoint. Without it, a
    /// forgotten `--resume` would silently delete a run's checkpoint.
    pub overwrite: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            eval_every: 1000,
            valid_batches: 30,
            log_every: 50,
            max_steps: None,
            resume: false,
            overwrite: false,
        }
    }
}

/// Where the loop is. Saved in every checkpoint so `--resume` can carry on mid-epoch.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct TrainState {
    /// Current epoch, 0-based.
    epoch: usize,
    /// Batches of the current epoch already trained on; the next one to train is this index.
    batch_in_epoch: usize,
    /// Optimizer steps since the start of the run.
    global_step: usize,
    best_valid_loss: Option<f64>,
    /// What `batch_in_epoch` indexes into: the epoch's batch → documents mapping depends on
    /// the train split's size and the documents per batch. A resume against a different split
    /// (another DB, other batch settings) would silently train on other documents, so these are
    /// checked on `--resume`. `Option` so checkpoints written before they existed still load.
    #[serde(default)]
    train_documents: Option<usize>,
    #[serde(default)]
    docs_per_batch: Option<usize>,
}

type CheckpointRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;

fn checkpoint_dir(artifact_dir: &str) -> PathBuf {
    Path::new(artifact_dir).join("checkpoint")
}

/// The directories a checkpoint can be in, most authoritative first. `checkpoint.tmp` and
/// `checkpoint.old` only hold the latest complete checkpoint if a crash interrupted the
/// renames at the end of `save_checkpoint`; `state.json` is written last, so its presence
/// marks a complete one.
fn checkpoint_candidates(artifact_dir: &str) -> [PathBuf; 3] {
    let dir = Path::new(artifact_dir);
    [dir.join("checkpoint"), dir.join("checkpoint.tmp"), dir.join("checkpoint.old")]
}

fn find_checkpoint(artifact_dir: &str) -> Option<PathBuf> {
    checkpoint_candidates(artifact_dir)
        .into_iter()
        .find(|dir| dir.join("state.json").exists())
}

/// Flush every file in `dir`, and `dir` itself, to disk.
fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        fs::File::open(entry?.path())?.sync_all()?;
    }
    fs::File::open(dir)?.sync_all()
}

/// Save `model` to `<artifact_dir>/<name>.mpk` without ever leaving a truncated file there.
fn save_model_atomically<B: Backend>(model: &GPTModel<B>, artifact_dir: &str, name: &str) {
    let dir = Path::new(artifact_dir);
    model
        .clone()
        // Not `{name}.tmp`: the recorder replaces the extension rather than appending one.
        .save_file(dir.join(format!("{name}-tmp")), &CompactRecorder::new())
        .unwrap_or_else(|err| panic!("save {name}: {err}"));
    fs::rename(dir.join(format!("{name}-tmp.mpk")), dir.join(format!("{name}.mpk")))
        .unwrap_or_else(|err| panic!("move {name} into place: {err}"));
}

/// Write model + optimizer + loop state for `--resume`. Everything goes into
/// `checkpoint.tmp/` (state.json last) and is flushed; only then is the previous checkpoint
/// moved aside to `checkpoint.old/`, the new one renamed into place, and the old one deleted.
/// At every instant one of the three directories holds a complete checkpoint, and
/// `find_checkpoint` knows where to look.
fn save_checkpoint<B: AutodiffBackend, O: Optimizer<GPTModel<B>, B>>(
    artifact_dir: &str,
    model: &GPTModel<B>,
    optim: &O,
    state: &TrainState,
) {
    let [dir, tmp, old] = checkpoint_candidates(artifact_dir);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("create checkpoint dir");
    let recorder = CheckpointRecorder::new();
    model
        .clone()
        .save_file(tmp.join("model"), &recorder)
        .expect("save model checkpoint");
    recorder
        .record(optim.to_record(), tmp.join("optim"))
        .expect("save optimizer checkpoint");
    fs::write(
        tmp.join("state.json"),
        serde_json::to_string_pretty(state).unwrap(),
    )
    .expect("save loop state");
    fsync_dir(&tmp).expect("flush checkpoint to disk");

    let _ = fs::remove_dir_all(&old);
    if dir.exists() {
        fs::rename(&dir, &old).expect("move previous checkpoint aside");
    }
    fs::rename(&tmp, &dir).expect("move checkpoint into place");
    let _ = fs::remove_dir_all(&old);
}

fn load_checkpoint<B: AutodiffBackend, O: Optimizer<GPTModel<B>, B>>(
    artifact_dir: &str,
    model: GPTModel<B>,
    optim: O,
    device: &B::Device,
) -> (GPTModel<B>, O, TrainState) {
    let dir = find_checkpoint(artifact_dir)
        .unwrap_or_else(|| panic!("--resume: no checkpoint in '{artifact_dir}'"));
    if dir != checkpoint_dir(artifact_dir) {
        println!("Resuming from {} (the last save was interrupted)", dir.display());
    }
    let recorder = CheckpointRecorder::new();
    let model = model
        .load_file(dir.join("model"), &recorder, device)
        .expect("load model checkpoint");
    let optim = optim.load_record(
        recorder
            .load(dir.join("optim"), device)
            .expect("load optimizer checkpoint"),
    );
    let state = serde_json::from_str(
        &fs::read_to_string(dir.join("state.json")).expect("read checkpoint state.json"),
    )
    .expect("parse checkpoint state.json");
    (model, optim, state)
}

/// Appends one row per log/eval event to `<artifact_dir>/metrics.csv`.
struct MetricsLog {
    path: PathBuf,
}

impl MetricsLog {
    const HEADER: &str = "step,epoch,tokens,split,loss,tokens_per_sec";

    /// Open (or create) the log. On a resume, rows past `resume_step` are dropped: they were
    /// logged after the checkpoint being resumed from and are about to be logged again.
    fn new(artifact_dir: &str, resume_step: Option<usize>) -> Self {
        let path = Path::new(artifact_dir).join("metrics.csv");
        let existing = fs::read_to_string(&path).unwrap_or_default();
        let kept: Vec<&str> = existing
            .lines()
            .skip(1)
            .filter(|row| {
                let step = row.split(',').next().and_then(|s| s.parse::<usize>().ok());
                matches!((step, resume_step), (Some(step), Some(resume)) if step <= resume)
            })
            .collect();
        let mut contents = format!("{}\n", Self::HEADER);
        for row in kept {
            contents.push_str(row);
            contents.push('\n');
        }
        fs::write(&path, contents).expect("write metrics.csv");
        Self { path }
    }

    fn append(&self, step: usize, epoch: f64, tokens: usize, split: &str, loss: f64, tps: f64) {
        let mut f = OpenOptions::new()
            .append(true)
            .open(&self.path)
            .expect("open metrics.csv");
        writeln!(f, "{step},{epoch:.5},{tokens},{split},{loss:.5},{tps:.0}").unwrap();
    }
}

fn reset_artifact_dir(artifact_dir: &str) {
    if let Err(err) = fs::remove_dir_all(artifact_dir)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        panic!("Failed to clear artifact dir '{artifact_dir}': {err}");
    }
    fs::create_dir_all(artifact_dir)
        .unwrap_or_else(|err| panic!("Failed to create artifact dir '{artifact_dir}': {err}"));
}

/// Mean loss over the (pre-packed) validation batches.
fn validate<B: Backend, T: Tokenizer>(
    model: &GPTModel<B>,
    batcher: &TextBatcher<T>,
    batches: &[PackedBatch],
    device: &B::Device,
) -> f64 {
    let mut total: Option<Tensor<B, 1>> = None;
    for packed in batches {
        let batch = batcher.upload::<B>(packed.clone(), device);
        let loss = model.loss(batch.inputs, batch.targets);
        total = Some(match total {
            Some(t) => t + loss,
            None => loss,
        });
    }
    total.map_or(f64::NAN, |t| {
        t.into_scalar().elem::<f64>() / batches.len() as f64
    })
}

fn format_duration(secs: f64) -> String {
    let secs = secs.max(0.0) as u64;
    format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
}

/// Run training against the given tokenizer and train/valid datasets. Kept generic over the
/// dataset type so tests can supply small in-memory synthetic data instead of the real
/// (very large) fineweb dataset.
///
/// Every epoch walks the whole train split in a fresh shuffled order. Every `eval_every` steps
/// the first `valid_batches` batches of the validation split are scored, `model` (latest) and
/// `best-model` (lowest validation loss) are written for `generate-text`, and a full-precision
/// checkpoint is written to `checkpoint/` for `--resume`.
pub fn train<B: AutodiffBackend, T, D: Dataset<TextItem> + 'static>(
    artifact_dir: &str,
    config: TrainingConfig,
    options: RunOptions,
    device: B::Device,
    tokenizer: T,
    train_dataset: D,
    valid_dataset: D,
) where
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    if options.resume {
        assert!(
            find_checkpoint(artifact_dir).is_some(),
            "--resume: no checkpoint in '{artifact_dir}'"
        );
    } else {
        assert!(
            options.overwrite || find_checkpoint(artifact_dir).is_none(),
            "'{artifact_dir}' holds a training checkpoint. Pass --resume to continue that run, \
             or --overwrite to delete it and start over."
        );
        reset_artifact_dir(artifact_dir);
    }
    // Rewritten on resume too, so it always describes the run (e.g. a changed --num-epochs).
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let context_size = config.model.embedding_config.context_size;
    let batcher = TextBatcher::new(tokenizer, context_size, config.batch_size);
    let tokens_per_step = config.batch_size * context_size;

    // The loader hands the batcher a pool of *documents* per step, sized to cover the token
    // budget of `batch_size` sequences; see `documents_per_batch`.
    let docs_per_batch = documents_per_batch(config.batch_size, context_size);
    let train_dataset = Arc::new(train_dataset);
    let valid_dataset: Arc<dyn Dataset<TextItem>> = Arc::new(valid_dataset);
    let steps_per_epoch = OrderedLoader::num_batches(train_dataset.len(), docs_per_batch);
    assert!(
        steps_per_epoch > 0,
        "train split ({} documents) is smaller than one batch ({docs_per_batch} documents)",
        train_dataset.len()
    );
    let available_valid_batches = OrderedLoader::num_batches(valid_dataset.len(), docs_per_batch);
    assert!(
        available_valid_batches > 0,
        "validation split ({} documents) is smaller than one batch ({docs_per_batch} documents)",
        valid_dataset.len()
    );
    let valid_batches = options.valid_batches.min(available_valid_batches);
    let total_steps = (steps_per_epoch * config.num_epochs)
        .min(options.max_steps.unwrap_or(usize::MAX));
    let num_workers = config.num_workers.max(1);
    let prefetch = 4 * num_workers;

    println!(
        "Training: {} sequences x {context_size} tokens per step ({tokens_per_step} tokens, \
         {docs_per_batch} documents), {steps_per_epoch} steps per epoch, {} epochs, \
         {total_steps} steps total. Validating on {valid_batches} batches every {} steps.",
        config.batch_size, config.num_epochs, options.eval_every
    );

    // Validation batches never change, so pack them once.
    let valid_packed: Vec<PackedBatch> = OrderedLoader::new(
        valid_dataset,
        batcher.clone(),
        docs_per_batch,
        0,
        valid_batches,
        VALID_SEED,
        num_workers,
        prefetch,
    )
    .map(|(_, batch)| batch)
    .collect();

    let model = config.model.init::<B>(&device);
    let optim = config.optimizer.init::<B, GPTModel<B>>();
    let (mut model, mut optim, mut state) = if options.resume {
        let (model, optim, state) = load_checkpoint(artifact_dir, model, optim, &device);
        let check = |what: &str, saved: Option<usize>, now: usize| {
            if let Some(saved) = saved {
                assert_eq!(
                    saved, now,
                    "--resume: the checkpoint was taken with {what} = {saved}, but it is now \
                     {now}. Resuming would train on different documents than the interrupted \
                     run; use the same dataset and batch settings."
                );
            }
        };
        check("train documents", state.train_documents, train_dataset.len());
        check("documents per batch", state.docs_per_batch, docs_per_batch);
        assert!(
            state.batch_in_epoch < steps_per_epoch,
            "--resume: checkpoint is at batch {} of an epoch of {steps_per_epoch} batches",
            state.batch_in_epoch
        );
        println!(
            "Resuming at step {} (epoch {}, batch {}/{steps_per_epoch})",
            state.global_step, state.epoch, state.batch_in_epoch
        );
        (model, optim, state)
    } else {
        let state = TrainState {
            train_documents: Some(train_dataset.len()),
            docs_per_batch: Some(docs_per_batch),
            ..TrainState::default()
        };
        (model, optim, state)
    };
    // Checkpoints from before these fields existed get them filled in from here on.
    state.train_documents = Some(train_dataset.len());
    state.docs_per_batch = Some(docs_per_batch);

    let metrics = MetricsLog::new(artifact_dir, options.resume.then_some(state.global_step));
    let run_start = Instant::now();
    let run_start_step = state.global_step;

    // Runs validation and writes everything a later `generate-text` or `--resume` needs.
    let evaluate = |model: &GPTModel<B>, optim: &_, state: &mut TrainState| {
        let valid_loss = validate(&model.valid(), &batcher, &valid_packed, &device);
        let epoch_frac = state.epoch as f64 + state.batch_in_epoch as f64 / steps_per_epoch as f64;
        let improved = state.best_valid_loss.is_none_or(|best| valid_loss < best);
        println!(
            "[eval] step {} | epoch {epoch_frac:.3} | valid loss {valid_loss:.4} | ppl {:.2}{}",
            state.global_step,
            valid_loss.exp(),
            if improved { " (best)" } else { "" }
        );
        metrics.append(
            state.global_step,
            epoch_frac,
            state.global_step * tokens_per_step,
            "valid",
            valid_loss,
            0.0,
        );
        if improved {
            state.best_valid_loss = Some(valid_loss);
            save_model_atomically(model, artifact_dir, "best-model");
        }
        save_model_atomically(model, artifact_dir, "model");
        save_checkpoint(artifact_dir, model, optim, state);
    };

    let mut interval_loss: Option<Tensor<B::InnerBackend, 1>> = None;
    let mut interval_steps = 0usize;
    let mut interval_start = Instant::now();

    'epochs: while state.epoch < config.num_epochs && state.global_step < total_steps {
        // A different, reproducible order every epoch.
        let shuffled: Arc<dyn Dataset<TextItem>> = Arc::new(ShuffledDataset::new(
            train_dataset.clone(),
            config.seed.wrapping_add(state.epoch as u64),
        ));
        let batch_seed = config.seed ^ ((state.epoch as u64) << 32);
        let loader = OrderedLoader::new(
            shuffled,
            batcher.clone(),
            docs_per_batch,
            state.batch_in_epoch,
            steps_per_epoch,
            batch_seed,
            num_workers,
            prefetch,
        );

        for (index, packed) in loader {
            let batch = batcher.upload::<B>(packed, &device);
            let loss = model.loss(batch.inputs, batch.targets);

            // Accumulate on the device: reading the loss back every step would make the CPU wait
            // for the GPU to drain each step before it could queue the next one.
            let detached = loss.clone().detach().inner();
            interval_loss = Some(match interval_loss {
                Some(sum) => sum + detached,
                None => detached,
            });
            interval_steps += 1;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(config.learning_rate, model, grads);

            state.global_step += 1;
            state.batch_in_epoch = index + 1;
            if state.batch_in_epoch == steps_per_epoch {
                // Roll over now rather than after the loop, so a checkpoint taken on the last
                // step of an epoch already points at the start of the next one.
                state.epoch += 1;
                state.batch_in_epoch = 0;
            }

            if state.global_step.is_multiple_of(options.log_every) {
                let mean = interval_loss.take().unwrap().into_scalar().elem::<f64>()
                    / interval_steps as f64;
                let elapsed = interval_start.elapsed().as_secs_f64();
                let tps = (interval_steps * tokens_per_step) as f64 / elapsed;
                let run_rate = (state.global_step - run_start_step) as f64
                    / run_start.elapsed().as_secs_f64();
                let eta = (total_steps - state.global_step) as f64 / run_rate;
                let epoch_frac =
                    state.epoch as f64 + state.batch_in_epoch as f64 / steps_per_epoch as f64;
                println!(
                    "step {}/{total_steps} | epoch {epoch_frac:.3} | loss {mean:.4} | \
                     {tps:.0} tok/s | {:.0} ms/step | eta {}",
                    state.global_step,
                    1000.0 * elapsed / interval_steps as f64,
                    format_duration(eta)
                );
                metrics.append(
                    state.global_step,
                    epoch_frac,
                    state.global_step * tokens_per_step,
                    "train",
                    mean,
                    tps,
                );
                interval_steps = 0;
                interval_start = Instant::now();
            }

            if state.global_step.is_multiple_of(options.eval_every) {
                evaluate(&model, &optim, &mut state);
                // Don't charge the evaluation to the next interval's throughput.
                interval_start = Instant::now();
            }

            if state.global_step >= total_steps {
                break 'epochs;
            }
        }
    }

    if !state.global_step.is_multiple_of(options.eval_every) {
        evaluate(&model, &optim, &mut state);
    }
    println!(
        "Done: {} steps in {}",
        state.global_step - run_start_step,
        format_duration(run_start.elapsed().as_secs_f64())
    );
}

/// Load the tokenizer and the full fineweb dataset from disk, then run `train`.
pub fn train_from_disk<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    options: RunOptions,
    device: B::Device,
) {
    // The config is the single source of truth for which tokenizer this run uses, so that the
    // `config.json` written next to the checkpoints always describes the tokens they were
    // trained on.
    let (kind, path) = config.tokenizer_spec();
    let tokenizer = AnyTokenizer::load(kind, Path::new(&path));
    println!(
        "Tokenizer: {kind} from {path} ({} tokens)",
        tokenizer.get_vocab_size()
    );
    assert_eq!(
        tokenizer.get_vocab_size(),
        config.model.embedding_config.vocab_size,
        "tokenizer vocab size does not match the model's embedding vocab size; the model would \
         be trained against ids it has no embedding row for"
    );

    let fine_web_dataset = load_default_fineweb_dataset();
    let (train_ds, valid_ds, _test_ds) = split_dataset(fine_web_dataset);
    println!(
        "Dataset: {} train / {} validation documents",
        train_ds.len(),
        valid_ds.len()
    );
    train::<B, _, _>(artifact_dir, config, options, device, tokenizer, train_ds, valid_ds);
}

// ----------- TESTS -----------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        EmbeddingModuleConfig, GPTModelConfig, MultiHeadAttentionConfig, TransformerBlockConfig,
    };
    use crate::tokenizer::{SimpleTokenizer, Token, Tokenizer, Vocab};
    use burn::backend::{Autodiff, LibTorch};
    use burn::data::dataset::InMemDataset;
    use std::collections::HashMap;

    type TestBackend = Autodiff<LibTorch<f32>>;

    /// A tiny hand-built vocabulary covering just the words used by `synthetic_dataset`,
    /// so the test doesn't need a real `vocab.json` or the fineweb dataset.
    fn synthetic_tokenizer() -> SimpleTokenizer {
        let tokens2words: Vec<String> = ["<UNK>", "<EOT>", "a", "b", "c", "d", "e", "f", "g", "h"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let words2tokens: HashMap<String, Token> = tokens2words
            .iter()
            .enumerate()
            .map(|(i, word)| (word.clone(), Token(i as u32)))
            .collect();

        SimpleTokenizer::new(Vocab {
            words2tokens,
            tokens2words,
        })
    }

    /// A handful of short synthetic texts, long enough to slice a few tokens of context from.
    fn synthetic_dataset() -> InMemDataset<TextItem> {
        InMemDataset::new(
            [
                "a b c d e f g h",
                "b c d e f g h a",
                "c d e f g h a b",
                "d e f g h a b c",
            ]
            .iter()
            .map(|text| TextItem {
                text: text.to_string(),
            })
            .collect(),
        )
    }

    fn tiny_config() -> TrainingConfig {
        let vocab_size = synthetic_tokenizer().get_vocab_size();
        let embedding_config = EmbeddingModuleConfig::new(4, vocab_size, 8);
        let mha_config = MultiHeadAttentionConfig::new(8).with_num_heads(2);
        let transformer_config = TransformerBlockConfig::new(mha_config);
        let model_config =
            GPTModelConfig::new(embedding_config, transformer_config).with_num_transformer_layers(1);

        TrainingConfig::new(model_config, AdamWConfig::new())
            .with_num_epochs(1)
            .with_batch_size(2)
            .with_num_workers(1)
    }

    fn run(artifact_dir: &str, config: TrainingConfig, options: RunOptions) {
        let _rng = crate::torch_rng_lock();
        train::<TestBackend, _, _>(
            artifact_dir,
            config,
            options,
            Default::default(),
            synthetic_tokenizer(),
            synthetic_dataset(),
            synthetic_dataset(),
        );
    }

    fn state(artifact_dir: &str) -> TrainState {
        let json = std::fs::read_to_string(format!("{artifact_dir}/checkpoint/state.json")).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    /// One full epoch over tiny synthetic data with a tiny model, so it runs in well under a
    /// second instead of the many hours a real epoch over the fineweb dataset would take.
    #[test]
    fn test_train_runs_one_full_epoch_on_synthetic_data() {
        let artifact_dir = std::env::temp_dir().join("llm-from-scratch-test-train-smoke");
        let artifact_dir = artifact_dir.to_str().unwrap();

        run(artifact_dir, tiny_config(), RunOptions { eval_every: 3, overwrite: true, ..RunOptions::default() });

        for file in ["model.mpk", "best-model.mpk", "config.json", "metrics.csv"] {
            assert!(
                Path::new(&format!("{artifact_dir}/{file}")).exists(),
                "{file} should have been written"
            );
        }
        // 4 documents, one document per batch at this size: a full epoch is 4 steps.
        let state = state(artifact_dir);
        assert_eq!(state.global_step, 4);
        assert_eq!(state.epoch, 1);

        std::fs::remove_dir_all(artifact_dir).ok();
    }

    /// Stopping at `--max-steps` and resuming must pick up at the next batch of the same epoch.
    #[test]
    fn test_resume_continues_mid_epoch() {
        let artifact_dir = std::env::temp_dir().join("llm-from-scratch-test-train-resume");
        let artifact_dir = artifact_dir.to_str().unwrap();
        let options = RunOptions { eval_every: 100, ..RunOptions::default() };

        run(artifact_dir, tiny_config().with_num_epochs(2), RunOptions { max_steps: Some(3), overwrite: true, ..options.clone() });
        let first = state(artifact_dir);
        assert_eq!((first.epoch, first.batch_in_epoch, first.global_step), (0, 3, 3));

        run(artifact_dir, tiny_config().with_num_epochs(2), RunOptions { resume: true, ..options });
        let second = state(artifact_dir);
        assert_eq!((second.epoch, second.global_step), (2, 8));

        std::fs::remove_dir_all(artifact_dir).ok();
    }

    /// Logits of the checkpointed model on a fixed input, to compare two runs' weights.
    fn checkpoint_logits(artifact_dir: &str) -> Vec<f32> {
        let device = Default::default();
        let model = tiny_config()
            .model
            .init::<TestBackend>(&device)
            .load_file(
                format!("{artifact_dir}/checkpoint/model"),
                &CheckpointRecorder::new(),
                &device,
            )
            .unwrap();
        let input = Tensor::<TestBackend, 1, Int>::from_data([2, 3, 4, 5].as_slice(), &device)
            .reshape([1, 4]);
        model.forward(input).into_data().to_vec::<f32>().unwrap()
    }

    /// Stop-and-resume must end at the *same weights* as an uninterrupted run, not just the
    /// same step counter: same batches, same AdamW state, same epoch boundaries.
    #[test]
    fn test_resumed_run_matches_uninterrupted_run() {
        let straight = std::env::temp_dir().join("llm-from-scratch-test-resume-straight");
        let split = std::env::temp_dir().join("llm-from-scratch-test-resume-split");
        let (straight, split) = (straight.to_str().unwrap(), split.to_str().unwrap());
        let config = || tiny_config().with_num_epochs(2);
        let options = RunOptions { eval_every: 100, overwrite: true, ..RunOptions::default() };

        run(straight, config(), options.clone());
        run(split, config(), RunOptions { max_steps: Some(5), ..options.clone() });
        run(split, config(), RunOptions { resume: true, overwrite: false, ..options });

        assert_eq!(checkpoint_logits(straight), checkpoint_logits(split));
        // The resumed run's metrics have one row per logged step, no duplicates.
        let metrics = std::fs::read_to_string(format!("{split}/metrics.csv")).unwrap();
        let valid_rows = metrics.lines().filter(|l| l.contains(",valid,")).count();
        assert_eq!(valid_rows, 2, "evaluations at step 5 and at the end:\n{metrics}");

        std::fs::remove_dir_all(straight).ok();
        std::fs::remove_dir_all(split).ok();
    }

    /// A crash between the renames at the end of `save_checkpoint` leaves the latest complete
    /// checkpoint in `checkpoint.tmp/` (or `checkpoint.old/`); `--resume` must find it.
    #[test]
    fn test_resume_finds_checkpoint_left_by_interrupted_save() {
        let artifact_dir = std::env::temp_dir().join("llm-from-scratch-test-resume-interrupted");
        let artifact_dir = artifact_dir.to_str().unwrap();
        let options = RunOptions { eval_every: 100, ..RunOptions::default() };

        run(artifact_dir, tiny_config(), RunOptions { max_steps: Some(2), overwrite: true, ..options.clone() });
        std::fs::rename(
            format!("{artifact_dir}/checkpoint"),
            format!("{artifact_dir}/checkpoint.tmp"),
        )
        .unwrap();

        run(artifact_dir, tiny_config(), RunOptions { resume: true, ..options });
        assert_eq!(state(artifact_dir).global_step, 4);

        std::fs::remove_dir_all(artifact_dir).ok();
    }

    /// Forgetting `--resume` must not wipe a run's checkpoint.
    #[test]
    #[should_panic(expected = "holds a training checkpoint")]
    fn test_fresh_run_refuses_to_delete_a_checkpoint() {
        let artifact_dir = std::env::temp_dir().join("llm-from-scratch-test-refuse-overwrite");
        let artifact_dir = artifact_dir.to_str().unwrap();
        let options = RunOptions { eval_every: 100, ..RunOptions::default() };

        run(artifact_dir, tiny_config(), RunOptions { max_steps: Some(1), overwrite: true, ..options.clone() });
        run(artifact_dir, tiny_config(), options);
    }
}

#[cfg(test)]
mod tokenizer_selection_tests {
    use super::*;
    use crate::model::{
        EmbeddingModuleConfig, GPTModelConfig, MultiHeadAttentionConfig, TransformerBlockConfig,
    };
    use burn::optim::AdamWConfig;

    fn dummy_config() -> TrainingConfig {
        let gpt = GPTModelConfig::new(
            EmbeddingModuleConfig::new(8, 32, 4),
            TransformerBlockConfig::new(MultiHeadAttentionConfig::new(4).with_num_heads(2)),
        );
        TrainingConfig::new(gpt, AdamWConfig::new())
    }

    /// A `config.json` written before the tokenizer fields existed described a word-level run,
    /// so that is what it must deserialize back to — not the new CLI default of `bpe`.
    #[test]
    fn legacy_config_without_tokenizer_fields_loads_as_simple() {
        let mut value = serde_json::to_value(dummy_config()).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.remove("tokenizer_kind");
        obj.remove("tokenizer_path");

        let loaded: TrainingConfig = serde_json::from_value(value).unwrap();
        assert_eq!(loaded.tokenizer_spec(), (TokenizerKind::Simple, "vocab.json".to_string()));
    }

    /// A config written today must round-trip its tokenizer choice, since `generate-text`
    /// rebuilds the tokenizer from exactly these two fields.
    #[test]
    fn tokenizer_choice_round_trips_through_the_config() {
        let config = dummy_config()
            .with_tokenizer_kind(Some(TokenizerKind::Bpe))
            .with_tokenizer_path(Some("tokenizer.json".to_string()));

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"bpe\""), "kind should serialize lowercase: {json}");

        let loaded: TrainingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.tokenizer_spec(), (TokenizerKind::Bpe, "tokenizer.json".to_string()));
    }
}

//! Command-line entry point for the llm-from-scratch project.
//!
//! Subcommands:
//! - `create-vocab`: build the legacy word-level vocabulary from the dataset and write it to
//!   `vocab.json`.
//! - `train-tokenizer [--vocab-size N] [--num-docs N] [--min-frequency N] [--output PATH]`:
//!   train a byte-level BPE tokenizer on the dataset's train split and write it to
//!   `tokenizer.json`.
//! - `train [--d-model N] [--num-heads N] [--num-layers N] [--context-length N] [--batch-size N]
//!   [--tokenizer simple|bpe] [--tokenizer-path PATH]`: train the GPT model. The tokenizer
//!   choice is recorded in `artifacts/config.json`.
//! - `inspect-batch [--batch-size N] [--context-size N]`: initialize the dataloader,
//!   generate a single batch, detokenize it and display it.
//! - `generate-text [--prompt "..."] [--max-new-tokens N] [--config-path PATH] [--model-path PATH]`:
//!   load a trained model (default `artifacts/config.json` / `artifacts/model`) and run
//!   inference with argmax, displaying the top-5 next-token probabilities at each step and
//!   waiting for a key press to advance.

use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use crate::dataset::{
    TextBatch, TextBatcher, documents_per_batch, load_default_fineweb_dataset, split_dataset,
};
use crate::model::{
    EmbeddingModuleConfig, GPTModel, GPTModelConfig, MultiHeadAttentionConfig,
    TransformerBlockConfig,
};
use crate::tokenizer::{AnyTokenizer, BpeTokenizer, SimpleTokenizer, Token, Tokenizer, TokenizerKind};
use crate::training::TrainingConfig;

use burn::Tensor;
use burn::backend::Autodiff;
use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::config::Config;
use burn::data::dataloader::DataLoader;
use burn::data::dataloader::Dataset;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::tensor::Int;
use burn::tensor::activation;

mod dataset;
mod model;
mod tokenizer;
mod training;

const VOCAB_PATH: &str = "vocab.json";
const TOKENIZER_PATH: &str = "tokenizer.json";

// Defaults for `train-tokenizer`.
// A power of two, and small enough that the (untied) embedding + output head stay a sane
// fraction of a 124M-param model: 32768 * 768 * 2 = 50M params of pure lookup table.
const TOKENIZER_VOCAB_SIZE: usize = 32_768;
// Number of *documents* from the train split to fit the merges on. fineweb-edu documents
// average a few KB, so this is on the order of 10^8-10^9 characters — plenty for a 32k
// vocabulary, and the point past which more data stops moving the merge table.
const TOKENIZER_NUM_DOCS: usize = 100_000;
// Drop merges seen fewer than this many times, so one-off noise in the corpus (mojibake,
// base64 blobs, repeated boilerplate hashes) doesn't win vocabulary slots.
const TOKENIZER_MIN_FREQUENCY: u64 = 2;

// Default architecture parameters, tuned to fit comfortably in 6 GB on a Mac.
// `d_model` must be divisible by `num_heads` (so head_dim = d_model / num_heads).
const D_MODEL: usize = 768;
const NUM_HEADS: usize = 12;
const NUM_LAYERS: usize = 12;
const CONTEXT_LENGTH: usize = 1024;
const BATCH_SIZE: usize = 64;
// Burn's checkpointing strategy only fires at epoch boundaries, so a full-corpus epoch on a
// multi-million-row dataset means waiting days for the first checkpoint. Cap the epoch to a
// fixed number of optimizer steps instead; see `split_dataset` for the tradeoff this makes.
// 300 steps at the default batch size is ~19.7M tokens, i.e. a checkpoint every few minutes.
const EPOCH_BATCHES: usize = 300;
// Validation runs every epoch too, so it needs the same kind of cap — otherwise it's the full
// (uncapped) 10% validation split running against a training epoch that's now a small slice of
// the corpus. Keeps EPOCH_BATCHES' 10:1 train/valid ratio.
const VALID_BATCHES: usize = 30;
// High ceiling; in practice runs are stopped manually once the checkpointed loss looks good,
// relying on burn-train's default keep-best-by-validation-loss checkpointing strategy.
const NUM_EPOCHS: usize = 200;
const NUM_WORKERS: usize = 32;

#[derive(Parser, Debug)]
#[command(
    name = "llm-from-scratch",
    about = "Train and inspect a small GPT model from scratch",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build the legacy word-level vocabulary from the dataset and write it to `vocab.json`.
    CreateVocab,
    /// Train a byte-level BPE tokenizer on the dataset and write it to `tokenizer.json`.
    TrainTokenizer {
        /// Target vocabulary size, including the 256 byte tokens and `<|endoftext|>`.
        #[arg(long, default_value_t = TOKENIZER_VOCAB_SIZE)]
        vocab_size: usize,
        /// Number of documents from the train split to fit the merges on. The trainer holds
        /// the whole word-frequency table in RAM, so this is the knob to turn down if you run
        /// out of memory — not `--vocab-size`.
        #[arg(long, default_value_t = TOKENIZER_NUM_DOCS)]
        num_docs: usize,
        /// Ignore merge candidates occurring fewer than this many times in the sample.
        #[arg(long, default_value_t = TOKENIZER_MIN_FREQUENCY)]
        min_frequency: u64,
        /// Where to write the trained tokenizer.
        #[arg(long, default_value = TOKENIZER_PATH)]
        output: String,
    },
    /// Train the GPT model.
    Train {
        /// Embedding dimension (must be divisible by `num-heads`).
        #[arg(long, default_value_t = D_MODEL)]
        d_model: usize,
        /// Number of attention heads per transformer block.
        #[arg(long, default_value_t = NUM_HEADS)]
        num_heads: usize,
        /// Number of transformer blocks in the model.
        #[arg(long, default_value_t = NUM_LAYERS)]
        num_layers: usize,
        /// Context length (max sequence length).
        #[arg(long, default_value_t = CONTEXT_LENGTH)]
        context_length: usize,
        /// Training batch size.
        #[arg(long, default_value_t = BATCH_SIZE)]
        batch_size: usize,
        /// Device to train on: `cpu`, `cuda` (or `cuda:N`), `mps`, `vulkan`.
        #[arg(long, default_value = "cpu", value_parser = parse_device)]
        device: LibTorchDevice,
        /// Number of batches (optimizer steps) in one epoch, so an epoch — and therefore a
        /// checkpoint, since burn-train only checkpoints at epoch boundaries — lands at a
        /// practical cadence instead of requiring a full sweep of the dataset. Each batch is
        /// `--batch-size * --context-length` tokens. 0 means a full pass over the train split.
        #[arg(long, default_value_t = EPOCH_BATCHES)]
        epoch_size: usize,
        /// Number of batches in the validation pass, which runs every epoch and so needs the
        /// same kind of cap as --epoch-size or it dwarfs a capped training epoch. 0 means a
        /// full pass over the validation split.
        #[arg(long, default_value_t = VALID_BATCHES)]
        valid_size: usize,
        /// Number of epochs (bounded train-split passes) before training stops on its own.
        #[arg(long, default_value_t = NUM_EPOCHS)]
        num_epochs: usize,
        /// Number of background dataloader worker threads.
        #[arg(long, default_value_t = NUM_WORKERS)]
        num_workers: usize,
        /// Which tokenizer to train against. Recorded in `artifacts/config.json` so
        /// `generate-text` reconstructs the same one.
        #[arg(long, value_enum, default_value_t = TokenizerKind::Bpe)]
        tokenizer: TokenizerKind,
        /// Override where the tokenizer is loaded from (defaults to `tokenizer.json` for `bpe`,
        /// `vocab.json` for `simple`).
        #[arg(long)]
        tokenizer_path: Option<String>,
    },
    /// Initialize the dataloader, generate a batch, detokenize it and display it.
    InspectBatch {
        /// Number of items in the batch.
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        /// Maximum sequence length per item.
        #[arg(long, default_value_t = 128)]
        context_size: usize,
        /// Which tokenizer to decode the batch with.
        #[arg(long, value_enum, default_value_t = TokenizerKind::Bpe)]
        tokenizer: TokenizerKind,
        /// Override where the tokenizer is loaded from.
        #[arg(long)]
        tokenizer_path: Option<String>,
    },
    /// Load a trained model and run inference on `--prompt`.
    GenerateText {
        /// Prompt text used to seed the generation.
        #[arg(long, default_value = "A Hello. This is a text to transform.")]
        prompt: String,
        /// Maximum number of new tokens to generate.
        #[arg(long, default_value_t = 20)]
        max_new_tokens: usize,
        /// Path to the training config JSON file.
        #[arg(long, default_value = "artifacts/config.json")]
        config_path: String,
        /// Path to the model record to load (no file extension — the recorder appends its
        /// own), e.g. `artifacts/checkpoint/model-4` to load a specific epoch checkpoint
        /// instead of a completed run's final saved model.
        #[arg(long, default_value = "artifacts/model")]
        model_path: String,
        /// Device to run inference on: `cpu`, `cuda` (or `cuda:N`), `mps`, `vulkan`.
        #[arg(long, default_value = "cpu", value_parser = parse_device)]
        device: LibTorchDevice,
    },
}

/// Parse a `--device` CLI value into a `LibTorchDevice`.
fn parse_device(s: &str) -> Result<LibTorchDevice, String> {
    let lower = s.to_ascii_lowercase();
    match lower.as_str() {
        "cpu" => Ok(LibTorchDevice::Cpu),
        "mps" => Ok(LibTorchDevice::Mps),
        "vulkan" => Ok(LibTorchDevice::Vulkan),
        other => {
            if let Some(idx) = other.strip_prefix("cuda") {
                let idx = idx.trim_start_matches(':');
                let index: usize = if idx.is_empty() {
                    0
                } else {
                    idx.parse()
                        .map_err(|_| format!("invalid CUDA device index in '{s}'"))?
                };
                Ok(LibTorchDevice::Cuda(index))
            } else {
                Err(format!(
                    "unknown device '{s}': expected one of cpu, cuda[:N], mps, vulkan"
                ))
            }
        }
    }
}

/// Build the GPT model config from the given architecture parameters.
fn gpt_model_config(
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
) -> GPTModelConfig {
    let embedding_config = EmbeddingModuleConfig::new(context_length, vocab_size, d_model);
    let mha_config = MultiHeadAttentionConfig::new(d_model).with_num_heads(num_heads);
    let transformer_config = TransformerBlockConfig::new(mha_config);
    GPTModelConfig::new(embedding_config, transformer_config)
        .with_num_transformer_layers(num_layers)
}

/// Build the tokenizer vocabulary from the dataset and write it to `vocab.json`.
fn create_vocab() {
    let dataset = load_default_fineweb_dataset();
    let vocab = SimpleTokenizer::build_vocab(&dataset);
    vocab.to_file(Path::new(VOCAB_PATH));
}

/// Train a byte-level BPE tokenizer on the dataset and write it to `output`.
fn train_tokenizer(vocab_size: usize, num_docs: usize, min_frequency: u64, output: &str) {
    let dataset = load_default_fineweb_dataset();

    // Fit the merges on the *train* split only. The tokenizer is part of the model: letting it
    // see validation/test documents leaks those into every downstream eval, because the merge
    // table is then shaped by text the model is later scored on.
    let (train, valid, _test) = split_dataset(dataset, None, None);

    let num_docs = num_docs.min(train.len());
    println!(
        "Training a byte-level BPE tokenizer: vocab_size={vocab_size}, \
         min_frequency={min_frequency}, {num_docs} documents from the train split"
    );

    // Sample with a stride across the whole train split rather than taking the first N rows.
    // fineweb-edu is stored roughly in crawl order, so a contiguous prefix is a biased slice of
    // the corpus (a handful of dumps, and whatever sites dominate them) and would bake that
    // bias into the merge table. Striding costs random-access seeks instead of a sequential
    // scan, which is the slow part of this command on a network volume — but it only happens
    // once, and the merge table it produces is the one you keep.
    let stride = (train.len() / num_docs).max(1);
    let texts = (0..num_docs).map(move |i| {
        train
            .get(i * stride)
            .expect("strided index stays within the train split")
            .text
    });
    let tokenizer = BpeTokenizer::train(texts, vocab_size, min_frequency);

    let path = Path::new(output);
    tokenizer.to_file(path);
    println!(
        "Wrote {} ({} tokens, <|endoftext|> = {})",
        path.display(),
        tokenizer.get_vocab_size(),
        tokenizer.end_of_text().0
    );

    // Score it on held-out documents the merges were never fitted on. This is the number to
    // compare against a pretrained tokenizer, or against another --vocab-size.
    let sample: Vec<String> = (0..1_000.min(valid.len()))
        .map(|i| valid.get(i).expect("index is within the valid split").text)
        .collect();
    if !sample.is_empty() {
        println!(
            "Held-out compression: {:.3} bytes/token over {} validation documents",
            tokenizer.bytes_per_token(sample.iter().map(String::as_str)),
            sample.len()
        );
    }
}

/// Resolve `--tokenizer` / `--tokenizer-path` into a loaded tokenizer and the path it came from.
fn load_tokenizer(kind: TokenizerKind, path: Option<String>) -> (AnyTokenizer, String) {
    let path = path.unwrap_or_else(|| kind.default_path().to_string());
    let tokenizer = AnyTokenizer::load(kind, Path::new(&path));
    println!(
        "Tokenizer: {kind} from {path} ({} tokens)",
        tokenizer.get_vocab_size()
    );
    (tokenizer, path)
}

/// Train the GPT model with the given architecture and training parameters.
fn run_train(
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    context_length: usize,
    batch_size: usize,
    device: LibTorchDevice,
    epoch_size: usize,
    valid_size: usize,
    num_epochs: usize,
    num_workers: usize,
    tokenizer_kind: TokenizerKind,
    tokenizer_path: Option<String>,
) {
    type MyBackend = LibTorch<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let artifact_dir = "artifacts";

    // Load the tokenizer up front purely to size the embedding and output head. The training
    // run reloads it from the config, so there is exactly one place that decides which
    // tokenizer a checkpoint belongs to.
    let (tokenizer, tokenizer_path) = load_tokenizer(tokenizer_kind, tokenizer_path);
    let vocab_size = tokenizer.get_vocab_size();
    drop(tokenizer);

    let gpt_config = gpt_model_config(vocab_size, context_length, d_model, num_heads, num_layers);

    // `--epoch-size` and `--valid-size` are counts of *batches*, because a batch is one
    // optimizer step and that is the unit the checkpoint cadence is actually measured in.
    // `split_dataset` caps by document, though, so convert: the dataloader hands the batcher
    // `documents_per_batch` documents per batch, and that pool size already accounts for
    // `--batch-size` and `--context-length`. Note the consequence of counting steps rather than
    // documents — halving `--batch-size` now halves the tokens an epoch sees, where capping by
    // document held tokens-per-epoch roughly fixed and doubled the step count instead.
    let docs_per_batch = documents_per_batch(batch_size, context_length);
    let batches_to_items = |batches: usize| {
        if batches == 0 {
            None
        } else {
            Some(batches * docs_per_batch)
        }
    };
    let max_train_items = batches_to_items(epoch_size);
    let max_valid_items = batches_to_items(valid_size);

    if let (Some(train_items), Some(valid_items)) = (max_train_items, max_valid_items) {
        println!(
            "Epoch: {epoch_size} batches ({train_items} documents, {} tokens), \
             validation: {valid_size} batches ({valid_items} documents)",
            epoch_size * batch_size * context_length
        );
    }

    crate::training::train_from_disk::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(gpt_config, AdamWConfig::new())
            .with_batch_size(batch_size)
            .with_num_epochs(num_epochs)
            .with_num_workers(num_workers)
            .with_tokenizer_kind(Some(tokenizer_kind))
            .with_tokenizer_path(Some(tokenizer_path)),
        device.clone(),
        max_train_items,
        max_valid_items,
    );
}

/// Initialize the dataloader, generate a single batch, detokenize it and print it.
fn run_inspect_batch(
    batch_size: usize,
    context_size: usize,
    tokenizer_kind: TokenizerKind,
    tokenizer_path: Option<String>,
) {
    type MyBackend = LibTorch<f32>;

    let (tokenizer, _) = load_tokenizer(tokenizer_kind, tokenizer_path);
    let batcher = TextBatcher::new(tokenizer.clone(), context_size, batch_size);

    let dataset = load_default_fineweb_dataset();
    let (train_ds, _valid_ds, _test_ds) = split_dataset(dataset, None, None);

    // Documents per call, not sequences per batch — see `documents_per_batch`.
    let docs_per_batch = documents_per_batch(batch_size, context_size);
    println!(
        "Batching: {batch_size} sequences x {context_size} tokens, packed from a pool of {docs_per_batch}"
    );

    let dataloader: Arc<dyn DataLoader<MyBackend, TextBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(docs_per_batch)
            .shuffle(42)
            .num_workers(1)
            .build(train_ds);

    let batch = dataloader
        .iter()
        .next()
        .expect("Dataloader should yield at least one batch");

    let [actual_batch_size, seq_len] = batch.inputs.dims();
    let inputs_data = batch.inputs.into_data().to_vec::<i64>().unwrap();
    let targets_data = batch.targets.into_data().to_vec::<i64>().unwrap();

    println!(
        "Inspected batch: actual_batch_size={}, seq_len={} (capped by context_size={})",
        actual_batch_size, seq_len, context_size
    );
    println!();

    for i in 0..actual_batch_size {
        let start = i * seq_len;
        let end = start + seq_len;
        let input_tokens: Vec<Token> = inputs_data[start..end]
            .iter()
            .map(|&t| Token(t as u32))
            .collect();
        let target_tokens: Vec<Token> = targets_data[start..end]
            .iter()
            .map(|&t| Token(t as u32))
            .collect();

        println!("=== Batch item {i} ===");
        println!("Input:  {}", tokenizer.decode(&input_tokens));
        println!("Target: {}", tokenizer.decode(&target_tokens));
        println!();
    }
}

/// Generate text with a trained GPT model, one token at a time, using argmax
/// sampling. At each step, the top-5 candidate next tokens (with their softmax
/// probabilities) are printed and the user is prompted to press Enter to reveal
/// the chosen next token.
fn run_generate_text(
    prompt: String,
    max_new_tokens: usize,
    config_path: String,
    model_path: String,
    device: LibTorchDevice,
) {
    type MyBackend = LibTorch<f32>;

    // Load the saved config and trained model
    let config = TrainingConfig::load(config_path)
        .expect("Config should exist at --config-path; run `train` first");
    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .expect("Trained model should exist at --model-path; run `train` first");

    let model: GPTModel<MyBackend> = config.model.init(&device).load_record(record);
    let context_length = config.model.embedding_config.context_size;

    // The tokenizer comes from the saved config, never from a flag: decoding a checkpoint with
    // a different tokenizer than it was trained on produces confident nonsense rather than an
    // error, because the two id spaces overlap.
    let (kind, path) = config.tokenizer_spec();
    let (tokenizer, _) = load_tokenizer(kind, Some(path));
    let vocab_size = tokenizer.get_vocab_size();
    assert_eq!(
        vocab_size, config.model.embedding_config.vocab_size,
        "tokenizer vocab size does not match the trained model's embedding vocab size"
    );

    let mut tokens = tokenizer.encode(&prompt);
    if tokens.is_empty() {
        eprintln!("Prompt produced no tokens; nothing to generate.");
        return;
    }

    println!("Prompt: {}", tokenizer.decode(&tokens));
    println!("Generating up to {max_new_tokens} tokens (argmax, top-5 shown each step).");
    println!();

    for step in 0..max_new_tokens {
        // Truncate context if needed
        let start = if tokens.len() > context_length {
            tokens.len() - context_length
        } else {
            0
        };
        let input_tokens = &tokens[start..];

        // Build input tensor [1, seq_len]
        let indices: Vec<i64> = input_tokens.iter().map(|t| t.0 as i64).collect();
        let input_tensor = Tensor::<MyBackend, 1, Int>::from_data(indices.as_slice(), &device)
            .reshape([1, input_tokens.len()]);

        // Forward pass -> [1, seq_len, vocab_size]
        let output = model.forward(input_tensor);

        // Get logits for the last position -> [1, 1, vocab_size]
        let last_logits = output.slice([
            0..1,
            input_tokens.len() - 1..input_tokens.len(),
            0..vocab_size,
        ]);
        // Reshape to [vocab_size] for softmax
        let last_logits: Tensor<MyBackend, 1> = last_logits.reshape([vocab_size]);

        // Softmax to get probabilities
        let probs = activation::softmax(last_logits, 0);
        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();

        // Find top 5 (token_id, probability) pairs
        let mut indexed: Vec<(usize, f32)> = probs_data
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();

        println!("--- Step {} ---", step + 1);
        println!("Top 5 candidate next tokens:");
        for (rank, (token_id, prob)) in top5.iter().enumerate() {
            let word = tokenizer
                .token_to_piece(Token(*token_id as u32))
                .unwrap_or_else(|| "<invalid id>".to_string());
            println!(
                "  #{}: \"{}\" (id={}, prob={:.4})",
                rank + 1,
                word,
                token_id,
                prob
            );
        }

        // Prompt the user to pick one of the top candidates, defaulting to #1
        // (argmax). The loop re-prompts on invalid input. Typing `q` quits the
        // whole generation, in which case the running text is still printed
        // below.
        let mut quit = false;
        let choice: usize = loop {
            print!(
                "\nChoose a token (1-{}, default 1, or 'q' to quit): ",
                top5.len()
            );
            io::stdout().flush().unwrap();
            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_err() {
                // EOF on stdin: just default to argmax.
                break 1;
            }
            let trimmed = line.trim();

            if trimmed.is_empty() {
                break 1;
            }
            if trimmed.eq_ignore_ascii_case("q") {
                quit = true;
                break 1; // value is unused; we break the for-loop below
            }
            match trimmed.parse::<usize>() {
                Ok(n) if (1..=top5.len()).contains(&n) => break n,
                Ok(n) => println!(
                    "Invalid choice: {} (must be between 1 and {})",
                    n,
                    top5.len()
                ),
                Err(_) => println!(
                    "Invalid choice: \"{}\" (must be a number between 1 and {})",
                    trimmed,
                    top5.len()
                ),
            }
        };

        if quit {
            println!("\nQuitting.");
            break;
        }

        // Pick the chosen candidate (1-indexed in the prompt, 0-indexed in `top5`)
        let best_idx = top5[choice - 1].0;
        let next_token = Token(best_idx as u32);
        tokens.push(next_token);

        let label = if choice == 1 { "default" } else { "user pick" };
        println!(
            "Chosen: \"{}\" (id={}, #{}) [{}]",
            tokenizer
                .token_to_piece(next_token)
                .unwrap_or_else(|| "<invalid id>".to_string()),
            best_idx,
            choice,
            label
        );
        println!("Text so far: {}", tokenizer.decode(&tokens));
        println!();

        // Stop at EOT
        if next_token == tokenizer.end_of_text() {
            println!("Reached end-of-text, stopping.");
            break;
        }
    }

    println!();
    println!("Final text: {}", tokenizer.decode(&tokens));
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::CreateVocab => create_vocab(),
        Commands::TrainTokenizer {
            vocab_size,
            num_docs,
            min_frequency,
            output,
        } => train_tokenizer(vocab_size, num_docs, min_frequency, &output),
        Commands::Train {
            d_model,
            num_heads,
            num_layers,
            context_length,
            batch_size,
            device,
            epoch_size,
            valid_size,
            num_epochs,
            num_workers,
            tokenizer,
            tokenizer_path,
        } => run_train(
            d_model,
            num_heads,
            num_layers,
            context_length,
            batch_size,
            device,
            epoch_size,
            valid_size,
            num_epochs,
            num_workers,
            tokenizer,
            tokenizer_path,
        ),
        Commands::InspectBatch {
            batch_size,
            context_size,
            tokenizer,
            tokenizer_path,
        } => run_inspect_batch(batch_size, context_size, tokenizer, tokenizer_path),
        Commands::GenerateText {
            prompt,
            max_new_tokens,
            config_path,
            model_path,
            device,
        } => run_generate_text(prompt, max_new_tokens, config_path, model_path, device),
    }
}

//! Command-line entry point for the llm-from-scratch project.
//!
//! Subcommands:
//! - `create-vocab`: build the tokenizer vocabulary from the dataset and write it to `vocab.json`.
//! - `train`: train the GPT model.
//! - `inspect-batch [--batch-size N] [--context-size N]`: initialize the dataloader,
//!   generate a single batch, detokenize it and display it.
//! - `generate-text [--prompt "..."] [--max-new-tokens N] [--context-length N]`:
//!   generate text with an untrained model using argmax, displaying the top-5
//!   next-token probabilities at each step and waiting for a key press to advance.

use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use crate::dataset::{TextBatch, TextBatcher, load_fineweb_dataset_from_disk, split_dataset};
use crate::model::{
    EmbeddingModuleConfig, GPTModelConfig, MultiHeadAttentionConfig, TransformerBlockConfig,
};
use crate::tokenizer::{SimpleTokenizer, Token, Tokenizer};
use crate::training::TrainingConfig;

use burn::Tensor;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::data::dataloader::DataLoader;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamWConfig;
use burn::prelude::Backend;
use burn::tensor::Int;
use burn::tensor::activation;

mod dataset;
mod model;
mod tokenizer;
mod training;

const DATASET_PATH: &str = "sample_20pct.db";
const VOCAB_PATH: &str = "vocab.json";
const D_MODEL: usize = 144;
const CONTEXT_LENGTH: usize = 512;

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
    /// Build the tokenizer vocabulary from the dataset and write it to `vocab.json`.
    CreateVocab,
    /// Train the GPT model.
    Train,
    /// Initialize the dataloader, generate a batch, detokenize it and display it.
    InspectBatch {
        /// Number of items in the batch.
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        /// Maximum sequence length per item.
        #[arg(long, default_value_t = 128)]
        context_size: usize,
    },
    /// Generate text with an untrained GPT model (argmax, top-5 visible, step-by-step).
    GenerateText {
        /// Prompt text used to seed the generation.
        #[arg(long, default_value = "A Hello. This is a text to transform.")]
        prompt: String,
        /// Maximum number of new tokens to generate.
        #[arg(long, default_value_t = 20)]
        max_new_tokens: usize,
        /// Context length used to truncate the input window.
        #[arg(long, default_value_t = CONTEXT_LENGTH)]
        context_length: usize,
    },
}

/// Build the default GPT model config used by training and generation.
fn gpt_model_config(vocab_size: usize, context_length: usize) -> GPTModelConfig {
    let embedding_config = EmbeddingModuleConfig::new(context_length, vocab_size, D_MODEL);
    let mha_config = MultiHeadAttentionConfig::new(D_MODEL, D_MODEL);
    let transformer_config = TransformerBlockConfig::new(mha_config);
    GPTModelConfig::new(embedding_config, transformer_config)
}

/// Build the tokenizer vocabulary from the dataset and write it to `vocab.json`.
fn create_vocab() {
    let dataset = load_fineweb_dataset_from_disk(DATASET_PATH);
    let vocab = SimpleTokenizer::build_vocab(&dataset);
    vocab.to_file(Path::new(VOCAB_PATH));
}

/// Train the GPT model using the parameters currently configured below.
fn run_train() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let artifact_dir = "artifacts";

    let tokenizer = SimpleTokenizer::from_vocab_file(Path::new(VOCAB_PATH));
    let vocab_size = tokenizer.get_vocab_size();

    let gpt_config = gpt_model_config(vocab_size, CONTEXT_LENGTH);

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(gpt_config, AdamWConfig::new()).with_batch_size(2),
        device.clone(),
    );
}

/// Initialize the dataloader, generate a single batch, detokenize it and print it.
fn run_inspect_batch(batch_size: usize, context_size: usize) {
    type MyBackend = Wgpu<f32, i32>;

    let tokenizer = SimpleTokenizer::from_vocab_file(Path::new(VOCAB_PATH));
    let batcher = TextBatcher::new(tokenizer.clone(), context_size);

    let dataset = load_fineweb_dataset_from_disk(DATASET_PATH);
    let (train_ds, _valid_ds, _test_ds) = split_dataset(dataset);

    let dataloader: Arc<dyn DataLoader<MyBackend, TextBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .shuffle(42)
            .num_workers(1)
            .build(train_ds);

    let batch = dataloader
        .iter()
        .next()
        .expect("Dataloader should yield at least one batch");

    let [actual_batch_size, seq_len] = batch.inputs.dims();
    let inputs_data = batch.inputs.into_data().to_vec::<i32>().unwrap();
    let targets_data = batch.targets.into_data().to_vec::<i32>().unwrap();

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

/// Generate text with an untrained GPT model, one token at a time, using argmax
/// sampling. At each step, the top-5 candidate next tokens (with their softmax
/// probabilities) are printed and the user is prompted to press Enter to reveal
/// the chosen next token.
fn run_generate_text(prompt: String, max_new_tokens: usize, context_length: usize) {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let tokenizer = SimpleTokenizer::from_vocab_file(Path::new(VOCAB_PATH));
    let vocab_size = tokenizer.get_vocab_size();

    let model = gpt_model_config(vocab_size, context_length).init::<MyBackend>(&device);

    let mut tokens = tokenizer.encode(&prompt);
    if tokens.is_empty() {
        eprintln!("Prompt produced no tokens; nothing to generate.");
        return;
    }

    println!("Prompt: {}", tokenizer.decode(&tokens));
    println!(
        "Generating up to {max_new_tokens} tokens (argmax, top-5 shown each step)."
    );
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
        let indices: Vec<i32> = input_tokens.iter().map(|t| t.0 as i32).collect();
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
            let word = &tokenizer.vocab.tokens2words[*token_id];
            println!("  #{}: \"{}\" (id={}, prob={:.4})", rank + 1, word, token_id, prob);
        }

        // Wait for key press to reveal the chosen token
        print!("\nPress Enter to see the chosen token (or 'q' + Enter to quit)... ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        io::stdin().read_line(&mut line).unwrap();
        if line.trim().eq_ignore_ascii_case("q") {
            break;
        }

        // Argmax: pick the highest-probability token
        let best_idx = top5[0].0;
        let next_token = Token(best_idx as u32);
        tokens.push(next_token);

        println!(
            "Chosen: \"{}\" (id={})",
            tokenizer.vocab.tokens2words[best_idx], best_idx
        );
        println!("Text so far: {}", tokenizer.decode(&tokens));
        println!();

        // Stop at EOT
        if next_token.0 == Token::END_OF_TEXT.0 {
            println!("Reached <EOT>, stopping.");
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
        Commands::Train => run_train(),
        Commands::InspectBatch {
            batch_size,
            context_size,
        } => run_inspect_batch(batch_size, context_size),
        Commands::GenerateText {
            prompt,
            max_new_tokens,
            context_length,
        } => run_generate_text(prompt, max_new_tokens, context_length),
    }
}

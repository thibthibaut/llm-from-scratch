//! Command-line entry point for the llm-from-scratch project.
//!
//! Subcommands:
//! - `create-vocab`: build the tokenizer vocabulary from the dataset and write it to `vocab.json`.
//! - `train`: train the GPT model.
//! - `inspect-batch [--batch-size N] [--context-size N]`: initialize the dataloader,
//!   generate a single batch, detokenize it and display it.

use std::path::Path;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use crate::dataset::{TextBatch, TextBatcher, load_fineweb_dataset_from_disk, split_dataset};
use crate::model::{
    EmbeddingModuleConfig, GPTModelConfig, MultiHeadAttentionConfig, TransformerBlockConfig,
};
use crate::tokenizer::{SimpleTokenizer, Token, Tokenizer};
use crate::training::TrainingConfig;

use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::data::dataloader::DataLoader;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamWConfig;

mod dataset;
mod model;
mod tokenizer;
mod training;

const DATASET_PATH: &str = "sample_20pct.db";
const VOCAB_PATH: &str = "vocab.json";

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
        #[arg(long, default_value_t = 32)]
        context_size: usize,
    },
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
    let context_length = 512;
    let d_model = 144;
    let batch_size = 2;

    let embedding_config = EmbeddingModuleConfig::new(context_length, vocab_size, d_model);
    let mha_config = MultiHeadAttentionConfig::new(d_model, d_model);
    let transformer_config = TransformerBlockConfig::new(mha_config);
    let gpt_config = GPTModelConfig::new(embedding_config, transformer_config);

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(gpt_config, AdamWConfig::new()).with_batch_size(batch_size),
        device.clone(),
    );
}

/// Initialize the dataloader, generate a single batch, detokenize it and print it.
fn run_inspect_batch(batch_size: usize, context_size: usize) {
    type MyBackend = Wgpu<f32, i32>;
    let _device = WgpuDevice::default();

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

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::CreateVocab => create_vocab(),
        Commands::Train => run_train(),
        Commands::InspectBatch {
            batch_size,
            context_size,
        } => run_inspect_batch(batch_size, context_size),
    }
}

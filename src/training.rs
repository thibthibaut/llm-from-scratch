use std::path::Path;

use crate::dataset::{TextBatch, TextBatcher, load_fineweb_dataset_from_disk, split_dataset};
use crate::model::{GPTModel, GPTModelConfig};
use crate::tokenizer::SimpleTokenizer;
use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{LossMetric, PerplexityMetric};
use burn::train::{
    InferenceStep, Learner, SequenceOutput, SupervisedTraining, TrainOutput, TrainStep,
};

impl<B: Backend> GPTModel<B> {
    /// Compute the language modeling loss and predictions for a batch.
    pub fn forward_loss(&self, batch: TextBatch<B>) -> SequenceOutput<B> {
        let [batch_size, seq_len] = batch.inputs.dims();

        // Forward pass: [batch_size, seq_len] -> [batch_size, seq_len, vocab_size]
        let logits = self.forward(batch.inputs);
        let [_batch_size, _seq_len, vocab_size] = logits.dims();

        // Flatten for cross-entropy: [batch_size * seq_len, vocab_size]
        let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);

        // Flatten targets: [batch_size * seq_len]
        let targets_flat = batch.targets.clone().reshape([batch_size * seq_len]);

        // Compute loss
        let device = &logits_flat.device();
        let loss_fn = CrossEntropyLossConfig::new().init(device);
        let loss = loss_fn.forward(logits_flat, targets_flat);

        // Compute predictions: argmax over vocab dimension
        // argmax returns [batch_size, seq_len, 1], squeeze to [batch_size, seq_len]
        let predictions = logits.clone().argmax(2).squeeze::<2>();

        SequenceOutput {
            loss,
            logits,
            predictions: Some(predictions),
            targets: batch.targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for GPTModel<B> {
    type Input = TextBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: TextBatch<B>) -> TrainOutput<SequenceOutput<B>> {
        let item = self.forward_loss(batch);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for GPTModel<B> {
    type Input = TextBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: TextBatch<B>) -> SequenceOutput<B> {
        self.forward_loss(batch)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: GPTModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 256)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 123456)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let tokenizer = SimpleTokenizer::from_vocab_file(Path::new("vocab.json"));
    let context_size = config.model.embedding_config.context_size;
    let batcher = TextBatcher::new(tokenizer, context_size);

    let fine_web_dataset = load_fineweb_dataset_from_disk("sample_20pct.db");
    let (train_ds, valid_ds, test_ds) = split_dataset(fine_web_dataset);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_ds);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_ds);

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((LossMetric::new(), PerplexityMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

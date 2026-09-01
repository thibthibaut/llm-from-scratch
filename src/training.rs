use std::path::Path;

use crate::dataset::{TextBatch, TextBatcher, TextItem, load_default_fineweb_dataset, split_dataset};
use crate::model::{GPTModel, GPTModelConfig};
use crate::tokenizer::{AnyTokenizer, Tokenizer, TokenizerKind};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
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
    /// BPE-trained model with the word-level vocab (or vice versa) does not necessarily fail
    /// loudly — the id spaces overlap — it just silently produces nonsense.
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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    if let Err(err) = std::fs::remove_dir_all(artifact_dir)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        panic!("Failed to clear artifact dir '{artifact_dir}': {err}");
    }
    std::fs::create_dir_all(artifact_dir)
        .unwrap_or_else(|err| panic!("Failed to create artifact dir '{artifact_dir}': {err}"));
}

/// Run training against the given tokenizer and train/valid datasets. Kept generic over the
/// dataset type so tests can supply small in-memory synthetic data instead of the real
/// (very large) fineweb dataset.
pub fn train<B: AutodiffBackend, T, D: Dataset<TextItem> + 'static>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: T,
    train_dataset: D,
    valid_dataset: D,
) where
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let context_size = config.model.embedding_config.context_size;
    let batcher = TextBatcher::new(tokenizer, context_size);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(valid_dataset);

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

/// Load the tokenizer and the full fineweb dataset from disk, then run `train`.
///
/// `max_train_items`/`max_valid_items` cap the train/validation split sizes — see
/// `split_dataset` for why that matters for checkpoint cadence.
pub fn train_from_disk<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    max_train_items: Option<usize>,
    max_valid_items: Option<usize>,
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
    let (train_ds, valid_ds, _test_ds) =
        split_dataset(fine_web_dataset, max_train_items, max_valid_items);
    train::<B, _, _>(artifact_dir, config, device, tokenizer, train_ds, valid_ds);
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

    /// Exercises a full train step and a full validation step against tiny synthetic data and
    /// a tiny model, so it runs in well under a second instead of the many hours a real epoch
    /// over the fineweb dataset would take.
    #[test]
    fn test_train_runs_one_epoch_on_synthetic_data() {
        let device = Default::default();
        let tokenizer = synthetic_tokenizer();
        let vocab_size = tokenizer.get_vocab_size();

        let embedding_config = EmbeddingModuleConfig::new(4, vocab_size, 8);
        let mha_config = MultiHeadAttentionConfig::new(8).with_num_heads(2);
        let transformer_config = TransformerBlockConfig::new(mha_config);
        let model_config =
            GPTModelConfig::new(embedding_config, transformer_config).with_num_transformer_layers(1);

        let config = TrainingConfig::new(model_config, AdamWConfig::new())
            .with_num_epochs(1)
            .with_batch_size(2)
            .with_num_workers(0);

        let artifact_dir = std::env::temp_dir().join("llm-from-scratch-test-train-smoke");
        let artifact_dir = artifact_dir.to_str().unwrap();

        train::<TestBackend, _, _>(
            artifact_dir,
            config,
            device,
            tokenizer,
            synthetic_dataset(),
            synthetic_dataset(),
        );

        assert!(
            Path::new(&format!("{artifact_dir}/model.mpk")).exists(),
            "trained model should have been saved after the train + valid step"
        );

        std::fs::remove_dir_all(artifact_dir).ok();
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

use burn::Tensor;
use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::config::Config;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};

use crate::model::{GPTModel, Trace};
use crate::tokenizer::{AnyTokenizer, Token, Tokenizer, load_tokenizer};
use crate::training::TrainingConfig;

/// Load a trained model, its saved config and the tokenizer that config names.
pub fn load_trained_model(
    config_path: &str,
    model_path: &str,
    device: &LibTorchDevice,
) -> (TrainingConfig, GPTModel<LibTorch<f32>>, AnyTokenizer) {
    // Load the saved config and trained model
    let config = TrainingConfig::load(config_path)
        .expect("Config should exist at --config-path; run `train` first");
    let record = CompactRecorder::new()
        .load(model_path.into(), device)
        .expect("Trained model should exist at --model-path; run `train` first");

    let model: GPTModel<LibTorch<f32>> = config.model.init(device).load_record(record);

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

    (config, model, tokenizer)
}

/// Trace one forward pass over `tokens` (already capped to the context size).
pub fn trace_tokens(
    model: &GPTModel<LibTorch<f32>>,
    tokens: &[Token],
    device: &LibTorchDevice,
) -> Trace {
    let ids: Vec<i64> = tokens.iter().map(|t| t.0 as i64).collect();
    let indices = Tensor::<LibTorch<f32>, 1, burn::tensor::Int>::from_data(ids.as_slice(), device)
        .reshape([1, ids.len()]);
    model.forward_traced(indices)
}

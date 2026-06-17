use std::path::Path;

use crate::dataset::load_fineweb_dataset_from_disk;
use crate::model::EmbeddingModuleConfig;
use crate::model::GPTModel;
use crate::model::GPTModelConfig;
use crate::model::MultiHeadAttentionConfig;
use crate::model::TransformerBlockConfig;
use crate::tokenizer::SimpleTokenizer;
use crate::tokenizer::Token;
use crate::tokenizer::Tokenizer;
use crate::training::TrainingConfig;
use burn::Tensor;
use burn::backend::Autodiff;
use burn::optim::AdamWConfig;
use burn::prelude::Backend;
use burn::tensor::ElementConversion;
use burn::tensor::Int;
// use burn::backend::Autodiff;
use burn::backend::Wgpu;
mod dataset;
mod model;
mod tokenizer;
mod training;

fn generate_tokenizer_vocab() {
    // let dataset = load_fineweb_dataset();
    let dataset =
        load_fineweb_dataset_from_disk("/Users/thibaut/ws/llm-from-scratch/sample_20pct.db");
    let vocab = SimpleTokenizer::build_vocab(&dataset);
    vocab.to_file(Path::new("vocab.json"));
}

/*
GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}
*/
fn build_gpt_model<B: Backend>(
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    device: &B::Device,
) -> GPTModel<B> {
    let embedding_config = EmbeddingModuleConfig::new(vocab_size, d_model, context_length);

    let mha_config = MultiHeadAttentionConfig::new(d_model, d_model);

    let transformer_config = TransformerBlockConfig::new(mha_config);

    let gpt_config = GPTModelConfig::new(embedding_config, transformer_config);
    gpt_config.init(device)
}

fn generate_text<B: Backend>(
    model: &GPTModel<B>,
    tokenizer: &SimpleTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    context_length: usize,
    device: &B::Device,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    let vocab_size = tokenizer.get_vocab_size();

    for _ in 0..max_new_tokens {
        // Truncate if longer than context length
        let start = if tokens.len() > context_length {
            tokens.len() - context_length
        } else {
            0
        };
        let input_tokens = &tokens[start..];

        // Create input tensor [1, seq_len]
        let indices: Vec<i32> = input_tokens.iter().map(|t| t.0 as i32).collect();
        let input_tensor = Tensor::<B, 1, Int>::from_data(indices.as_slice(), device)
            .reshape([1, input_tokens.len()]);

        // Forward pass
        let output = model.forward(input_tensor); // [1, seq_len, vocab_size]

        // Get logits for the last position and find the token with highest score
        let last_logits = output.slice([
            0..1,
            input_tokens.len() - 1..input_tokens.len(),
            0..vocab_size,
        ]);
        let next_token_tensor = last_logits.argmax(2); // [1, 1, Int]

        // Extract token id from the 1-element tensor
        let next_token_id = next_token_tensor.into_scalar();
        let next_token = Token(next_token_id.elem::<u32>());
        tokens.push(next_token);

        // Stop if end of text token
        if next_token.0 == Token::END_OF_TEXT.0 {
            break;
        }
    }

    tokenizer.decode(&tokens)
}

fn main() {
    //generate_tokenizer_vocab();

    let tok = SimpleTokenizer::from_vocab_file(Path::new("vocab.json"));

    let vocab_size = tok.get_vocab_size();
    let context_length = 512;
    let d_model = 144;
    let batch_size = 2;

    // println!("Vocab size {}", vocab_size);

    // let device = Default::default();
    // let gpt_model = build_gpt_model::<Wgpu>(vocab_size, 1024, 768, &device);

    // let prompt = "A Hello. This is a text to transform.";
    // println!("Prompt: {}", prompt);

    // let generated = generate_text(&gpt_model, &tok, prompt, 20, 1024, &device);
    // println!("Generated: {}", generated);

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "artifacts";

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

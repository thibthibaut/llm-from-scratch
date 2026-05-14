use std::path::Path;

use crate::dataset::load_fineweb_dataset_from_disk;
use crate::model::EmbeddingModuleConfig;
use crate::model::GPTModel;
use crate::model::GPTModelConfig;
use crate::model::MultiHeadAttentionConfig;
use crate::model::TransformerBlockConfig;
use crate::tokenizer::SimpleTokenizer;
use crate::tokenizer::Tokenizer;
use burn::Tensor;
use burn::prelude::Backend;
// use burn::backend::Autodiff;
use burn::backend::Wgpu;
mod dataset;
mod model;
mod tokenizer;

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

fn main() {
    //generate_tokenizer_vocab();

    let tok = SimpleTokenizer::from_vocab_file(Path::new("vocab.json"));

    let vocab_size = tok.get_vocab_size();

    println!("Vocab size {}", vocab_size);
    let prompt = "A Hello. This is a text to transform. Why are we doing that? Banana. 12. a1a. I don't know";
    let tokens = tok.encode(prompt);
    let text = tok.decode(&tokens);
    println!("original {:?}", prompt);
    println!("tokens {:?}", tokens);
    println!("text {:?}", text);

    // type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();

    let gpt_model = build_gpt_model::<Wgpu>(vocab_size, 1024, 768, &device);

    // Create an input tensor
    let indices: Vec<i32> = tokens.iter().map(|x| x.0 as i32).collect();
    let indices: Tensor<Wgpu, 1, burn::tensor::Int> =
        Tensor::from_data(indices.as_slice(), &device);

    let indices = indices.reshape([2, 15]);
    let output = gpt_model.forward(indices);

    println!("Model output{}", output);
}

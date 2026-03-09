use std::path::Path;

use crate::dataset::load_fineweb_dataset_from_disk;
use crate::model::EmbeddingModuleConfig;
use crate::tokenizer::SimpleTokenizer;
use crate::tokenizer::Tokenizer;
use burn::Tensor;
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

fn main() {
    generate_tokenizer_vocab();

    let tok = SimpleTokenizer::from_vocab_file(Path::new("vocab.json"));

    let prompt = "A Hello. This is a text to transform. Why are we doing that? Banana. 12. a1a. I don't know";

    let tokens = tok.encode(prompt);

    let text = tok.decode(&tokens);
    println!("original {:?}", prompt);
    println!("tokens {:?}", tokens);
    println!("text {:?}", text);

    // type MyBackend = Wgpu<f32, i32>;
    let config = EmbeddingModuleConfig::new(32000, 4, 12);

    let device = Default::default();

    let module = config.init::<Wgpu>(&device);

    println!("{:?}", module);

    // Create an input tensor
    let indices: Vec<i32> = tokens.iter().map(|x| x.0 as i32).collect();
    let indices: Tensor<Wgpu, 1, burn::tensor::Int> =
        Tensor::from_data(indices.as_slice(), &device);

    let indices = indices.reshape([2, 15]);
    let embeddings = module.forward(indices);

    println!(" Embeddings {}", embeddings);
}

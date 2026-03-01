use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::dataset::load_fineweb_dataset;
use crate::dataset::load_gutenberg_dataset;
use crate::model::EmbeddingModule;
use crate::model::ModelConfig;
use crate::tokenizer::SimpleTokenizer;
use crate::tokenizer::Tokenizer;
use burn::Tensor;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::data::dataloader::Dataset;
mod dataset;
mod model;
mod tokenizer;

fn generate_tokenizer_vocab() {
    let dataset = load_fineweb_dataset();
    let vocab = SimpleTokenizer::build_vocab(&dataset);
    vocab.to_file(Path::new("vocab.json"));
}

fn main() {
    // generate_tokenizer_vocab();
    //

    let tok = SimpleTokenizer::from_vocab_file(Path::new("vocab.json"));

    let prompt = "Hello. This is a text to tokenize. What the fuck is this? Banana. 12. a1a";

    let tokens = tok.encode(prompt);

    let text = tok.decode(&tokens);
    println!("original {:?}", prompt);
    println!("tokens {:?}", tokens);
    println!("text {:?}", text);

    type MyBackend = Wgpu<f32, i32>;
    let config = ModelConfig::new(); //.with_vocab_size(50000).with_d_model(512);

    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;

    let module = config.init::<MyBackend>(&device);

    println!("{:?}", module);

    // Create an input tensor
    let indices: Vec<i32> = tokens.iter().map(|x| x.0 as i32).collect();
    let indices: Tensor<MyBackend, 1, burn::tensor::Int> =
        Tensor::from_data(indices.as_slice(), &device);

    let indices = indices.reshape([1, -1]);
    let embeddings = module.forward(indices);

    println!(" Embeddings {:?}", embeddings);
}

use burn::Tensor;
use burn::config::Config;
use burn::tensor::Int;
use burn::tensor::backend::Backend;
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
};

#[derive(Module, Debug)]
pub struct EmbeddingModule<B: Backend> {
    embedding: Embedding<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 32000)]
    vocab_size: usize,
    #[config(default = 3)]
    d_model: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingModule<B> {
        // note: default initializer for this layer is  Initializer::Normal{mean:0.0, std:1.0}
        let embedding_config = EmbeddingConfig::new(self.vocab_size, self.d_model);
        EmbeddingModule {
            embedding: embedding_config.init(device),
        }
    }
}

impl<B: Backend> EmbeddingModule<B> {
    pub fn forward(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(indices)
    }
}

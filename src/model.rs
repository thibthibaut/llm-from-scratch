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
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
}

#[derive(Config, Debug)]
pub struct EmbeddingModuleConfig {
    vocab_size: usize,
    d_model: usize,
    context_size: usize,
}

impl EmbeddingModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingModule<B> {
        // note: default initializer for this layer is  Initializer::Normal{mean:0.0, std:1.0}
        let token_embedding_config = EmbeddingConfig::new(self.vocab_size, self.d_model);
        let pos_embedding_config = EmbeddingConfig::new(self.context_size, self.d_model);
        EmbeddingModule {
            token_embedding: token_embedding_config.init(device),
            position_embedding: pos_embedding_config.init(device),
        }
    }
}

impl<B: Backend> EmbeddingModule<B> {
    pub fn forward(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Get the sequence length
        let [batch_size, sequence_length] = indices.dims();
        // Generate an input for positions [0, 1, 2 ...]
        let positions = Tensor::arange(0..sequence_length as i64, &indices.device());
        // expand positions to the batch dimension
        let positions = positions.expand([batch_size, sequence_length]);
        // Add the two embeddings
        self.token_embedding.forward(indices) + self.position_embedding.forward(positions)
    }
}

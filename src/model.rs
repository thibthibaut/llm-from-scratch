use burn::Tensor;
use burn::config::Config;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{DropoutConfig, LayerNorm, LayerNormConfig, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, activation};
use burn::{
    module::Module,
    nn::{Dropout, Embedding, EmbeddingConfig, Linear},
};

use crate::tokenizer::Vocab;

// --- GPT model ---
#[derive(Module, Debug)]
pub struct GPTModel<B: Backend> {
    embedding_layer: EmbeddingModule<B>,
    transformer_layers: Vec<TransformerBlock<B>>,
    layer_norm: LayerNorm<B>,
    out_head: Linear<B>, // Final projection from embedding space to vocab space
}

#[derive(Config, Debug)]
pub struct GPTModelConfig {
    pub embedding_config: EmbeddingModuleConfig,
    pub transformer_config: TransformerBlockConfig,
    #[config(default = 12)]
    num_transformer_layers: usize,
}

impl GPTModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPTModel<B> {
        let embedding_layer = self.embedding_config.init(device);
        let transformer_layers = (0..self.num_transformer_layers)
            .map(|_| self.transformer_config.init(device))
            .collect();
        let layer_norm = LayerNormConfig::new(self.embedding_config.d_model).init(device);
        let out_head = LinearConfig::new(
            self.embedding_config.d_model,
            self.embedding_config.vocab_size,
        )
        .init(device);

        GPTModel {
            embedding_layer,
            transformer_layers,
            layer_norm,
            out_head,
        }
    }
}

impl<B: Backend> GPTModel<B> {
    pub fn forward(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // 1. Pass through embeddings
        let mut x = self.embedding_layer.forward(indices);

        // 2. Pass through all transformer blocks
        for layer in self.transformer_layers.iter() {
            x = layer.forward(x);
        }

        // 3. Final layer norm
        x = self.layer_norm.forward(x);

        // 4. Output projection to vocabulary size
        self.out_head.forward(x)
    }
}

// --- Embedding layer ---
#[derive(Module, Debug)]
pub struct EmbeddingModule<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
}

#[derive(Config, Debug)]
pub struct EmbeddingModuleConfig {
    pub context_size: usize,
    vocab_size: usize,
    d_model: usize,
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

// --- MultiHeadSelfAttention ---------------------------------------------------

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    query_weights: Linear<B>,  // [d_in -> d_out]
    key_weights: Linear<B>,    // [d_in -> d_out]
    value_weights: Linear<B>,  // [d_in -> d_out]
    out_projection: Linear<B>, // [d_out -> d_out]
    dropout: Dropout,
    // Causal mask so that the model cannot look into the future:
    // Shape: [context_length, context_length], True = keep, False = mask out
    // mask: Tensor<B, 2, Bool>,
    num_heads: usize,
    head_dim: usize, // d_out / num_heads
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    d_in: usize,  // Input embedding dimension
    d_out: usize, // Total output dimension (must be divisible by num_heads)
    // context_length: usize, // Max sequence length, needed for the causal mask
    #[config(default = 12)]
    num_heads: usize, // Number of attention heads
    #[config(default = 0.1)]
    dropout_prob: f64, // Dropout probability
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert!(
            self.d_out.is_multiple_of(self.num_heads),
            "d_out ({}) must be divisible by num_heads ({})",
            self.d_out,
            self.num_heads
        );

        let head_dim = self.d_out / self.num_heads;

        // All three projections go from d_in -> d_out (full dimension),
        // then we split into heads during the forward pass via reshape.
        let query_weights = LinearConfig::new(self.d_in, self.d_out)
            .with_bias(false)
            .init(device);
        let key_weights = LinearConfig::new(self.d_in, self.d_out)
            .with_bias(false)
            .init(device);
        let value_weights = LinearConfig::new(self.d_in, self.d_out)
            .with_bias(false)
            .init(device);

        // Final projection after concatenating heads
        let out_projection = LinearConfig::new(self.d_out, self.d_out).init(device);

        let dropout = DropoutConfig::new(self.dropout_prob).init();

        // Causal mask: lower triangular, shape [context_length, context_length]
        // tril()[i][j] = true  means token i CAN attend to token j (j <= i)
        // tril()[i][j] = false means token i CANNOT attend to token j (future)
        // let mask = Tensor::<B, 2>::ones([self.context_length, self.context_length], device)
        //     .tril(0) // keep diagonal and below
        //     .bool(); // cast to Bool tensor

        MultiHeadAttention {
            query_weights,
            key_weights,
            value_weights,
            out_projection,
            dropout,
            // mask,
            num_heads: self.num_heads,
            head_dim,
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, sequence: Tensor<B, 3>) -> Tensor<B, 3> {
        // Input shape is [Batch Size, Sequence Lenght, Input Dimension]
        let [batch_size, seq_len, d_in] = sequence.dims();

        // Compute Q, K, V
        // Note: it's fine to clone tensors in burn, it's just a shallow clone
        let keys = self.key_weights.forward(sequence.clone());
        let values = self.value_weights.forward(sequence.clone());
        let queries = self.query_weights.forward(sequence.clone());
        // Shape is now [batch_size, seq_len, d_out]

        // We need to reshape to perform multi-head attention:
        // Basically we split the dimension d_out between all the heads
        // [B, seq_len, d_out] -> [B, seq, num_heads, head_dim]
        // And we rearange so that the last 2 dimensions are seq_len, head_dim
        // so that we can do matrix multiplication later
        // [B, seq, num_heads, head_dim] [B, num_heads, seq_len, head_dim]
        let keys = keys
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let values = values
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let queries = queries
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Compute the attention scores with Q @ K^T
        // Q @ K^T  : [B, NumHeads, SeqLen, HeadDim] @ [B, NumHeads, HeadDim, SeqLen]
        //          = [B, NumHeads, SeqLen, SeqLen]
        let scale = (self.head_dim as f64).sqrt();
        let attn_scores = queries.matmul(keys.swap_dims(2, 3)) / scale;

        let mask =
            generate_autoregressive_mask::<B>(batch_size, seq_len, &sequence.clone().device())
                .unsqueeze_dim::<4>(1);

        // Apply causal mask: slice to actual seq_len (may be < context_length)
        // let mask = self
        //     .mask
        //     .clone()
        //     .slice([0..seq_len, 0..seq_len]) // [SeqLen, SeqLen]
        //     .unsqueeze::<4>(); // [1, 1, SeqLen, SeqLen] — broadcasts over B and NumHeads

        let masking_value: f32 = -1.0e4_f32; // f32::NEG_INFINITY
        let attn_scores = attn_scores.mask_fill(mask.bool_not(), masking_value);

        // Softmax over last dim (the key dimension)
        let attn_weights = activation::softmax(attn_scores, 3); // dim=3 = SeqLen of keys
        // Apply dropout
        let attn_weights = self.dropout.forward(attn_weights);

        // attn_weights @ V : [B, NumHeads, SeqLen, SeqLen] @ [B, NumHeads, SeqLen, HeadDim]
        //                  = [B, NumHeads, SeqLen, HeadDim]
        let context = attn_weights.matmul(values);

        // [B, NumHeads, SeqLen, HeadDim] -> [B, SeqLen, NumHeads, HeadDim] -> [B, SeqLen, D_out]
        let context =
            context
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, self.num_heads * self.head_dim]);

        // Final projection [B, SeqLen, D_out] -> [B, SeqLen, D_out]
        self.out_projection.forward(context)
    }
}

// ---------- TransformerBlock

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    layer_norm1: LayerNorm<B>,
    mha: MultiHeadAttention<B>,
    dropout1: Dropout,
    layer_norm2: LayerNorm<B>,
    linear1: Linear<B>,
    activation: Activation<B>,
    linear2: Linear<B>,
    dropout2: Dropout,
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    mha_config: MultiHeadAttentionConfig,
    #[config(default = 4)]
    ff_expansion_factor: usize,
    #[config(default = "ActivationConfig::Gelu")]
    activation: ActivationConfig,
    #[config(default = 0.1)]
    dropout_prob: f64, // Dropout probability
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let d_model = self.mha_config.d_in;
        let ff_dim: usize = d_model * self.ff_expansion_factor;

        TransformerBlock {
            layer_norm1: LayerNormConfig::new(d_model).init(device),
            mha: self.mha_config.init(device),
            dropout1: DropoutConfig::new(self.dropout_prob).init(),
            layer_norm2: LayerNormConfig::new(d_model).init(device),

            linear1: LinearConfig::new(d_model, ff_dim).init(device),
            activation: self.activation.init(device),
            linear2: LinearConfig::new(ff_dim, d_model).init(device),
            dropout2: DropoutConfig::new(self.dropout_prob).init(),
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shortcut = x.clone();

        // Layer normalization
        let x = self.layer_norm1.forward(x);
        // Self attention
        let x = self.mha.forward(x);

        let x = self.dropout1.forward(x);

        // Ad the resudual
        let x = x + shortcut;

        let shortcut = x.clone();

        // Layer norm
        let x = self.layer_norm2.forward(x);

        // MLP
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);

        // Add the residual
        x + shortcut
    }
}

// ----------- TEST ---------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu<f32>;

    #[test]
    fn test_multihead_attention_output_shape() {
        let device = Default::default();

        let batch_size = 2;
        let seq_len = 5;
        let d_in = 16;
        let d_out = 16;
        let num_heads = 4;
        // let context_length = 10;

        let mha = MultiHeadAttentionConfig {
            d_in,
            d_out,
            // context_length,
            num_heads,
            dropout_prob: 0.0, // no dropout during testing
        }
        .init::<TestBackend>(&device);

        // Random input: [batch_size, seq_len, d_in]
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_in],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = mha.forward(input);

        // Output must be [batch_size, seq_len, d_out]
        assert_eq!(
            output.dims(),
            [batch_size, seq_len, d_out],
            "Output shape mismatch"
        );
    }

    #[test]
    fn test_multihead_attention_causal_masking() {
        let device = Default::default();

        let mha = MultiHeadAttentionConfig {
            d_in: 8,
            d_out: 8,
            // context_length: 6,
            num_heads: 2,
            dropout_prob: 0.0,
        }
        .init::<TestBackend>(&device);

        // Two identical sequences — output should be identical across the batch
        let seq = Tensor::<TestBackend, 3>::ones([1, 4, 8], &device);
        let batch = Tensor::cat(vec![seq.clone(), seq], 0); // [2, 4, 8]

        let output = mha.forward(batch);
        let out0 = output.clone().slice([0..1, 0..4, 0..8]);
        let out1 = output.slice([1..2, 0..4, 0..8]);

        // Both rows in the batch should produce the same result
        let diff = (out0 - out1).abs().max();
        let max_diff = diff.into_scalar();
        assert!(
            max_diff < 1e-5,
            "Identical inputs should produce identical outputs, got diff {max_diff}"
        );
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_invalid_config_panics() {
        let device = Default::default();
        // d_out=10 is not divisible by num_heads=3 → should panic
        MultiHeadAttentionConfig {
            d_in: 8,
            d_out: 10,
            // context_length: 6,
            num_heads: 3,
            dropout_prob: 0.0,
        }
        .init::<TestBackend>(&device);
    }
}

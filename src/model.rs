use burn::Tensor;
use burn::config::Config;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::{DropoutConfig, LayerNorm, LayerNormConfig, LinearConfig};
use burn::tensor::{FloatDType, Int};
use burn::{
    module::Module,
    nn::{Dropout, Embedding, EmbeddingConfig, Linear},
};
use serde::{Deserialize, Serialize};

use crate::ops::{FusedOps, causal_attention};

// --- Numeric precision --------------------------------------------------------

/// What the matrix multiplications run in. Weights, optimizer state, layer norms, residual
/// stream and loss always stay f32; only the matmul inputs (and attention) change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Plain f32 everywhere. The same math as before this option existed, though not the same
    /// bits: attention and the loss now run as fused kernels, which round differently.
    #[default]
    F32,
    /// f32 tensors, but cuBLAS may run f32 matmuls on TF32 tensor cores (10-bit mantissa
    /// inputs, f32 accumulation). Set process-wide, see `Precision::configure_process`.
    Tf32,
    /// Mixed precision in the style of PyTorch autocast: matmul and attention inputs are cast to
    /// bf16 (f32 accumulation), everything else stays f32, and the f32 master weights are what
    /// the optimizer updates. *Not* the same as running the model in bf16, which loses small
    /// weight updates to bf16's 8-bit mantissa.
    Bf16,
}

impl Precision {
    /// The dtype matmuls run in, when it isn't f32.
    fn compute_dtype(self) -> Option<FloatDType> {
        match self {
            Precision::Bf16 => Some(FloatDType::BF16),
            Precision::F32 | Precision::Tf32 => None,
        }
    }

    /// Apply the process-wide part of this precision. Must run before libtorch creates its
    /// cuBLAS handle, i.e. before the first CUDA op.
    pub fn configure_process(self) {
        if self != Precision::F32 {
            // libtorch reads this when it decides whether f32 matmuls may use TF32; it's the C++
            // equivalent of Python's `torch.backends.cuda.matmul.allow_tf32 = True`. In bf16 mode
            // it only affects the few matmuls that stay f32.
            // SAFETY: called at startup, before any other thread exists.
            unsafe { std::env::set_var("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1") };
        }
    }
}

/// Cast to the compute dtype (no-op in f32 modes, and for a tensor already in that dtype).
fn to_compute<B: Backend, const D: usize>(x: Tensor<B, D>, precision: Precision) -> Tensor<B, D> {
    match precision.compute_dtype() {
        None => x,
        Some(dtype) => x.cast(dtype),
    }
}

/// `linear.forward(x)`, with the inputs cast to the compute dtype. The output stays in the
/// compute dtype, so consecutive low-precision ops don't bounce through f32.
fn linear_lowp<B: Backend, const D: usize>(
    linear: &Linear<B>,
    x: Tensor<B, D>,
    precision: Precision,
) -> Tensor<B, D> {
    match precision.compute_dtype() {
        None => linear.forward(x),
        Some(dtype) => burn::tensor::module::linear(
            x.cast(dtype),
            linear.weight.val().cast(dtype),
            linear.bias.as_ref().map(|b| b.val().cast(dtype)),
        ),
    }
}

/// Back to f32 for the residual stream and layer norms.
fn to_f32<B: Backend, const D: usize>(x: Tensor<B, D>, precision: Precision) -> Tensor<B, D> {
    match precision.compute_dtype() {
        None => x,
        Some(_) => x.cast(FloatDType::F32),
    }
}

use burn::tensor::backend::Backend;

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

impl<B: FusedOps> GPTModel<B> {
    /// Logits in f32.
    pub fn forward(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_with(indices, Precision::F32)
    }

    /// Logits, in the compute dtype of `precision` (bf16 logits are half the memory of the
    /// largest activation in the model; the loss upcasts them chunk by chunk).
    pub fn forward_with(&self, indices: Tensor<B, 2, Int>, precision: Precision) -> Tensor<B, 3> {
        // 1. Pass through embeddings
        let mut x = self.embedding_layer.forward(indices);

        // 2. Pass through all transformer blocks
        for layer in self.transformer_layers.iter() {
            x = layer.forward_with(x, precision);
        }

        // 3. Final layer norm
        x = self.layer_norm.forward(x);

        // 4. Output projection to vocabulary size
        linear_lowp(&self.out_head, x, precision)
    }
}

// --- Embedding layer ---
#[derive(Module, Debug)]
pub struct EmbeddingModule<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    dropout: Dropout,
    context_size: usize,
}

#[derive(Config, Debug)]
pub struct EmbeddingModuleConfig {
    pub context_size: usize,
    pub vocab_size: usize,
    d_model: usize,
    #[config(default = 0.1)]
    dropout_prob: f64,
}

impl EmbeddingModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingModule<B> {
        // note: default initializer for this layer is  Initializer::Normal{mean:0.0, std:1.0}
        let token_embedding_config = EmbeddingConfig::new(self.vocab_size, self.d_model);
        let pos_embedding_config = EmbeddingConfig::new(self.context_size, self.d_model);
        EmbeddingModule {
            token_embedding: token_embedding_config.init(device),
            position_embedding: pos_embedding_config.init(device),
            dropout: DropoutConfig::new(self.dropout_prob).init(),
            context_size: self.context_size,
        }
    }
}

impl<B: Backend> EmbeddingModule<B> {
    pub fn forward(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Get the sequence length
        let [batch_size, sequence_length] = indices.dims();
        assert!(
            sequence_length <= self.context_size,
            "sequence length ({}) exceeds context size ({})",
            sequence_length,
            self.context_size
        );
        // Generate an input for positions [0, 1, 2 ...]
        let positions = Tensor::arange(0..sequence_length as i64, &indices.device());
        // expand positions to the batch dimension
        let positions = positions.expand([batch_size, sequence_length]);
        // Add the two embeddings
        let x = self.token_embedding.forward(indices) + self.position_embedding.forward(positions);
        x
        // self.dropout.forward(x)
    }
}

// --- MultiHeadSelfAttention ---------------------------------------------------

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    query_weights: Linear<B>,  // [d_model -> d_model]
    key_weights: Linear<B>,    // [d_model -> d_model]
    value_weights: Linear<B>,  // [d_model -> d_model]
    out_projection: Linear<B>, // [d_model -> d_model]
    dropout: Dropout,
    num_heads: usize,
    head_dim: usize, // d_model / num_heads
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    d_model: usize, // Total model dimension (must be divisible by num_heads)
    #[config(default = 12)]
    num_heads: usize, // Number of attention heads
    #[config(default = 0.1)]
    dropout_prob: f64, // Dropout probability
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert!(
            self.d_model.is_multiple_of(self.num_heads),
            "d_model ({}) must be divisible by num_heads ({})",
            self.d_model,
            self.num_heads
        );

        let head_dim = self.d_model / self.num_heads;

        // All three projections go from d_model -> d_model (full dimension),
        // then we split into heads during the forward pass via reshape.
        let query_weights = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(false)
            .init(device);
        let key_weights = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(false)
            .init(device);
        let value_weights = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(false)
            .init(device);

        // Final projection after concatenating heads
        let out_projection = LinearConfig::new(self.d_model, self.d_model).init(device);

        let dropout = DropoutConfig::new(self.dropout_prob).init();

        MultiHeadAttention {
            query_weights,
            key_weights,
            value_weights,
            out_projection,
            dropout,
            num_heads: self.num_heads,
            head_dim,
        }
    }
}

impl<B: FusedOps> MultiHeadAttention<B> {
    #[cfg(test)]
    pub fn forward(&self, sequence: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with(sequence, Precision::F32)
    }

    pub fn forward_with(&self, sequence: Tensor<B, 3>, precision: Precision) -> Tensor<B, 3> {
        // Input shape is [Batch Size, Sequence Lenght, Input Dimension]
        let [batch_size, seq_len, _d_model] = sequence.dims();

        // Compute Q, K, V
        // Cast once here rather than inside each of the three projections (casting a tensor
        // that is already bf16 is a no-op).
        let sequence = to_compute(sequence, precision);
        // Note: it's fine to clone tensors in burn, it's just a shallow clone
        let keys = linear_lowp(&self.key_weights, sequence.clone(), precision);
        let values = linear_lowp(&self.value_weights, sequence.clone(), precision);
        let queries = linear_lowp(&self.query_weights, sequence, precision);
        // Shape is now [batch_size, seq_len, d_model]

        // We need to reshape to perform multi-head attention:
        // Basically we split the model dimension between all the heads
        // [B, seq_len, d_model] -> [B, seq, num_heads, head_dim]
        // And we rearange so that the last 2 dimensions are seq_len, head_dim
        // [B, seq, num_heads, head_dim] [B, num_heads, seq_len, head_dim]
        let split_heads = |t: Tensor<B, 3>| {
            t.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let (queries, keys, values) = (split_heads(queries), split_heads(keys), split_heads(values));

        // softmax(Q Kᵀ / √head_dim + causal mask) V, as one fused kernel (FlashAttention for
        // bf16, memory-efficient attention for f32). Writing it out op by op made autodiff keep
        // several [B, NumHeads, SeqLen, SeqLen] tensors per layer alive for the backward pass,
        // which is what capped the batch size; see `crate::ops`.
        // Attention dropout isn't applied (it was already disabled).
        let context = causal_attention(queries, keys, values);

        // [B, NumHeads, SeqLen, HeadDim] -> [B, SeqLen, NumHeads, HeadDim] -> [B, SeqLen, D_out]
        let context =
            context
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, self.num_heads * self.head_dim]);

        // Final projection [B, SeqLen, D_out] -> [B, SeqLen, D_out]
        to_f32(linear_lowp(&self.out_projection, context, precision), precision)
    }
}

// ---------- TransformerBlock

/// A pre-norm transformer block: attention and MLP sublayers, each wrapped in a residual.
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
        let d_model = self.mha_config.d_model;
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

impl<B: FusedOps> TransformerBlock<B> {
    pub fn forward_with(&self, x: Tensor<B, 3>, precision: Precision) -> Tensor<B, 3> {
        let shortcut = x.clone();

        // Layer normalization
        let x = self.layer_norm1.forward(x);
        // Self attention
        let x = self.mha.forward_with(x, precision);

        // let x = self.dropout1.forward(x);

        // Ad the resudual
        let x = x + shortcut;

        let shortcut = x.clone();

        // Layer norm
        let x = self.layer_norm2.forward(x);

        // MLP (the activation runs in the compute dtype, like autocast does)
        let x = linear_lowp(&self.linear1, x, precision);
        let x = self.activation.forward(x);
        let x = to_f32(linear_lowp(&self.linear2, x, precision), precision);

        // let x = self.dropout2.forward(x);

        // Add the residual
        x + shortcut
    }
}

// ----------- TEST ---------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::LibTorch;

    type TestBackend = LibTorch<f32>;

    fn test_gpt_config(context_size: usize) -> GPTModelConfig {
        let embedding_config =
            EmbeddingModuleConfig::new(context_size, 16, 8).with_dropout_prob(0.0);
        let mha_config = MultiHeadAttentionConfig::new(8)
            .with_num_heads(2)
            .with_dropout_prob(0.0);
        let transformer_config = TransformerBlockConfig::new(mha_config).with_dropout_prob(0.0);

        GPTModelConfig::new(embedding_config, transformer_config).with_num_transformer_layers(1)
    }

    #[test]
    fn test_gpt_model_output_shape() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let model = test_gpt_config(4).init::<TestBackend>(&device);
        let indices = Tensor::<TestBackend, 1, Int>::from_data([0, 1, 2, 3].as_slice(), &device)
            .reshape([1, 4]);

        let output = model.forward(indices);

        assert_eq!(output.dims(), [1, 4, 16]);
    }

    #[test]
    #[should_panic(expected = "exceeds context size")]
    fn test_gpt_model_panics_when_sequence_exceeds_context_size() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let model = test_gpt_config(4).init::<TestBackend>(&device);
        let indices = Tensor::<TestBackend, 1, Int>::from_data([0, 1, 2, 3, 4].as_slice(), &device)
            .reshape([1, 5]);

        model.forward(indices);
    }

    #[test]
    fn test_multihead_attention_output_shape() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();

        let batch_size = 2;
        let seq_len = 5;
        let d_model = 16;
        let num_heads = 4;

        let mha = MultiHeadAttentionConfig {
            d_model,
            num_heads,
            dropout_prob: 0.0, // no dropout during testing
        }
        .init::<TestBackend>(&device);

        // Random input: [batch_size, seq_len, d_model]
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = mha.forward(input);

        // Output must be [batch_size, seq_len, d_model]
        assert_eq!(
            output.dims(),
            [batch_size, seq_len, d_model],
            "Output shape mismatch"
        );
    }

    #[test]
    fn test_multihead_attention_causal_masking() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();

        let mha = MultiHeadAttentionConfig {
            d_model: 8,
            num_heads: 2,
            dropout_prob: 0.0,
        }
        .init::<TestBackend>(&device);

        // The first two positions are identical across both sequences, while
        // future positions are intentionally very different. With a causal mask,
        // outputs for the shared prefix must not depend on those future tokens.
        let data = [
            1.0, 0.0, 0.5, -0.5, 0.2, -0.2, 0.7, -0.7, // seq 0, pos 0
            0.3, 0.4, -0.1, 0.8, -0.9, 0.6, -0.4, 0.1, // seq 0, pos 1
            10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, // seq 0, pos 2
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, // seq 0, pos 3
            1.0, 0.0, 0.5, -0.5, 0.2, -0.2, 0.7, -0.7, // seq 1, pos 0
            0.3, 0.4, -0.1, 0.8, -0.9, 0.6, -0.4, 0.1, // seq 1, pos 1
            -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, // seq 1, pos 2
            -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, // seq 1, pos 3
        ];
        let batch =
            Tensor::<TestBackend, 1>::from_data(data.as_slice(), &device).reshape([2, 4, 8]);

        let output = mha.forward(batch);
        let out0 = output.clone().slice([0..1, 0..2, 0..8]);
        let out1 = output.slice([1..2, 0..2, 0..8]);

        let diff = (out0 - out1).abs().max();
        let max_diff = diff.into_scalar();
        assert!(
            max_diff < 1e-5,
            "Future tokens should not affect prefix outputs, got diff {max_diff}"
        );
    }

    #[test]
    #[should_panic(expected = "d_model")]
    fn test_invalid_config_panics() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        // d_model=10 is not divisible by num_heads=3 -> should panic
        MultiHeadAttentionConfig {
            d_model: 10,
            num_heads: 3,
            dropout_prob: 0.0,
        }
        .init::<TestBackend>(&device);
    }
}

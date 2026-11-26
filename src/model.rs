use burn::Tensor;
use burn::config::Config;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::{DropoutConfig, LayerNorm, LayerNormConfig, LinearConfig};
use burn::tensor::{Bool, FloatDType, Int};
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
        self.forward_impl(indices, precision, None)
    }

    /// One f32 forward pass over a single sequence (`indices` is `[1, seq]`), recording the
    /// embeddings, every block's attention weights and residual stream, and the last logits.
    pub fn forward_traced(&self, indices: Tensor<B, 2, Int>) -> Trace {
        assert_eq!(
            indices.dims()[0],
            1,
            "forward_traced takes a batch of one sequence"
        );
        let mut trace = Trace::default();
        let logits = self.forward_impl(indices, Precision::F32, Some(&mut trace));
        let [_, seq_len, vocab_size] = logits.dims();
        trace.last_logits =
            to_vecs(logits.slice([0..1, seq_len - 1..seq_len, 0..vocab_size])).remove(0);
        trace
    }

    pub fn num_layers(&self) -> usize {
        self.transformer_layers.len()
    }

    pub fn num_heads(&self) -> usize {
        self.transformer_layers.first().map_or(0, |b| b.mha.num_heads)
    }

    /// The token embedding table, `[vocab, d_model]`.
    pub fn token_embedding_weights(&self) -> Tensor<B, 2> {
        self.embedding_layer.token_embedding.weight.val()
    }

    fn forward_impl(
        &self,
        indices: Tensor<B, 2, Int>,
        precision: Precision,
        mut trace: Option<&mut Trace>,
    ) -> Tensor<B, 3> {
        // 1. Pass through embeddings
        let mut x = self.embedding_layer.forward(indices);
        if let Some(trace) = trace.as_deref_mut() {
            trace.embedding = to_vecs(x.clone());
        }

        // 2. Pass through all transformer blocks
        for layer in self.transformer_layers.iter() {
            x = match trace.as_deref_mut() {
                None => layer.forward_impl(x, precision, None),
                Some(trace) => {
                    let mut block = BlockTrace::default();
                    let x = layer.forward_impl(x, precision, Some(&mut block));
                    trace.blocks.push(block);
                    x
                }
            };
        }

        // 3. Final layer norm
        x = self.layer_norm.forward(x);

        // 4. Output projection to vocabulary size
        linear_lowp(&self.out_head, x, precision)
    }
}

/// What `GPTModel::forward_traced` records, for batch 0, in f32.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    /// `[seq][d_model]`, token + position embedding.
    pub embedding: Vec<Vec<f32>>,
    /// One per transformer block, in order.
    pub blocks: Vec<BlockTrace>,
    /// `[vocab]`, logits at the last position.
    pub last_logits: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct BlockTrace {
    /// `[heads][seq][seq]`, softmax(QKᵀ/√head_dim + causal mask).
    pub attention: Vec<Vec<Vec<f32>>>,
    /// `[seq][d_model]`, residual stream after the block.
    pub residual: Vec<Vec<f32>>,
    /// `[heads][seq][head_dim]`, the split-head queries, keys and values.
    pub q: Vec<Vec<Vec<f32>>>,
    pub k: Vec<Vec<Vec<f32>>>,
    pub v: Vec<Vec<Vec<f32>>>,
    /// `[heads][seq][seq]`, QKᵀ/√head_dim, masked (future) cells are `f32::NEG_INFINITY`.
    pub scores: Vec<Vec<Vec<f32>>>,
    /// `[seq][d_model]`, layer_norm2 output (the MLP input).
    pub mlp_in: Vec<Vec<f32>>,
    /// `[seq][ff_dim]`, activation output.
    pub mlp_hidden: Vec<Vec<f32>>,
    /// `[seq][d_model]`, linear2 output (added to the residual).
    pub mlp_out: Vec<Vec<f32>>,
}

/// Batch 0 of a `[batch, rows, cols]` tensor, as f32 rows.
fn to_vecs<B: Backend>(t: Tensor<B, 3>) -> Vec<Vec<f32>> {
    let [_, _, cols] = t.dims();
    let data: Vec<f32> = t
        .slice(0..1)
        .cast(FloatDType::F32)
        .into_data()
        .to_vec()
        .unwrap();
    data.chunks(cols).map(<[f32]>::to_vec).collect()
}

/// Batch 0 of a `[batch, heads, rows, cols]` tensor, as f32, with one device sync.
fn to_heads<B: Backend>(t: Tensor<B, 4>) -> Vec<Vec<Vec<f32>>> {
    let [_, heads, rows, cols] = t.dims();
    to_vecs(t.slice(0..1).reshape([1, heads * rows, cols]))
        .chunks(rows)
        .map(<[Vec<f32>]>::to_vec)
        .collect()
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
        self.forward_impl(sequence, precision, None)
    }

    /// `trace`, when given, receives batch 0's `q`, `k`, `v`, `scores` and `attention`.
    fn forward_impl(
        &self,
        sequence: Tensor<B, 3>,
        precision: Precision,
        trace: Option<&mut BlockTrace>,
    ) -> Tensor<B, 3> {
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

        // The fused kernel below never materializes the weights, so recompute them from the
        // same queries and keys.
        if let Some(trace) = trace {
            let q = queries.clone().cast(FloatDType::F32);
            let k = keys.clone().cast(FloatDType::F32);
            let scores =
                q.clone().matmul(k.clone().swap_dims(2, 3)) / (self.head_dim as f64).sqrt();
            let future = Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, &scores.device());
            let scores = scores.mask_fill(future.unsqueeze::<4>(), f32::NEG_INFINITY);
            trace.attention = to_heads(burn::tensor::activation::softmax(scores.clone(), 3));
            trace.scores = to_heads(scores);
            trace.q = to_heads(q);
            trace.k = to_heads(k);
            trace.v = to_heads(values.clone());
        }

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
        self.forward_impl(x, precision, None)
    }

    fn forward_impl(
        &self,
        x: Tensor<B, 3>,
        precision: Precision,
        mut trace: Option<&mut BlockTrace>,
    ) -> Tensor<B, 3> {
        let shortcut = x.clone();

        // Layer normalization
        let x = self.layer_norm1.forward(x);
        // Self attention
        let x = self.mha.forward_impl(x, precision, trace.as_deref_mut());

        // let x = self.dropout1.forward(x);

        // Ad the resudual
        let x = x + shortcut;

        let shortcut = x.clone();

        // Layer norm
        let x = self.layer_norm2.forward(x);
        if let Some(trace) = trace.as_deref_mut() {
            trace.mlp_in = to_vecs(x.clone());
        }

        // MLP (the activation runs in the compute dtype, like autocast does)
        let x = linear_lowp(&self.linear1, x, precision);
        let x = self.activation.forward(x);
        if let Some(trace) = trace.as_deref_mut() {
            trace.mlp_hidden = to_vecs(x.clone());
        }
        let x = to_f32(linear_lowp(&self.linear2, x, precision), precision);
        if let Some(trace) = trace.as_deref_mut() {
            trace.mlp_out = to_vecs(x.clone());
        }

        // let x = self.dropout2.forward(x);

        // Add the residual
        let x = x + shortcut;
        if let Some(trace) = trace {
            trace.residual = to_vecs(x.clone());
        }
        x
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

    fn indices(
        ids: &[i64],
        device: &burn::backend::libtorch::LibTorchDevice,
    ) -> Tensor<TestBackend, 2, Int> {
        Tensor::<TestBackend, 1, Int>::from_data(ids, device).reshape([1, ids.len()])
    }

    #[test]
    fn traced_logits_match_forward() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let model = test_gpt_config(4).init::<TestBackend>(&device);
        let x = indices(&[0, 1, 2, 3], &device);

        let expected: Vec<f32> = model
            .forward(x.clone())
            .slice([0..1, 3..4, 0..16])
            .into_data()
            .to_vec()
            .unwrap();
        let trace = model.forward_traced(x);

        assert_eq!(trace.last_logits.len(), expected.len());
        for (a, b) in trace.last_logits.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn traced_attention_is_causal_softmax() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let model = test_gpt_config(4).init::<TestBackend>(&device);
        let trace = model.forward_traced(indices(&[0, 1, 2, 3], &device));

        assert_eq!(trace.blocks.len(), 1);
        let attention = &trace.blocks[0].attention;
        assert_eq!(attention.len(), 2);
        for head in attention {
            assert_eq!(head.len(), 4);
            for (i, row) in head.iter().enumerate() {
                assert_eq!(row.len(), 4);
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "row {i} sums to {sum}");
                for &w in &row[i + 1..] {
                    assert_eq!(w, 0.0);
                }
            }
        }
    }

    #[test]
    fn traced_attention_reproduces_mha_output() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(8)
            .with_num_heads(2)
            .with_dropout_prob(0.0)
            .init::<TestBackend>(&device);
        let x = Tensor::<TestBackend, 3>::random(
            [1, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let mut trace = BlockTrace::default();
        let expected = mha.forward_impl(x.clone(), Precision::F32, Some(&mut trace));

        let weights: Vec<f32> = trace.attention.into_iter().flatten().flatten().collect();
        let weights =
            Tensor::<TestBackend, 1>::from_data(weights.as_slice(), &device).reshape([1, 2, 4, 4]);
        let v = mha
            .value_weights
            .forward(x)
            .reshape([1, 4, 2, 4])
            .swap_dims(1, 2);
        let context = weights.matmul(v).swap_dims(1, 2).reshape([1, 4, 8]);
        let rebuilt = mha.out_projection.forward(context);

        let diff = (rebuilt - expected).abs().max().into_scalar();
        assert!(diff < 1e-4, "rebuilt output differs by {diff}");
    }

    #[test]
    fn traced_residual_shapes() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let model = test_gpt_config(4).init::<TestBackend>(&device);
        let trace = model.forward_traced(indices(&[0, 1, 2, 3], &device));

        assert_eq!(trace.embedding.len(), 4);
        assert!(trace.embedding.iter().all(|row| row.len() == 8));
        assert_eq!(trace.blocks[0].residual.len(), 4);
        assert!(trace.blocks[0].residual.iter().all(|row| row.len() == 8));
        assert_eq!(trace.last_logits.len(), 16);
    }

    #[test]
    fn one_token_trace() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let model = test_gpt_config(4).init::<TestBackend>(&device);
        let trace = model.forward_traced(indices(&[5], &device));

        let attention = &trace.blocks[0].attention;
        assert_eq!(attention.len(), 2);
        for head in attention {
            assert_eq!(head.len(), 1);
            assert_eq!(head[0].len(), 1);
            assert!((head[0][0] - 1.0).abs() < 1e-6);
        }
    }

    fn traced_block(ids: &[i64]) -> (GPTModel<TestBackend>, BlockTrace) {
        let device = Default::default();
        let model = test_gpt_config(4).init::<TestBackend>(&device);
        let mut trace = model.forward_traced(indices(ids, &device));
        (model, trace.blocks.remove(0))
    }

    fn rows_to_tensor(rows: &[Vec<f32>]) -> Tensor<TestBackend, 3> {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::<TestBackend, 1>::from_data(flat.as_slice(), &Default::default()).reshape([
            1,
            rows.len(),
            rows[0].len(),
        ])
    }

    fn assert_rows_close(a: &[Vec<f32>], b: Tensor<TestBackend, 3>) {
        let b = to_vecs(b);
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(&b) {
            assert_eq!(ra.len(), rb.len());
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < 1e-5, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn traced_qkv_and_mlp_shapes() {
        let _rng = crate::torch_rng_lock();
        let (_, block) = traced_block(&[0, 1, 2]);

        for t in [&block.q, &block.k, &block.v] {
            assert_eq!(t.len(), 2);
            assert!(
                t.iter()
                    .all(|h| h.len() == 3 && h.iter().all(|r| r.len() == 4))
            );
        }
        assert_eq!(block.scores.len(), 2);
        for head in &block.scores {
            assert_eq!(head.len(), 3);
            for (i, row) in head.iter().enumerate() {
                assert_eq!(row.len(), 3);
                for &s in &row[i + 1..] {
                    assert_eq!(s, f32::NEG_INFINITY);
                }
            }
        }
        assert_eq!(block.mlp_in.len(), 3);
        assert!(block.mlp_in.iter().all(|r| r.len() == 8));
        assert_eq!(block.mlp_hidden.len(), 3);
        assert!(block.mlp_hidden.iter().all(|r| r.len() == 32));
        assert_eq!(block.mlp_out.len(), 3);
        assert!(block.mlp_out.iter().all(|r| r.len() == 8));
    }

    #[test]
    fn softmax_of_scores_is_attention() {
        let _rng = crate::torch_rng_lock();
        let (_, block) = traced_block(&[0, 1, 2, 3]);

        for (scores, attention) in block.scores.iter().zip(&block.attention) {
            for (i, (s, a)) in scores.iter().zip(attention).enumerate() {
                let max = s[..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = s[..=i].iter().map(|x| (x - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (j, e) in exps.iter().enumerate() {
                    let w = e / sum;
                    assert!((w - a[j]).abs() < 1e-5, "[{i}][{j}] {w} vs {}", a[j]);
                }
            }
        }
    }

    #[test]
    fn scores_are_scaled_q_dot_k() {
        let _rng = crate::torch_rng_lock();
        let (_, block) = traced_block(&[0, 1, 2, 3]);

        for h in 0..2 {
            for i in 0..4 {
                for j in 0..=i {
                    let dot: f32 = block.q[h][i]
                        .iter()
                        .zip(&block.k[h][j])
                        .map(|(a, b)| a * b)
                        .sum();
                    let expected = dot / 4f32.sqrt();
                    let got = block.scores[h][i][j];
                    assert!((got - expected).abs() < 1e-5, "{got} vs {expected}");
                }
            }
        }
    }

    #[test]
    fn mlp_fields_are_what_the_block_used() {
        let _rng = crate::torch_rng_lock();
        let (model, block) = traced_block(&[0, 1, 2, 3]);
        let layer = &model.transformer_layers[0];

        let mid: Vec<Vec<f32>> = block
            .residual
            .iter()
            .zip(&block.mlp_out)
            .map(|(r, o)| r.iter().zip(o).map(|(a, b)| a - b).collect())
            .collect();
        assert_rows_close(
            &block.mlp_in,
            layer.layer_norm2.forward(rows_to_tensor(&mid)),
        );
        let hidden = layer
            .activation
            .forward(layer.linear1.forward(rows_to_tensor(&block.mlp_in)));
        assert_rows_close(&block.mlp_hidden, hidden);
        assert_rows_close(
            &block.mlp_out,
            layer.linear2.forward(rows_to_tensor(&block.mlp_hidden)),
        );
    }
}

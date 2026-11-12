//! Fused kernels burn does not provide in a form we can train with.
//!
//! - **Causal attention.** burn's `attention` module op dispatches to PyTorch's fused
//!   `scaled_dot_product_attention` on LibTorch, *but only without autodiff*: `Autodiff<B>`
//!   implements it with `attention_fallback`, i.e. `softmax(QKᵀ)V` written out as separate ops.
//!   Training then materializes the `[batch, heads, seq, seq]` score matrix several times per
//!   layer (scores, mask, softmax output, and their gradients): 3.2 GB per copy per layer at
//!   batch 64 × 1024 tokens in f32. That's memory-bound work for a 4090, and it caps the batch.
//!   `causal_attention` runs PyTorch's fused kernel forward *and* backward instead (FlashAttention
//!   for bf16 inputs, memory-efficient attention for f32), which never builds that matrix.
//!
//! - **Cross-entropy.** burn's `CrossEntropyLoss` is `log_softmax` + gather, so autodiff keeps the
//!   `[tokens, vocab]` log-probabilities on top of the logits: 2 × 8.6 GB at batch 64 × 1024 with a
//!   32k vocabulary in f32, plus the gradient. `cross_entropy` saves only the logits and a
//!   per-token log-sum-exp, and writes the gradient `softmax − onehot` directly, in row chunks so
//!   its f32 temporaries stay small even when the logits are bf16.
//!
//! Both are exposed through the `FusedOps` backend extension trait, implemented for `LibTorch`
//! (the actual kernels) and for `Autodiff<B>` (the gradient bookkeeping), following burn's
//! "custom backend operation" pattern.

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::libtorch::{LibTorch, TchElement, TchTensor};
use burn::prelude::*;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ops::{FloatTensor, IntTensor};

/// Rows of logits processed at once by the cross-entropy kernels. Bounds the f32 temporaries
/// to `CE_CHUNK_ROWS × vocab × 4` bytes (256 MB at a 32k vocabulary).
const CE_CHUNK_ROWS: i64 = 2048;

pub trait FusedOps: Backend {
    /// Whatever the backward pass of `causal_attention` needs to keep from the forward pass.
    type AttentionCtx: Clone + Send + core::fmt::Debug + 'static;

    /// Causal scaled-dot-product attention, `softmax(QKᵀ/√d + causal mask) V`.
    /// `q`, `k`, `v`: `[batch, heads, seq, head_dim]`.
    fn causal_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Same as `causal_attention`, also returning what `causal_attention_backward` needs.
    fn causal_attention_with_ctx(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, Self::AttentionCtx);

    /// Gradients with respect to `(q, k, v)` given the gradient of the output.
    fn causal_attention_backward(
        ctx: Self::AttentionCtx,
        grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    /// Mean cross-entropy of `logits` `[tokens, vocab]` (any float dtype) against `targets`
    /// `[tokens]`, plus the per-token log-sum-exp the backward pass reuses. The loss is a
    /// one-element f32 tensor whatever the dtype of `logits`.
    fn cross_entropy(
        logits: FloatTensor<Self>,
        targets: IntTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// Gradient with respect to `logits`, in the dtype of `logits`.
    fn cross_entropy_backward(
        logits: FloatTensor<Self>,
        targets: IntTensor<Self>,
        log_sum_exp: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// A backend usable for training with the fused ops: both it and its inner (validation)
/// backend implement them.
pub trait FusedAutodiffBackend: AutodiffBackend<InnerBackend: FusedOps> + FusedOps {}
impl<B: AutodiffBackend<InnerBackend: FusedOps> + FusedOps> FusedAutodiffBackend for B {}

// --- LibTorch: the kernels ---------------------------------------------------------------

/// Torch-autograd state for the attention backward. The forward pass runs PyTorch's fused SDPA
/// on detached leaf copies of q/k/v with gradients enabled, so PyTorch itself records what its
/// fused backward kernel needs (the output and the softmax log-sum-exp); `out` carries that
/// graph. burn only ever sees a detached copy of the output.
#[derive(Clone, Debug)]
pub struct TorchAttentionCtx {
    q: TchTensor,
    k: TchTensor,
    v: TchTensor,
    out: TchTensor,
}

fn sdpa(q: &tch::Tensor, k: &tch::Tensor, v: &tch::Tensor) -> tch::Tensor {
    tch::Tensor::scaled_dot_product_attention::<tch::Tensor>(q, k, v, None, 0.0, true, None, false)
}

impl<E: TchElement> FusedOps for LibTorch<E> {
    type AttentionCtx = TorchAttentionCtx;

    fn causal_attention(q: TchTensor, k: TchTensor, v: TchTensor) -> TchTensor {
        TchTensor::new(tch::no_grad(|| sdpa(&q.tensor, &k.tensor, &v.tensor)))
    }

    fn causal_attention_with_ctx(
        q: TchTensor,
        k: TchTensor,
        v: TchTensor,
    ) -> (TchTensor, TorchAttentionCtx) {
        let leaf = |t: &TchTensor| t.tensor.detach().set_requires_grad(true);
        let (q, k, v) = (leaf(&q), leaf(&k), leaf(&v));
        let out = tch::with_grad(|| sdpa(&q, &k, &v));
        let result = TchTensor::new(out.detach());
        let ctx = TorchAttentionCtx {
            q: TchTensor::new(q),
            k: TchTensor::new(k),
            v: TchTensor::new(v),
            out: TchTensor::new(out),
        };
        (result, ctx)
    }

    fn causal_attention_backward(
        ctx: TorchAttentionCtx,
        grad: TchTensor,
    ) -> (TchTensor, TchTensor, TchTensor) {
        // `run_backward` only differentiates scalars, so seed it with `sum(out * grad)`, whose
        // gradient with respect to `out` is exactly `grad`.
        let grad = grad.tensor.to_kind(ctx.out.tensor.kind());
        let grads = tch::with_grad(|| {
            let seed = (&ctx.out.tensor * &grad).sum(None);
            tch::Tensor::run_backward(
                &[seed],
                &[&ctx.q.tensor, &ctx.k.tensor, &ctx.v.tensor],
                false,
                false,
            )
        });
        let [dq, dk, dv]: [tch::Tensor; 3] = grads.try_into().expect("three input gradients");
        (TchTensor::new(dq), TchTensor::new(dk), TchTensor::new(dv))
    }

    fn cross_entropy(logits: TchTensor, targets: TchTensor) -> (TchTensor, TchTensor) {
        let (loss, lse) = tch::no_grad(|| {
            let logits = &logits.tensor;
            let targets = &targets.tensor;
            let rows = logits.size()[0];
            // log-sum-exp in f32, a chunk of rows at a time, so bf16 logits are never upcast
            // all at once.
            let lse = tch::Tensor::cat(
                &logits
                    .split(CE_CHUNK_ROWS, 0)
                    .iter()
                    .map(|chunk| chunk.to_kind(tch::Kind::Float).logsumexp([-1], false))
                    .collect::<Vec<_>>(),
                0,
            );
            let target_logit = logits
                .gather(1, &targets.unsqueeze(1), false)
                .squeeze_dim(1)
                .to_kind(tch::Kind::Float);
            let loss = (&lse - target_logit).sum(tch::Kind::Float) / rows as f64;
            (loss.reshape([1]), lse)
        });
        (TchTensor::new(loss), TchTensor::new(lse))
    }

    fn cross_entropy_backward(
        logits: TchTensor,
        targets: TchTensor,
        lse: TchTensor,
        grad: TchTensor,
    ) -> TchTensor {
        let out = tch::no_grad(|| {
            let logits = &logits.tensor;
            let rows = logits.size()[0];
            // d(mean CE)/d logits = (softmax(logits) − onehot(target)) · grad / rows
            let scale = grad.tensor.to_kind(tch::Kind::Float) / rows as f64;
            let out = logits.empty_like();
            for ((chunk, target), (lse, mut dst)) in logits
                .split(CE_CHUNK_ROWS, 0)
                .iter()
                .zip(targets.tensor.split(CE_CHUNK_ROWS, 0))
                .zip(
                    lse.tensor
                        .split(CE_CHUNK_ROWS, 0)
                        .into_iter()
                        .zip(out.split(CE_CHUNK_ROWS, 0)),
                )
            {
                let mut probs = (chunk.to_kind(tch::Kind::Float) - lse.unsqueeze(1)).exp();
                let _ = probs.scatter_add_(
                    1,
                    &target.unsqueeze(1),
                    &tch::Tensor::full(
                        [target.size()[0], 1],
                        -1.0,
                        (tch::Kind::Float, probs.device()),
                    ),
                );
                let _ = probs.g_mul_(&scale);
                dst.copy_(&probs);
            }
            out
        });
        TchTensor::new(out)
    }
}

// --- Autodiff: the gradient bookkeeping ---------------------------------------------------

impl<B: FusedOps, C: CheckpointStrategy> FusedOps for Autodiff<B, C> {
    type AttentionCtx = B::AttentionCtx;

    fn causal_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct CausalAttention;

        impl<B: FusedOps> Backward<B, 3> for CausalAttention {
            type State = B::AttentionCtx;

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_q, node_k, node_v] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (dq, dk, dv) = B::causal_attention_backward(ops.state, grad);
                for (node, grad) in [(node_q, dq), (node_k, dk), (node_v, dv)] {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, grad);
                    }
                }
            }
        }

        match CausalAttention
            .prepare::<C>([q.node.clone(), k.node.clone(), v.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (out, ctx) = B::causal_attention_with_ctx(q.primitive, k.primitive, v.primitive);
                prep.finish(ctx, out)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::causal_attention(q.primitive, k.primitive, v.primitive))
            }
        }
    }

    fn causal_attention_with_ctx(
        _q: FloatTensor<Self>,
        _k: FloatTensor<Self>,
        _v: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, Self::AttentionCtx) {
        unreachable!("second-order gradients through attention are not supported")
    }

    fn causal_attention_backward(
        _ctx: Self::AttentionCtx,
        _grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        unreachable!("second-order gradients through attention are not supported")
    }

    fn cross_entropy(
        logits: FloatTensor<Self>,
        targets: IntTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct CrossEntropy;

        impl<B: FusedOps> Backward<B, 1> for CrossEntropy {
            type State = (FloatTensor<B>, IntTensor<B>, FloatTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_logits] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                if let Some(node) = node_logits {
                    let (logits, targets, lse) = ops.state;
                    let grad = B::cross_entropy_backward(logits, targets, lse, grad);
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        let (loss, lse) = B::cross_entropy(logits.primitive.clone(), targets.clone());
        let loss = match CrossEntropy
            .prepare::<C>([logits.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((logits.primitive, targets, lse.clone()), loss),
            OpsKind::UnTracked(prep) => prep.finish(loss),
        };
        // The log-sum-exp is an implementation detail of the backward pass, not differentiable.
        (loss, <Self as AutodiffBackend>::from_inner(lse))
    }

    fn cross_entropy_backward(
        _logits: FloatTensor<Self>,
        _targets: IntTensor<Self>,
        _lse: FloatTensor<Self>,
        _grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unreachable!("second-order gradients through cross-entropy are not supported")
    }
}

// --- Tensor-level wrappers ----------------------------------------------------------------

pub fn causal_attention<B: FusedOps>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    Tensor::from_primitive(TensorPrimitive::Float(B::causal_attention(
        q.into_primitive().tensor(),
        k.into_primitive().tensor(),
        v.into_primitive().tensor(),
    )))
}

/// Mean cross-entropy between `logits` `[tokens, vocab]` and `targets` `[tokens]`.
pub fn cross_entropy<B: FusedOps>(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let (loss, _lse) = B::cross_entropy(logits.into_primitive().tensor(), targets.into_primitive());
    Tensor::from_primitive(TensorPrimitive::Float(loss))
}

// ----------- TESTS -----------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, LibTorch};
    use burn::nn::loss::CrossEntropyLossConfig;
    use burn::tensor::Distribution;
    use burn::tensor::Tolerance;
    use burn::tensor::module::attention_fallback;
    use burn::tensor::ops::AttentionModuleOptions;

    type B = Autodiff<LibTorch<f32>>;

    fn assert_close<const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, what: &str) {
        let diff = (a.clone() - b.clone()).abs().max().into_scalar();
        assert!(diff < 1e-4, "{what}: max abs diff {diff}");
        a.into_data()
            .assert_approx_eq::<f32>(&b.into_data(), Tolerance::absolute(1e-4));
    }

    /// The fused attention must match softmax(QKᵀ/√d + causal mask)V written out op by op, in
    /// value and in all three input gradients.
    #[test]
    fn causal_attention_matches_reference_forward_and_backward() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let shape = [2, 3, 5, 4];
        let make = || Tensor::<B, 4>::random(shape, Distribution::Normal(0.0, 1.0), &device);
        let (q, k, v) = (make(), make(), make());
        let upstream = Tensor::<B, 4>::random(shape, Distribution::Normal(0.0, 1.0), &device);

        let run = |fused: bool| {
            let (q, k, v) = (q.clone().require_grad(), k.clone().require_grad(), v.clone().require_grad());
            let out = if fused {
                causal_attention(q.clone(), k.clone(), v.clone())
            } else {
                let options = AttentionModuleOptions { is_causal: true, ..Default::default() };
                attention_fallback(q.clone(), k.clone(), v.clone(), None, None, options)
            };
            let grads = (out.clone() * upstream.clone()).sum().backward();
            let grad = |t: &Tensor<B, 4>| Tensor::<B, 4>::from_inner(t.grad(&grads).unwrap());
            (out, grad(&q), grad(&k), grad(&v))
        };

        let (out, dq, dk, dv) = run(true);
        let (out_ref, dq_ref, dk_ref, dv_ref) = run(false);
        assert_close(out, out_ref, "output");
        assert_close(dq, dq_ref, "dq");
        assert_close(dk, dk_ref, "dk");
        assert_close(dv, dv_ref, "dv");
    }

    /// The fused cross-entropy must match burn's `CrossEntropyLoss`, in value and gradient,
    /// including across several row chunks.
    #[test]
    fn cross_entropy_matches_reference_forward_and_backward() {
        let _rng = crate::torch_rng_lock();
        let device = Default::default();
        let rows = CE_CHUNK_ROWS as usize * 2 + 3; // spans three chunks, the last one partial
        let vocab = 11;
        let logits = Tensor::<B, 2>::random([rows, vocab], Distribution::Normal(0.0, 3.0), &device);
        let targets = Tensor::<B, 1, Int>::random([rows], Distribution::Uniform(0.0, vocab as f64), &device);

        let run = |fused: bool| {
            let logits = logits.clone().require_grad();
            let loss = if fused {
                cross_entropy(logits.clone(), targets.clone())
            } else {
                CrossEntropyLossConfig::new().init(&device).forward(logits.clone(), targets.clone())
            };
            let grads = (loss.clone() * 2.5).backward(); // non-unit upstream gradient
            (loss, Tensor::<B, 2>::from_inner(logits.grad(&grads).unwrap()))
        };

        let (loss, grad) = run(true);
        let (loss_ref, grad_ref) = run(false);
        assert_close(loss, loss_ref, "loss");
        assert_close(grad, grad_ref, "grad");
    }
}

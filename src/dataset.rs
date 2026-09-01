use burn::data::dataset::Dataset;
use burn::data::dataset::transform::PartialDataset;
use burn::data::dataset::{HuggingfaceDatasetLoader, SqliteDataset};
use burn::tensor::Int;
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TextItem {
    pub text: String,
}

/// Load the dataset
pub fn _load_gutenberg_dataset() -> SqliteDataset<TextItem> {
    HuggingfaceDatasetLoader::new("NNEngine/Gutenberg-Clean")
        .dataset("train") // There's only a train split in this dataset
        .expect("Failed to load the dataset")
}

/// Path to the cached fineweb-edu sqlite file that `burn-dataset` downloaded
/// under `~/.cache/burn-dataset`.
pub fn default_fineweb_dataset_path() -> PathBuf {
    // let home = std::env::var("HOME").expect("HOME environment variable must be set");
    PathBuf::from("/")
        .join("workspace")
        .join("HuggingFaceFWfineweb-edu-sample-10BT.db")
}

pub fn load_fineweb_dataset_from_disk(path: &str) -> SqliteDataset<TextItem> {
    SqliteDataset::from_db_file(path, "train").expect("Failed to load SQLite dataset")
}

/// Load the fineweb-edu dataset directly from its cached `.db` file in
/// `~/.cache/burn-dataset`, skipping `HuggingfaceDatasetLoader`'s network/cache check.
pub fn load_default_fineweb_dataset() -> SqliteDataset<TextItem> {
    let path = default_fineweb_dataset_path();
    let path = path.to_str().expect("dataset path should be valid UTF-8");
    load_fineweb_dataset_from_disk(path)
}

/// Helper to split the dataset.
///
/// `max_train_items`/`max_valid_items` optionally cap the size of the train/validation splits
/// (taking each one's first N items) without moving the underlying 80/10/10 boundaries — the
/// test split's start point is always the true (uncapped) validation end, so it stays fixed
/// and non-overlapping regardless of capping.
///
/// The caps are in *documents*, which is what a split is made of. The `train` command takes its
/// `--epoch-size`/`--valid-size` in batches instead — a batch being one optimizer step, which is
/// the unit checkpoint cadence is measured in — and multiplies by `documents_per_batch` to get
/// the numbers passed here.
///
/// Burn's checkpointing strategy only fires at epoch boundaries, so a full-corpus epoch (and
/// full-corpus validation pass) on a dataset this size means waiting days for the first
/// checkpoint; capping both to a smaller slice makes epochs (and therefore checkpoints) land at
/// a practical cadence. The tradeoff: within a single run, every epoch reuses the same capped
/// slices rather than sweeping fresh data.
pub fn split_dataset(
    dataset: SqliteDataset<TextItem>,
    max_train_items: Option<usize>,
    max_valid_items: Option<usize>,
) -> (
    PartialDataset<Arc<SqliteDataset<TextItem>>, TextItem>,
    PartialDataset<Arc<SqliteDataset<TextItem>>, TextItem>,
    PartialDataset<Arc<SqliteDataset<TextItem>>, TextItem>,
) {
    let len = dataset.len();
    let arc_dataset = Arc::new(dataset);

    // Define standard 80/10/10 split indices
    let train_end_full = (len as f32 * 0.8) as usize;
    let val_end_full = train_end_full + ((len as f32 * 0.1) as usize);

    // Optionally cap the train/validation splits; the test split boundary is unaffected.
    let train_end = match max_train_items {
        Some(cap) => cap.min(train_end_full),
        None => train_end_full,
    };
    let val_end = match max_valid_items {
        Some(cap) => (train_end_full + cap).min(val_end_full),
        None => val_end_full,
    };

    // Create partial datasets using slice indices
    let train_dataset = PartialDataset::new(arc_dataset.clone(), 0, train_end);
    let val_dataset = PartialDataset::new(arc_dataset.clone(), train_end_full, val_end);
    let test_dataset = PartialDataset::new(arc_dataset, val_end_full, len);

    (train_dataset, val_dataset, test_dataset)
}

//--- Batcher ---

/// Rough tokens-per-document estimate, used to size the pool of documents feeding each batch.
///
/// The measured mean over 300k fineweb-edu rows is ~1031 tokens, but sizing the pool on the
/// mean means it falls short about half the time. A deliberately low estimate buys headroom.
/// Measured trade-off at `--batch-size 64 --context-length 1024` (65537 tokens needed):
///
/// | estimate | documents | pool too small | token utilisation | mean duplication |
/// |----------|-----------|----------------|-------------------|------------------|
/// |      700 |        94 |           0.4% |             69.7% |            0.02% |
/// |  **800** |    **82** |       **7.9%** |         **79.8%** |        **0.43%** |
/// |      900 |        73 |          28.5% |             88.1% |            2.41% |
/// |     1031 |        64 |          56.3% |             93.5% |            7.69% |
///
/// 800 is the sweet spot: duplication stays under half a percent while still using ~80% of
/// every token it tokenizes. Tokens left unused are not lost — those documents come round
/// again in later epochs, at a different random offset.
const ESTIMATED_TOKENS_PER_DOC: usize = 800;

/// How many documents the dataloader should hand to `TextBatcher::batch` to build one batch of
/// `batch_size` sequences of `context_length` tokens.
///
/// This is deliberately *not* `batch_size`. The batcher packs documents end to end and cuts
/// fixed-size windows out of the result, so what it needs is a token budget, not a document
/// count — and a document is worth far more than one training sequence.
pub fn documents_per_batch(batch_size: usize, context_length: usize) -> usize {
    (batch_size * context_length + 1).div_ceil(ESTIMATED_TOKENS_PER_DOC)
}

/// Turns a pool of documents into one batch of fixed-size training sequences.
///
/// Documents are concatenated into a single token stream separated by `<|endoftext|>`, and the
/// batch is cut from that stream as `batch_size` consecutive windows of `context_length` tokens.
///
/// ```text
///   [doc A ......][EOT][doc B ..........][EOT][doc C ....][EOT][doc D ...
///   |--- window 0 ---||--- window 1 ---||--- window 2 ---|
/// ```
///
/// Packing rather than slicing each document separately matters for three reasons:
///
/// 1. **Every batch has the same shape.** The previous batcher set the sequence length from the
///    *shortest* document in the batch, and the minimum of 64 draws from a heavy-tailed length
///    distribution is tiny: measured on fineweb-edu it was ~106 tokens against a median document
///    of 624, so a 1024-token context trained at ~106 and no batch ever reached the full window.
/// 2. **Position embeddings past the shortest document used to get no gradient at all.** With
///    the sequence length pinned near 106, positions above ~209 were never trained, leaving most
///    of the position table at its random initialisation.
/// 3. **`<|endoftext|>` appears in the training data.** Slicing documents individually never
///    emitted it, so the model could not learn to stop and generation ran until its token
///    budget expired.
///
/// A window may straddle a document boundary, so the model can attend across `<|endoftext|>`
/// into unrelated text. This is the same trade-off GPT-2, nanoGPT and Llama make: the separator
/// token is what teaches the boundary. Masking attention per document is a refinement, not a
/// prerequisite.
#[derive(Clone)]
pub struct TextBatcher<T: Tokenizer> {
    tokenizer: T,
    /// Tokens per training sequence — the width of the returned tensors.
    context_length: usize,
    /// Sequences per batch — the height of the returned tensors.
    batch_size: usize,
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<T: Tokenizer> TextBatcher<T> {
    pub fn new(tokenizer: T, context_length: usize, batch_size: usize) -> Self {
        assert!(context_length > 0, "context_length must be non-zero");
        assert!(batch_size > 0, "batch_size must be non-zero");
        Self {
            tokenizer,
            context_length,
            batch_size,
        }
    }
}

impl<T: Tokenizer + Send + Sync, B: Backend> Batcher<B, TextItem, TextBatch<B>> for TextBatcher<T> {
    fn batch(&self, items: Vec<TextItem>, device: &B::Device) -> TextBatch<B> {
        // 1. Pack every document into one flat token stream, with an end-of-text token marking
        //    each boundary so the model can learn where documents stop.
        let end_of_text = self.tokenizer.end_of_text().0 as i64;
        let mut stream: Vec<i64> = Vec::new();
        for item in &items {
            stream.extend(self.tokenizer.encode(&item.text).iter().map(|t| t.0 as i64));
            stream.push(end_of_text);
        }
        assert!(
            !stream.is_empty(),
            "batch received no documents; the dataloader should never hand out an empty pool"
        );

        // 2. Read the batch out of the stream as a *ring*, starting at a random offset.
        //
        //    The random offset means successive epochs cut the same documents at different
        //    points instead of replaying identical windows. Wrapping with `%` also covers the
        //    case where this pool of documents did not supply enough tokens: the stream simply
        //    repeats, rather than the batcher emitting a short batch and reintroducing the
        //    varying tensor shapes this design exists to eliminate. `documents_per_batch` sizes
        //    the pool so that wrap-around stays rare (~8% of batches, duplicating <0.5% of
        //    tokens) — it is a safety net, not the normal path.
        let start = rand::thread_rng().gen_range(0..stream.len());
        let token_at = |offset: usize| stream[(start + offset) % stream.len()];

        // 3. Cut `batch_size` windows laid end to end. Each window reads one token past its own
        //    width so the target is the input shifted left by one; consecutive windows therefore
        //    share exactly one token at their boundary.
        let token_count = self.batch_size * self.context_length;
        let mut inputs = Vec::with_capacity(token_count);
        let mut targets = Vec::with_capacity(token_count);
        for window in 0..self.batch_size {
            let window_start = window * self.context_length;
            for i in 0..self.context_length {
                inputs.push(token_at(window_start + i));
                targets.push(token_at(window_start + i + 1));
            }
        }

        // 4. Upload each tensor once, rather than once per sequence plus a stack. The previous
        //    batcher uploaded every document in full and then sliced ~10% of it back out.
        let shape = [self.batch_size, self.context_length];
        TextBatch {
            inputs: Tensor::<B, 1, Int>::from_data(inputs.as_slice(), device).reshape(shape),
            targets: Tensor::<B, 1, Int>::from_data(targets.as_slice(), device).reshape(shape),
        }
    }
}

// ----------- TESTS -----------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{SimpleTokenizer, Token, Vocab};
    use burn::backend::LibTorch;
    use std::collections::HashMap;

    type TestBackend = LibTorch<f32>;

    fn make_test_tokenizer() -> SimpleTokenizer {
        let mut words2tokens = HashMap::new();
        let tokens2words = vec![
            "<UNK>".to_string(),
            "<EOT>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
            "f".to_string(),
            "g".to_string(),
            "h".to_string(),
            "i".to_string(),
            "j".to_string(),
            "k".to_string(),
            "l".to_string(),
            "m".to_string(),
            "n".to_string(),
            "o".to_string(),
            "p".to_string(),
            "q".to_string(),
            "r".to_string(),
            "s".to_string(),
            "t".to_string(),
            "u".to_string(),
            "v".to_string(),
            "w".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ];

        for (i, word) in tokens2words.iter().enumerate() {
            words2tokens.insert(word.clone(), Token(i as u32));
        }

        let vocab = Vocab {
            words2tokens,
            tokens2words,
        };

        SimpleTokenizer::new(vocab)
    }

    /// Decode a batch tensor back into a `Vec<Vec<i64>>`, one inner vec per sequence.
    fn rows(t: Tensor<TestBackend, 2, Int>) -> Vec<Vec<i64>> {
        let [batch_size, seq_len] = t.dims();
        let flat = t.into_data().to_vec::<i64>().unwrap();
        flat.chunks(seq_len).map(<[i64]>::to_vec).collect()
    }

    fn items(texts: &[&str]) -> Vec<TextItem> {
        texts
            .iter()
            .map(|t| TextItem {
                text: (*t).to_string(),
            })
            .collect()
    }

    /// The whole point of the packing batcher: the shape is what you asked for, no matter what
    /// the documents look like. The old batcher derived it from the shortest document, so a
    /// single short document collapsed the sequence length for the entire batch.
    #[test]
    fn shape_is_fixed_regardless_of_document_lengths() {
        let device = Default::default();
        let batcher = TextBatcher::new(make_test_tokenizer(), 4, 3);

        // One long document mixed with documents far shorter than the context length.
        let batch: TextBatch<TestBackend> = batcher.batch(
            items(&["a b c d e f g h i j k l m n o p", "q r", "s", "t u v w x y z"]),
            &device,
        );

        assert_eq!(batch.inputs.dims(), [3, 4]);
        assert_eq!(batch.targets.dims(), [3, 4]);
    }

    /// A two-token document would have made the old batcher emit a single-token sequence.
    #[test]
    fn a_tiny_document_does_not_shrink_the_batch() {
        let device = Default::default();
        let batcher = TextBatcher::new(make_test_tokenizer(), 8, 2);

        let batch: TextBatch<TestBackend> =
            batcher.batch(items(&["a b", "c d e f g h i j k l m n o p q r s t"]), &device);

        assert_eq!(batch.inputs.dims(), [2, 8]);
    }

    /// Targets must be inputs shifted left by one, everywhere — that shift *is* the language
    /// modelling objective, including across the boundary between two windows.
    #[test]
    fn targets_are_inputs_shifted_by_one() {
        let device = Default::default();
        let batcher = TextBatcher::new(make_test_tokenizer(), 5, 3);

        let batch: TextBatch<TestBackend> = batcher.batch(
            items(&[
                "a b c d e f g h i j",
                "k l m n o p q r s t",
                "u v w x y z a b c d",
            ]),
            &device,
        );

        let inputs = rows(batch.inputs);
        let targets = rows(batch.targets);

        for (input, target) in inputs.iter().zip(&targets) {
            // Within a window, target[i] is the token after input[i].
            assert_eq!(&input[1..], &target[..target.len() - 1]);
        }
        // Consecutive windows are laid end to end, so window n's last target is window n+1's
        // first input.
        for w in 0..inputs.len() - 1 {
            assert_eq!(*targets[w].last().unwrap(), inputs[w + 1][0]);
        }
    }

    /// Document boundaries have to be visible in the token stream, otherwise the model never
    /// learns to emit end-of-text and generation cannot terminate on its own.
    #[test]
    fn documents_are_separated_by_end_of_text() {
        let device = Default::default();
        let tokenizer = make_test_tokenizer();
        let eot = tokenizer.end_of_text().0 as i64;
        // Context of 2 over many short documents guarantees the windows cover several boundaries.
        let batcher = TextBatcher::new(tokenizer, 2, 8);

        let batch: TextBatch<TestBackend> =
            batcher.batch(items(&["a b"; 12]), &device);

        let seen: Vec<i64> = batch.inputs.into_data().to_vec::<i64>().unwrap();
        assert!(
            seen.contains(&eot),
            "packed stream should contain the end-of-text separator, got {seen:?}"
        );
    }

    /// When the documents supply fewer tokens than the batch needs, the stream wraps instead of
    /// the batcher emitting a short batch — varying shapes are exactly what this design removes.
    #[test]
    fn a_short_document_pool_wraps_instead_of_shrinking() {
        let device = Default::default();
        // Needs 8 * 16 = 128 tokens; one 3-token document supplies 4 (3 + end-of-text).
        let batcher = TextBatcher::new(make_test_tokenizer(), 16, 8);

        let batch: TextBatch<TestBackend> = batcher.batch(items(&["a b c"]), &device);

        assert_eq!(batch.inputs.dims(), [8, 16]);
        let flat = batch.inputs.into_data().to_vec::<i64>().unwrap();
        assert_eq!(flat.len(), 128);
    }

    /// The random start offset is what stops successive epochs from replaying identical windows.
    #[test]
    fn successive_batches_start_at_different_offsets() {
        let device = Default::default();
        let batcher = TextBatcher::new(make_test_tokenizer(), 3, 1);
        let docs = items(&["a b c d e f g h i j k l m n o p q r s t u v w x y z"]);

        let first: TextBatch<TestBackend> = batcher.batch(docs.clone(), &device);
        let first = first.inputs.into_data().to_vec::<i64>().unwrap();

        let differs = (0..20).any(|_| {
            let next: TextBatch<TestBackend> = batcher.batch(docs.clone(), &device);
            next.inputs.into_data().to_vec::<i64>().unwrap() != first
        });
        assert!(differs, "batches should not always start at the same offset");
    }

    /// The document pool must be sized by token budget, not by sequence count — a document is
    /// worth far more than one training sequence.
    #[test]
    fn documents_per_batch_covers_the_token_budget() {
        // 64 * 1024 + 1 = 65537 tokens, at an assumed 800 tokens per document.
        assert_eq!(documents_per_batch(64, 1024), 82);
        // Always at least one document, even for a tiny batch.
        assert_eq!(documents_per_batch(1, 1), 1);
        // Scales with the token budget, not the sequence count.
        assert_eq!(
            documents_per_batch(128, 1024),
            2 * documents_per_batch(64, 1024)
        );
    }
}

use burn::data::dataset::Dataset;
use burn::data::dataset::transform::PartialDataset;
use burn::data::dataset::{HuggingfaceDatasetLoader, SqliteDataset};
use burn::tensor::Int;
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{SimpleTokenizer, Tokenizer, Vocab};
use std::collections::HashMap;
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

/// Load the dataset
pub fn _load_fineweb_dataset() -> SqliteDataset<TextItem> {
    HuggingfaceDatasetLoader::new("HuggingFaceFW/fineweb-edu")
        .with_subset("sample-10BT")
        .dataset("train")
        .expect("Failed to load the dataset")
}
pub fn load_fineweb_dataset_from_disk(path: &str) -> SqliteDataset<TextItem> {
    SqliteDataset::from_db_file(path, "train").expect("Failed to load SQLite dataset")
}

/// Helper to split the dataset
pub fn split_dataset(
    dataset: SqliteDataset<TextItem>,
) -> (
    PartialDataset<Arc<SqliteDataset<TextItem>>, TextItem>,
    PartialDataset<Arc<SqliteDataset<TextItem>>, TextItem>,
    PartialDataset<Arc<SqliteDataset<TextItem>>, TextItem>,
) {
    let len = dataset.len();
    let arc_dataset = Arc::new(dataset);

    // Define standard 80/10/10 split indices
    let train_end = (len as f32 * 0.8) as usize;
    let val_end = train_end + ((len as f32 * 0.1) as usize);

    // Create partial datasets using slice indices
    let train_dataset = PartialDataset::new(arc_dataset.clone(), 0, train_end);
    let val_dataset = PartialDataset::new(arc_dataset.clone(), train_end, val_end);
    let test_dataset = PartialDataset::new(arc_dataset, val_end, len);

    (train_dataset, val_dataset, test_dataset)
}

//--- Batcher ---
#[derive(Clone, Default)]
pub struct TextBatcher<T: Tokenizer> {
    tokenizer: T,
    context_length: usize,
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<T: Tokenizer> TextBatcher<T> {
    pub fn new(tokenizer: T, context_length: usize) -> Self {
        Self {
            tokenizer,
            context_length,
        }
    }
}

impl<T: Tokenizer + Send + Sync, B: Backend> Batcher<B, TextItem, TextBatch<B>> for TextBatcher<T> {
    fn batch(&self, items: Vec<TextItem>, device: &B::Device) -> TextBatch<B> {
        // 1. Tokenize all items and create per-text tensors on the device
        let tokenized: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                let tokens: Vec<i32> = self
                    .tokenizer
                    .encode(&item.text)
                    .iter()
                    .map(|t| t.0 as i32)
                    .collect();
                Tensor::<B, 1, Int>::from_data(tokens.as_slice(), device)
            })
            .collect();

        // 2. Find the shortest tokenized text length
        let min_len = tokenized.iter().map(|t| t.dims()[0]).min().unwrap_or(0);

        // 3. Determine seq_len (input/target length)
        // We need seq_len + 1 tokens from each text (for the LM shift)
        let seq_len = (min_len.saturating_sub(1)).min(self.context_length);

        assert!(
            seq_len > 0,
            "All texts in the batch are too short (need at least 2 tokens)"
        );

        let mut rng = rand::thread_rng();

        // 4. For each text: pick random start and slice input/target on the GPU
        let mut inputs: Vec<Tensor<B, 1, Int>> = Vec::new();
        let mut targets: Vec<Tensor<B, 1, Int>> = Vec::new();

        for t in tokenized {
            let text_len = t.dims()[0];
            assert!(
                text_len >= seq_len + 1,
                "Text length {} is shorter than required seq_len + 1 = {}",
                text_len,
                seq_len + 1
            );

            let max_start = text_len - seq_len - 1;
            let start = if max_start > 0 {
                rng.gen_range(0..=max_start)
            } else {
                0
            };

            let input = t.clone().slice(start..start + seq_len);
            let target = t.slice(start + 1..start + seq_len + 1);

            inputs.push(input);
            targets.push(target);
        }

        // 5. Stack all per-text slices into batch tensors [batch_size, seq_len]
        TextBatch {
            inputs: Tensor::stack::<2>(inputs, 0),
            targets: Tensor::stack::<2>(targets, 0),
        }
    }
}

// ----------- TESTS -----------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Token;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu<f32>;

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

    #[test]
    fn test_batch_shape_and_shift() {
        let device = <TestBackend as Backend>::Device::default();
        let tokenizer = make_test_tokenizer();
        let batcher = TextBatcher::new(tokenizer, 1024);

        // "a b c d" -> tokens [2, 3, 4, 5] (4 tokens)
        // "e f"     -> tokens [6, 7] (2 tokens)
        let items = vec![
            TextItem {
                text: "a b c d".to_string(),
            },
            TextItem {
                text: "e f".to_string(),
            },
        ];

        let batch: TextBatch<TestBackend> = batcher.batch(items, &device);

        // seq_len = min(4, 2) - 1 = 1
        assert_eq!(batch.inputs.dims(), [2, 1]);
        assert_eq!(batch.targets.dims(), [2, 1]);

        // Verify input/target shift: target should be input + 1
        let input_data = batch.inputs.into_data().to_vec::<i32>().unwrap();
        let target_data = batch.targets.into_data().to_vec::<i32>().unwrap();

        // For text 1: input = [2] (a), target = [3] (b)  OR input = [3], target = [4] etc.
        // For text 2: input = [6] (e), target = [7] (f)
        assert_eq!(input_data[0] + 1, target_data[0]);
        assert_eq!(input_data[1] + 1, target_data[1]);
    }

    #[test]
    fn test_seq_len_capped_by_context_length() {
        let device = <TestBackend as Backend>::Device::default();
        let tokenizer = make_test_tokenizer();
        let batcher = TextBatcher::new(tokenizer, 2); // small context_length

        // "a b c d e f" -> tokens [2, 3, 4, 5, 6, 7] (6 tokens)
        // "g h i j"     -> tokens [8, 9, 10, 11] (4 tokens)
        let items = vec![
            TextItem {
                text: "a b c d e f".to_string(),
            },
            TextItem {
                text: "g h i j".to_string(),
            },
        ];

        let batch: TextBatch<TestBackend> = batcher.batch(items, &device);

        // seq_len = min(6, 4) - 1 = 3, but capped by context_length = 2
        assert_eq!(batch.inputs.dims(), [2, 2]);
        assert_eq!(batch.targets.dims(), [2, 2]);
    }

    #[test]
    fn test_long_text_random_slicing() {
        let device = <TestBackend as Backend>::Device::default();
        let tokenizer = make_test_tokenizer();
        // Use small context_length so random slicing has room to vary
        let batcher = TextBatcher::new(tokenizer, 3);

        // Long text: 10 tokens -> [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        // seq_len = min(10 - 1, 3) = 3
        // max_start = 10 - 3 - 1 = 6 -> multiple possible starting positions
        let items = vec![TextItem {
            text: "a b c d e f g h i j".to_string(),
        }];

        let batch: TextBatch<TestBackend> = batcher.batch(items.clone(), &device);

        assert_eq!(batch.inputs.dims(), [1, 3]);
        assert_eq!(batch.targets.dims(), [1, 3]);

        // Run multiple times and check we get different slices
        let input_data_1 = batch.inputs.into_data().to_vec::<i32>().unwrap();

        let mut found_different = false;
        for _ in 0..20 {
            let batch2: TextBatch<TestBackend> = batcher.batch(items.clone(), &device);
            let input_data_2 = batch2.inputs.into_data().to_vec::<i32>().unwrap();
            if input_data_1[0] != input_data_2[0] {
                found_different = true;
                break;
            }
        }

        assert!(
            found_different,
            "Random slicing should produce different starting positions over multiple runs"
        );
    }

    #[test]
    fn test_single_text_exact_length() {
        let device = <TestBackend as Backend>::Device::default();
        let tokenizer = make_test_tokenizer();
        let batcher = TextBatcher::new(tokenizer, 1024);

        // Exactly 3 tokens: "a b c" -> [2, 3, 4]
        // seq_len = 3 - 1 = 2
        let items = vec![TextItem {
            text: "a b c".to_string(),
        }];

        let batch: TextBatch<TestBackend> = batcher.batch(items, &device);

        assert_eq!(batch.inputs.dims(), [1, 2]);
        assert_eq!(batch.targets.dims(), [1, 2]);

        let input_data = batch.inputs.into_data().to_vec::<i32>().unwrap();
        let target_data = batch.targets.into_data().to_vec::<i32>().unwrap();

        // Should be deterministic since there's only one possible slice
        assert_eq!(input_data, vec![2, 3]);
        assert_eq!(target_data, vec![3, 4]);
    }
}

//! A prefetching batch loader that yields batches in a fixed, reproducible order.
//!
//! Why not burn's `MultiThreadDataLoader`: it hands each worker a contiguous chunk of the
//! dataset and yields whichever worker finishes first. That's fine for epoch-sized runs, but it
//! means "the first N batches consumed" is not a prefix of anything, so a run stopped mid-epoch
//! can't be resumed at the right place. Here batch `i` is always built from documents
//! `[i * docs_per_batch, (i + 1) * docs_per_batch)` of the (already shuffled) dataset, with a
//! batch-specific seed for the packing offset. Workers build batches in parallel, and the
//! iterator puts them back in order. Resuming at step `k` means starting at batch `k`.
//!
//! Workers only tokenize and pack on the CPU. The tensors are uploaded by the thread that
//! consumes the iterator (see `TextBatcher::upload`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use burn::data::dataset::Dataset;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::dataset::{PackedBatch, TextBatcher, TextItem};
use crate::tokenizer::Tokenizer;

pub struct OrderedLoader {
    receiver: Option<mpsc::Receiver<(usize, PackedBatch)>>,
    workers: Vec<thread::JoinHandle<()>>,
    /// Batches that arrived before the one the iterator is waiting for.
    pending: BTreeMap<usize, PackedBatch>,
    /// Index of the next batch to yield.
    next: usize,
    end: usize,
    /// Shared with the workers: the batch index the consumer is waiting for, for backpressure.
    consumed: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
}

impl OrderedLoader {
    /// Yield batches `start..end` of `dataset`, each built from `docs_per_batch` consecutive
    /// documents. At most `prefetch` batches are built ahead of the consumer.
    #[allow(clippy::too_many_arguments)]
    pub fn new<T>(
        dataset: Arc<dyn Dataset<TextItem>>,
        batcher: TextBatcher<T>,
        docs_per_batch: usize,
        start: usize,
        end: usize,
        seed: u64,
        num_workers: usize,
        prefetch: usize,
    ) -> Self
    where
        T: Tokenizer + Clone + Send + Sync + 'static,
    {
        assert!(num_workers > 0, "OrderedLoader needs at least one worker");
        let (sender, receiver) = mpsc::channel();
        let next_index = Arc::new(AtomicUsize::new(start));
        let consumed = Arc::new(AtomicUsize::new(start));
        let stop = Arc::new(AtomicBool::new(false));
        let prefetch = prefetch.max(num_workers);

        let workers = (0..num_workers)
            .map(|worker| {
                let dataset = dataset.clone();
                let batcher = batcher.clone();
                let sender = sender.clone();
                let next_index = next_index.clone();
                let consumed = consumed.clone();
                let stop = stop.clone();
                thread::Builder::new()
                    .name(format!("loader-{worker}"))
                    .spawn(move || {
                        loop {
                            let index = next_index.fetch_add(1, Ordering::Relaxed);
                            if index >= end {
                                break;
                            }
                            // Backpressure: don't run more than `prefetch` batches ahead.
                            while index >= consumed.load(Ordering::Relaxed) + prefetch {
                                if stop.load(Ordering::Relaxed) {
                                    return;
                                }
                                thread::sleep(Duration::from_millis(2));
                            }
                            if stop.load(Ordering::Relaxed) {
                                return;
                            }
                            let first = index * docs_per_batch;
                            let items: Vec<TextItem> = (first..first + docs_per_batch)
                                .map(|i| dataset.get(i).expect("batch index within dataset"))
                                .collect();
                            let mut rng = StdRng::seed_from_u64(seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                            let packed = batcher.pack(&items, &mut rng);
                            if sender.send((index, packed)).is_err() {
                                return; // consumer dropped
                            }
                        }
                    })
                    .expect("spawn loader worker")
            })
            .collect();

        Self {
            receiver: Some(receiver),
            workers,
            pending: BTreeMap::new(),
            next: start,
            end,
            consumed,
            stop,
        }
    }

    /// Number of whole batches `dataset_len` documents make.
    pub fn num_batches(dataset_len: usize, docs_per_batch: usize) -> usize {
        dataset_len / docs_per_batch
    }
}

impl Iterator for OrderedLoader {
    /// `(batch index, batch)`.
    type Item = (usize, PackedBatch);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.end {
            return None;
        }
        let receiver = self.receiver.as_ref()?;
        let batch = loop {
            if let Some(batch) = self.pending.remove(&self.next) {
                break batch;
            }
            let (index, batch) = receiver.recv().expect("loader workers exited early");
            self.pending.insert(index, batch);
        };
        let index = self.next;
        self.next += 1;
        self.consumed.store(self.next, Ordering::Relaxed);
        Some((index, batch))
    }
}

impl Drop for OrderedLoader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.receiver.take(); // unblocks any worker stuck in `send`
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{SimpleTokenizer, Token, Vocab};
    use burn::data::dataset::InMemDataset;
    use std::collections::HashMap;

    /// A letters-only word per document index (the word-level tokenizer splits digits apart).
    fn word(i: usize) -> String {
        let letter = |n: usize| char::from(b'a' + n as u8);
        format!("w{}{}", letter(i / 26), letter(i % 26))
    }

    fn tokenizer() -> SimpleTokenizer {
        let tokens2words: Vec<String> = ["<UNK>", "<EOT>"]
            .into_iter()
            .map(String::from)
            .chain((0..100).map(word))
            .collect();
        let words2tokens: HashMap<String, Token> = tokens2words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), Token(i as u32)))
            .collect();
        SimpleTokenizer::new(Vocab {
            words2tokens,
            tokens2words,
        })
    }

    /// Document `i` is the single word `word(i)`, so a batch's tokens identify its documents.
    fn dataset(n: usize) -> Arc<dyn Dataset<TextItem>> {
        Arc::new(InMemDataset::new(
            (0..n).map(|i| TextItem { text: word(i) }).collect(),
        ))
    }

    fn collect(start: usize, end: usize, workers: usize) -> Vec<(usize, PackedBatch)> {
        let batcher = TextBatcher::new(tokenizer(), 2, 1);
        OrderedLoader::new(dataset(40), batcher, 4, start, end, 7, workers, 3).collect()
    }

    #[test]
    fn yields_every_batch_in_order() {
        let indices: Vec<usize> = collect(0, 10, 4).into_iter().map(|(i, _)| i).collect();
        assert_eq!(indices, (0..10).collect::<Vec<_>>());
    }

    /// Resuming at batch k must produce exactly what an uninterrupted run produced from k on,
    /// whatever the worker count.
    #[test]
    fn resuming_reproduces_the_same_batches() {
        let full = collect(0, 10, 1);
        let resumed = collect(6, 10, 3);
        for ((i, a), (j, b)) in full[6..].iter().zip(&resumed) {
            assert_eq!(i, j);
            assert_eq!(a.inputs, b.inputs);
            assert_eq!(a.targets, b.targets);
        }
    }

    /// Batch i only sees documents [4i, 4i + 4): ids 2 + 4i .. 2 + 4i + 4, or end-of-text.
    #[test]
    fn batch_reads_its_own_documents() {
        for (index, batch) in collect(0, 10, 2) {
            for &t in &batch.inputs {
                let doc = (t - 2) as usize;
                assert!(t == 1 || (4 * index..4 * index + 4).contains(&doc), "batch {index} got token {t}");
            }
        }
    }

    #[test]
    fn dropping_early_does_not_hang() {
        let batcher = TextBatcher::new(tokenizer(), 2, 1);
        let mut loader = OrderedLoader::new(dataset(40), batcher, 4, 0, 10, 7, 4, 2);
        loader.next();
        drop(loader);
    }
}

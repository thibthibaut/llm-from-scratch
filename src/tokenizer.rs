use crate::dataset::TextItem;
use burn::data::dataloader::Dataset;
use burn::data::dataset::SqliteDataset;
use indicatif::ProgressIterator;
use rayon::prelude::*;
use regex::Regex;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::BufWriter, path::Path, sync::Arc, sync::LazyLock};
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

/// Token type
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token(pub u32);
impl Token {
    // Define some special tokens
    pub const UNKNOWN: Token = Token(0);
    pub const END_OF_TEXT: Token = Token(1);
}

/// Vocabulary with lookup and reverse lookup from text to tokens
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Vocab {
    pub words2tokens: HashMap<String, Token>,
    pub tokens2words: Vec<String>, // reverse lookup, indices are the tokens
}
impl Vocab {
    // TODO: Use serde_any instead, serde-file-formats or savefile
    pub fn from_file(path: &Path) -> Self {
        let file = std::fs::File::open(path).expect("Failed to open vocab file");
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).expect("Failed to deserialize vocab")
    }

    pub fn to_file(&self, path: &Path) {
        let file = File::create(path).expect("Failed to create file");
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self).expect("Failed to write JSON");
    }
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<Token>;
    fn decode(&self, tokens: &[Token]) -> String;
    fn get_vocab_size(&self) -> usize;

    /// The end-of-text token for *this* tokenizer.
    ///
    /// Not a constant: a trained BPE tokenizer assigns special-token ids at training time, so
    /// the generation loop has to ask the tokenizer rather than compare against a hardcoded id.
    fn end_of_text(&self) -> Token;

    /// The surface string a single token stands for, for inspection/debugging (top-k candidate
    /// listings and the like). Returns `None` for ids outside the vocabulary.
    ///
    /// This is *not* `decode` on a one-element slice: for a byte-level BPE tokenizer the piece
    /// is the byte-level-escaped form (`"\u{c4}\u{a0}the"`), which is exactly what you want when
    /// showing which token the model picked, and is not necessarily valid standalone text.
    fn token_to_piece(&self, token: Token) -> Option<String>;
}

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    pub vocab: Vocab,
}

impl SimpleTokenizer {
    fn token_regex() -> &'static Regex {
        static RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r#"[a-zA-Z]+|\d|[.,!?;:'"()]"#).unwrap());
        &RE
    }
    fn punct_regex() -> &'static Regex {
        static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" ([.,!?;:'])").unwrap());
        &RE
    }

    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    pub fn from_vocab_file(vocab_path: &Path) -> Self {
        let vocab = Vocab::from_file(vocab_path);
        Self::new(vocab)
    }

    fn split_words(text: &str) -> Vec<String> {
        Self::token_regex()
            .find_iter(&text.to_ascii_lowercase())
            .map(|match_| match_.as_str().to_owned())
            .collect()
    }

    /// Build vocabulary from a dataset
    pub fn build_vocab(dataset: &SqliteDataset<TextItem>) -> Vocab {
        let num_samples = dataset.len() as u64;
        let occurrence_counter: HashMap<String, u32> = dataset
            .iter()
            .take(num_samples as usize)
            .progress_count(num_samples)
            .par_bridge()
            .flat_map(|sample| Self::split_words(&sample.text))
            .fold(HashMap::new, |mut counts: HashMap<String, u32>, text| {
                *counts.entry(text).or_insert(0) += 1;
                counts
            })
            .reduce(HashMap::new, |mut counts: HashMap<String, u32>, b| {
                for (k, v) in b {
                    *counts.entry(k).or_insert(0) += v;
                }
                counts
            });

        println!("Vocab Size: {}", occurrence_counter.len());

        // Limit the size of the vocabulary by dropping low-frequency words
        let limit = 32_000;

        // Convert HashMap to Vec for sorting
        let mut sorted_entries: Vec<_> = occurrence_counter.into_iter().collect();

        //  Sort descending by count (most frequent first)
        sorted_entries.sort_by(|a, b| b.1.cmp(&a.1));

        // Keep only the top <limit>
        sorted_entries.truncate(limit);

        // Special tokens we want to add to our vocab
        let special_tokens = [("<UNK>", Token::UNKNOWN), ("<EOT>", Token::END_OF_TEXT)];

        // Convert the sorted Vec to the Vocab HashMap
        let mut vocab_map: HashMap<String, Token> = sorted_entries
            .into_iter()
            .enumerate()
            .map(|(index, (word, _freq))| (word, Token((index + special_tokens.len()) as u32)))
            .collect();

        // Insert special tokens
        for (word, tok) in special_tokens.iter() {
            vocab_map.insert(word.to_string(), *tok);
        }

        // Create the reverse lookup
        let mut reverse_lookup = vec![String::new(); vocab_map.len()];
        for (word, token) in &vocab_map {
            reverse_lookup[token.0 as usize] = word.clone();
        }
        Vocab {
            words2tokens: vocab_map,
            tokens2words: reverse_lookup,
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Vec<Token> {
        Self::split_words(text)
            .iter()
            .map(|entry| {
                self.vocab
                    .words2tokens
                    .get(entry)
                    .copied()
                    .unwrap_or(Token::UNKNOWN)
            })
            .collect()
    }

    fn decode(&self, tokens: &[Token]) -> String {
        let text: String = tokens
            .iter()
            .map(|tok| self.vocab.tokens2words[tok.0 as usize].as_str())
            .collect::<Vec<&str>>()
            .join(" ");

        // Remove spaces before puctuation
        Self::punct_regex().replace_all(&text, "$1").to_string()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.tokens2words.len()
    }

    fn end_of_text(&self) -> Token {
        Token::END_OF_TEXT
    }

    fn token_to_piece(&self, token: Token) -> Option<String> {
        self.vocab.tokens2words.get(token.0 as usize).cloned()
    }
}

// ----------- BYTE-LEVEL BPE TOKENIZER ----------------------------------------
//
// Backed by HuggingFace's `tokenizers` crate. Unlike `SimpleTokenizer` this is *lossless*:
// `decode(encode(x)) == x` for any input, including case, whitespace, punctuation the regex
// never covered (`-`, `/`, `=`, `$`, `{`, ...) and non-ASCII bytes. It also has no `<UNK>`,
// because the trainer is seeded with all 256 byte-level characters (see `train`), so every
// possible input byte is representable.

/// The single special token in a trained vocabulary: the document separator, also used as the
/// stop signal in the generation loop.
///
/// Deliberately *not* a CLI flag. Special tokens are a contract between the tokenizer and the
/// model code (`Tokenizer::end_of_text`, the generation loop's stop condition), not a tuning
/// knob — a flag would let you produce a tokenizer whose special tokens the rest of the
/// codebase doesn't know about. Its *id* is not hardcoded: it's read back from the trained
/// tokenizer via `end_of_text`.
pub const END_OF_TEXT_PIECE: &str = "<|endoftext|>";

/// Byte-level BPE tokenizer, either trained by `BpeTokenizer::train` or loaded from a
/// `tokenizer.json` (ours, or a pretrained one downloaded from the HuggingFace Hub).
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    // `Arc` because `TextBatcher` is cloned once per dataloader worker (`--num-workers` is 32
    // by default) and a deep copy of a 32k vocab plus its merge table per worker is pure waste.
    // The tokenizer is immutable after construction, so sharing it is free.
    inner: Arc<HfTokenizer>,
    end_of_text: Token,
}

impl BpeTokenizer {
    fn wrap(inner: HfTokenizer) -> Self {
        let end_of_text = inner
            .token_to_id(END_OF_TEXT_PIECE)
            .map(Token)
            .unwrap_or_else(|| {
                panic!("tokenizer has no `{END_OF_TEXT_PIECE}` token; the model needs one to know when to stop")
            });
        Self {
            inner: Arc::new(inner),
            end_of_text,
        }
    }

    /// Load a tokenizer from a `tokenizer.json`.
    // Exercised by the tests; unused in the binary until `train`/`generate-text` are switched
    // over from `SimpleTokenizer` (which invalidates the existing checkpoints under artifacts/).
    #[allow(dead_code)]
    pub fn from_file(path: &Path) -> Self {
        let inner = HfTokenizer::from_file(path)
            .unwrap_or_else(|e| panic!("failed to load tokenizer from {}: {e}", path.display()));
        Self::wrap(inner)
    }

    /// Save this tokenizer to a `tokenizer.json`.
    pub fn to_file(&self, path: &Path) {
        self.inner
            .save(path, true)
            .unwrap_or_else(|e| panic!("failed to write tokenizer to {}: {e}", path.display()));
    }

    /// Train a byte-level BPE tokenizer over `texts`.
    ///
    /// `min_frequency` drops merges seen fewer than that many times, which keeps one-off noise
    /// (mojibake, base64 blobs, boilerplate hashes — fineweb-edu has plenty) out of the merge
    /// table.
    ///
    /// Memory: the trainer holds the word-frequency table for the *entire* input in RAM before
    /// it starts merging, so peak usage scales with the number of distinct pre-tokens in
    /// `texts`, not with the merge count. Cap the input rather than the vocab if you run out.
    pub fn train<I>(texts: I, vocab_size: usize, min_frequency: u64) -> Self
    where
        I: Iterator<Item = String> + Send,
    {
        // `add_prefix_space: false` matches GPT-2: a leading space is part of the *following*
        // token ("the" and " the" are distinct), and nothing is injected at the start of input.
        let byte_level = ByteLevel::default().add_prefix_space(false);

        let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .show_progress(true)
            .special_tokens(vec![AddedToken::from(END_OF_TEXT_PIECE, true)])
            // Seed the vocabulary with all 256 byte-level characters. This is what makes the
            // tokenizer total: every byte has a token, so no input can ever produce an <UNK>,
            // no matter how little of it the training sample saw.
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .build()
            .into();

        let mut inner = HfTokenizer::new(BPE::default());
        inner.with_pre_tokenizer(Some(byte_level));
        inner.with_decoder(Some(byte_level));
        inner.with_post_processor(Some(byte_level));

        inner
            .train(&mut trainer, texts)
            .unwrap_or_else(|e| panic!("BPE training failed: {e}"));

        Self::wrap(inner)
    }

    /// Average bytes of source text encoded per token, over `texts`.
    ///
    /// The metric to compare tokenizers on: higher means the same context window holds more
    /// text, and the same corpus costs fewer training steps. Compare two candidates on a slice
    /// the tokenizer was *not* trained on.
    pub fn bytes_per_token<'a, I>(&self, texts: I) -> f64
    where
        I: Iterator<Item = &'a str>,
    {
        let (bytes, tokens) = texts.fold((0usize, 0usize), |(b, t), text| {
            (b + text.len(), t + self.encode(text).len())
        });
        if tokens == 0 {
            0.0
        } else {
            bytes as f64 / tokens as f64
        }
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<Token> {
        // `encode_fast` rather than `encode`: the only difference is that it skips computing a
        // byte offset for every token, and nothing here consumes offsets — we want token ids and
        // nothing else. Worth ~13% on fineweb-edu documents; the merge loop dominates the rest.
        self.inner
            .encode_fast(text, false)
            .unwrap_or_else(|e| panic!("failed to encode text: {e}"))
            .get_ids()
            .iter()
            .map(|id| Token(*id))
            .collect()
    }

    fn decode(&self, tokens: &[Token]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|t| t.0).collect();
        // `skip_special_tokens: false` — when inspecting generation we want to *see* that the
        // model emitted <|endoftext|>, not have it silently disappear.
        self.inner
            .decode(&ids, false)
            .unwrap_or_else(|e| panic!("failed to decode tokens: {e}"))
    }

    fn get_vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn end_of_text(&self) -> Token {
        self.end_of_text
    }

    fn token_to_piece(&self, token: Token) -> Option<String> {
        self.inner.id_to_token(token.0)
    }
}

// ----------- TESTS -----------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// A corpus with enough repetition for BPE to find real merges.
    fn corpus() -> Vec<String> {
        let sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Education is the most powerful weapon which you can use to change the world.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "The mitochondria is the powerhouse of the cell, as every student learns.",
        ];
        sentences
            .iter()
            .cycle()
            .take(400)
            .map(|s| s.to_string())
            .collect()
    }

    fn tiny_tokenizer() -> BpeTokenizer {
        BpeTokenizer::train(corpus().into_iter(), 500, 2)
    }

    #[test]
    fn round_trip_is_lossless() {
        let tokenizer = tiny_tokenizer();
        // Case, runs of whitespace, punctuation the old regex dropped, and non-ASCII — all of
        // which `SimpleTokenizer` silently destroys.
        for text in [
            "The quick brown fox jumps over the lazy dog.",
            "MiXeD CaSe  with   irregular\tspacing\nand newlines",
            "symbols the old regex dropped: - / = % $ [ ] { } # @ ~ ^ * + < > |",
            "non-ascii: café, naïve, 日本語, emoji 🚀, em—dash",
            "let x: Vec<u32> = (0..10).map(|i| i * 2).collect();",
        ] {
            assert_eq!(tokenizer.decode(&tokenizer.encode(text)), text);
        }
    }

    #[test]
    fn every_byte_is_representable() {
        // The byte-level initial alphabet means there is no such thing as an unknown token,
        // even for text sharing nothing with the training corpus.
        let tokenizer = tiny_tokenizer();
        let text = "ЖЖЖ ᚠᚡᚢ \u{0}\u{1}\u{7f} ⣿⣿";
        let tokens = tokenizer.encode(text);
        assert!(!tokens.is_empty());
        assert_eq!(tokenizer.decode(&tokens), text);
    }

    #[test]
    fn learns_merges_beyond_the_byte_alphabet() {
        let tokenizer = tiny_tokenizer();
        // 256 byte tokens + <|endoftext|> is the floor; anything above that is learned merges.
        assert!(tokenizer.get_vocab_size() > 257);
        // Frequent whole words should have collapsed into single tokens.
        assert_eq!(tokenizer.encode("The").len(), 1);
        assert_eq!(tokenizer.encode(" energy").len(), 1);
    }

    #[test]
    fn end_of_text_is_read_from_the_tokenizer() {
        let tokenizer = tiny_tokenizer();
        let eot = tokenizer.end_of_text();
        assert_eq!(
            tokenizer.token_to_piece(eot).as_deref(),
            Some(END_OF_TEXT_PIECE)
        );
        // Special tokens are added first, so the id happens to be 0 here — but nothing in the
        // codebase may assume that, which is why it goes through `end_of_text()`.
        assert_eq!(tokenizer.encode(END_OF_TEXT_PIECE), vec![eot]);
    }

    #[test]
    fn survives_a_save_load_round_trip() {
        let tokenizer = tiny_tokenizer();
        let path = std::env::temp_dir().join(format!(
            "llm-from-scratch-tokenizer-{}.json",
            std::process::id()
        ));
        tokenizer.to_file(&path);

        let loaded = BpeTokenizer::from_file(&path);
        let text = "The quick brown fox — café 🚀";
        assert_eq!(loaded.encode(text), tokenizer.encode(text));
        assert_eq!(loaded.end_of_text().0, tokenizer.end_of_text().0);
        assert_eq!(loaded.get_vocab_size(), tokenizer.get_vocab_size());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn compresses_better_than_one_byte_per_token() {
        let tokenizer = tiny_tokenizer();
        let held_out = ["The lazy dog learns about energy in the world."];
        assert!(tokenizer.bytes_per_token(held_out.into_iter()) > 2.0);
    }
}

// ----------- RUNTIME SELECTION -----------------------------------------------

/// Which tokenizer implementation to use.
///
/// `Default` is `Simple` on purpose, and it is *not* the CLI default (which is `Bpe`). The two
/// defaults answer different questions: serde's applies when reading a `config.json` written
/// before this field existed, and such a run was necessarily word-level — while a new run
/// should get the byte-level BPE tokenizer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerKind {
    /// Word-level tokenizer backed by `vocab.json`. Lossy — kept so old runs stay reproducible.
    #[default]
    Simple,
    /// Byte-level BPE tokenizer backed by `tokenizer.json`. Lossless.
    Bpe,
}

impl TokenizerKind {
    /// The artifact this tokenizer is loaded from, when no path is given explicitly.
    pub fn default_path(&self) -> &'static str {
        match self {
            Self::Simple => "vocab.json",
            Self::Bpe => "tokenizer.json",
        }
    }
}

impl std::fmt::Display for TokenizerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Bpe => write!(f, "bpe"),
        }
    }
}

/// A tokenizer chosen at runtime.
///
/// An enum rather than `Box<dyn Tokenizer>`: `TextBatcher` must be `Clone` (the dataloader
/// clones it per worker) and a boxed trait object is not, and `encode` sits in the batching hot
/// path where a vtable indirection per item buys nothing. Adding a third tokenizer means adding
/// a variant here and a line to each `match` — the compiler will point at every one.
#[derive(Debug, Clone)]
pub enum AnyTokenizer {
    Simple(SimpleTokenizer),
    Bpe(BpeTokenizer),
}

impl AnyTokenizer {
    /// Load the tokenizer of the given kind from `path`.
    pub fn load(kind: TokenizerKind, path: &Path) -> Self {
        match kind {
            TokenizerKind::Simple => Self::Simple(SimpleTokenizer::from_vocab_file(path)),
            TokenizerKind::Bpe => Self::Bpe(BpeTokenizer::from_file(path)),
        }
    }
}

impl Tokenizer for AnyTokenizer {
    fn encode(&self, text: &str) -> Vec<Token> {
        match self {
            Self::Simple(t) => t.encode(text),
            Self::Bpe(t) => t.encode(text),
        }
    }

    fn decode(&self, tokens: &[Token]) -> String {
        match self {
            Self::Simple(t) => t.decode(tokens),
            Self::Bpe(t) => t.decode(tokens),
        }
    }

    fn get_vocab_size(&self) -> usize {
        match self {
            Self::Simple(t) => t.get_vocab_size(),
            Self::Bpe(t) => t.get_vocab_size(),
        }
    }

    fn end_of_text(&self) -> Token {
        match self {
            Self::Simple(t) => t.end_of_text(),
            Self::Bpe(t) => t.end_of_text(),
        }
    }

    fn token_to_piece(&self, token: Token) -> Option<String> {
        match self {
            Self::Simple(t) => t.token_to_piece(token),
            Self::Bpe(t) => t.token_to_piece(token),
        }
    }
}

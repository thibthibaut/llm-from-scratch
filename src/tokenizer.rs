use crate::dataset::TextItem;
use burn::data::dataloader::Dataset;
use burn::data::dataset::SqliteDataset;
use indicatif::ProgressIterator;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::BufWriter, path::Path, sync::LazyLock};

/// Token type
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Token(pub u32);
impl Token {
    // Define some special tokens
    pub const UNKNOWN: Token = Token(0);
    pub const END_OF_TEXT: Token = Token(1);
}

/// Vocabulary with lookup and reverse lookup from text to tokens
#[derive(Serialize, Deserialize, Debug)]
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
}

#[derive(Debug)]
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
        // let num_samples = 1_000_00 as u64;
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
}

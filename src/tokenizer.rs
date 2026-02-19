use crate::dataset::load_gutenberg_dataset;
use burn::data::dataloader::Dataset;
use indicatif::ProgressIterator;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

/// Token type
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Token(pub u32);
impl Token {
    pub const UNKNOWN: Token = Token(0);
    pub const END_OF_TEXT: Token = Token(1);
}

/// Vocabulary
#[derive(Serialize, Deserialize, Debug)]
pub struct Vocab {
    pub words2tokens: HashMap<String, Token>,
    pub tokens2words: Vec<String>, // reverse lookup, indices are the tokens
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<Token>;
    fn decode(&self, tokens: &[Token]) -> String;
}

#[derive(Debug)]
pub struct SimpleTokenizer {
    token_regex: Regex,
    vocab: Vocab,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: Vocab {
                words2tokens: HashMap::new(),
                tokens2words: vec![],
            },
            token_regex: Regex::new(r#"[a-zA-Z]+|\d|[.,!?;:'\"()]"#).unwrap(),
        }
    }

    pub fn from_vocab_file(vocab_path: &Path) -> Self {
        let file = std::fs::File::open(vocab_path).expect("Failed to open vocab file");
        let reader = std::io::BufReader::new(file);
        let vocab = serde_json::from_reader(reader).expect("Failed to deserialize vocab");
        Self {
            vocab,
            token_regex: Regex::new(r#"[a-zA-Z]+|\d|[.,!?;:'\"()]"#).unwrap(),
        }
    }

    fn split_words(&self, text: &str) -> Vec<String> {
        self.token_regex
            .find_iter(&text.to_ascii_lowercase())
            .map(|match_| match_.as_str().to_owned())
            .collect()
    }

    /// Build vocabulary from a dataset
    pub fn build_vocab(&self) -> Vocab {
        let dataset = load_gutenberg_dataset();

        let num_samples = 1000 as u64; // dataset.len() as u64;
        let occurence_counter = dataset
            .iter()
            .take(num_samples as usize)
            .progress_count(num_samples)
            .flat_map(|sample| self.split_words(&sample.text))
            .fold(HashMap::new(), |mut counts, token| {
                *counts.entry(token).or_insert(0) += 1;
                counts
            });

        // Limit the size of the vocabulary by dropping low-frequency words
        let limit = 32_000;

        // Convert HashMap to Vec for sorting
        let mut sorted_entries: Vec<_> = occurence_counter.into_iter().collect();

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
        self.token_regex
            .find_iter(text)
            .map(|match_| {
                self.vocab
                    .words2tokens
                    .get(match_.as_str())
                    .copied()
                    .unwrap_or(Token(0))
            })
            .collect()
    }

    fn decode(&self, tokens: &[Token]) -> String {
        todo!()
    }
}

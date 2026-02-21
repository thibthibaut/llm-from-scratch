use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::dataset::load_fineweb_dataset;
use crate::dataset::load_gutenberg_dataset;
use crate::tokenizer::SimpleTokenizer;
use crate::tokenizer::Tokenizer;
use burn::data::dataloader::Dataset;
mod dataset;
mod tokenizer;

fn generate_tokenizer_vocab() {
    let dataset = load_fineweb_dataset();
    let vocab = SimpleTokenizer::build_vocab(&dataset);
    vocab.to_file(Path::new("vocab.json"));
}

fn main() {
    // generate_tokenizer_vocab();
    //

    let tok = SimpleTokenizer::from_vocab_file(Path::new("vocab.json"));

    let prompt = "Hello. This is a text to tokenize. What the fuck is this? Banana. 12. a1a";

    let tokens = tok.encode(prompt);

    let text = tok.decode(&tokens);
    println!("original {:?}", prompt);
    println!("tokens {:?}", tokens);
    println!("text {:?}", text);
}

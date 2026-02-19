use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::dataset::load_gutenberg_dataset;
use crate::tokenizer::SimpleTokenizer;
use crate::tokenizer::Tokenizer;
use burn::data::dataloader::Dataset;
mod dataset;
mod tokenizer;
fn main() {
    // let tokenizer = SimpleTokenizer::new();

    // println!("building vocab...");
    // let vocab = Path("vocab1.json");
    let tokenizer = SimpleTokenizer::from_vocab_file(Path::new("vocab1.json"));
    println!("tokenizer {:?}", tokenizer);
    // let file = File::create("vocab1.json").expect("Failed to create file");
    // let writer = BufWriter::new(file);
    // serde_json::to_writer_pretty(writer, &vocab).expect("Failed to write JSON");
}

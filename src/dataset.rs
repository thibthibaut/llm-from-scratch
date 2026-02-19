use burn::data::dataset::{HuggingfaceDatasetLoader, SqliteDataset};
use serde::{Deserialize, Serialize};

/// Data structure matching NNEngine/Gutenberg-Clean format
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GutenbergItem {
    /// Unique sample identifier (e.g., "twg_000000012345")
    pub id: String,
    /// Clean English text segment (30-60 words)
    pub text: String,
    /// Number of words in the segment
    pub word_count: u32,
    /// Data source identifier (typically "gutenberg")
    pub source: String,
}

/// Load the dataset
pub fn load_gutenberg_dataset() -> SqliteDataset<GutenbergItem> {
    HuggingfaceDatasetLoader::new("NNEngine/Gutenberg-Clean")
        .dataset("train") // There's only a train split in this dataset
        .expect("Failed to load the dataset")
}

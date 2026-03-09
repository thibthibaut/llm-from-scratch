use burn::data::dataset::{HuggingfaceDatasetLoader, SqliteDataset};
use serde::{Deserialize, Serialize};

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

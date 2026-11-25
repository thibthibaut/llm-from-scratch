pub mod dataset;
pub mod inference;
pub mod loader;
pub mod model;
pub mod ops;
pub mod sampling;
pub mod tokenizer;
pub mod training;

use burn::backend::libtorch::LibTorchDevice;

/// Held by every test that draws from libtorch's global random generator (model init, random
/// tensors). Tests run in parallel, and a test that seeds the generator and compares two
/// training runs needs nothing else drawing from it in between.
#[cfg(test)]
pub fn torch_rng_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Parse a `--device` CLI value into a `LibTorchDevice`.
pub fn parse_device(s: &str) -> Result<LibTorchDevice, String> {
    let lower = s.to_ascii_lowercase();
    match lower.as_str() {
        "cpu" => Ok(LibTorchDevice::Cpu),
        "mps" => Ok(LibTorchDevice::Mps),
        "vulkan" => Ok(LibTorchDevice::Vulkan),
        other => {
            if let Some(idx) = other.strip_prefix("cuda") {
                let idx = idx.trim_start_matches(':');
                let index: usize = if idx.is_empty() {
                    0
                } else {
                    idx.parse()
                        .map_err(|_| format!("invalid CUDA device index in '{s}'"))?
                };
                Ok(LibTorchDevice::Cuda(index))
            } else {
                Err(format!(
                    "unknown device '{s}': expected one of cpu, cuda[:N], mps, vulkan"
                ))
            }
        }
    }
}

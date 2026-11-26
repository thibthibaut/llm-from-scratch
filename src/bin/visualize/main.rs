mod app;
mod cache;
mod pca;
mod ui;
mod zoom;

use app::{App, Effect};
use clap::Parser;
use llm_from_scratch::inference::{load_trained_model, trace_tokens};
use llm_from_scratch::parse_device;
use pca::{PCA_COMPONENTS, Pca};
use rand::SeedableRng;
use rand::rngs::StdRng;
use ratatui::crossterm::event::{self, Event, KeyEventKind};

/// Step through a trained model's forward pass on a prompt you type.
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "artifacts/config.json")]
    config_path: String,
    #[arg(long, default_value = "artifacts/model")]
    model_path: String,
    #[arg(long, default_value = "cpu", value_parser = parse_device)]
    device: burn::backend::libtorch::LibTorchDevice,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let (config, model, tokenizer) =
        load_trained_model(&args.config_path, &args.model_path, &args.device);
    let table = model.token_embedding_weights();
    let [vocab, dim] = table.dims();
    let pca = Pca::fit(
        &table.into_data().to_vec::<f32>().unwrap(),
        vocab,
        dim,
        PCA_COMPONENTS,
    );
    let mut app = App::new(
        tokenizer,
        model.num_layers(),
        model.num_heads(),
        config.model.embedding_config.context_size,
        StdRng::from_entropy(),
    );

    let mut terminal = ratatui::init();
    let result = loop {
        match terminal.size() {
            Ok(size) => {
                let (main, zoom) = ui::window_for(size.into());
                app.set_window(main, zoom)
            }
            Err(e) => break Err(e),
        }
        if let Err(e) = terminal.draw(|f| ui::draw(f, &app, &pca)) {
            break Err(e);
        }
        let key = match event::read() {
            Ok(Event::Key(k)) if k.kind == KeyEventKind::Press => k,
            Ok(_) => continue,
            Err(e) => break Err(e),
        };
        match app.on_key(key) {
            Some(Effect::RunForward) => {
                app.set_trace(trace_tokens(&model, app.model_input(), &args.device))
            }
            Some(Effect::Quit) => break Ok(()),
            None => {}
        }
    };
    ratatui::restore();
    result
}

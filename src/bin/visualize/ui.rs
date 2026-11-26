use crate::app::{App, Stage, WINDOW, Zoom};
use crate::pca::Pca;
use crate::zoom;
use llm_from_scratch::sampling::Sampler;
use llm_from_scratch::tokenizer::{Token, Tokenizer};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

const TOP: usize = 10;
const LABEL: usize = 8;

/// GPT-2 `bytes_to_unicode` inverse: the byte a byte-level BPE char stands for.
fn bpe_byte(c: char) -> Option<u8> {
    let c = c as u32;
    if matches!(c, 33..=126 | 161..=172 | 174..=255) {
        return Some(c as u8);
    }
    let n = c.checked_sub(256)?;
    (0..=255u32)
        .filter(|b| !matches!(b, 33..=126 | 161..=172 | 174..=255))
        .nth(n as usize)
        .map(|b| b as u8)
}

pub fn display_piece(piece: &str) -> String {
    let decoded = piece
        .chars()
        .map(bpe_byte)
        .collect::<Option<Vec<u8>>>()
        .map_or_else(
            || piece.to_string(),
            |b| String::from_utf8_lossy(&b).into_owned(),
        );
    decoded
        .replace('Ġ', "␣")
        .replace('Ċ', "↵")
        .replace(' ', "␣")
        .replace('\n', "↵")
}

/// 0 → mid-grey, ±`row_max_abs` → white / near-black.
pub fn embedding_grey(v: f32, row_max_abs: f32) -> u8 {
    if row_max_abs == 0.0 {
        return 128;
    }
    (128.0 + 127.0 * v / row_max_abs).round().clamp(0.0, 255.0) as u8
}

/// Diverging grey for a pre-softmax score; masked cells are not passed in.
pub fn score_grey(s: f32, max_abs: f32) -> u8 {
    embedding_grey(s, max_abs)
}

pub fn attention_grey(w: f32) -> u8 {
    (255.0 * w).round().clamp(0.0, 255.0) as u8
}

pub(crate) fn grey(g: u8) -> Style {
    Style::default().bg(Color::Rgb(g, g, g))
}

pub(crate) fn piece<T: Tokenizer>(app: &App<T>, t: Token) -> String {
    display_piece(&app.tokenizer.token_to_piece(t).unwrap_or_default())
}

/// `s` cut or padded to exactly `w` chars.
pub(crate) fn fit(s: &str, w: usize) -> String {
    format!("{:w$}", s.chars().take(w).collect::<String>())
}

pub fn draw<T: Tokenizer>(frame: &mut Frame, app: &App<T>, pca: &Pca) {
    let [title, prompt, tokens, middle, output, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Min(WINDOW as u16 + 3),
        // header + top 10 + drawn row + borders
        Constraint::Length(14),
        Constraint::Length(2),
    ])
    .areas(frame.area());
    title_bar(frame, app, title);
    prompt_panel(frame, app, prompt);
    tokens_panel(frame, app, tokens);
    if app.zoom == Zoom::Off {
        let [emb, att] = Layout::horizontal([Constraint::Fill(1); 2]).areas(middle);
        embeddings_panel(frame, app, pca, emb);
        attention_panel(frame, app, att);
        output_panel(frame, app, output);
    } else {
        zoom_panel(frame, app, middle.union(output));
    }
    footer_bar(frame, app, footer);
}

/// The zoom takes the middle band and the output rows.
fn zoom_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    match app.zoom {
        Zoom::Head => zoom::head_zoom_panel(frame, app, area),
        _ => {
            panel(frame, area, " MLP zoom ".into(), Style::default());
        }
    }
}
/// Renders a titled block and returns its inner area.
pub(crate) fn panel(frame: &mut Frame, area: Rect, title: String, style: Style) -> Rect {
    let block = Block::bordered().title(title).border_style(style);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    inner
}

fn title_bar<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    let (badge, color) = if app.is_editing() {
        (" EDIT ", Color::Yellow)
    } else {
        (" INFERENCE ", Color::Cyan)
    };
    let line = Line::from(vec![
        Span::styled(badge, Style::default().fg(Color::Black).bg(color)),
        Span::raw(" GPT-2 forward pass"),
    ]);
    frame.render_widget(line, area);
}

fn prompt_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    let editing = app.is_editing();
    let style = if editing {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };
    let inner = panel(frame, area, " Prompt ".into(), style);
    let text = if editing {
        format!("{}▌", app.text)
    } else {
        app.text.clone()
    };
    // Show the end of the text: newlines as ↵, then the last line-width chars.
    let chars: Vec<char> = text.replace('\n', "↵").chars().collect();
    let tail: String = chars[chars.len().saturating_sub(inner.width as usize)..]
        .iter()
        .collect();
    frame.render_widget(Paragraph::new(tail), inner);
}

fn tokens_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    let inner = panel(frame, area, " Tokens ".into(), Style::default());
    // Pack chips into lines by hand, then keep the last ones so new tokens stay visible.
    let width = inner.width as usize;
    let mut lines: Vec<Vec<Span>> = vec![vec![]];
    let mut used = 0;
    for &t in &app.tokens {
        let chip = format!("[{} {}] ", piece(app, t), t.0);
        let len = chip.chars().count();
        if used > 0 && used + len > width {
            lines.push(vec![]);
            used = 0;
        }
        used += len;
        lines.last_mut().unwrap().push(Span::raw(chip));
    }
    let skip = lines.len().saturating_sub(inner.height as usize);
    let lines: Vec<Line> = lines.into_iter().skip(skip).map(Line::from).collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The last `rows` tokens of the model input, aligned with trace rows.
pub(crate) fn window_labels<T: Tokenizer>(app: &App<T>, rows: usize) -> Vec<String> {
    let input = app.model_input();
    let offset = input.len().saturating_sub(rows);
    (rows.saturating_sub(WINDOW)..rows)
        .map(|i| {
            input
                .get(offset + i)
                .map_or(String::new(), |&t| piece(app, t))
        })
        .collect()
}

fn embeddings_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, pca: &Pca, area: Rect) {
    let inner = panel(
        frame,
        area,
        format!(" Embeddings  layer {} ", app.layer),
        Style::default(),
    );
    let Some(trace) = &app.trace else { return };
    let rows = match app.layer {
        0 => &trace.embedding,
        n => match trace.blocks.get(n - 1) {
            Some(b) => &b.residual,
            None => return,
        },
    };
    let start = rows.len().saturating_sub(WINDOW);
    let labels = window_labels(app, rows.len());
    let lines: Vec<Line> = rows[start..]
        .iter()
        .zip(labels)
        .map(|(row, label)| {
            let proj = pca.project(row);
            let max = proj.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let cell = (inner.width as usize)
                .saturating_sub(LABEL)
                .checked_div(proj.len())
                .unwrap_or(1)
                .clamp(1, 2);
            let mut spans = vec![Span::raw(fit(&label, LABEL))];
            spans.extend(
                proj.iter()
                    .map(|&v| Span::styled(" ".repeat(cell), grey(embedding_grey(v, max)))),
            );
            Line::from(spans)
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

fn attention_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    let inner = panel(
        frame,
        area,
        format!(
            " Attention  layer {} · head {} · {} ",
            app.layer,
            app.head + 1,
            if app.show_scores { "scores" } else { "weights" }
        ),
        Style::default(),
    );
    let Some(trace) = &app.trace else { return };
    let Some(weights) = app
        .layer
        .checked_sub(1)
        .and_then(|l| trace.blocks.get(l))
        .and_then(|b| b.attention.get(app.head))
    else {
        return;
    };
    let start = weights.len().saturating_sub(WINDOW);
    let labels = window_labels(app, weights.len());
    let scores = app
        .show_scores
        .then(|| &trace.blocks[app.layer - 1].scores[app.head]);
    // scale by the visible window's largest finite |score|
    let max = scores.map_or(0.0, |sc| {
        sc[start..]
            .iter()
            .flat_map(|r| &r[start..])
            .filter(|v| v.is_finite())
            .fold(0.0f32, |m, v| m.max(v.abs()))
    });
    // Cells adapt to the panel: up to 3 shaded cols + 1 separator, down to 1 col.
    let n = weights.len().min(WINDOW);
    let step = (inner.width as usize)
        .saturating_sub(LABEL)
        .checked_div(n)
        .unwrap_or(1)
        .clamp(1, 4);
    let (cell, sep) = if step > 1 { (step - 1, " ") } else { (1, "") };
    let mut header = vec![Span::raw(" ".repeat(LABEL))];
    header.extend(labels.iter().map(|l| Span::raw(fit(l, cell) + sep)));
    let mut lines = vec![Line::from(header)];
    for (i, (row, label)) in weights[start..].iter().zip(&labels).enumerate() {
        let mut spans = vec![Span::raw(fit(label, LABEL))];
        for (j, &w) in row[start..].iter().enumerate() {
            let g = match scores {
                Some(_) if j > i => 0, // masked
                Some(sc) => score_grey(sc[start + i][start + j], max),
                None => attention_grey(w),
            };
            spans.push(Span::styled(" ".repeat(cell), grey(g)));
            spans.push(Span::raw(sep));
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

/// `█` bar of `p · width` cells followed by the value.
pub(crate) fn bar(p: f32, width: usize, style: Style) -> Vec<Span<'static>> {
    let n = (p * width as f32).round() as usize;
    vec![
        Span::styled("█".repeat(n), style),
        Span::raw(format!("{} {p:.2}", " ".repeat(width - n.min(width)))),
    ]
}

/// The last logits once the stage reaches `from`.
fn logits_from<T: Tokenizer>(app: &App<T>, from: Stage) -> Option<&[f32]> {
    let order = |s| match s {
        Stage::Output => 1,
        Stage::Sampler => 2,
        Stage::Drawn => 3,
        _ => 0,
    };
    (order(app.stage) >= order(from)).then_some(app.trace.as_ref()?.last_logits.as_slice())
}

fn output_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    let off = |o: Option<String>| o.unwrap_or_else(|| "off".into());
    let title = format!(
        " Next token · T={:.1} top-k {} top-p {} ",
        app.temperature,
        off(app.top_k.map(|k| k.to_string())),
        off(app.top_p.map(|p| format!("{p:.2}"))),
    );
    let inner = panel(frame, area, title, Style::default());
    let Some(logits) = logits_from(app, Stage::Output) else {
        return;
    };
    let model = Sampler::default().distribution(logits);
    let filtered = app.sampler().distribution(logits);
    let drawn = if app.stage == Stage::Drawn {
        app.drawn
    } else {
        None
    };
    let mut rows: Vec<(usize, f32)> = model.iter().take(TOP).copied().collect();
    if let Some(d) = drawn.filter(|d| !rows.iter().any(|r| r.0 == *d)) {
        rows.extend(model.iter().find(|r| r.0 == d));
    }
    // label, then two columns of (bar + " 0.00"), then the marker
    let width = ((inner.width as usize).saturating_sub(LABEL + 2 * 6 + 10) / 2).max(1);
    let mut lines = vec![Line::raw(format!(
        "{}{}{}",
        " ".repeat(LABEL),
        fit("model (T=1)", width + 6),
        "after T, k, p"
    ))];
    for (id, p) in rows {
        let mut spans = vec![Span::raw(fit(&piece(app, Token(id as u32)), LABEL))];
        spans.extend(bar(p, width, Style::default().fg(Color::DarkGray)));
        spans.push(Span::raw(" "));
        match filtered.iter().find(|r| r.0 == id) {
            Some(&(_, q)) => spans.extend(bar(q, width, Style::default().fg(Color::Green))),
            None => spans.push(Span::styled(
                fit("✗ cut", width + 5),
                Style::default().fg(Color::Red),
            )),
        }
        if drawn == Some(id) {
            spans.push(Span::styled(
                "  ◀ drawn",
                Style::default().add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn footer_bar<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    let text = if app.is_editing() {
        "type to edit · Enter run · Ctrl+C quit".to_string()
    } else {
        let stage = match app.stage {
            Stage::Block(n) => {
                format!(
                    "block {n}/{}",
                    app.trace.as_ref().map_or(0, |t| t.blocks.len())
                )
            }
            Stage::Output => "output".into(),
            Stage::Sampler => "sampler".into(),
            Stage::Drawn => "drawn".into(),
            Stage::Embedding | Stage::Editing => "embedding".into(),
        };
        let keys = match app.zoom {
            Zoom::Off => "Space sampler · ↑↓ layer · ←→ head · z head · m mlp · s scores",
            Zoom::Head => "↑↓ query · ←→ head · [ ] layer · m mlp · z/Esc close",
            Zoom::Mlp => "↑↓ token · [ ] layer · z head · m/Esc close",
        };
        let tail = if app.zoom == Zoom::Off {
            " · Esc edit"
        } else {
            ""
        };
        format!(
            "{stage} · Enter next · {keys} · t/T temp · k/K top-k · p/P top-p{tail} · Ctrl+C quit"
        )
    };
    let footer = Paragraph::new(text)
        .style(Style::default().fg(Color::DarkGray))
        .wrap(Wrap { trim: true });
    frame.render_widget(footer, area);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::app::test_support::{CharTokenizer, fake_trace};
    use crate::app::{App, Stage};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    pub(crate) fn pca() -> Pca {
        let data: Vec<f32> = (0..32).map(|i| ((i * 7) % 11) as f32).collect();
        Pca::fit(&data, 8, 4, 4)
    }
    pub(crate) fn press(a: &mut App<CharTokenizer>, code: KeyCode) {
        a.on_key(KeyEvent::new(code, KeyModifiers::NONE));
    }
    /// Typed `text`, ran, and got `fake_trace(text.len())`.
    pub(crate) fn running(text: &str) -> App<CharTokenizer> {
        let mut a = App::new(CharTokenizer, 2, 2, 256, StdRng::seed_from_u64(0));
        for c in text.chars() {
            press(&mut a, KeyCode::Char(c));
        }
        press(&mut a, KeyCode::Enter);
        a.set_trace(fake_trace(text.chars().count()));
        a
    }
    pub(crate) fn render(a: &App<CharTokenizer>, w: u16, h: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| draw(f, a, &pca())).unwrap();
        t.backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn display_piece_maps_bpe_markers() {
        assert_eq!(display_piece("Ġcat"), "␣cat");
        assert_eq!(display_piece("Ċ"), "↵");
    }

    #[test]
    fn display_piece_decodes_bytes() {
        assert_eq!(display_piece("Ã©"), "é");
        assert_eq!(display_piece("âĢĻ"), "’");
        assert_eq!(display_piece("ĠÃ©"), "␣é");
        assert!(!display_piece("Ã").is_empty());
    }

    /// Right half (the attention panel at 120 cols) of the row after its title.
    fn attention_header(text: &str) -> String {
        let rows: Vec<Vec<char>> = text
            .chars()
            .collect::<Vec<_>>()
            .chunks(120)
            .map(|r| r.to_vec())
            .collect();
        let t = rows
            .iter()
            .position(|r| r[60..].iter().collect::<String>().contains("Attention"))
            .unwrap();
        rows[t + 1][60..].iter().collect()
    }

    #[test]
    fn core_panels_visible_at_120x40() {
        let mut a = running("abcdefgh");
        while a.stage != Stage::Block(2) {
            press(&mut a, KeyCode::Enter);
        }
        let text = render(&a, 120, 40);
        assert!(text.contains("abcdefgh"));
        assert!(text.contains("Embeddings"));
        // CharTokenizer maps 'a'..'h' to pieces '!'..'(': first embedding row under the title
        let rows: Vec<String> = text
            .chars()
            .collect::<Vec<_>>()
            .chunks(120)
            .map(|r| r.iter().collect())
            .collect();
        let e = rows.iter().position(|r| r.contains("Embeddings")).unwrap();
        assert!(rows[e + 1].starts_with("│!"));
        assert!(attention_header(&text).contains('!'));
    }

    #[test]
    fn attention_last_column_visible_at_120_cols() {
        let mut a = running("abcdefgh");
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.stage, Stage::Block(1));
        assert!(attention_header(&render(&a, 120, 40)).contains('('));
    }

    #[test]
    fn greys() {
        assert_eq!(embedding_grey(0.0, 2.0), 128);
        assert_eq!(embedding_grey(2.0, 2.0), 255);
        assert_eq!(embedding_grey(-2.0, 2.0), 1);
        assert_eq!(embedding_grey(1.0, 0.0), 128);
        assert_eq!(attention_grey(0.0), 0);
        assert_eq!(attention_grey(1.0), 255);
    }

    #[test]
    fn renders_without_panic() {
        for (w, h) in [(40, 12), (80, 24), (200, 60)] {
            let mut a = App::new(CharTokenizer, 2, 2, 256, StdRng::seed_from_u64(0));
            render(&a, w, h);
            a = running("abc");
            loop {
                render(&a, w, h);
                let was_drawn = a.stage == Stage::Drawn;
                press(&mut a, KeyCode::Enter);
                if was_drawn {
                    // committed: `drawn` is taken, the trace is not yet replaced
                    assert!(a.drawn.is_none());
                    render(&a, w, h);
                    break;
                }
            }
        }
    }

    #[test]
    fn renders_one_token() {
        let mut a = running("a");
        press(&mut a, KeyCode::Enter);
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.stage, Stage::Block(2));
        render(&a, 80, 24);
    }

    #[test]
    fn mode_badge_and_footer() {
        let mut a = App::new(CharTokenizer, 2, 2, 256, StdRng::seed_from_u64(0));
        press(&mut a, KeyCode::Char('a'));
        let text = render(&a, 120, 30);
        assert!(text.contains("EDIT") && text.contains("type to edit"));
        press(&mut a, KeyCode::Enter);
        a.set_trace(fake_trace(1));
        let text = render(&a, 120, 30);
        assert!(text.contains("INFERENCE") && text.contains("Esc edit"));
    }

    #[test]
    fn footer_shows_exit_keys_at_80_cols() {
        let text = render(&running("ab"), 80, 24);
        assert!(text.contains("Esc edit") && text.contains("Ctrl+C quit"));
    }

    #[test]
    fn output_panels_visible_at_120x40() {
        let mut a = running("abc");
        while a.stage != Stage::Drawn {
            press(&mut a, KeyCode::Enter);
        }
        let text = render(&a, 120, 40);
        assert!(text.contains("◀ drawn"));
        // 9 near-zero tokens of the top 10: both columns of the merged panel
        assert!(text.matches("0.00").count() >= 18);
    }

    #[test]
    fn score_grey_diverges() {
        assert_eq!(score_grey(0.0, 2.0), 128);
        assert_eq!(score_grey(2.0, 2.0), 255);
        assert_eq!(score_grey(-2.0, 2.0), 1);
        assert_eq!(score_grey(1.0, 0.0), 128);
    }

    #[test]
    fn output_panel_merged() {
        let mut a = running("abc");
        while a.stage != Stage::Drawn {
            press(&mut a, KeyCode::Enter);
        }
        let text = render(&a, 120, 40);
        assert!(text.contains("Next token · T=1.0"));
        assert!(text.contains("◀ drawn"));
        assert!(!text.contains(" Sampler "));
    }

    #[test]
    fn output_columns_from_output_stage() {
        let mut a = running("abc");
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.stage, Stage::Block(1));
        assert!(!render(&a, 120, 40).contains('█'));
        while a.stage != Stage::Output {
            press(&mut a, KeyCode::Enter);
        }
        press(&mut a, KeyCode::Char('k'));
        press(&mut a, KeyCode::Char('k'));
        assert_eq!(a.top_k, Some(1));
        assert!(render(&a, 120, 40).contains("✗ cut"));
    }

    #[test]
    fn attention_title_shows_mode() {
        let mut a = running("abc");
        press(&mut a, KeyCode::Enter);
        assert!(render(&a, 120, 40).contains("weights"));
        press(&mut a, KeyCode::Char('s'));
        assert!(render(&a, 120, 40).contains("scores"));
    }

    #[test]
    fn footer_lists_zoom_keys() {
        let mut a = running("abc");
        press(&mut a, KeyCode::Enter);
        assert!(render(&a, 120, 40).contains("z head · m mlp · s scores"));
        press(&mut a, KeyCode::Char('z'));
        assert!(render(&a, 120, 40).contains("z/Esc close"));
        press(&mut a, KeyCode::Char('m'));
        assert!(render(&a, 120, 40).contains("m/Esc close"));
    }

    #[test]
    fn stage_gating() {
        let mut a = running("abc");
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.stage, Stage::Block(1));
        assert!(!render(&a, 120, 40).contains('█'));
    }

    #[test]
    fn cut_marker_shown() {
        let mut a = running("abc");
        press(&mut a, KeyCode::Char(' '));
        press(&mut a, KeyCode::Char('k'));
        press(&mut a, KeyCode::Char('k'));
        assert_eq!(a.top_k, Some(1));
        assert!(render(&a, 120, 40).contains('✗'));
    }

    pub(crate) fn zoomed(text: &str, zoom: Zoom) -> App<CharTokenizer> {
        let mut a = running(text);
        while a.stage != Stage::Block(2) {
            press(&mut a, KeyCode::Enter);
        }
        press(
            &mut a,
            KeyCode::Char(if zoom == Zoom::Head { 'z' } else { 'm' }),
        );
        assert_eq!(a.zoom, zoom);
        a
    }
    pub(crate) fn lines(text: &str, w: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chars.chunks(w).map(|r| r.iter().collect()).collect()
    }
}

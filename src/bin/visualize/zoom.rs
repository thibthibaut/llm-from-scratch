use crate::app::App;
use crate::cache::CacheTag;
use crate::pca::Pca;
use crate::ui::{bar, fit, grey, panel, piece, score_grey, window_labels};
use llm_from_scratch::tokenizer::Tokenizer;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

/// Kind marker + token label, the left column of every zoom row.
const ZOOM_LABEL: usize = 10;
/// Weight bar cells; with " 0.00" the bar column is `BAR + 5` wide.
const BAR: usize = 6;

/// One diverging-grey cell per element, scaled by `max_abs`.
fn cells(v: &[f32], max_abs: f32) -> Vec<Span<'static>> {
    v.iter()
        .map(|&x| Span::styled(" ", grey(score_grey(x, max_abs))))
        .collect()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

fn tag_span(tag: Option<CacheTag>) -> Span<'static> {
    match tag {
        Some(CacheTag::Read) => Span::styled("read", Style::default().fg(Color::DarkGray)),
        Some(CacheTag::Write) => Span::styled("write", Style::default().fg(Color::White)),
        Some(CacheTag::Recomputed) => Span::styled("recomputed", Style::default().fg(Color::Red)),
        None => Span::raw(""),
    }
}

/// `Σ weight × v` over all keys `0..=query`, from recorded values.
fn head_out(weights: &[f32], v: &[Vec<f32>], query: usize) -> Vec<f32> {
    let mut out = vec![0.0; v.get(query).map_or(0, |r| r.len())];
    for j in 0..=query {
        for (o, x) in out.iter_mut().zip(&v[j]) {
            *o += weights[j] * x;
        }
    }
    out
}

pub(crate) fn head_zoom_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, area: Rect) {
    let input = app.model_input();
    let (start, query) = (app.zoom_start(), app.zoom_pos);
    let label = |p: usize| input.get(p).map_or(String::new(), |&t| piece(app, t));
    let title = format!(
        " Head {}·{} · query \"{}\" ",
        app.layer,
        app.head + 1,
        label(query)
    );
    let inner = panel(frame, area, title, Style::default());
    let Some(trace) = &app.trace else { return };
    let Some(b) = app.layer.checked_sub(1).and_then(|l| trace.blocks.get(l)) else {
        return;
    };
    // after a commit the input has grown but the trace is still the old pass
    if b.q.get(app.head).is_none_or(|q| q.len() != input.len()) || query >= input.len() {
        return;
    }
    let (q, k, v) = (&b.q[app.head], &b.k[app.head], &b.v[app.head]);
    let (scores, weights) = (&b.scores[app.head][query], &b.attention[app.head][query]);
    let tags = app.cache.as_ref().and_then(|c| c.get(app.layer - 1));
    let tag = |p: usize| tags.and_then(|t| t.get(p).copied());
    let head = |kind: &str, p: usize| Span::raw(fit(&format!("{kind}  {}", label(p)), ZOOM_LABEL));
    let d = q[query].len();
    let col = d.max("q ⊙ k".chars().count()); // q ⊙ k column, at least its heading

    let mut lines = vec![];
    let mut row = vec![head("q", query)];
    row.extend(cells(&q[query], max_abs(&q[query])));
    lines.push(Line::from(row));
    lines.push(Line::raw(format!(
        "{}{}  {}{}",
        " ".repeat(ZOOM_LABEL),
        fit("q ⊙ k", col),
        fit("score", 7),
        "softmax"
    )));
    let qk: Vec<Vec<f32>> = (start..=query)
        .map(|j| q[query].iter().zip(&k[j]).map(|(a, b)| a * b).collect())
        .collect();
    let qk_max = qk.iter().fold(0.0f32, |m, r| m.max(max_abs(r)));
    // keys before the display window, summarised as one weight
    let earlier = || {
        let mut row = vec![Span::raw(fit(&format!("…{start} earlier"), ZOOM_LABEL))];
        row.push(Span::raw(" ".repeat(col + 2 + 7)));
        row.extend(bar(
            weights[..start].iter().sum(),
            BAR,
            Style::default().fg(Color::Cyan),
        ));
        row
    };
    if start > 0 {
        lines.push(Line::from(earlier()));
    }
    for (j, prod) in (start..=query).zip(&qk) {
        let mut row = vec![head("k", j)];
        row.extend(cells(prod, qk_max));
        row.push(Span::raw(" ".repeat(col - d)));
        row.push(Span::raw(format!("  {:<7}", format!("{:+.1}", scores[j]))));
        row.extend(bar(weights[j], BAR, Style::default().fg(Color::Cyan)));
        row.push(Span::raw("  "));
        row.push(tag_span(tag(j)));
        lines.push(Line::from(row));
    }
    // v cells start after the bar column + " × "; pad so tags line up with the key rows
    let v_pad = " ".repeat(col + 2 + 7 + BAR + 5 + 2 - (BAR + 5 + 3 + d));
    if start > 0 {
        let mut row = vec![Span::raw(fit(&format!("…{start} earlier"), ZOOM_LABEL))];
        row.extend(bar(
            weights[..start].iter().sum(),
            BAR,
            Style::default().fg(Color::Cyan),
        ));
        lines.push(Line::from(row));
    }
    for j in start..=query {
        let mut row = vec![head("v", j)];
        row.extend(bar(weights[j], BAR, Style::default().fg(Color::Cyan)));
        row.push(Span::raw(" × "));
        row.extend(cells(&v[j], max_abs(&v[j])));
        row.push(Span::raw(v_pad.clone()));
        row.push(tag_span(tag(j)));
        lines.push(Line::from(row));
    }
    let out = head_out(weights, v, query);
    let mut row = vec![Span::raw(fit("= out", ZOOM_LABEL + BAR + 5 + 3))];
    row.extend(cells(&out, max_abs(&out)));
    row.push(Span::raw("  head output (before W_O)"));
    lines.push(Line::from(row));

    let map = cache_map(app, app.cache.as_deref());
    // the map keeps at least its header + the selected layer's row
    let map_h = (map.len() as u16).min(inner.height.saturating_sub(lines.len() as u16).max(2));
    // the map follows the pipeline rows directly
    let top_h = (lines.len() as u16).min(inner.height.saturating_sub(map_h));
    let top = Rect {
        height: top_h,
        ..inner
    };
    let bottom = Rect {
        y: inner.y + top_h,
        height: map_h.min(inner.height),
        ..inner
    };
    frame.render_widget(Paragraph::new(lines), top);
    let rows = (map_h as usize).saturating_sub(1);
    frame.render_widget(Paragraph::new(map_window(map, rows, app.layer)), bottom);
}

/// Header + one row per reached layer: dim `Read`, bright `Write`, red `Recomputed`.
fn cache_map<T: Tokenizer>(app: &App<T>, cache: Option<&[Vec<CacheTag>]>) -> Vec<Line<'static>> {
    let Some(cache) = cache else { return vec![] };
    let n = app.model_input().len();
    let mut header = vec![Span::raw(fit("KV cache", ZOOM_LABEL))];
    header.extend(
        window_labels(app, n)
            .iter()
            .map(|l| Span::raw(fit(l, 4) + " ")),
    );
    let mut lines = vec![Line::from(header)];
    for (l, tags) in cache.iter().enumerate().take(app.reached_layer()) {
        let mut row = vec![Span::raw(fit(&format!("layer {:2}", l + 1), ZOOM_LABEL))];
        for &t in tags.get(app.zoom_start()..).unwrap_or(&[]) {
            let style = match t {
                CacheTag::Read => grey(60),
                CacheTag::Write => grey(230),
                CacheTag::Recomputed => Style::default().bg(Color::Red),
            };
            row.push(Span::styled("   ", style));
            row.push(Span::raw("  "));
        }
        if l + 1 == app.layer && tags.last() == Some(&CacheTag::Write) {
            row.push(Span::raw("← write"));
        }
        lines.push(Line::from(row));
    }
    lines
}

/// The header plus the `rows` layer rows nearest the selected (1-based) layer.
fn map_window(mut map: Vec<Line<'static>>, rows: usize, layer: usize) -> Vec<Line<'static>> {
    if map.is_empty() {
        return map;
    }
    let layers = map.split_off(1);
    let rows = rows.min(layers.len());
    let first = layer
        .saturating_sub(1)
        .saturating_sub(rows / 2)
        .min(layers.len() - rows);
    map.extend(layers.into_iter().skip(first).take(rows));
    map
}

/// Black for `a ≤ 0`, white at the token's `max`, linear between.
pub fn neuron_grey(a: f32, max: f32) -> u8 {
    if a <= 0.0 || max <= 0.0 {
        return 0;
    }
    (255.0 * a / max).round().clamp(0.0, 255.0) as u8
}

/// Neurons per grid row; each `▀` cell holds two grid rows (top = fg, bottom = bg).
const GRID_COLS: usize = 64;

pub(crate) fn mlp_zoom_panel<T: Tokenizer>(frame: &mut Frame, app: &App<T>, pca: &Pca, area: Rect) {
    let input = app.model_input();
    let pos = app.zoom_pos;
    let label = input.get(pos).map_or(String::new(), |&t| piece(app, t));
    let title = format!(" MLP · layer {} · token \"{label}\" ", app.layer);
    let inner = panel(frame, area, title, Style::default());
    let Some(b) = (app.trace.as_ref())
        .zip(app.layer.checked_sub(1))
        .and_then(|(t, l)| t.blocks.get(l))
    else {
        return;
    };
    // after a commit the input has grown but the trace is still the old pass
    if b.mlp_in.len() != input.len() || pos >= input.len() {
        return;
    }
    let (hidden, pad) = (&b.mlp_hidden[pos], " ".repeat(ZOOM_LABEL));
    let row = |kind: &str, v: &[f32], note: &str| {
        let mut r = vec![Span::raw(fit(kind, ZOOM_LABEL))];
        r.extend(cells(v, max_abs(v)));
        r.push(Span::raw(format!("  {note}")));
        Line::from(r)
    };
    let mut lines = vec![row(
        "in",
        &pca.project_direction(&b.mlp_in[pos]),
        "after LayerNorm",
    )];
    let max = hidden.iter().fold(0.0f32, |m, &x| m.max(x));
    let at = |i: usize| neuron_grey(hidden.get(i).copied().unwrap_or(0.0), max);
    for r in 0..hidden.len().div_ceil(2 * GRID_COLS) {
        let mut cells = vec![Span::raw(if r == 0 {
            fit("hidden", ZOOM_LABEL)
        } else {
            pad.clone()
        })];
        for c in 0..GRID_COLS {
            let (top, bottom) = (at(2 * r * GRID_COLS + c), at((2 * r + 1) * GRID_COLS + c));
            let style = Style::default()
                .fg(Color::Rgb(top, top, top))
                .bg(Color::Rgb(bottom, bottom, bottom));
            cells.push(Span::styled("▀", style));
        }
        lines.push(Line::from(cells));
    }
    lines.push(row(
        "out",
        &pca.project_direction(&b.mlp_out[pos]),
        "added to residual",
    ));
    frame.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::fake_trace;
    use crate::app::{Stage, Zoom};
    use crate::ui::tests::{lines, press, render, running, zoomed};
    use ratatui::crossterm::event::KeyCode;

    #[test]
    fn head_zoom_shows_pipeline() {
        let a = zoomed("abc", Zoom::Head);
        let text = render(&a, 120, 40);
        assert!(text.contains("head output (before W_O)"));
        assert!(text.contains("Head 2·1"));
        assert!(text.contains("softmax"));
        let b: Vec<char> = text.chars().collect();
        assert!(b.windows(2).any(|w| w[0] == '+' && w[1].is_ascii_digit()));
        assert!(text.contains("write"));
        // out = Σ_j attention[0][2][j] · v[0][j], block 2, rebuilt from fake_trace's closed form
        let blk = &a.trace.as_ref().unwrap().blocks[1];
        let out = head_out(&blk.attention[0][2], &blk.v[0], 2);
        assert_eq!(out.len(), 3);
        for (d, o) in out.iter().enumerate() {
            // v = 2 + (1000 + 10p + d)·0.01, weights 1/3 over p = 0..=2
            assert!((o - (12.1 + 0.01 * d as f32)).abs() < 1e-4, "{d}: {o}");
        }
    }

    #[test]
    fn head_zoom_reads_after_append() {
        let mut a = running("ab");
        while a.stage != Stage::Drawn {
            press(&mut a, KeyCode::Enter);
        }
        press(&mut a, KeyCode::Enter);
        a.set_trace(fake_trace(3));
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.stage, Stage::Block(1));
        press(&mut a, KeyCode::Char('z'));
        let text = render(&a, 120, 40);
        assert!(text.contains("read"));
        assert!(text.contains("write"));
        assert!(text.contains("← write"));
    }

    #[test]
    fn head_zoom_rows_follow_query() {
        let mut a = zoomed("abc", Zoom::Head);
        press(&mut a, KeyCode::Up);
        assert_eq!(a.zoom_pos, 1);
        let rows = lines(&render(&a, 120, 40), 120);
        let keys: Vec<&String> = rows.iter().filter(|r| r.starts_with("│k ")).collect();
        assert_eq!(keys.len(), 2);
        // CharTokenizer: 'a' → "!", 'b' → "\"", 'c' → "#"
        assert!(keys[0].starts_with("│k  !"));
        assert!(keys[1].starts_with("│k  \""));
        assert!(!keys.iter().any(|r| r.starts_with("│k  #")));
    }

    #[test]
    fn zooms_render_one_token() {
        let mut a = running("a");
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.stage, Stage::Block(1));
        press(&mut a, KeyCode::Char('z'));
        render(&a, 120, 40);
        render(&a, 40, 12);
        press(&mut a, KeyCode::Char('m'));
        render(&a, 120, 40);
        render(&a, 40, 12);
    }

    #[test]
    fn zooms_render_in_every_stage() {
        for (w, h) in [(40, 12), (80, 24), (120, 40), (200, 60)] {
            let mut a = running("abc");
            press(&mut a, KeyCode::Enter);
            press(&mut a, KeyCode::Char('z'));
            assert_eq!(a.zoom, Zoom::Head);
            loop {
                render(&a, w, h);
                let was_drawn = a.stage == Stage::Drawn;
                press(&mut a, KeyCode::Enter);
                if was_drawn {
                    render(&a, w, h); // committed, trace not yet replaced
                    break;
                }
            }
            a.set_trace(fake_trace(4));
            let text = render(&a, w, h);
            assert_eq!(a.stage, Stage::Embedding);
            assert!(text.contains("Head"));
            assert!(!text.contains("q ⊙ k"));
            press(&mut a, KeyCode::Enter);
            assert!(render(&a, w, h).contains("q ⊙ k") || h < 24);
        }
    }

    #[test]
    fn head_out_includes_keys_before_window() {
        let w = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25];
        let v: Vec<Vec<f32>> = (0..10).map(|p| vec![p as f32, 1.0]).collect();
        let out = head_out(&w, &v, 9);
        // the window-only sum would give 0.5 in dim 1
        assert_eq!(out, vec![4.25, 1.0]);
    }

    #[test]
    fn head_zoom_shows_earlier_keys_row() {
        let text = render(&zoomed("abcdefghij", Zoom::Head), 120, 40);
        assert!(text.contains("2 earlier"), "{text}");
        assert!(text.contains("0.20"), "{text}");
        let text = render(&zoomed("abc", Zoom::Head), 120, 40);
        assert!(!text.contains("earlier"));
    }

    #[test]
    fn neuron_grey_maps() {
        assert_eq!(neuron_grey(-0.1, 2.0), 0);
        assert_eq!(neuron_grey(2.0, 2.0), 255);
        assert_eq!(neuron_grey(1.0, 0.0), 0);
    }

    #[test]
    fn mlp_zoom_shows_in_hidden_out() {
        let text = render(&zoomed("abc", Zoom::Mlp), 120, 40);
        assert!(text.contains("after LayerNorm"));
        assert!(text.contains("added to residual"));
        assert!(text.contains("MLP · layer 2"));
        assert!(text.matches('▀').count() >= 24);
    }
}

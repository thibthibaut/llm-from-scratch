use llm_from_scratch::model::Trace;
use llm_from_scratch::sampling::Sampler;
use llm_from_scratch::tokenizer::{Token, Tokenizer};
use rand::rngs::StdRng;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub const MAX_MODEL_TOKENS: usize = 256;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Editing,
    Embedding,
    Block(usize), // 1-based
    Output,
    Sampler,
    Drawn,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    RunForward,
    Quit,
}

pub struct App<T: Tokenizer> {
    pub text: String,
    pub tokens: Vec<Token>,
    pub stage: Stage,
    pub layer: usize, // 0 = input embedding, n = after block n
    pub head: usize,  // 0-based
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub trace: Option<Trace>,
    pub drawn: Option<usize>,
    pub tokenizer: T,
    rng: StdRng,
    num_layers: usize,
    num_heads: usize,
    max_input: usize,
}

fn round_to(x: f32, scale: f32) -> f32 {
    (x * scale).round() / scale
}

impl<T: Tokenizer> App<T> {
    pub fn new(
        tokenizer: T,
        num_layers: usize,
        num_heads: usize,
        context_size: usize,
        rng: StdRng,
    ) -> Self {
        Self {
            text: String::new(),
            tokens: Vec::new(),
            stage: Stage::Editing,
            layer: 0,
            head: 0,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            trace: None,
            drawn: None,
            tokenizer,
            rng,
            num_layers,
            num_heads,
            max_input: MAX_MODEL_TOKENS.min(context_size),
        }
    }

    pub fn is_editing(&self) -> bool {
        self.stage == Stage::Editing
    }

    pub fn model_input(&self) -> &[Token] {
        &self.tokens[self.tokens.len().saturating_sub(self.max_input)..]
    }

    pub fn sampler(&self) -> Sampler {
        let mut s = Sampler::default().with_temperature(self.temperature);
        if let Some(k) = self.top_k {
            s = s.with_top_k(k);
        }
        if let Some(p) = self.top_p {
            s = s.with_top_p(p);
        }
        s
    }

    pub fn reached_layer(&self) -> usize {
        match self.stage {
            Stage::Editing | Stage::Embedding => 0,
            Stage::Block(n) => n,
            _ => self.num_layers,
        }
    }

    pub fn set_trace(&mut self, trace: Trace) {
        self.trace = Some(trace);
        self.drawn = None;
        self.go(Stage::Embedding);
    }

    fn go(&mut self, stage: Stage) {
        self.stage = stage;
        self.layer = self.reached_layer();
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Option<Effect> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Some(Effect::Quit);
        }
        if self.is_editing() {
            return self.edit_key(key.code);
        }
        match key.code {
            KeyCode::Enter => return self.advance(),
            KeyCode::Char(' ') if self.stage != Stage::Drawn => self.go(Stage::Sampler),
            KeyCode::Esc => {
                self.stage = Stage::Editing;
                self.trace = None;
                self.drawn = None;
            }
            KeyCode::Up => self.layer = (self.layer + 1).min(self.reached_layer()),
            KeyCode::Down => self.layer = self.layer.saturating_sub(1),
            KeyCode::Left => self.head = self.head.saturating_sub(1),
            KeyCode::Right => self.head = (self.head + 1).min(self.num_heads.saturating_sub(1)),
            KeyCode::Char(c) if matches!(self.stage, Stage::Output | Stage::Sampler) => {
                self.sampler_key(c)
            }
            _ => {}
        }
        None
    }

    fn edit_key(&mut self, code: KeyCode) -> Option<Effect> {
        match code {
            KeyCode::Char(c) => self.text.push(c),
            KeyCode::Backspace => {
                self.text.pop();
            }
            KeyCode::Enter => {
                return (!self.tokens.is_empty()).then_some(Effect::RunForward);
            }
            _ => return None,
        }
        self.tokens = self.tokenizer.encode(&self.text);
        None
    }

    fn advance(&mut self) -> Option<Effect> {
        match self.stage {
            Stage::Embedding if self.num_layers > 0 => self.go(Stage::Block(1)),
            Stage::Embedding => self.go(Stage::Output),
            Stage::Block(n) if n < self.num_layers => self.go(Stage::Block(n + 1)),
            Stage::Block(_) => self.go(Stage::Output),
            Stage::Output => self.go(Stage::Sampler),
            Stage::Sampler => {
                let logits = self.trace.as_ref().map(|t| t.last_logits.as_slice())?;
                self.drawn = Some(self.sampler().sample(logits, &mut self.rng));
                self.go(Stage::Drawn);
            }
            Stage::Drawn => {
                self.tokens.push(Token(self.drawn.take()? as u32));
                self.text = self.tokenizer.decode(&self.tokens);
                return Some(Effect::RunForward);
            }
            Stage::Editing => {}
        }
        None
    }

    fn sampler_key(&mut self, c: char) {
        match c {
            't' => self.temperature = round_to(self.temperature - 0.1, 10.0).max(0.1),
            'T' => self.temperature = round_to(self.temperature + 0.1, 10.0).min(3.0),
            'k' | 'K' => {
                // None, 1, 5, 10, 15, ...
                self.top_k = match (c, self.top_k) {
                    ('k', None | Some(1)) => None,
                    ('k', Some(5)) => Some(1),
                    ('k', Some(k)) => Some(k - 5),
                    (_, None) => Some(1),
                    (_, Some(1)) => Some(5),
                    (_, Some(k)) => Some(k + 5),
                };
            }
            'p' => {
                let p = round_to(self.top_p.unwrap_or(1.0) - 0.05, 100.0).max(0.05);
                self.top_p = Some(p);
            }
            'P' => {
                let p = round_to(self.top_p.unwrap_or(1.0) + 0.05, 100.0);
                self.top_p = (p < 1.0).then_some(p);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use llm_from_scratch::model::BlockTrace;

    pub(crate) struct CharTokenizer;

    impl Tokenizer for CharTokenizer {
        fn encode(&self, text: &str) -> Vec<Token> {
            text.chars().map(|c| Token(c as u32 % 64)).collect()
        }
        fn decode(&self, tokens: &[Token]) -> String {
            tokens
                .iter()
                .map(|t| char::from_u32(t.0).unwrap())
                .collect()
        }
        fn get_vocab_size(&self) -> usize {
            64
        }
        fn end_of_text(&self) -> Token {
            Token(0)
        }
        fn token_to_piece(&self, t: Token) -> Option<String> {
            char::from_u32(t.0).map(|c| c.to_string())
        }
    }

    /// 2 blocks, 2 heads, d = 4, 64 logits with logit 7 the largest.
    pub(crate) fn fake_trace(seq: usize) -> Trace {
        let block = BlockTrace {
            attention: vec![vec![vec![1.0 / seq as f32; seq]; seq]; 2],
            residual: vec![vec![0.5; 4]; seq],
        };
        let mut last_logits = vec![0.0; 64];
        last_logits[7] = 10.0;
        Trace {
            embedding: vec![vec![0.1; 4]; seq],
            blocks: vec![block.clone(), block],
            last_logits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use rand::SeedableRng;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }
    fn app() -> App<CharTokenizer> {
        App::new(CharTokenizer, 2, 2, 256, StdRng::seed_from_u64(0))
    }
    fn type_str(a: &mut App<CharTokenizer>, s: &str) {
        for c in s.chars() {
            a.on_key(key(KeyCode::Char(c)));
        }
    }
    fn enter(a: &mut App<CharTokenizer>) -> Option<Effect> {
        a.on_key(key(KeyCode::Enter))
    }
    /// Editing "ab" then running to the Embedding stage.
    fn at_embedding() -> App<CharTokenizer> {
        let mut a = app();
        type_str(&mut a, "ab");
        assert_eq!(enter(&mut a), Some(Effect::RunForward));
        a.set_trace(fake_trace(2));
        a
    }
    fn at_sampler() -> App<CharTokenizer> {
        let mut a = at_embedding();
        a.on_key(key(KeyCode::Char(' ')));
        a
    }

    #[test]
    fn typing_updates_tokens() {
        let mut a = app();
        type_str(&mut a, "ab");
        assert_eq!(a.text, "ab");
        assert_eq!(a.tokens.len(), 2);
        assert_eq!(a.stage, Stage::Editing);
    }

    #[test]
    fn backspace_removes_a_char() {
        let mut a = app();
        type_str(&mut a, "aé");
        a.on_key(key(KeyCode::Backspace));
        assert_eq!(a.text, "a");
    }

    #[test]
    fn enter_on_empty_prompt_does_nothing() {
        let mut a = app();
        assert_eq!(enter(&mut a), None);
        assert_eq!(a.stage, Stage::Editing);
    }

    #[test]
    fn full_walk() {
        let mut a = at_embedding();
        assert_eq!((a.stage, a.layer), (Stage::Embedding, 0));
        assert_eq!(enter(&mut a), None);
        assert_eq!((a.stage, a.layer), (Stage::Block(1), 1));
        enter(&mut a);
        assert_eq!((a.stage, a.layer), (Stage::Block(2), 2));
        enter(&mut a);
        assert_eq!(a.stage, Stage::Output);
        enter(&mut a);
        assert_eq!(a.stage, Stage::Sampler);
        enter(&mut a);
        assert_eq!(a.stage, Stage::Drawn);
        assert!(a.drawn.is_some());
        assert_eq!(enter(&mut a), Some(Effect::RunForward));
        assert_eq!(a.tokens.len(), 3);
    }

    #[test]
    fn letters_do_not_edit_in_inference() {
        let mut a = at_embedding();
        a.on_key(key(KeyCode::Char('x')));
        assert_eq!(a.text, "ab");
        a.on_key(key(KeyCode::Char(' ')));
        assert_eq!(a.stage, Stage::Sampler);
        assert_eq!(a.text, "ab");
    }

    #[test]
    fn esc_returns_to_editing_keeping_text() {
        let mut a = at_embedding();
        a.on_key(key(KeyCode::Esc));
        assert!(a.is_editing());
        assert!(a.trace.is_none());
        assert_eq!(a.text, "ab");
        assert!(a.drawn.is_none());
    }

    #[test]
    fn layer_clamped_to_reached() {
        let mut a = at_embedding();
        enter(&mut a);
        for _ in 0..2 {
            a.on_key(key(KeyCode::Up));
        }
        assert_eq!(a.layer, 1);
        for _ in 0..2 {
            a.on_key(key(KeyCode::Down));
        }
        assert_eq!(a.layer, 0);
    }

    #[test]
    fn head_clamped() {
        let mut a = at_embedding();
        for _ in 0..5 {
            a.on_key(key(KeyCode::Right));
        }
        assert_eq!(a.head, 1);
        for _ in 0..5 {
            a.on_key(key(KeyCode::Left));
        }
        assert_eq!(a.head, 0);
    }

    #[test]
    fn sampler_keys_clamp() {
        let mut a = at_sampler();
        let press = |a: &mut App<CharTokenizer>, c, n| {
            for _ in 0..n {
                a.on_key(key(KeyCode::Char(c)));
            }
        };
        press(&mut a, 't', 20);
        assert_eq!(a.temperature, 0.1);
        press(&mut a, 'T', 40);
        assert_eq!(a.temperature, 3.0);
        for (c, want) in [
            ('K', Some(1)),
            ('K', Some(5)),
            ('k', Some(1)),
            ('k', None),
            ('k', None),
        ] {
            press(&mut a, c, 1);
            assert_eq!(a.top_k, want);
        }
        press(&mut a, 'p', 1);
        assert_eq!(a.top_p, Some(0.95));
        press(&mut a, 'P', 1);
        assert_eq!(a.top_p, None);
    }

    #[test]
    fn top_k_steps_on_grid() {
        let mut a = at_sampler();
        let mut up = vec![];
        for _ in 0..20 {
            a.on_key(key(KeyCode::Char('K')));
            up.push(a.top_k);
        }
        let grid: Vec<Option<usize>> = [1, 5]
            .into_iter()
            .chain((10..=95).step_by(5))
            .map(Some)
            .collect();
        assert_eq!(up, grid);
        for want in grid.iter().rev().skip(1).chain([&None]) {
            a.on_key(key(KeyCode::Char('k')));
            assert_eq!(&a.top_k, want);
        }
    }

    #[test]
    fn space_in_drawn_is_noop() {
        let mut a = at_sampler();
        enter(&mut a);
        assert_eq!(a.stage, Stage::Drawn);
        let drawn = a.drawn;
        a.on_key(key(KeyCode::Char(' ')));
        assert_eq!((a.stage, a.drawn), (Stage::Drawn, drawn));
        assert_eq!(enter(&mut a), Some(Effect::RunForward));
        assert_eq!(a.tokens.len(), 3);
    }

    #[test]
    fn sampler_keys_inactive_in_embedding() {
        let mut a = at_embedding();
        a.on_key(key(KeyCode::Char('t')));
        assert_eq!(a.temperature, 1.0);
    }

    #[test]
    fn greedy_draw_with_top_k_1() {
        let mut a = at_sampler();
        a.on_key(key(KeyCode::Char('K')));
        assert_eq!(a.top_k, Some(1));
        enter(&mut a);
        assert_eq!(a.drawn, Some(7));
    }

    #[test]
    fn model_input_caps_at_256() {
        let mut a = app();
        type_str(&mut a, &"a".repeat(300));
        assert_eq!(a.model_input().len(), 256);
        assert_eq!(a.model_input(), &a.tokens[44..]);
    }

    #[test]
    fn model_input_caps_at_context_size() {
        let mut a = App::new(CharTokenizer, 2, 2, 16, StdRng::seed_from_u64(0));
        type_str(&mut a, &"abcdefghij".repeat(4));
        assert_eq!(a.model_input().len(), 16);
        assert_eq!(a.model_input(), &a.tokens[24..]);
    }

    #[test]
    fn ctrl_c_quits_in_both_modes() {
        let mut a = app();
        assert_eq!(a.on_key(ctrl('c')), Some(Effect::Quit));
        let mut a = at_sampler();
        assert_eq!(a.on_key(ctrl('c')), Some(Effect::Quit));
    }
}

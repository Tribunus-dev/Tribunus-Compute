//! Grammar-guided generation (GBNF) support.
//!
//! Provides:
//! - GBNF grammar parser (llama.cpp-compatible)
//! - NFA construction from AST (Thompson construction)
//! - DFA compilation for fast token masking
//! - Token validity checking against the grammar
//!
//! GBNF format example:
//! ```text
//! root ::= "{" ws "name" ws ":" ws string ws "," ws "age" ws ":" ws number ws "}"
//! string ::= "\"" ([^"]*) "\""
//! number ::= [0-9]+
//! ws ::= [ \t\n]*
//! ```
//!
//! Rules:
//! ```text
//! name ::= expression
//! "literal"          - string literal
//! [chars]            - character class
//! [^chars]           - negated character class
//! .                  - any single character
//! (expression)       - grouping
//! A B                - sequence
//! A | B              - alternation
//! A*                 - zero or more
//! A+                 - one or more
//! A?                 - optional
//! rule_name          - reference to another rule
//! \xNN               - hex byte literal
//! # comment          - line comment
//! ```

use std::collections::{BTreeSet, HashMap, VecDeque};

// ── Grammar AST Nodes ─────────────────────────────────────────────────────

/// A node in the GBNF grammar AST.
#[derive(Debug, Clone)]
pub enum GrammarNode {
    /// Literal string match: `"hello"`
    Lit(String),
    /// Character class: `[a-z]`, `[^0-9]`
    CharClass {
        /// Character ranges (start, end) inclusive.
        chars: Vec<(char, char)>,
        /// True if the class is negated (`[^...]`).
        negated: bool,
    },
    /// Any single character (`.`). Equivalent to `[^\n]` in most GBNF
    /// implementations.
    Any,
    /// Sequence: `A B C`
    Seq(Vec<GrammarNode>),
    /// Alternation: `A | B | C`
    Alt(Vec<GrammarNode>),
    /// Repetition: `A*`
    Star(Box<GrammarNode>),
    /// One or more: `A+`
    Plus(Box<GrammarNode>),
    /// Optional: `A?`
    Opt(Box<GrammarNode>),
    /// Reference to another rule: `rule_name`
    Ref(String),
    /// Hex byte literal `\xNN`
    HexByte(u8),
}

/// A complete GBNF grammar definition.
///
/// Contains a set of named rules plus the root rule name.
/// The root rule is the entry point for generation.
#[derive(Debug, Clone)]
pub struct Grammar {
    /// Named grammar rules. The key is the rule name (without `::=`).
    pub rules: HashMap<String, GrammarNode>,
    /// Name of the root rule (entry point).
    pub root: String,
}

// ── GBNF Parser ───────────────────────────────────────────────────────────

/// Parse GBNF text into a [`Grammar`].
///
/// `text` is the complete GBNF grammar definition, with one or more rules
/// separated by newlines. Comments (`# ...`) and blank lines are ignored.
pub fn parse_gbnf(text: &str) -> Result<Grammar, String> {
    let mut rules: HashMap<String, GrammarNode> = HashMap::new();
    let mut root: Option<String> = None;

    for line in text.lines() {
        let line = line.trim();
        // Skip blank lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Look for `name ::= expr`
        if let Some(pos) = line.find("::=") {
            let name = line[..pos].trim().to_string();
            let body = line[pos + 3..].trim();

            if name.is_empty() {
                return Err("empty rule name".to_string());
            }
            if body.is_empty() {
                return Err(format!("empty body in rule '{}'", name));
            }

            let node = parse_expression(body, &rules, &name)?;

            if root.is_none() {
                root = Some(name.clone());
            }
            rules.insert(name, node);
        } else {
            // Lines without `::=` that aren't comments — could be continuation
            // of a multi-line rule. For now, treat as error.
            // (Simplification: GBNF traditionally allows multi-line, but we
            //  require inline rules for simplicity.)
            return Err(format!("expected '::=' in grammar line: '{}'", line));
        }
    }

    let root = root.ok_or_else(|| "no rules defined".to_string())?;
    if !rules.contains_key(&root) {
        return Err(format!(
            "root rule '{}' is defined but no rule body was found",
            root
        ));
    }

    Ok(Grammar { rules, root })
}

/// Parse a single GBNF expression string into a [`GrammarNode`].
///
/// This is a recursive descent parser for GBNF expressions.
fn parse_expression(
    s: &str,
    rules: &HashMap<String, GrammarNode>,
    current_rule: &str,
) -> Result<GrammarNode, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty expression".to_string());
    }

    // Top-level: split on `|` outside parentheses and brackets
    let alt_parts = split_alternation(s)?;
    if alt_parts.len() > 1 {
        let mut alts = Vec::new();
        for part in alt_parts {
            alts.push(parse_sequence(part, rules, current_rule)?);
        }
        return Ok(GrammarNode::Alt(alts));
    }

    parse_sequence(s, rules, current_rule)
}

/// Split an expression on `|` that are not inside `(...)`, `[...]`, or `"..."`.
fn split_alternation(s: &str) -> Result<Vec<&str>, String> {
    let mut parts = Vec::new();
    let mut depth_paren = 0i32;
    let mut depth_bracket = 0i32;
    let mut in_string = false;
    let mut in_hex = false;
    let mut start = 0usize;

    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if in_string {
            if chars[i] == '"' {
                // Check for escaped quote
                if i > 0 && chars[i - 1] == '\\' {
                    // escaped quote inside string, continue
                } else {
                    in_string = false;
                }
            }
            i += 1;
            continue;
        }
        if in_hex {
            // \xNN is exactly 2 hex digits
            let hex_start = i;
            while i < chars.len() && i < hex_start + 2 {
                i += 1;
            }
            in_hex = false;
            continue;
        }

        match chars[i] {
            '"' => {
                in_string = true;
                i += 1;
            }
            '(' => {
                depth_paren += 1;
                i += 1;
            }
            ')' => {
                depth_paren -= 1;
                if depth_paren < 0 {
                    return Err("unbalanced ')'".to_string());
                }
                i += 1;
            }
            '[' => {
                depth_bracket += 1;
                i += 1;
            }
            ']' => {
                depth_bracket -= 1;
                if depth_bracket < 0 {
                    return Err("unbalanced ']'".to_string());
                }
                i += 1;
            }
            '\\' if i + 1 < chars.len() && chars[i + 1] == 'x' => {
                in_hex = true;
                i += 2; // skip \x
            }
            '|' if depth_paren == 0 && depth_bracket == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    if depth_paren != 0 {
        return Err("unbalanced '('".to_string());
    }
    if depth_bracket != 0 {
        return Err("unbalanced '['".to_string());
    }
    if in_string {
        return Err("unclosed string literal".to_string());
    }

    parts.push(&s[start..]);
    Ok(parts)
}

/// Parse a sequence of terms (no alternation at this level).
fn parse_sequence(
    s: &str,
    rules: &HashMap<String, GrammarNode>,
    current_rule: &str,
) -> Result<GrammarNode, String> {
    let terms = split_sequence_terms(s)?;
    if terms.is_empty() {
        return Err("empty sequence".to_string());
    }
    if terms.len() == 1 {
        return parse_term(terms[0], rules, current_rule);
    }

    let mut seq = Vec::new();
    for term in terms {
        seq.push(parse_term(term, rules, current_rule)?);
    }
    Ok(GrammarNode::Seq(seq))
}

/// Split a sequence into individual terms (handling suffixes *, +, ?).
fn split_sequence_terms(s: &str) -> Result<Vec<&str>, String> {
    let mut terms = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Skip whitespace between terms
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        let term_start = i;

        // Determine the extent of the term
        match chars[i] {
            '"' => {
                // String literal: consume until closing "
                i += 1;
                while i < len {
                    if chars[i] == '\\' && i + 1 < len {
                        i += 2; // skip escaped char
                        continue;
                    }
                    if chars[i] == '"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            '[' => {
                // Character class: consume until closing ]
                let mut depth = 1;
                i += 1;
                while i < len && depth > 0 {
                    if chars[i] == '[' {
                        depth += 1;
                    } else if chars[i] == ']' {
                        depth -= 1;
                    } else if chars[i] == '\\' && i + 1 < len {
                        i += 1; // skip next char in escape
                    }
                    i += 1;
                }
            }
            '(' => {
                // Group: consume balanced ()
                let mut depth = 1;
                i += 1;
                while i < len && depth > 0 {
                    if chars[i] == '(' {
                        depth += 1;
                    } else if chars[i] == ')' {
                        depth -= 1;
                    } else if chars[i] == '"' {
                        // Jump over string inside group
                        i += 1;
                        while i < len {
                            if chars[i] == '\\' && i + 1 < len {
                                i += 2;
                                continue;
                            }
                            if chars[i] == '"' {
                                break;
                            }
                            i += 1;
                        }
                    }
                    i += 1;
                }
            }
            '\\' if i + 1 < len && chars[i + 1] == 'x' => {
                // Hex byte literal \xNN
                i += 4; // \xNN
            }
            '.' => {
                // Any single char
                i += 1;
            }
            _ => {
                // Identifier or single character
                while i < len
                    && !chars[i].is_whitespace()
                    && !matches!(chars[i], '|' | '*' | '+' | '?' | ')' | ']')
                {
                    i += 1;
                }
            }
        }

        // Consume suffix operators *, +, ?
        while i < len && matches!(chars[i], '*' | '+' | '?') {
            i += 1;
        }

        let term_str = &s[term_start..i];
        let trimmed = term_str.trim();
        if !trimmed.is_empty() {
            terms.push(trimmed);
        }
    }

    // Check for standalone suffix operators (they should be attached to a term)
    for t in &terms {
        if *t == "*" || *t == "+" || *t == "?" {
            return Err(format!("suffix operator '{}' without preceding term", t));
        }
    }

    Ok(terms)
}

/// Parse a single term (without alternation or sequence) into a GrammarNode.
fn parse_term(
    s: &str,
    rules: &HashMap<String, GrammarNode>,
    current_rule: &str,
) -> Result<GrammarNode, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty term".to_string());
    }

    // Check for suffix operators
    let (base, suffix) = if s.ends_with('*') {
        (&s[..s.len() - 1], Some('*'))
    } else if s.ends_with('+') {
        (&s[..s.len() - 1], Some('+'))
    } else if s.ends_with('?') {
        (&s[..s.len() - 1], Some('?'))
    } else {
        (s, None)
    };

    let base = base.trim();
    let node = parse_atom(base, rules, current_rule)?;

    match suffix {
        Some('*') => Ok(GrammarNode::Star(Box::new(node))),
        Some('+') => Ok(GrammarNode::Plus(Box::new(node))),
        Some('?') => Ok(GrammarNode::Opt(Box::new(node))),
        _ => Ok(node),
    }
}

/// Parse an atomic expression (no suffix operators).
fn parse_atom(
    s: &str,
    rules: &HashMap<String, GrammarNode>,
    _current_rule: &str,
) -> Result<GrammarNode, String> {
    let s = s.trim();

    if s.is_empty() {
        return Err("empty atom".to_string());
    }

    let chars: Vec<char> = s.chars().collect();

    // String literal
    if chars[0] == '"' && chars[chars.len() - 1] == '"' && chars.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        let unescaped = unescape_string(inner)?;
        return Ok(GrammarNode::Lit(unescaped));
    }

    // Character class [...]
    if chars[0] == '[' && chars[chars.len() - 1] == ']' {
        return parse_char_class(&s[1..s.len() - 1]);
    }

    // Any character
    if s == "." {
        return Ok(GrammarNode::Any);
    }

    // Hex byte literal \xNN
    if s.starts_with("\\x") && s.len() == 4 {
        let hex_str = &s[2..];
        let byte = u8::from_str_radix(hex_str, 16)
            .map_err(|e| format!("invalid hex byte '{}': {}", s, e))?;
        return Ok(GrammarNode::HexByte(byte));
    }

    // Group (...)
    if chars[0] == '(' && chars[chars.len() - 1] == ')' {
        let inner = &s[1..s.len() - 1];
        return parse_expression(inner, rules, _current_rule);
    }

    // Reference to another rule: identifier
    if is_identifier(s) {
        if !rules.contains_key(s) {
            // Forward reference - allow it; will be resolved at compile time
            // or by the caller. Since our parser processes rules in order,
            // forward refs to rules not yet seen may be resolved later.
        }
        return Ok(GrammarNode::Ref(s.to_string()));
    }

    Err(format!("unrecognized grammar term '{}'", s))
}

/// Parse a character class body (without `[` and `]`).
fn parse_char_class(body: &str) -> Result<GrammarNode, String> {
    let body = body.trim();
    let negated = body.starts_with('^');
    let content = if negated { &body[1..] } else { body };

    let mut ranges: Vec<(char, char)> = Vec::new();
    let chars: Vec<char> = content.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() && chars[i + 1] == '-' {
            // Range: a-z
            let start = chars[i];
            let end = chars[i + 2];
            if start > end {
                return Err(format!("invalid char range '{}-{}'", start, end));
            }
            ranges.push((start, end));
            i += 3;
        } else {
            // Single character
            let ch = if chars[i] == '\\' && i + 1 < chars.len() {
                i += 1;
                match chars[i] {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '"' => '"',
                    c => c,
                }
            } else {
                chars[i]
            };
            ranges.push((ch, ch));
            i += 1;
        }
    }

    if ranges.is_empty() {
        return Err("empty character class".to_string());
    }

    Ok(GrammarNode::CharClass {
        chars: ranges,
        negated,
    })
}

fn unescape_string(s: &str) -> Result<String, String> {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            i += 1;
            match chars[i] {
                'n' => result.push('\n'),
                't' => result.push('\t'),
                'r' => result.push('\r'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                'x' => {
                    // Hex byte: \xNN
                    if i + 2 < chars.len() {
                        let hex = &s[i + 1..i + 3];
                        let byte = u8::from_str_radix(hex, 16)
                            .map_err(|e| format!("invalid hex escape '\\x{}': {}", hex, e))?;
                        result.push(byte as char);
                        i += 2;
                    } else {
                        return Err("incomplete hex escape".to_string());
                    }
                }
                c => {
                    result.push(c);
                }
            }
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }
    Ok(result)
}

fn is_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    for c in chars {
        if !c.is_ascii_alphanumeric() && c != '_' && c != '-' {
            return false;
        }
    }
    true
}

// ── Internal NFA ───────────────────────────────────────────────────────────

/// Internal NFA state (Thompson construction).
#[derive(Debug, Clone)]
struct NFAState {
    /// Transitions: (character or None for epsilon, target state index).
    transitions: Vec<(Option<char>, usize)>,
    /// Is this an accept (terminal) state?
    is_accept: bool,
}

/// A Thompson NFA fragment.
#[derive(Debug, Clone)]
struct NFAGraph {
    states: Vec<NFAState>,
    start: usize,
    accept: usize,
}

impl NFAGraph {
    fn new() -> Self {
        let start = 0;
        let accept = 1;
        NFAGraph {
            states: vec![
                NFAState {
                    transitions: vec![],
                    is_accept: false,
                },
                NFAState {
                    transitions: vec![],
                    is_accept: true,
                },
            ],
            start,
            accept,
        }
    }

    fn add_state(&mut self, is_accept: bool) -> usize {
        let id = self.states.len();
        self.states.push(NFAState {
            transitions: vec![],
            is_accept,
        });
        id
    }

    fn add_transition(&mut self, from: usize, ch: Option<char>, to: usize) {
        self.states[from].transitions.push((ch, to));
    }

    fn add_epsilon(&mut self, from: usize, to: usize) {
        self.add_transition(from, None, to);
    }

    fn add_char(&mut self, from: usize, ch: char, to: usize) {
        self.add_transition(from, Some(ch), to);
    }
}

// ── AST to NFA Compilation ─────────────────────────────────────────────────

/// Compile a grammar node into an NFA fragment.
///
/// Uses Thompson construction: each node produces an NFA fragment with
/// exactly one start and one accept state.
fn compile_node(
    node: &GrammarNode,
    rules: &HashMap<String, GrammarNode>,
    visited: &mut Vec<String>,
) -> Result<NFAGraph, String> {
    match node {
        GrammarNode::Lit(s) => compile_lit(s),
        GrammarNode::CharClass { chars, negated } => compile_char_class(chars, *negated),
        GrammarNode::Any => compile_any(),
        GrammarNode::Seq(nodes) => compile_seq(nodes, rules, visited),
        GrammarNode::Alt(nodes) => compile_alt(nodes, rules, visited),
        GrammarNode::Star(inner) => compile_star(inner, rules, visited),
        GrammarNode::Plus(inner) => compile_plus(inner, rules, visited),
        GrammarNode::Opt(inner) => compile_opt(inner, rules, visited),
        GrammarNode::Ref(name) => compile_ref(name, rules, visited),
        GrammarNode::HexByte(byte) => compile_lit(&(*byte as char).to_string()),
    }
}

fn compile_lit(s: &str) -> Result<NFAGraph, String> {
    if s.is_empty() {
        // Empty string: epsilon transition
        let mut nfa = NFAGraph::new();
        nfa.add_epsilon(nfa.start, nfa.accept);
        return Ok(nfa);
    }

    let chars: Vec<char> = s.chars().collect();
    let mut nfa = NFAGraph::new();

    // Replace the default start/accept with a chain
    let mut prev = nfa.start;
    for &ch in &chars {
        let next = nfa.add_state(false);
        nfa.add_char(prev, ch, next);
        prev = next;
    }
    // Mark last state as accept
    nfa.states[nfa.start].is_accept = false;
    nfa.states[prev].is_accept = true;
    nfa.accept = prev;

    Ok(nfa)
}

fn compile_char_class(ranges: &[(char, char)], negated: bool) -> Result<NFAGraph, String> {
    let mut nfa = NFAGraph::new();
    nfa.states[nfa.start].is_accept = false;
    let accept = nfa.add_state(true);
    if negated {
        // Negated character class [^...]: match any character NOT in the set.
        // We build a practical character set covering printable ASCII plus
        // common whitespace, then exclude the specified characters.
        let excluded_chars: std::collections::BTreeSet<char> = ranges
            .iter()
            .flat_map(|&(start, end)| {
                if start == end {
                    vec![start]
                } else {
                    ((start as u32)..=(end as u32))
                        .filter_map(std::char::from_u32)
                        .collect()
                }
            })
            .collect();

        // Build the practical character universe
        let mut universe: Vec<char> = Vec::new();
        // Printable ASCII range (0x20-0x7E)
        for c in 0x20u32..=0x7Eu32 {
            if let Some(ch) = std::char::from_u32(c) {
                universe.push(ch);
            }
        }
        // Whitespace: tab, newline, carriage return
        universe.push('\t');
        universe.push('\n');
        universe.push('\r');
        // Common Unicode whitespace / zero-width chars
        for c in [0xA0u32, 0x200Bu32, 0x200Cu32, 0x200Du32, 0xFEFFu32] {
            if let Some(ch) = std::char::from_u32(c) {
                universe.push(ch);
            }
        }

        // Add transitions for each character NOT in the excluded set
        for &ch in &universe {
            if !excluded_chars.contains(&ch) {
                nfa.add_char(nfa.start, ch, accept);
            }
        }
    } else {
        // Non-negated: add transitions for each character in the ranges
        for &(start, end) in ranges {
            if start == end {
                nfa.add_char(nfa.start, start, accept);
            } else {
                let start_u = start as u32;
                let end_u = end as u32;
                for c in char_range(start_u, end_u) {
                    nfa.add_char(nfa.start, c, accept);
                }
            }
        }
    }
    nfa.accept = accept;

    Ok(nfa)
}

fn compile_any() -> Result<NFAGraph, String> {
    // . matches any single character
    // Add explicit transitions for common printable characters and whitespace.
    // During DFA construction, these transitions create a proper DFA where
    // any input character from the grammar is accepted.
    let mut nfa = NFAGraph::new();
    nfa.states[nfa.start].is_accept = false;
    let accept = nfa.add_state(true);
    // Printable ASCII (0x20-0x7E) + tab, newline, carriage return
    for c in 0x20u32..=0x7Eu32 {
        if let Some(ch) = char::from_u32(c) {
            nfa.add_char(nfa.start, ch, accept);
        }
    }
    nfa.add_char(nfa.start, '\t', accept);
    nfa.add_char(nfa.start, '\n', accept);
    nfa.add_char(nfa.start, '\r', accept);
    // Include common Unicode characters that might appear in token text
    for c in [0xA0u32, 0x200Bu32, 0x200Cu32, 0x200Du32, 0xFEFFu32] {
        if let Some(ch) = char::from_u32(c) {
            nfa.add_char(nfa.start, ch, accept);
        }
    }
    nfa.accept = accept;
    Ok(nfa)
}

fn compile_seq(
    nodes: &[GrammarNode],
    rules: &HashMap<String, GrammarNode>,
    visited: &mut Vec<String>,
) -> Result<NFAGraph, String> {
    if nodes.is_empty() {
        let mut nfa = NFAGraph::new();
        nfa.add_epsilon(nfa.start, nfa.accept);
        return Ok(nfa);
    }

    let first = compile_node(&nodes[0], rules, visited)?;
    let mut nfa = first;

    for node in &nodes[1..] {
        let next = compile_node(node, rules, visited)?;
        // Connect: current accept -> next start
        nfa.add_epsilon(
            nfa.accept,
            next.start + nfa.states.len().saturating_sub(next.states.len().max(1)),
        );
        // Actually, we need to merge properly.
        // Simple approach: offset next's states and connect
        let offset = nfa.states.len();
        for state in &next.states {
            let mut new_state = NFAState {
                transitions: vec![],
                is_accept: state.is_accept,
            };
            for (ch, target) in &state.transitions {
                let new_target = if target == &next.start {
                    // Pointers to start within next are offset
                    *target + offset
                } else {
                    *target + offset
                };
                new_state.transitions.push((*ch, new_target));
            }
            nfa.states.push(new_state);
        }
        // Adjust references: next.start -> nfa start's transitions need fixing
        // The connection from nfa's old accept to next's start
        let old_accept = nfa.accept;
        // Mark old accept as non-accepting
        nfa.states[old_accept].is_accept = false;
        nfa.add_epsilon(old_accept, offset);
        nfa.accept = offset + (next.accept - next.start);
    }

    Ok(nfa)
}

fn compile_alt(
    nodes: &[GrammarNode],
    rules: &HashMap<String, GrammarNode>,
    visited: &mut Vec<String>,
) -> Result<NFAGraph, String> {
    if nodes.is_empty() {
        let mut nfa = NFAGraph::new();
        nfa.add_epsilon(nfa.start, nfa.accept);
        return Ok(nfa);
    }

    let mut nfa = NFAGraph::new();
    let entry = nfa.start;

    for node in nodes {
        let fragment = compile_node(node, rules, visited)?;
        let offset = nfa.states.len();
        // Append fragment states
        for state in &fragment.states {
            nfa.states.push(state.clone());
        }
        // Epsilon from entry to fragment start
        nfa.add_epsilon(entry, offset + fragment.start);
        // Epsilon from fragment accept to nfa accept
        nfa.states[offset + fragment.accept].is_accept = false;
        nfa.add_epsilon(offset + fragment.accept, nfa.accept);
    }

    Ok(nfa)
}

fn compile_star(
    inner: &GrammarNode,
    rules: &HashMap<String, GrammarNode>,
    visited: &mut Vec<String>,
) -> Result<NFAGraph, String> {
    let inner_frag = compile_node(inner, rules, visited)?;
    let mut nfa = NFAGraph::new();
    let entry = nfa.start;
    let exit = nfa.accept;

    let offset = nfa.states.len();
    for state in &inner_frag.states {
        nfa.states.push(state.clone());
    }

    // Epsilon: entry -> exit (zero repetitions)
    nfa.add_epsilon(entry, exit);
    // Epsilon: entry -> inner start
    nfa.add_epsilon(entry, offset + inner_frag.start);
    // Epsilon: inner accept -> inner start (loop)
    nfa.states[offset + inner_frag.accept].is_accept = false;
    nfa.add_epsilon(offset + inner_frag.accept, offset + inner_frag.start);
    // Epsilon: inner accept -> exit
    nfa.add_epsilon(offset + inner_frag.accept, exit);

    Ok(nfa)
}

fn compile_plus(
    inner: &GrammarNode,
    rules: &HashMap<String, GrammarNode>,
    visited: &mut Vec<String>,
) -> Result<NFAGraph, String> {
    let inner_frag = compile_node(inner, rules, visited)?;
    let mut nfa = NFAGraph::new();
    let entry = nfa.start;

    let offset = nfa.states.len();
    for state in &inner_frag.states {
        nfa.states.push(state.clone());
    }

    // Epsilon: entry -> inner start
    nfa.add_epsilon(entry, offset + inner_frag.start);
    // Epsilon: inner accept -> inner start (loop for one or more)
    nfa.states[offset + inner_frag.accept].is_accept = false;
    nfa.add_epsilon(offset + inner_frag.accept, offset + inner_frag.start);
    // Epsilon: inner accept -> nfa accept
    nfa.add_epsilon(offset + inner_frag.accept, nfa.accept);

    Ok(nfa)
}

fn compile_opt(
    inner: &GrammarNode,
    rules: &HashMap<String, GrammarNode>,
    visited: &mut Vec<String>,
) -> Result<NFAGraph, String> {
    let inner_frag = compile_node(inner, rules, visited)?;
    let mut nfa = NFAGraph::new();

    // Epsilon: entry -> exit (skip inner)
    nfa.add_epsilon(nfa.start, nfa.accept);

    let offset = nfa.states.len();
    for state in &inner_frag.states {
        nfa.states.push(state.clone());
    }

    // Epsilon: entry -> inner start
    nfa.add_epsilon(nfa.start, offset + inner_frag.start);
    // Epsilon: inner accept -> exit
    nfa.states[offset + inner_frag.accept].is_accept = false;
    nfa.add_epsilon(offset + inner_frag.accept, nfa.accept);

    Ok(nfa)
}

fn compile_ref(
    name: &str,
    rules: &HashMap<String, GrammarNode>,
    visited: &mut Vec<String>,
) -> Result<NFAGraph, String> {
    if visited.contains(&name.to_string()) {
        return Err(format!(
            "circular rule reference detected: {}",
            visited.join(" -> ")
        ));
    }

    let rule = rules
        .get(name)
        .ok_or_else(|| format!("undefined rule '{}'", name))?;

    visited.push(name.to_string());
    let result = compile_node(rule, rules, visited);
    visited.pop();

    result
}

fn char_range(start: u32, end: u32) -> Vec<char> {
    (start..=end).filter_map(std::char::from_u32).collect()
}

// ── NFA → DFA (Subset Construction) ──────────────────────────────────────

/// A deterministic finite automaton state.
#[derive(Debug, Clone, PartialEq)]
pub struct DFAState {
    /// Unique state ID.
    pub id: usize,
    /// Valid character ranges from this state (sorted, non-overlapping).
    pub valid_chars: Vec<(char, char)>,
    /// Transitions: (single character, next_state_id)
    pub transitions: Vec<(char, usize)>,
    /// Is this an accept (terminal) state?
    pub is_accept: bool,
}

/// Compiled deterministic finite automaton for grammar-guided generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GrammarFSM {
    /// DFA states indexed by ID.
    states: Vec<DFAState>,
    /// Current state ID during generation.
    current_state: usize,
    /// Start state ID.
    start_state: usize,
}

impl GrammarFSM {
    /// Create a new FSM from a compiled DFA.
    fn new(states: Vec<DFAState>, start: usize) -> Self {
        GrammarFSM {
            states,
            current_state: start,
            start_state: start,
        }
    }

    /// Get a mask of valid tokens given the current FSM state.
    ///
    /// For each token ID, checks if the token's decoded text matches
    /// a valid continuation from the current FSM state. Returns a
    /// boolean mask: true = allowed, false = forbidden.
    pub fn valid_token_mask(&self, tokenizer: &GrammarTokenizer, vocab_size: usize) -> Vec<bool> {
        let mut mask = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size {
            let text = tokenizer.decode(id as u32);
            mask.push(self.is_valid_token(text));
        }
        mask
    }

    /// Check if a single token's text is a valid continuation from the
    /// current FSM state.
    fn is_valid_token(&self, text: &str) -> bool {
        let mut state_id = self.current_state;
        for ch in text.chars() {
            let next = self.find_transition(state_id, ch);
            match next {
                Some(next_id) => state_id = next_id,
                None => return false,
            }
        }
        true
    }

    /// Find the next state for a given character from the current DFA state.
    fn find_transition(&self, state_id: usize, ch: char) -> Option<usize> {
        let state = &self.states[state_id];
        // Binary search on transitions (sorted by char)
        state
            .transitions
            .binary_search_by(|(c, _)| c.cmp(&ch))
            .ok()
            .map(|idx| state.transitions[idx].1)
    }

    /// Advance the FSM given an accepted token's decoded text.
    pub fn advance(&mut self, token_text: &str) -> Result<(), String> {
        for ch in token_text.chars() {
            let next = self
                .find_transition(self.current_state, ch)
                .ok_or_else(|| {
                    format!(
                        "no transition for character '{}' (U+{:X}) from state {}",
                        ch, ch as u32, self.current_state
                    )
                })?;
            self.current_state = next;
        }
        Ok(())
    }

    /// Reset the FSM to the start state (new sequence).
    pub fn reset(&mut self) {
        self.current_state = self.start_state;
    }

    /// Returns the current DFA state ID.
    pub fn current_state(&self) -> usize {
        self.current_state
    }

    /// Returns the start DFA state ID.
    pub fn start_state(&self) -> usize {
        self.start_state
    }

    /// Is the FSM in an accept state?
    pub fn is_accepting(&self) -> bool {
        self.current_state < self.states.len() && self.states[self.current_state].is_accept
    }

    /// Apply grammar mask to logits in-place.
    /// Sets forbidden token logits to -infinity (f32::NEG_INFINITY).
    pub fn apply_mask_to_logits(&self, logits: &mut [f32], tokenizer: &GrammarTokenizer) {
        debug_assert_eq!(logits.len(), tokenizer.id_to_text.len());
        for (i, logit) in logits.iter_mut().enumerate() {
            let text = tokenizer.decode(i as u32);
            if !self.is_valid_token(text) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

/// Compute the epsilon closure of a set of NFA states.
fn epsilon_closure(nfa: &[NFAState], states: &BTreeSet<usize>) -> BTreeSet<usize> {
    let mut closure: BTreeSet<usize> = states.clone();
    let mut stack: Vec<usize> = states.iter().copied().collect();

    while let Some(state_id) = stack.pop() {
        for (ch, target) in &nfa[state_id].transitions {
            if ch.is_none() && closure.insert(*target) {
                stack.push(*target);
            }
        }
    }

    closure
}

/// Compute the transition from a set of NFA states on a character.
fn nfa_transition(nfa: &[NFAState], states: &BTreeSet<usize>, ch: char) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();
    for &state_id in states {
        for (cond, target) in &nfa[state_id].transitions {
            if let Some(c) = cond {
                if *c == ch {
                    result.insert(*target);
                }
            }
        }
    }
    epsilon_closure(nfa, &result)
}

/// Build the set of all valid characters from a set of NFA states.
fn valid_chars_from_nfa_states(nfa: &[NFAState], states: &BTreeSet<usize>) -> Vec<(char, char)> {
    let mut char_set: BTreeSet<char> = BTreeSet::new();
    for &state_id in states {
        for (cond, _target) in &nfa[state_id].transitions {
            if let Some(c) = cond {
                char_set.insert(*c);
            }
        }
    }

    // Convert to ranges
    let mut chars: Vec<char> = char_set.into_iter().collect();
    chars.sort();
    let mut ranges: Vec<(char, char)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = chars[i];
        let mut end = start;
        while i + 1 < chars.len() && (chars[i + 1] as u32) == (end as u32) + 1 {
            i += 1;
            end = chars[i];
        }
        ranges.push((start, end));
        i += 1;
    }
    ranges
}

/// Convert an NFA to a DFA via subset construction.
fn nfa_to_dfa(nfa: &[NFAState], start_state: usize) -> Vec<DFAState> {
    let start_closure = epsilon_closure(nfa, &BTreeSet::from([start_state]));
    let mut dfa_states: Vec<DFAState> = Vec::new();
    let mut state_map: HashMap<BTreeSet<usize>, usize> = HashMap::new();

    // Collect all characters that appear in any transition
    let mut all_chars: BTreeSet<char> = BTreeSet::new();
    for state in nfa {
        for (cond, _) in &state.transitions {
            if let Some(c) = cond {
                all_chars.insert(*c);
            }
        }
    }

    // Add the UTF-8 continuation bytes and common characters
    // Extend with common printable ASCII if no transitions exist
    if all_chars.is_empty() {
        // Add a broad range so we can still build a DFA
        for c in 0x20u32..=0x7Eu32 {
            if let Some(ch) = char::from_u32(c) {
                all_chars.insert(ch);
            }
        }
        // Add newline
        all_chars.insert('\n');
        all_chars.insert('\t');
        all_chars.insert('\r');
    }

    let chars_vec: Vec<char> = all_chars.into_iter().collect();

    let start_id = 0;
    state_map.insert(start_closure.clone(), start_id);
    let mut queue: VecDeque<BTreeSet<usize>> = VecDeque::new();
    queue.push_back(start_closure.clone());

    dfa_states.push(DFAState {
        id: start_id,
        valid_chars: vec![],
        transitions: vec![],
        is_accept: start_closure.iter().any(|&s| nfa[s].is_accept),
    });

    while let Some(current_set) = queue.pop_front() {
        let current_id = state_map[&current_set];

        for &ch in &chars_vec {
            let next_set = nfa_transition(nfa, &current_set, ch);
            if next_set.is_empty() {
                continue;
            }

            let next_id = if let Some(&id) = state_map.get(&next_set) {
                id
            } else {
                let id = dfa_states.len();
                state_map.insert(next_set.clone(), id);
                dfa_states.push(DFAState {
                    id,
                    valid_chars: vec![],
                    transitions: vec![],
                    is_accept: next_set.iter().any(|&s| nfa[s].is_accept),
                });
                queue.push_back(next_set.clone());
                id
            };

            dfa_states[current_id].transitions.push((ch, next_id));
        }

        // Sort transitions for binary search
        dfa_states[current_id]
            .transitions
            .sort_by(|(a, _), (b, _)| a.cmp(b));
        dfa_states[current_id]
            .transitions
            .dedup_by(|(a, _), (b, _)| a == b);

        // Build valid_chars from transitions
        let chars_from_transitions: Vec<char> = dfa_states[current_id]
            .transitions
            .iter()
            .map(|(c, _)| *c)
            .collect();
        let mut ranges: Vec<(char, char)> = Vec::new();
        let mut i = 0;
        while i < chars_from_transitions.len() {
            let start = chars_from_transitions[i];
            let mut end = start;
            while i + 1 < chars_from_transitions.len()
                && (chars_from_transitions[i + 1] as u32) == (end as u32) + 1
            {
                i += 1;
                end = chars_from_transitions[i];
            }
            ranges.push((start, end));
            i += 1;
        }
        dfa_states[current_id].valid_chars = ranges;
    }

    dfa_states
}

// ── GrammarTokenizer ──────────────────────────────────────────────────────

use std::path::Path;

/// Minimal tokenizer for grammar masking.
///
/// Just needs the token_id → text mapping, not the full tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct GrammarTokenizer {
    /// token_id → decoded text
    pub id_to_text: Vec<String>,
}

impl GrammarTokenizer {
    /// Load tokenizer from a tokenizer.json file.
    ///
    /// Expects the standard HuggingFace tokenizer.json format with
    /// a `model.vocab` dictionary mapping strings to integers,
    /// or `added_tokens` for special tokens.
    pub fn load(tokenizer_path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(tokenizer_path)
            .map_err(|e| format!("failed to read tokenizer.json: {}", e))?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("invalid tokenizer.json: {}", e))?;

        // Determine vocab size
        let mut id_to_text: Vec<String> = Vec::new();

        // Try model.vocab first (standard HF format)
        if let Some(vocab) = json.get("model").and_then(|m| m.get("vocab")) {
            if let Some(obj) = vocab.as_object() {
                for (_token, id_val) in obj {
                    if let Some(id) = id_val.as_u64() {
                        let id = id as usize;
                        if id >= id_to_text.len() {
                            id_to_text.resize(id + 1, String::new());
                        }
                        id_to_text[id] = _token.to_string();
                    }
                }
            }
        }

        // Also check for added_tokens
        if let Some(added) = json.get("added_tokens").and_then(|a| a.as_array()) {
            for entry in added {
                if let (Some(id), Some(content)) = (
                    entry.get("id").and_then(|v| v.as_u64()),
                    entry.get("content").and_then(|v| v.as_str()),
                ) {
                    let id = id as usize;
                    if id >= id_to_text.len() {
                        id_to_text.resize(id + 1, String::new());
                    }
                    id_to_text[id] = content.to_string();
                }
            }
        }

        if id_to_text.is_empty() {
            return Err("tokenizer.json has no vocabulary entries".to_string());
        }

        Ok(GrammarTokenizer { id_to_text })
    }

    /// Create a new tokenizer from an existing id→text mapping.
    pub fn new(id_to_text: Vec<String>) -> Self {
        GrammarTokenizer { id_to_text }
    }

    /// Decode a token ID to its text representation.
    pub fn decode(&self, token_id: u32) -> &str {
        let id = token_id as usize;
        if id < self.id_to_text.len() {
            &self.id_to_text[id]
        } else {
            ""
        }
    }

    /// The vocabulary size (number of known tokens).
    pub fn vocab_size(&self) -> usize {
        self.id_to_text.len()
    }
}

// ── Public API ────────────────────────────────────────────────────────────

impl Grammar {
    /// Parse GBNF text into a grammar.
    pub fn parse(text: &str) -> Result<Self, String> {
        parse_gbnf(text)
    }

    /// Compile grammar to a deterministic finite automaton for fast
    /// token masking during generation.
    pub fn compile(&self) -> Result<GrammarFSM, String> {
        let mut visited = Vec::new();
        let nfa = compile_ref(&self.root, &self.rules, &mut visited)?;
        let dfa_states = nfa_to_dfa(&nfa.states, nfa.start);
        Ok(GrammarFSM::new(dfa_states, 0))
    }

    /// Convenience: parse + compile in one step.
    pub fn compile_from_text(text: &str) -> Result<GrammarFSM, String> {
        let grammar = Self::parse(text)?;
        grammar.compile()
    }

    /// Build a JSON schema grammar for structured output.
    ///
    /// Converts a JSON Schema object to a GBNF grammar that generates
    /// matching JSON.
    pub fn from_json_schema(name: &str, schema: &serde_json::Value) -> Result<Self, String> {
        let gbnf = json_schema_to_gbnf(name, schema)?;
        Self::parse(&gbnf)
    }
}

// ── JSON Schema → GBNF ─────────────────────────────────────────────────────

/// Convert a JSON Schema to GBNF grammar text.
fn json_schema_to_gbnf(name: &str, schema: &serde_json::Value) -> Result<String, String> {
    let mut grammar = String::new();
    let root = format!("root ::= {}", name);
    grammar.push_str(&root);
    grammar.push('\n');
    json_schema_emit_rule(name, schema, &mut grammar, 0)?;
    Ok(grammar)
}

fn json_schema_emit_rule(
    name: &str,
    schema: &serde_json::Value,
    out: &mut String,
    depth: usize,
) -> Result<(), String> {
    if depth > 20 {
        return Err("json schema nesting too deep (>20)".to_string());
    }

    // Determine the type from schema
    let schema_type = schema.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match schema_type {
        "object" => {
            let properties = match schema.get("properties").and_then(|v| v.as_object()) {
                Some(props) => props,
                None => {
                    // No properties defined — accept any object
                    out.push_str(&format!("{} ::= \"{{\" ws \"}}\"\n", name));
                    return Ok(());
                }
            };

            let required: Vec<&str> = schema
                .get("required")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();

            let _rule_lines: Vec<String> = Vec::new();
            let mut sub_rules: Vec<(String, serde_json::Value)> = Vec::new();
            let mut added_prop_names: Vec<String> = Vec::new();

            for (prop_name, prop_schema) in properties {
                let rule_name = format!("{}_{}", name, prop_name);
                let sub_rule_name = format!("_{}_value", &rule_name);

                // Emit the property value rule
                json_schema_emit_rule(&sub_rule_name, prop_schema, out, depth + 1)?;
                sub_rules.push((prop_name.clone(), prop_schema.clone()));

                let is_required = required.contains(&prop_name.as_str());
                let quoted_name = escape_json_string(prop_name);
                let pair = format!(" \"{}\" ws \":\" ws {} ", quoted_name, sub_rule_name);

                if !is_required {
                    let opt_name = format!("_{}_opt", &rule_name);
                    out.push_str(&format!("{} ::= {} | \"\"\n", opt_name, pair));
                    added_prop_names.push(opt_name);
                } else {
                    added_prop_names.push(pair);
                }
            }

            if added_prop_names.is_empty() {
                out.push_str(&format!("{} ::= \"{{}}\"\n", name));
            } else {
                // Build object: { prop1, prop2, ... } with optional trailing comma
                // For simplicity: comma-separated, no trailing comma, fixed order
                let props_seq: String = added_prop_names.join(" \",\" ws ");
                out.push_str(&format!("{} ::= \"{{\" ws {} \"}}\"\n", name, props_seq));
            }
        }

        "array" => {
            let items = schema.get("items");
            match items {
                Some(item_schema) => {
                    let item_rule = format!("{}_item", name);
                    json_schema_emit_rule(&item_rule, item_schema, out, depth + 1)?;
                    out.push_str(&format!(
                        "{} ::= \"[\" ws ({} (\",\" ws {})*) ws \"]\"\n",
                        name, item_rule, item_rule
                    ));
                }
                None => {
                    out.push_str(&format!("{} ::= \"[\" ws \"]\"\n", name));
                }
            }
        }

        "string" => {
            out.push_str(&format!("{} ::= string\n", name));
            if !out.contains("string ::=") {
                out.push_str("string ::= \"\\\"\" ([^\"]*) \"\\\"\"\n");
            }
        }

        "integer" => {
            out.push_str(&format!("{} ::= integer\n", name));
            if !out.contains("integer ::=") {
                out.push_str("integer ::= (\"-\" | \"\") [0-9]+\n");
            }
        }

        "number" => {
            out.push_str(&format!("{} ::= number\n", name));
            if !out.contains("number ::=") {
                out.push_str("number ::= (\"-\" | \"\") [0-9]+ (\".\" [0-9]+)?\n");
            }
        }

        "boolean" => {
            out.push_str(&format!("{} ::= \"true\" | \"false\"\n", name));
        }

        "null" => {
            out.push_str(&format!("{} ::= \"null\"\n", name));
        }

        // If no type, try enum
        _ => {
            if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
                let alts: Vec<String> = enum_values
                    .iter()
                    .map(|v| {
                        let s = serde_json::to_string(v).unwrap_or_else(|_| "null".to_string());
                        format!("\"{}\"", escape_json_string(&s))
                    })
                    .collect();
                out.push_str(&format!("{} ::= {}\n", name, alts.join(" | ")));
            } else if let Some(_ref_val) = schema.get("$ref").and_then(|v| v.as_str()) {
                // Handle $ref references (simplified: just treat as string)
                out.push_str(&format!("{} ::= string\n", name));
            } else {
                // Fallback: accept any value
                out.push_str(&format!("{} ::= any\n", name));
                if !out.contains("any ::=") {
                    out.push_str("any ::= string | number | \"true\" | \"false\" | \"null\" | \"[\" ws \"]\" | \"{\" ws \"}\"\n");
                }
            }
        }
    }

    // Add whitespace rule if not already present
    if !out.contains("ws ::=") {
        out.push_str("ws ::= [ \\t\\n]*\n");
    }

    Ok(())
}

/// Escape a string for use as a JSON string literal in GBNF.
fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\t' => result.push_str("\\t"),
            '\r' => result.push_str("\\r"),
            c => result.push(c),
        }
    }
    result
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_json() {
        let gbnf = r#"
root ::= "{" ws "}" ws
ws ::= [ \t\n]*
"#;
        let grammar = Grammar::parse(gbnf).expect("parse failed");
        assert_eq!(grammar.root, "root");
        assert!(grammar.rules.contains_key("root"));
        assert!(grammar.rules.contains_key("ws"));
    }

    #[test]
    fn test_parse_person_grammar() {
        let gbnf = r#"
root ::= "{" ws "name" ws ":" ws string ws "," ws "age" ws ":" ws number ws "}"
string ::= "\"" ([^"]*) "\""
number ::= [0-9]+
ws ::= [ \t\n]*
"#;
        let grammar = Grammar::parse(gbnf).expect("parse failed");
        assert_eq!(grammar.root, "root");
        assert!(grammar.rules.contains_key("string"));
        assert!(grammar.rules.contains_key("number"));
        assert!(grammar.rules.contains_key("ws"));
    }

    #[test]
    fn test_compile_simple() {
        let gbnf = r#"
root ::= [a-z]+
"#;
        let grammar = Grammar::parse(gbnf).expect("parse failed");
        let fsm = grammar.compile().expect("compile failed");
        assert!(fsm.states.len() >= 1);
        assert!(!fsm.is_accepting()); // after consuming nothing, we need at least one char
    }

    #[test]
    fn test_parse_with_alternation() {
        let gbnf = "root ::= \"hello\" | \"world\"\n";
        let grammar = Grammar::parse(gbnf).expect("parse failed");
        let _fsm = grammar.compile().expect("compile failed");
    }

    #[test]
    fn test_parse_with_repetition() {
        let gbnf = "root ::= [a-z]*\n";
        let grammar = Grammar::parse(gbnf).expect("parse failed");
        let fsm = grammar.compile().expect("compile failed");
        assert!(fsm.is_accepting()); // * means zero or more, so empty string is valid
    }

    #[test]
    fn test_empty_grammar_fails() {
        assert!(Grammar::parse("").is_err());
    }

    #[test]
    fn test_comment_line() {
        let gbnf = "# this is a comment\nroot ::= \"a\"\n";
        let grammar = Grammar::parse(gbnf).expect("parse failed");
        assert_eq!(grammar.root, "root");
    }

    #[test]
    fn test_json_schema_object() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let grammar = Grammar::from_json_schema("person", &schema).expect("json schema failed");
        let _fsm = grammar.compile().expect("compile failed");
    }

    #[test]
    fn test_json_schema_nested() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    },
                    "required": ["street"]
                }
            },
            "required": ["name"]
        });
        let grammar = Grammar::from_json_schema("person", &schema).expect("json schema failed");
        let _fsm = grammar.compile().expect("compile failed");
    }

    #[test]
    fn test_tokenizer_from_vocab() {
        let tokenizer = GrammarTokenizer::new(vec![
            "hello".to_string(),
            "world".to_string(),
            " ".to_string(),
            "a".to_string(),
        ]);
        assert_eq!(tokenizer.decode(0), "hello");
        assert_eq!(tokenizer.decode(1), "world");
        assert_eq!(tokenizer.decode(3), "a");
        assert_eq!(tokenizer.decode(99), "");
    }

    #[test]
    fn test_fsm_advance_and_reset() {
        let gbnf = r#"root ::= "hello""#;
        let fsm = Grammar::compile_from_text(gbnf).expect("compile failed");
        let mut fsm = fsm;
        assert_eq!(fsm.current_state, fsm.start_state);

        fsm.advance("hello").expect("advance failed");
        assert!(fsm.is_accepting());

        fsm.reset();
        assert_eq!(fsm.current_state, fsm.start_state);
    }

    #[test]
    fn test_fsm_invalid_advance() {
        let gbnf = r#"root ::= "hello""#;
        let mut fsm = Grammar::compile_from_text(gbnf).expect("compile failed");
        assert!(fsm.advance("world").is_err());
    }
}

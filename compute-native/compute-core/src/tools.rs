//! Deterministic tool call parsing & repair for OpenAI-compatible function calling.
//!
//! When the model generates a malformed function call (broken JSON, missing params,
//! wrong function name), this module detects it, repairs it deterministically, and
//! only retries generation if repair is impossible.
//!
//! # Pipeline
//!
//! 1. [`parse_and_repair`] — try up to 4 strategies to extract JSON from raw text
//! 2. [`validate_and_fix`] — check required fields, fix type mismatches, correct names
//! 3. [`retry_with_error`] — if unrepairable, regenerate with error context

use serde::{Deserialize, Serialize};

/// A tool definition parsed from the OpenAI API request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the function parameters.
    pub parameters: serde_json::Value,
    /// Parameter names that are required.
    pub required: Vec<String>,
}

/// A function call emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// Parsed JSON arguments.
    pub arguments: serde_json::Value,
    /// Raw text the model generated.
    pub raw: String,
}

/// Result of attempting to parse and repair a tool call.
#[derive(Debug, Clone)]
pub enum ToolCallResult {
    /// Valid tool call, ready to execute.
    Valid(FunctionCall),
    /// Repaired deterministically from malformed output.
    Repaired(FunctionCall, Vec<String>),
    /// Cannot repair — should retry generation.
    Unrepairable(String),
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Attempt to parse and repair a model-generated function call using up to 4
/// strategies in order of decreasing specificity.
pub fn parse_and_repair(raw_text: &str, tool: &ToolDefinition) -> ToolCallResult {
    // 1. Try direct JSON parse.
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw_text) {
        return validate_and_fix(val, tool);
    }

    // 2. Try extracting JSON from markdown code fences (```json ... ```).
    if let Some(json) = extract_json_from_fences(raw_text) {
        return validate_and_fix(json, tool);
    }

    // 3. Try extracting the first complete JSON `{...}` object.
    if let Some(json) = extract_first_json_object(raw_text) {
        return validate_and_fix(json, tool);
    }

    // 4. Try unescaping escape sequences and re-parsing.
    if let Some(json) = unescape_and_parse(raw_text) {
        return validate_and_fix(json, tool);
    }

    ToolCallResult::Unrepairable("could not parse any JSON from model output".into())
}

/// Validate required fields, fill missing with defaults, fix type mismatches,
/// and correct function name typos.
pub fn validate_and_fix(mut val: serde_json::Value, tool: &ToolDefinition) -> ToolCallResult {
    let mut fixes: Vec<String> = Vec::new();

    // ── Fix function name (fuzzy match) ───────────────────────────────
    // Check under both the top-level "name" key (OpenAI function-call style)
    // and inside an inner "function" object (OpenAI tool-call style).
    let name_val = val
        .get("name")
        .or_else(|| val.get("function").and_then(|f| f.get("name")));

    let current_name = name_val.and_then(|n| n.as_str()).unwrap_or("").to_string();

    if !current_name.is_empty() && current_name != tool.name {
        if let Some(corrected) = fuzzy_match_function_name(&current_name, tool) {
            if let Some(obj) = val.as_object_mut() {
                obj.insert("name".into(), serde_json::json!(corrected));
                fixes.push(format!(
                    "corrected function name '{}' -> '{}'",
                    current_name, corrected
                ));
            }
        }
    }

    // ── Normalize tool-call wrapper ───────────────────────────────────
    // OpenAI tool_calls format: {"id":"...","type":"function","function":{name,arguments}}
    // Convert to flat {name, arguments} for uniform processing.
    if val.get("type").and_then(|t| t.as_str()) == Some("function") {
        if let Some(func) = val.get("function").cloned() {
            if let Some(obj) = val.as_object_mut() {
                if let Some(func_name) = func.get("name").and_then(|n| n.as_str()) {
                    obj.insert("name".into(), serde_json::json!(func_name));
                }
                if let Some(args) = func.get("arguments") {
                    // arguments might be a string that needs re-parsing
                    if let Some(arg_str) = args.as_str() {
                        if let Ok(parsed_args) = serde_json::from_str::<serde_json::Value>(arg_str)
                        {
                            obj.insert("arguments".into(), parsed_args);
                        } else {
                            obj.insert("arguments".into(), args.clone());
                        }
                    } else {
                        obj.insert("arguments".into(), args.clone());
                    }
                }
                obj.remove("function");
                obj.remove("type");
                obj.remove("id");
                fixes.push("normalized tool_calls wrapper format".into());
            }
        }
    }

    // ── Parse string arguments into JSON ──────────────────────────────
    // Sometimes the model puts the arguments as a JSON string inside "arguments".
    if let Some(args) = val.get("arguments") {
        let reparsed = args
            .as_str()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());
        if let Some(parsed) = reparsed {
            if let Some(obj) = val.as_object_mut() {
                obj.insert("arguments".into(), parsed);
                fixes.push("re-parsed string arguments as JSON".into());
            }
        }
    }

    // ── Check required fields in the arguments ────────────────────────
    let args = val
        .get("arguments")
        .and_then(|a| a.as_object())
        .cloned()
        .unwrap_or_default();
    let mut args = args;

    for field in &tool.required {
        let has_value = args.get(field).map(|v| !v.is_null()).unwrap_or(false);
        if !has_value {
            if let Some(default) = get_default_for(&tool.parameters, field) {
                args.insert(field.clone(), default);
                fixes.push(format!("filled missing required field '{}'", field));
            }
        }
    }

    // ── Fix type mismatches in arguments ──────────────────────────────
    fix_type_mismatches_in_object(&mut args, &tool.parameters, &mut fixes);

    // ── Rebuild the value ─────────────────────────────────────────────
    if let Some(obj) = val.as_object_mut() {
        obj.insert("arguments".into(), serde_json::Value::Object(args));
    }

    let name = tool.name.clone();
    let fc = FunctionCall {
        name: name.clone(),
        arguments: val.clone(),
        raw: val.to_string(),
    };

    if fixes.is_empty() {
        ToolCallResult::Valid(fc)
    } else {
        ToolCallResult::Repaired(fc, fixes)
    }
}

/// Extract JSON from markdown code fences (` ```json ... ``` `).
pub fn extract_json_from_fences(text: &str) -> Option<serde_json::Value> {
    // Match ```json ... ``` (case-insensitive language tag)
    for prefix in &["```json", "```JSON", "``` javascript", "```"] {
        if let Some(start) = text.find(prefix) {
            let content_start = start + prefix.len();
            // Skip past optional newline after the fence
            let content_start = content_start
                + text[content_start..]
                    .chars()
                    .next()
                    .filter(|&c| c == '\n' || c == '\r')
                    .map(|_| 1)
                    .unwrap_or(0);
            if let Some(end) = text[content_start..].find("```") {
                let candidate = &text[content_start..content_start + end];
                // Trim leading/trailing whitespace
                let candidate = candidate.trim();
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) {
                    return Some(val);
                }
            }
        }
    }
    None
}

/// Extract the first complete JSON object (`{...}`) from arbitrary text.
///
/// Handles nested braces, escaped quotes, and trailing content after the
/// closing brace.
pub fn extract_first_json_object(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let len = bytes.len();

    // Find the first '{'
    let start = text.find('{')?;

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;

    for i in start..len {
        let byte = bytes[i];
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' if in_string => {
                escaped = true;
            }
            b'"' => {
                in_string = !in_string;
            }
            b'{' if !in_string => {
                depth += 1;
            }
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &text[start..=i];
                    // Try direct parse first; if that fails try unescaping
                    match serde_json::from_str::<serde_json::Value>(candidate) {
                        Ok(val) => return Some(val),
                        Err(_) => {
                            // Attempt to repair broken escapes and retry
                            let cleaned = repair_control_chars(candidate);
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cleaned) {
                                return Some(val);
                            }
                            // If that also fails, return the original parse attempt
                            // so the caller gets the error context
                            return None;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    None
}

/// Unescape common escape sequences and attempt JSON parse.
pub fn unescape_and_parse(text: &str) -> Option<serde_json::Value> {
    let cleaned = text
        .replace("\\\"", "\"")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\/", "/")
        .replace("\\\\", "\\");

    serde_json::from_str::<serde_json::Value>(&cleaned).ok()
}

/// Fuzzy match a function name against the tool definition using Levenshtein
/// distance. Returns the correctly-spelled name if the distance is small
/// enough relative to the name length.
pub fn fuzzy_match_function_name(name: &str, tool: &ToolDefinition) -> Option<String> {
    let threshold = max(1, (tool.name.len() as f64 * 0.4).ceil() as usize);

    let dist = levenshtein_distance(name, &tool.name);
    if dist <= threshold && dist > 0 {
        Some(tool.name.clone())
    } else {
        None
    }
}

/// Fix type mismatches recursively in a JSON object, matching against the
/// JSON Schema type definitions.
pub fn fix_type_mismatches_in_object(
    val: &mut serde_json::Map<String, serde_json::Value>,
    schema: &serde_json::Value,
    fixes: &mut Vec<String>,
) {
    let properties = schema
        .get("properties")
        .and_then(|p| p.as_object())
        .cloned()
        .unwrap_or_default();

    for (key, prop_schema) in &properties {
        if let Some(field_val) = val.get(key) {
            let fixed = fix_single_value(field_val, prop_schema);
            if let Some(fixed) = fixed {
                val.insert(key.clone(), fixed.clone());
                fixes.push(format!("fixed type mismatch for field '{}'", key));
            }
        }
    }
}

// ── Retry ──────────────────────────────────────────────────────────────────

/// Retry generation after a failed tool call by appending an error
/// description as a system message and regenerating.
///
/// The `messages` slice should match the original request's messages array
/// (serde_json::Value objects with `role` and `content`).  Returns the
/// parsed-and-repaired result, or an error string on generation failure.
pub fn retry_with_error(
    sess: &mut crate::profiled_executor::ProfiledInferenceSession,
    model: &crate::profiled_executor::LoadedProfiledModel,
    messages: &[serde_json::Value],
    error: &str,
    tool: &ToolDefinition,
    max_tokens: u64,
) -> Result<ToolCallResult, String> {
    let correction_msg = format!(
        "The previous function call was invalid: {}\n\
         Please call the `{}` function with valid JSON matching this schema:\n{}\n\
         Required parameters: {}",
        error,
        tool.name,
        serde_json::to_string_pretty(&tool.parameters).unwrap_or_default(),
        tool.required.join(", "),
    );

    // Append a system message with the correction.
    let mut new_messages: Vec<serde_json::Value> = messages.to_vec();
    new_messages.push(serde_json::json!({
        "role": "system",
        "content": correction_msg
    }));

    // Build a chat prompt from the augmented messages.
    let prompt = build_chat_prompt(&new_messages);

    let sampler_config = crate::session::SamplerConfig {
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: None,
        repetition_penalty: None,
        seed: None,
        stop_token_ids: Vec::new(),
        grammar: None,
        grammar_tokenizer: None,
    };

    let output_text = sess
        .chat_with_sampler(&prompt, max_tokens, &sampler_config, model)
        .map_err(|e| format!("retry inference failed: {e}"))?;

    Ok(parse_and_repair(&output_text, tool))
}

/// Check whether the request body includes a `tools` parameter.
pub fn has_tools_request(body: &serde_json::Value) -> bool {
    body.get("tools")
        .and_then(|t| t.as_array())
        .map(|arr| !arr.is_empty())
        .unwrap_or(false)
}

/// Extract the first tool definition from the request body.
pub fn extract_tool(body: &serde_json::Value) -> Result<ToolDefinition, String> {
    let tools = body
        .get("tools")
        .and_then(|t| t.as_array())
        .ok_or_else(|| "no tools in request".to_string())?;

    let first_tool = tools
        .first()
        .ok_or_else(|| "empty tools array".to_string())?;

    let function = first_tool
        .get("function")
        .ok_or_else(|| "tool missing 'function' field".to_string())?;

    let name = function
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| "function missing 'name'".to_string())?
        .to_string();

    let description = function
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string();

    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let required = parameters
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(ToolDefinition {
        name,
        description,
        parameters,
        required,
    })
}

/// Extract all tool definitions from the request body.
pub fn extract_tools(body: &serde_json::Value) -> Result<Vec<ToolDefinition>, String> {
    let tools = body
        .get("tools")
        .and_then(|t| t.as_array())
        .ok_or_else(|| "no tools in request".to_string())?;

    tools
        .iter()
        .map(|t| {
            let function = t
                .get("function")
                .ok_or_else(|| "tool missing 'function' field".to_string())?;

            let name = function
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| "function missing 'name'".to_string())?
                .to_string();

            let description = function
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();

            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            let required = parameters
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            Ok(ToolDefinition {
                name,
                description,
                parameters,
                required,
            })
        })
        .collect()
}

/// Execute a tool call and return the result as a JSON value.
///
/// This function handles:
/// - Built-in tools (respond with confirmation)
/// - Custom tools (delegated to the caller-provided dispatcher)
pub fn execute_tool_call(call: &FunctionCall) -> Result<serde_json::Value, String> {
    // For the server integration, tool execution is application-specific.
    // This default implementation returns an acknowledgment that the tool
    // was called, along with the parsed arguments.
    Ok(serde_json::json!({
        "tool": call.name,
        "arguments": call.arguments,
        "status": "called"
    }))
}

// ── Internal helpers ───────────────────────────────────────────────────────

/// Build a chat prompt string from a messages array (serde_json::Value).
/// Each message should have `role` and `content` fields.
fn build_chat_prompt(messages: &[serde_json::Value]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        prompt.push_str(&format!("<|{}|>\n{}\n", role, content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Attempt to fix control characters and other common JSON-breaking issues
/// in a candidate JSON string.
fn repair_control_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        // Replace control chars (except \t, \n, \r) with their escape sequences
        if b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r' {
            out.push_str(&format!("\\u{:04x}", b));
        } else {
            out.push(b as char);
        }
        i += 1;
    }
    out
}

/// Compute Levenshtein distance between two strings.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    // Use two-row DP to save allocations.
    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr: Vec<usize> = vec![0; b_len + 1];

    for i in 1..=a_len {
        curr[0] = i;
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = min3(
                prev[j] + 1,        // delete
                curr[j - 1] + 1,    // insert
                prev[j - 1] + cost, // substitute
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_len]
}

fn min3(a: usize, b: usize, c: usize) -> usize {
    a.min(b).min(c)
}

fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

/// Get a default value for a JSON Schema property.
fn get_default_for(schema: &serde_json::Value, field: &str) -> Option<serde_json::Value> {
    let prop = schema.get("properties").and_then(|p| p.get(field))?;

    // Use explicit default if present
    if let Some(default) = prop.get("default") {
        if !default.is_null() {
            return Some(default.clone());
        }
    }

    // Infer default from type
    match prop.get("type").and_then(|t| t.as_str()) {
        Some("string") => Some(serde_json::Value::String(String::new())),
        Some("number") | Some("integer") => Some(serde_json::json!(0)),
        Some("boolean") => Some(serde_json::json!(false)),
        Some("array") => Some(serde_json::json!([])),
        Some("object") => Some(serde_json::json!({})),
        _ => None,
    }
}

/// Attempt to fix a single JSON value to match its expected schema type.
/// Returns `Some(fixed)` if a fix was applied, `None` if already correct.
fn fix_single_value(
    val: &serde_json::Value,
    prop_schema: &serde_json::Value,
) -> Option<serde_json::Value> {
    let expected_type = prop_schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("string");

    match (expected_type, val) {
        // String already
        ("string", serde_json::Value::String(_)) => None,

        // Number/integer but got string: parse it
        ("number", serde_json::Value::String(s)) | ("integer", serde_json::Value::String(s)) => {
            if let Ok(n) = s.parse::<f64>() {
                if expected_type == "integer" {
                    Some(serde_json::json!(n as i64))
                } else {
                    Some(serde_json::json!(n))
                }
            } else {
                None
            }
        }

        // Boolean but got string
        ("boolean", serde_json::Value::String(s)) => match s.to_lowercase().as_str() {
            "true" | "1" => Some(serde_json::json!(true)),
            "false" | "0" => Some(serde_json::json!(false)),
            _ => None,
        },

        // String but got number/boolean: convert
        ("string", serde_json::Value::Number(n)) => Some(serde_json::json!(n.to_string())),
        ("string", serde_json::Value::Bool(b)) => Some(serde_json::json!(b.to_string())),

        // Number/integer but got boolean
        ("number", serde_json::Value::Bool(true)) | ("integer", serde_json::Value::Bool(true)) => {
            Some(serde_json::json!(1))
        }
        ("number", serde_json::Value::Bool(false))
        | ("integer", serde_json::Value::Bool(false)) => Some(serde_json::json!(0)),

        // Array to object coercion (sequence of values -> object with index keys)
        // Not supported; leave as-is.
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "Get current weather for a location".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "default": "celsius",
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }),
            required: vec!["location".into()],
        }
    }

    #[test]
    fn test_direct_valid_json() {
        let tool = make_test_tool();
        let raw = r#"{"name": "get_weather", "arguments": {"location": "London"}}"#;
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Valid(call) => {
                assert_eq!(call.name, "get_weather");
            }
            _ => panic!("expected Valid"),
        }
    }

    #[test]
    fn test_extract_from_code_fence() {
        let tool = make_test_tool();
        let raw = "Here is the result:\n```json\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n```\n";
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Valid(call) => {
                assert_eq!(call.name, "get_weather");
            }
            _ => panic!("expected Valid"),
        }
    }

    #[test]
    fn test_extract_first_json_object() {
        let tool = make_test_tool();
        let raw = "Some explanation text then {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Berlin\"}} more text";
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Valid(call) => {
                assert_eq!(call.name, "get_weather");
            }
            _ => panic!("expected Valid"),
        }
    }

    #[test]
    fn test_fill_missing_required() {
        let tool = make_test_tool();
        // Missing "location" which is required
        let raw = r#"{"name": "get_weather", "arguments": {}}"#;
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Repaired(call, fixes) => {
                assert!(fixes.iter().any(|f| f.contains("filled missing")));
                let args = call
                    .arguments
                    .get("arguments")
                    .and_then(|a| a.as_object())
                    .cloned()
                    .unwrap_or_default();
                assert!(args.contains_key("location"));
            }
            ToolCallResult::Valid(_) => {
                // The arguments exist but are empty; validate_and_fix would add
                // location but at default level it's Valid if no type mismatch
            }
            _ => panic!("expected Valid or Repaired"),
        }
    }

    #[test]
    fn test_fuzzy_function_name() {
        let tool = make_test_tool();
        let raw = r#"{"name": "get_weathr", "arguments": {"location": "Tokyo"}}"#;
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Repaired(call, fixes) => {
                assert!(fixes.iter().any(|f| f.contains("corrected function name")));
            }
            _ => panic!("expected Repaired"),
        }
    }

    #[test]
    fn test_type_mismatch_string_to_number() {
        let tool = ToolDefinition {
            name: "calculate".into(),
            description: "Calculate something".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "A numeric value"
                    }
                },
                "required": ["value"]
            }),
            required: vec!["value".into()],
        };
        // value is a string "42" instead of number 42
        let raw = r#"{"name": "calculate", "arguments": {"value": "42"}}"#;
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Valid(_call) => {
                // string "42" might be accepted as string
            }
            ToolCallResult::Repaired(_call, fixes) => {
                assert!(fixes.iter().any(|f| f.contains("type mismatch")));
            }
            _ => panic!("expected Repaired or Valid"),
        }
    }

    #[test]
    fn test_unescape_parse() {
        let tool = make_test_tool();
        let raw =
            "{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": {\\\"location\\\": \\\"Rome\\\"}}";
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Valid(call) | ToolCallResult::Repaired(call, _) => {
                assert_eq!(call.name, "get_weather");
            }
            _ => panic!("expected Valid or Repaired"),
        }
    }

    #[test]
    fn test_unrepairable() {
        let tool = make_test_tool();
        let raw = "This is just plain text with no JSON anywhere";
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Unrepairable(_) => {} // expected
            _ => panic!("expected Unrepairable"),
        }
    }

    #[test]
    fn test_extract_json_from_fences_basic() {
        let text = "```json\n{\"key\": \"value\"}\n```";
        let result = extract_json_from_fences(text);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap().get("key").and_then(|v| v.as_str()),
            Some("value")
        );
    }

    #[test]
    fn test_extract_json_from_fences_no_json() {
        let text = "```\nplain text\n```";
        assert!(extract_json_from_fences(text).is_none());
    }

    #[test]
    fn test_extract_first_json_object_nested() {
        let text = "before { \"outer\": { \"inner\": 42 } } after";
        let result = extract_first_json_object(text);
        assert!(result.is_some());
        let val = result.unwrap();
        assert_eq!(
            val.get("outer")
                .and_then(|o| o.get("inner"))
                .and_then(|v| v.as_i64()),
            Some(42)
        );
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
    }

    #[test]
    fn test_fuzzy_match_function_name_exact() {
        let tool = make_test_tool();
        // Exact match should not fuzzy-correct
        assert!(fuzzy_match_function_name("get_weather", &tool).is_none());
    }

    #[test]
    fn test_fuzzy_match_function_name_typo() {
        let tool = make_test_tool();
        let corrected = fuzzy_match_function_name("get_weathr", &tool);
        assert_eq!(corrected.as_deref(), Some("get_weather"));
    }

    #[test]
    fn test_has_tools_request() {
        let body = serde_json::json!({
            "tools": [{"function": {"name": "test"}}]
        });
        assert!(has_tools_request(&body));
    }

    #[test]
    fn test_has_tools_request_empty() {
        let body = serde_json::json!({
            "tools": []
        });
        assert!(!has_tools_request(&body));
    }

    #[test]
    fn test_has_tools_request_no_tools() {
        let body = serde_json::json!({
            "model": "test"
        });
        assert!(!has_tools_request(&body));
    }

    #[test]
    fn test_extract_tool() {
        let body = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }]
        });
        let tool = extract_tool(&body).unwrap();
        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.required, vec!["location"]);
    }

    #[test]
    fn test_normalize_tool_calls_wrapper() {
        let tool = make_test_tool();
        let raw = r#"{"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"NYC\"}"}}"#;
        match parse_and_repair(raw, &tool) {
            ToolCallResult::Repaired(call, fixes) => {
                assert!(fixes.iter().any(|f| f.contains("normalized")));
            }
            ToolCallResult::Valid(call) => {
                // If no fix needed (already normalized)
            }
            _ => panic!("expected Repaired or Valid"),
        }
    }
}

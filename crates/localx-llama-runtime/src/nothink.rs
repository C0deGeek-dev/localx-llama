//! The no-think filter — an in-process replacement for the python sidecar.
//!
//! Pure transforms (no I/O) so they are fully unit-testable; the axum wiring that
//! streams them lives in `proxy.rs`. Four responsibilities, each carrying a
//! hard-won invariant:
//!
//! 1. Strip `<think>…</think>` from responses, **split-tag-safe** across SSE chunks.
//! 2. Substitute `[no output]` when a turn strips to empty (unterminated `<think>`).
//! 3. Strip Anthropic thinking-config keys from a request **at the root only** —
//!    never recursively, or tool payloads that happen to contain a `reasoning`
//!    field are silently corrupted.
//! 4. Merge mid-conversation system messages into the top-level `system` field,
//!    because Qwen chat templates hard-reject a system message after the start.

use serde_json::{Map, Value};

/// Substituted for an assistant turn that strips to nothing.
pub const EMPTY_AFTER_THINK: &str = "[no output]";

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
/// Longest tag is `</think>` (8); hold back 7 trailing chars so a tag split
/// across chunk boundaries is never emitted half-formed.
const HOLDBACK: usize = 7;

/// Strip every `<think>…</think>` span from a complete string.
///
/// An unterminated `<think>` drops everything from the tag onward.
pub fn strip_think(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        match rest.find(OPEN) {
            None => {
                out.push_str(rest);
                break;
            }
            Some(o) => {
                out.push_str(&rest[..o]);
                let after = &rest[o + OPEN.len()..];
                match after.find(CLOSE) {
                    None => break, // unterminated -> drop remainder
                    Some(c) => rest = &after[c + CLOSE.len()..],
                }
            }
        }
    }
    out
}

/// If a fully-stripped assistant text is blank, substitute `[no output]`.
pub fn fallback_if_empty(stripped: &str) -> String {
    if stripped.trim().is_empty() {
        EMPTY_AFTER_THINK.to_string()
    } else {
        stripped.to_string()
    }
}

/// Streaming `<think>` stripper for SSE. Feed chunks; it emits only text that is
/// safe to forward, holding back a short tail that might be a split tag.
#[derive(Debug, Default)]
pub struct ThinkStripper {
    buf: String,
    in_think: bool,
}

impl ThinkStripper {
    /// A fresh stripper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next chunk; returns the text safe to forward now.
    pub fn push(&mut self, chunk: &str) -> String {
        self.buf.push_str(chunk);
        let mut out = String::new();
        loop {
            if self.in_think {
                match self.buf.find(CLOSE) {
                    Some(j) => {
                        self.buf.drain(..j + CLOSE.len());
                        self.in_think = false;
                    }
                    None => {
                        // keep only a possible partial close tag
                        self.trim_to_holdback();
                        break;
                    }
                }
            } else {
                match self.buf.find(OPEN) {
                    Some(i) => {
                        out.push_str(&self.buf[..i]);
                        self.buf.drain(..i + OPEN.len());
                        self.in_think = true;
                    }
                    None => {
                        out.push_str(&self.emit_safe_prefix());
                        break;
                    }
                }
            }
        }
        out
    }

    /// Flush any held-back text at end of stream. An unterminated `<think>`
    /// yields nothing.
    pub fn finish(&mut self) -> String {
        if self.in_think {
            self.buf.clear();
            String::new()
        } else {
            std::mem::take(&mut self.buf)
        }
    }

    /// Emit everything except a trailing tail that could begin a tag.
    fn emit_safe_prefix(&mut self) -> String {
        let keep = self.holdback_start();
        let emitted: String = self.buf[..keep].to_string();
        self.buf.drain(..keep);
        emitted
    }

    fn trim_to_holdback(&mut self) {
        let keep = self.holdback_start();
        self.buf.drain(..keep);
    }

    /// Byte index up to which the buffer is safe to release.
    fn holdback_start(&self) -> usize {
        if self.buf.len() <= HOLDBACK {
            return 0;
        }
        let mut idx = self.buf.len() - HOLDBACK;
        while idx < self.buf.len() && !self.buf.is_char_boundary(idx) {
            idx += 1;
        }
        idx
    }
}

/// Anthropic request keys stripped from the root by default.
pub const DEFAULT_THINKING_KEYS: &[&str] = &["thinking"];

/// Remove thinking-config keys from a request object — **root only**.
pub fn strip_thinking_keys_root(body: &mut Value, keys: &[&str]) {
    if let Value::Object(map) = body {
        for k in keys {
            map.remove(*k);
        }
    }
}

fn block_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Fold any `role: system` messages in the `messages` array into the top-level
/// `system` field (Anthropic shape), preserving order. Default-on behaviour that
/// works around Qwen's "system message must be at the beginning" rejection.
pub fn merge_system_messages(body: &mut Value) {
    let Value::Object(map) = body else {
        return;
    };
    let Some(Value::Array(messages)) = map.get("messages") else {
        return;
    };

    let mut folded: Vec<String> = Vec::new();
    let mut kept: Vec<Value> = Vec::new();
    for m in messages {
        let is_system = m.get("role").and_then(Value::as_str) == Some("system");
        if is_system {
            let text = block_text(m.get("content").unwrap_or(&Value::Null));
            if !text.is_empty() {
                folded.push(text);
            }
        } else {
            kept.push(m.clone());
        }
    }

    if folded.is_empty() {
        return;
    }

    let mut system_parts: Vec<String> = Vec::new();
    if let Some(existing) = map.get("system") {
        let t = block_text(existing);
        if !t.is_empty() {
            system_parts.push(t);
        }
    }
    system_parts.extend(folded);

    map.insert(
        "system".to_string(),
        Value::String(system_parts.join("\n\n")),
    );
    map.insert("messages".to_string(), Value::Array(kept));
}

/// Apply the request-side transforms (key strip + system merge) to a JSON body.
pub fn transform_request(body: &mut Value, merge_system: bool) {
    strip_thinking_keys_root(body, DEFAULT_THINKING_KEYS);
    if merge_system {
        merge_system_messages(body);
    }
}

/// Convenience: an empty request map (used by callers building bodies).
pub fn empty_body() -> Value {
    Value::Object(Map::new())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_think_one_shot() {
        assert_eq!(strip_think("a<think>x</think>b"), "ab");
        assert_eq!(strip_think("no tags"), "no tags");
        assert_eq!(strip_think("open<think>never closed"), "open");
        assert_eq!(strip_think("<think>a</think><think>b</think>tail"), "tail");
    }

    #[test]
    fn streaming_strip_is_split_tag_safe() {
        // Feed the tags one char at a time; the stripper must not leak fragments.
        let mut s = ThinkStripper::new();
        let mut out = String::new();
        for ch in "Hello <think>secret</think> world".chars() {
            out.push_str(&s.push(&ch.to_string()));
        }
        out.push_str(&s.finish());
        assert_eq!(out, "Hello  world");
    }

    #[test]
    fn streaming_unterminated_think_drops_tail() {
        let mut s = ThinkStripper::new();
        let mut out = s.push("visible <thi");
        out.push_str(&s.push("nk>hidden forever"));
        out.push_str(&s.finish());
        assert_eq!(out, "visible ");
    }

    #[test]
    fn empty_after_think_fallback() {
        assert_eq!(
            fallback_if_empty(&strip_think("<think>all of it</think>")),
            "[no output]"
        );
        assert_eq!(fallback_if_empty("real answer"), "real answer");
    }

    #[test]
    fn thinking_keys_stripped_root_only() {
        let mut body = json!({
            "model": "x",
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "messages": [
                { "role": "user", "content": "keep this reasoning field", "reasoning": "not stripped" }
            ]
        });
        strip_thinking_keys_root(&mut body, DEFAULT_THINKING_KEYS);
        assert!(body.get("thinking").is_none());
        // a nested key of the same family must survive (not recursive).
        assert_eq!(body["messages"][0]["reasoning"], json!("not stripped"));
    }

    #[test]
    fn system_messages_merged_to_top_level() {
        let mut body = json!({
            "system": "base rules",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "system", "content": "mid-convo system" },
                { "role": "assistant", "content": "ok" }
            ]
        });
        merge_system_messages(&mut body);
        assert_eq!(body["system"], json!("base rules\n\nmid-convo system"));
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(msgs.iter().all(|m| m["role"] != json!("system")));
    }

    #[test]
    fn merge_noop_without_system_messages() {
        let mut body = json!({ "messages": [{ "role": "user", "content": "hi" }] });
        let before = body.clone();
        merge_system_messages(&mut body);
        assert_eq!(body, before);
    }
}

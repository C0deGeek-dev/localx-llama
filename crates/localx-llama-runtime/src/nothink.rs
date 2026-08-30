//! The no-think filter — an in-process replacement for the python sidecar.
//!
//! Pure transforms (no I/O) so they are fully unit-testable; the axum wiring that
//! streams them lives in `proxy.rs`. Four responsibilities, each carrying a
//! hard-won invariant:
//!
//! 1. Strip `<think>…</think>` from responses, **split-tag-safe** across SSE chunks.
//! 2. Substitute `[no output]` when a turn strips to empty.
//! 3. Strip Anthropic thinking-config keys from a request **at the root only** —
//!    never recursively, or tool payloads that happen to contain a `reasoning`
//!    field are silently corrupted.
//! 4. Merge mid-conversation system messages into the top-level `system` field,
//!    because Qwen chat templates hard-reject a system message after the start.

use std::collections::BTreeSet;

use serde_json::Value;

/// Substituted for an assistant turn that strips to nothing.
pub const EMPTY_AFTER_THINK: &str = "[no output]";

/// Whether an SSE event is a stream terminator: OpenAI's `data: [DONE]` or
/// Anthropic's `message_stop`. The proxy releases the think-stripper's held-back
/// tail in-band, just before this event, so the last characters are not stranded
/// after the marker a consumer stops reading at.
fn is_terminal_marker(event: &str) -> bool {
    for line in event.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        let Some(payload) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            return true;
        }
        if let Ok(v) = serde_json::from_str::<Value>(payload) {
            if v.get("type").and_then(Value::as_str) == Some("message_stop") {
                return true;
            }
        }
    }
    false
}

/// First index of `needle` in `haystack` (small-needle scan for SSE delimiters).
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
/// Longest tag is `</think>` (8); hold back 7 trailing chars so a tag split
/// across chunk boundaries is never emitted half-formed.
const HOLDBACK: usize = 7;

/// Safety valve for a `<think>` span that never closes. If the buffered
/// in-think span grows past this many bytes without a `</think>` arriving,
/// `push` gives up waiting, flips back to visible-text mode, and flushes the
/// whole span as ordinary text instead of holding it hostage until
/// [`ThinkStripper::finish`] — which never runs if the caller gives up on the
/// stream (kills the connection) before it ends naturally.
///
/// Tuned for two competing needs: large enough that a genuine long reasoning
/// trace is never truncated mid-thought (`keep_thinking = true` routes
/// straight to the server with real reasoning left in, and a verbose local
/// reasoning model can run several thousand tokens before closing
/// `</think>`), but small enough that a false-positive `<think>` match (the
/// bug this valve exists for) can't hold the visible reply hostage for
/// approximately the rest of the turn — the turn's hard ceiling is
/// `LocalModelMaxOutputTokens`, 16384 tokens by default (~60-90KB of text),
/// so this is deliberately well under half of that. Far larger than
/// `CLOSE.len()` (8), so there is no interaction with the `HOLDBACK`
/// split-tag math above: this only ever fires long after any plausible
/// tag-split ambiguity has resolved one way or the other.
const THINK_BAILOUT: usize = 32 * 1024;

/// Strip every `<think>…</think>` span from a complete string.
///
/// An unterminated `<think>` flushes its remainder as visible text instead of
/// dropping it — see [`ThinkStripper::finish`] for why.
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
                    None => {
                        out.push_str(after);
                        break;
                    }
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
///
/// A `<think>` span with no matching `</think>` is held in full until either
/// the close tag arrives or the stream ends — except past `THINK_BAILOUT`
/// bytes, at which point `push` gives up on it as reasoning and flushes it as
/// visible text right away, so a misclassified span can't stall the visible
/// reply for the rest of the turn if the caller gives up on the stream
/// before it ends naturally.
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
                    None if self.buf.len() > THINK_BAILOUT => {
                        // The span has run well past any plausible reasoning
                        // length without a `</think>` in sight — most likely
                        // a false-positive `<think>` match on ordinary text
                        // (or, at worst, a real one the model never closed).
                        // Give up waiting: the whole buffered span becomes
                        // ordinary visible text (the opening tag itself was
                        // already drained above when it was found, so `buf`
                        // holds exactly the text that followed it), and
                        // normal streaming resumes for anything after — this
                        // `continue` re-enters the loop with `in_think` now
                        // false, so a fresh `<think>` later in the stream is
                        // still detected correctly.
                        out.push_str(&self.buf);
                        self.buf.clear();
                        self.in_think = false;
                        continue;
                    }
                    None => {
                        // Nothing is emitted from this branch, so — unlike the
                        // visible-text branch below — there is no holdback
                        // hazard here: retain the WHOLE buffered span. It is
                        // dropped in one shot once `</think>` is found, or
                        // recovered by `finish()` as visible fallback text if
                        // the tag never closes (or, past `THINK_BAILOUT`
                        // bytes, flushed immediately above instead of waiting
                        // for either). Trimming this to a short tail on every
                        // push (the old behaviour) silently threw away almost
                        // all of a long unterminated `<think>` well before
                        // end-of-stream, so `finish()` alone could never
                        // recover it.
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

    /// Flush any held-back text at end of stream. An unterminated `<think>` —
    /// the model's generation ended before a matching `</think>` ever
    /// arrived — is no longer silently discarded: whatever is buffered (the
    /// content after the stray/incomplete opening tag) is flushed as
    /// ordinary visible text, exactly like the no-tag case. Losing a
    /// legitimate answer that happened to follow a stray or malformed
    /// `<think>` is worse than the alternative: real hidden reasoning that
    /// never closes by end of turn leaking into the visible reply as raw
    /// text. In practice this only ever sees a span shorter than
    /// `THINK_BAILOUT`: anything longer was already flushed by `push` itself,
    /// so a stream that's abandoned mid-turn (connection dropped) no longer
    /// strands buffered content unrecovered either way.
    pub fn finish(&mut self) -> String {
        self.in_think = false;
        std::mem::take(&mut self.buf)
    }

    /// Emit everything except a trailing tail that could begin a tag.
    fn emit_safe_prefix(&mut self) -> String {
        let keep = self.holdback_start();
        let emitted: String = self.buf[..keep].to_string();
        self.buf.drain(..keep);
        emitted
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

/// Strip `<think>` spans from a **non-streaming** JSON response body, in place
/// on the assistant text fields, and substitute `[no output]` when a turn strips
/// to nothing. Handles both the Anthropic (`content[].text`) and OpenAI
/// (`choices[].message.content`) response shapes. A body that is not one of those
/// shapes falls back to a whole-string strip (harmless for a model list / props).
pub fn strip_think_json_response(body: &[u8]) -> Vec<u8> {
    let Ok(mut v) = serde_json::from_slice::<Value>(body) else {
        // Not JSON — strip the raw text (covers a plain-text completion).
        return strip_think(&String::from_utf8_lossy(body)).into_bytes();
    };
    let mut touched = false;
    // Anthropic: { "content": [ { "type":"text", "text":"..." }, ... ] }
    if let Some(Value::Array(blocks)) = v.get_mut("content") {
        for b in blocks.iter_mut() {
            if b.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(Value::String(t)) = b.get_mut("text") {
                    *t = fallback_if_empty(&strip_think(t));
                    touched = true;
                }
            }
        }
    }
    // OpenAI: { "choices": [ { "message": { "content":"..." } }, ... ] }
    if let Some(Value::Array(choices)) = v.get_mut("choices") {
        for c in choices.iter_mut() {
            if let Some(Value::String(t)) = c.pointer_mut("/message/content") {
                *t = fallback_if_empty(&strip_think(t));
                touched = true;
            }
        }
    }
    if !touched {
        return body.to_vec();
    }
    serde_json::to_vec(&v).unwrap_or_else(|_| body.to_vec())
}

/// Stateful `<think>` filter for a **streaming** SSE response. Feeds raw response
/// bytes (arbitrary chunk boundaries), splits complete SSE events, and rewrites
/// the assistant text *inside* each event's `data:` JSON delta through a shared
/// [`ThinkStripper`] — so a `<think>` span crossing two `data:` events is stripped
/// without ever corrupting the SSE framing between them. Non-delta events, `:`
/// comments, and `data: [DONE]` pass through untouched.
#[derive(Debug, Default)]
pub struct SseThinkFilter {
    /// Bytes not yet forming a complete event (`\n\n`-terminated). Kept as bytes
    /// so a chunk boundary splitting a multibyte UTF-8 char is never lossily
    /// decoded — complete events end at an ASCII `\n\n`, so each is valid UTF-8.
    pending: Vec<u8>,
    stripper: ThinkStripper,
    shape: Option<DeltaShape>,
    /// Anthropic content blocks whose start has reached the downstream client.
    /// Some llama.cpp-compatible servers omit `content_block_start`; tracking
    /// indices lets the proxy synthesize that missing lifecycle event exactly
    /// once without duplicating starts from conforming servers.
    anthropic_open_blocks: BTreeSet<u64>,
}

#[derive(Debug, Clone, Copy)]
enum DeltaShape {
    /// OpenAI: `choices[0].delta.content`.
    OpenAi,
    /// Anthropic: `delta.text` (type `text_delta`).
    Anthropic { index: u64 },
}

impl SseThinkFilter {
    /// A fresh filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next raw chunk; returns transformed bytes safe to forward now.
    pub fn push(&mut self, chunk: impl AsRef<[u8]>) -> String {
        self.pending.extend_from_slice(chunk.as_ref());
        let mut out = String::new();
        // An SSE event ends at a blank line ("\n\n"). Process every complete one.
        while let Some(idx) = find_subslice(&self.pending, b"\n\n") {
            let event = String::from_utf8_lossy(&self.pending[..idx + 2]).into_owned();
            self.pending.drain(..idx + 2);
            out.push_str(&self.transform_event(&event));
        }
        out
    }

    /// Flush at end of stream: emit any held-back tail as a final delta in the
    /// detected shape, plus any trailing partial event verbatim.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        let tail = self.stripper.finish();
        if !tail.is_empty() {
            if let Some(frame) = self.shape.map(|s| s.data_frame(&tail)) {
                out.push_str(&frame);
            }
        }
        if !self.pending.is_empty() {
            out.push_str(&String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
        out
    }

    fn transform_event(&mut self, event: &str) -> String {
        let mut lines: Vec<String> = Vec::new();
        let event_json = event.lines().find_map(|line| {
            line.strip_prefix("data:")
                .and_then(|payload| serde_json::from_str::<Value>(payload.trim()).ok())
        });
        let event_type = event_json
            .as_ref()
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str);
        let event_index = event_json
            .as_ref()
            .and_then(|value| value.get("index"))
            .and_then(Value::as_u64)
            .unwrap_or(0);

        if event_type == Some("content_block_start") {
            self.anthropic_open_blocks.insert(event_index);
        } else if event_type == Some("content_block_delta")
            && event_json
                .as_ref()
                .and_then(|value| value.pointer("/delta/type"))
                .and_then(Value::as_str)
                == Some("text_delta")
            && self.anthropic_open_blocks.insert(event_index)
        {
            lines.push(anthropic_text_start_frame(event_index));
        }

        // Release the stripper's held-back tail (up to `HOLDBACK` bytes withheld
        // for split-tag safety) as an in-band content frame *before* a stream
        // terminator (`[DONE]` / `message_stop`) or Anthropic's block stop. A
        // consumer ends a content block at `content_block_stop`, so emitting a
        // held tail later would violate the lifecycle even if it still precedes
        // `message_stop`.
        if is_terminal_marker(event) || event_type == Some("content_block_stop") {
            let tail = self.stripper.finish();
            if !tail.is_empty() {
                if let Some(frame) = self.shape.map(|s| s.data_frame(&tail)) {
                    lines.push(frame);
                }
            }
        }
        for line in event.split_inclusive('\n') {
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if let Some(payload) = trimmed.strip_prefix("data:") {
                let payload = payload.trim_start();
                if payload == "[DONE]" || payload.is_empty() {
                    lines.push(line.to_string());
                    continue;
                }
                if let Some(rewritten) = self.rewrite_data(payload) {
                    lines.push(format!("data: {rewritten}\n"));
                } else {
                    lines.push(line.to_string());
                }
            } else {
                lines.push(line.to_string());
            }
        }
        if event_type == Some("content_block_stop") {
            self.anthropic_open_blocks.remove(&event_index);
            if matches!(self.shape, Some(DeltaShape::Anthropic { index }) if index == event_index) {
                self.shape = None;
            }
        }
        lines.concat()
    }

    /// Rewrite one `data:` JSON payload's delta text through the stripper.
    /// Returns `None` when the payload isn't a recognized text delta (pass through).
    fn rewrite_data(&mut self, payload: &str) -> Option<String> {
        let mut v: Value = serde_json::from_str(payload).ok()?;
        // OpenAI streaming: choices[].delta.content
        if let Some(Value::String(t)) = v.pointer_mut("/choices/0/delta/content") {
            self.shape = Some(DeltaShape::OpenAi);
            let stripped = self.stripper.push(t);
            *t = stripped;
            return serde_json::to_string(&v).ok();
        }
        // Anthropic streaming: delta.text (content_block_delta / text_delta)
        if v.get("delta")
            .and_then(|d| d.get("type"))
            .and_then(Value::as_str)
            == Some("text_delta")
        {
            let index = v.get("index").and_then(Value::as_u64).unwrap_or(0);
            if let Some(Value::String(t)) = v.pointer_mut("/delta/text") {
                self.shape = Some(DeltaShape::Anthropic { index });
                let stripped = self.stripper.push(t);
                *t = stripped;
                return serde_json::to_string(&v).ok();
            }
        }
        None
    }
}

impl DeltaShape {
    /// A minimal terminal `data:` frame carrying `text`, in this shape.
    fn data_frame(self, text: &str) -> String {
        let v = match self {
            DeltaShape::OpenAi => {
                serde_json::json!({ "choices": [ { "delta": { "content": text } } ] })
            }
            DeltaShape::Anthropic { index } => {
                serde_json::json!({ "type": "content_block_delta", "index": index, "delta": { "type": "text_delta", "text": text } })
            }
        };
        match self {
            DeltaShape::OpenAi => format!("data: {v}\n\n"),
            DeltaShape::Anthropic { .. } => {
                format!("event: content_block_delta\ndata: {v}\n\n")
            }
        }
    }
}

fn anthropic_text_start_frame(index: u64) -> String {
    let value = serde_json::json!({
        "type": "content_block_start",
        "index": index,
        "content_block": { "type": "text", "text": "" }
    });
    format!("event: content_block_start\ndata: {value}\n\n")
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_think_one_shot() {
        assert_eq!(strip_think("a<think>x</think>b"), "ab");
        assert_eq!(strip_think("no tags"), "no tags");
        assert_eq!(strip_think("open<think>never closed"), "opennever closed");
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
    fn streaming_unterminated_think_flushes_tail_as_visible_text() {
        let mut s = ThinkStripper::new();
        let mut out = s.push("visible <thi");
        out.push_str(&s.push("nk>hidden forever"));
        out.push_str(&s.finish());
        assert_eq!(out, "visible hidden forever");
    }

    #[test]
    fn streaming_unterminated_think_recovers_full_span_across_many_chunks() {
        // Regression test for the deeper bug: the old `push()` trimmed the
        // in-think buffer down to a short tail on EVERY chunk, not just at
        // `finish()` — so a `finish()`-only fix would still lose everything
        // but the last few bytes of a long unterminated span. Feed it one
        // character at a time and require the WHOLE thing back.
        let mut s = ThinkStripper::new();
        let mut out = s.push("answer <think>");
        for ch in "this is a long unterminated reasoning span".chars() {
            out.push_str(&s.push(&ch.to_string()));
        }
        out.push_str(&s.finish());
        assert_eq!(out, "answer this is a long unterminated reasoning span");
    }

    #[test]
    fn think_span_under_bailout_is_still_buffered_until_finish() {
        // Unchanged behavior: a genuinely long but still-plausible reasoning
        // span, one byte short of THINK_BAILOUT, is not affected by the new
        // safety valve — still held in full and only recovered by finish().
        let mut s = ThinkStripper::new();
        let long = "r".repeat(THINK_BAILOUT - 1);
        let mut out = s.push("answer <think>");
        out.push_str(&s.push(&long));
        assert_eq!(out, "answer ", "must not flush before finish(): {out}");
        out.push_str(&s.finish());
        assert_eq!(out, format!("answer {long}"));
    }

    #[test]
    fn think_span_over_bailout_flushes_immediately_as_visible_text() {
        let mut s = ThinkStripper::new();
        let mut out = s.push("answer <think>");
        let long = "r".repeat(THINK_BAILOUT + 1);
        out.push_str(&s.push(&long));
        // Flushed mid-stream, not held until finish().
        assert_eq!(out, format!("answer {long}"));
    }

    #[test]
    fn text_after_bailout_streams_normally_not_held_as_thinking() {
        let mut s = ThinkStripper::new();
        s.push("answer <think>");
        s.push(&"r".repeat(THINK_BAILOUT + 1));
        // Resumed in ordinary visible-text mode: this delta streams through
        // the normal small HOLDBACK-byte holdback, not the "withhold
        // everything until </think> or finish()" in-think behavior — proving
        // `in_think` was reset to `false`, not just flushed once.
        let out = s.push(" more visible text after bailout");
        assert!(
            out.starts_with(" more visible text after"),
            "text after bail-out did not stream normally: {out:?}"
        );
        let mut all = out;
        all.push_str(&s.finish());
        assert_eq!(all, " more visible text after bailout");
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

    #[test]
    fn json_response_strip_anthropic_shape() {
        let body =
            br#"{"content":[{"type":"text","text":"a<think>x</think>b"}],"role":"assistant"}"#;
        let out = strip_think_json_response(body);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["content"][0]["text"], json!("ab"));
    }

    #[test]
    fn json_response_strip_openai_shape() {
        let body = br#"{"choices":[{"message":{"content":"<think>all</think>hi"}}]}"#;
        let out = strip_think_json_response(body);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], json!("hi"));
    }

    #[test]
    fn json_response_all_think_becomes_no_output() {
        let body = br#"{"content":[{"type":"text","text":"<think>only</think>"}]}"#;
        let out = strip_think_json_response(body);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["content"][0]["text"], json!("[no output]"));
    }

    #[test]
    fn json_response_passes_through_a_model_list() {
        // Not a chat shape (no content/choices text) — must be byte-identical.
        let body = br#"{"models":[{"id":"m1"},{"id":"m2"}]}"#;
        assert_eq!(strip_think_json_response(body), body.to_vec());
    }

    #[test]
    fn sse_strips_think_within_one_openai_delta() {
        let mut f = SseThinkFilter::new();
        let mut out =
            f.push("data: {\"choices\":[{\"delta\":{\"content\":\"a<think>x</think>b\"}}]}\n\n");
        out.push_str(&f.finish());
        // Think content gone; visible "a" and "b" both survive (the trailing "b"
        // is held back for split-tag safety and flushed at finish as its own frame).
        assert!(
            !out.contains("<think>") && !out.contains('x'),
            "leaked: {out}"
        );
        let visible: String = out
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|p| serde_json::from_str::<Value>(p).ok())
            .filter_map(|v| {
                v.pointer("/choices/0/delta/content")
                    .and_then(|c| c.as_str().map(String::from))
            })
            .collect();
        assert_eq!(visible, "ab", "got: {out}");
        assert!(out.ends_with("\n\n"));
    }

    #[test]
    fn sse_strips_think_span_crossing_two_events_without_corrupting_framing() {
        // The <think> opens in event 1's delta and closes in event 3's delta.
        let mut f = SseThinkFilter::new();
        let mut out = String::new();
        out.push_str(
            &f.push("data: {\"choices\":[{\"delta\":{\"content\":\"keep <think>\"}}]}\n\n"),
        );
        out.push_str(&f.push("data: {\"choices\":[{\"delta\":{\"content\":\"secret \"}}]}\n\n"));
        out.push_str(
            &f.push("data: {\"choices\":[{\"delta\":{\"content\":\"more</think> done\"}}]}\n\n"),
        );
        out.push_str(&f.push("data: [DONE]\n\n"));
        out.push_str(&f.finish());
        // No think content leaks; SSE framing (one event per data line) intact.
        assert!(!out.contains("secret"), "leaked think: {out}");
        assert!(
            !out.contains("<think>") && !out.contains("</think>"),
            "leaked tag: {out}"
        );
        assert!(out.contains("[DONE]"));
        // Every data line is still a standalone \n\n-terminated event.
        assert_eq!(out.matches("data: ").count(), out.matches("\n\n").count());
        // The visible words survive.
        let visible: String = out.matches(|_c| true).collect();
        assert!(
            visible.contains("keep") && visible.contains("done"),
            "lost text: {out}"
        );
    }

    /// Visible text carried by content frames that appear *before* the first
    /// stream terminator (`[DONE]` / `message_stop`) — what a consumer that ends
    /// the stream at the terminator actually receives.
    fn visible_before_terminator(sse: &str) -> String {
        let mut text = String::new();
        for event in sse.split_inclusive("\n\n") {
            if is_terminal_marker(event) {
                break;
            }
            for line in event.split_inclusive('\n') {
                let trimmed = line.trim_end_matches(['\r', '\n']);
                let Some(payload) = trimmed.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload == "[DONE]" || payload.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(payload) {
                    if let Some(t) = v
                        .pointer("/choices/0/delta/content")
                        .or_else(|| v.pointer("/delta/text"))
                        .and_then(Value::as_str)
                    {
                        text.push_str(t);
                    }
                }
            }
        }
        text
    }

    #[test]
    fn openai_tail_is_flushed_before_done_not_after() {
        // The last ≤HOLDBACK chars of visible text ("or you?") were held back for
        // split-tag safety; they must reach the client *before* [DONE], or a
        // consumer that stops at [DONE] loses them (the "response capped off" bug).
        let mut f = SseThinkFilter::new();
        let mut out = f
            .push("data: {\"choices\":[{\"delta\":{\"content\":\"What can I do for you?\"}}]}\n\n");
        out.push_str(
            &f.push("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"),
        );
        out.push_str(&f.push("data: [DONE]\n\n"));
        out.push_str(&f.finish());
        assert_eq!(
            visible_before_terminator(&out),
            "What can I do for you?",
            "tail stranded after [DONE]: {out}"
        );
    }

    #[test]
    fn anthropic_tail_is_flushed_before_message_stop_not_after() {
        let mut f = SseThinkFilter::new();
        let mut out = f.push(
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Is there something specific\"}}\n\n",
        );
        out.push_str(&f.push(
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        ));
        out.push_str(&f.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        out.push_str(&f.finish());
        assert_eq!(
            visible_before_terminator(&out),
            "Is there something specific",
            "tail stranded after message_stop: {out}"
        );
        let start_at = out.find("content_block_start").expect("start synthesized");
        let tail_at = out.rfind("pecific").expect("held tail emitted");
        let block_stop_at = out.find("content_block_stop").expect("block stop present");
        assert!(
            start_at < tail_at && tail_at < block_stop_at,
            "bad order: {out}"
        );
        assert_eq!(
            out.matches(r#""type":"content_block_start""#).count(),
            1,
            "start must be synthesized exactly once: {out}"
        );
    }

    #[test]
    fn prism_anthropic_stream_is_normalized_to_a_complete_block_lifecycle() {
        // Prism's llama.cpp fork currently omits content_block_start. The
        // think filter also holds a short visible tail, so that tail must be
        // released before (not after) the upstream content_block_stop.
        let upstream = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"content\":[]}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"OK\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";
        let mut filter = SseThinkFilter::new();
        let out = filter.push(upstream) + &filter.finish();

        let start_at = out.find("content_block_start").expect("start synthesized");
        let text_at = out.rfind(r#""text":"OK""#).expect("text retained");
        let block_stop_at = out.find("content_block_stop").expect("block stop retained");
        let message_delta_at = out.find("message_delta").expect("message delta retained");
        let message_stop_at = out.find("message_stop").expect("message stop retained");
        assert!(
            start_at < text_at
                && text_at < block_stop_at
                && block_stop_at < message_delta_at
                && message_delta_at < message_stop_at,
            "invalid Anthropic lifecycle: {out}"
        );
        assert_eq!(
            out.matches(r#""type":"content_block_start""#).count(),
            1,
            "exactly one start expected: {out}"
        );
        assert!(
            out[text_at..block_stop_at].contains(r#""index":0"#),
            "synthetic tail must retain its block index: {out}"
        );
    }

    #[test]
    fn valid_anthropic_start_is_not_duplicated() {
        let mut filter = SseThinkFilter::new();
        let upstream = "event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"text_delta\",\"text\":\"complete\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":3}\n\n";
        let out = filter.push(upstream) + &filter.finish();

        assert_eq!(
            out.matches(r#""type":"content_block_start""#).count(),
            1,
            "valid start duplicated: {out}"
        );
        assert_eq!(visible_before_terminator(&out), "complete");
        let tail_at = out.rfind("omplete").expect("tail retained");
        let stop_at = out.find("content_block_stop").expect("stop retained");
        assert!(tail_at < stop_at, "tail emitted after block stop: {out}");
    }

    #[test]
    fn clean_eof_does_not_invent_an_anthropic_completion_marker() {
        let mut filter = SseThinkFilter::new();
        let mut out = filter.push(
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
        );
        out.push_str(&filter.finish());

        assert!(out.contains("partial"));
        assert!(!out.contains("message_stop"), "false completion: {out}");
        assert!(
            !out.contains("content_block_stop"),
            "false block stop: {out}"
        );
    }

    #[test]
    fn tail_after_terminator_is_not_double_emitted() {
        // The tail is flushed once (before the terminator); `finish()` must not
        // emit it a second time.
        let mut f = SseThinkFilter::new();
        let mut out = f.push("data: {\"choices\":[{\"delta\":{\"content\":\"abcdefghij\"}}]}\n\n");
        out.push_str(&f.push("data: [DONE]\n\n"));
        out.push_str(&f.finish());
        assert_eq!(
            out.matches("abcdefghij").count(),
            0,
            "no full copy expected"
        );
        // The held-back tail "defghij" appears exactly once, before [DONE].
        assert_eq!(
            out.matches("defghij").count(),
            1,
            "tail emitted once: {out}"
        );
        let done_at = out.find("[DONE]").expect("[DONE] present");
        let tail_at = out.find("defghij").expect("tail present");
        assert!(tail_at < done_at, "tail must precede [DONE]: {out}");
        // And the full text is recoverable in order before the terminator.
        assert_eq!(visible_before_terminator(&out), "abcdefghij");
    }

    #[test]
    fn sse_passes_through_done_and_non_delta_events() {
        let mut f = SseThinkFilter::new();
        let out = f.push("event: message_start\ndata: {\"type\":\"message_start\"}\n\n")
            + &f.push("data: [DONE]\n\n")
            + &f.finish();
        assert!(out.contains("message_start"));
        assert!(out.contains("[DONE]"));
    }

    #[test]
    fn sse_strips_anthropic_text_delta() {
        let mut f = SseThinkFilter::new();
        let out = f.push(
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"<think>h</think>hi\"}}\n\n",
        ) + &f.finish();
        assert!(out.contains(r#""text":"hi""#), "got: {out}");
        assert!(out.contains("content_block_delta"));
    }
}

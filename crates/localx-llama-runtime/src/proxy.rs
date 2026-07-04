//! The in-process no-think proxy (axum) — the replacement for the python sidecar.
//!
//! Composes the tested [`crate::nothink`] transforms into a real forwarding
//! proxy. The upstream is injected via the [`Upstream`] trait so the forward path
//! is unit-testable with a mock (no live server needed); [`ReqwestUpstream`] is
//! the production impl.
//!
//! The proxy is a faithful HTTP forwarder: it preserves method, path, query, and
//! headers in both directions, so `GET` endpoints (`/props`, `/v1/models`,
//! `/tokenize`) work through it as well as the `POST` chat path. `<think>`
//! stripping is content-type aware: streaming (`text/event-stream`) responses are
//! rewritten per SSE delta by [`crate::nothink::SseThinkFilter`] (framing-safe),
//! non-streaming chat JSON by [`crate::nothink::strip_think_json_response`], and
//! everything else (model lists, `/props`) passes through untouched.

use std::pin::Pin;
use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{Json, Response},
    routing::get,
    Router,
};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use http::{HeaderMap, Method};
use serde_json::{json, Value};

use crate::nothink::{strip_think_json_response, transform_request, SseThinkFilter};

/// A streamed response body: chunks of bytes from the upstream, errors surfaced.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, ProxyError>> + Send>>;

/// Hop-by-hop headers (RFC 7230 §6.1) plus length/host headers that must not be
/// copied verbatim across the proxy — the body may be re-encoded and reqwest sets
/// host/length itself.
const SKIP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
];

fn copy_headers(src: &HeaderMap, dst: &mut HeaderMap) {
    for (name, value) in src {
        if SKIP_HEADERS.contains(&name.as_str()) {
            continue;
        }
        dst.append(name.clone(), value.clone());
    }
}

/// Max request body accepted before returning 413 (mirrors the proxy ceiling).
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Errors from forwarding to the upstream llama-server.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// The upstream request failed.
    #[error("upstream request failed: {0}")]
    Upstream(String),
    /// The proxy could not bind or serve its listen socket.
    #[error("proxy listen failed: {0}")]
    Listen(String),
}

/// A request to forward upstream, preserving method/path+query/headers.
pub struct ForwardRequest {
    /// HTTP method (GET, POST, …).
    pub method: Method,
    /// Path and query string, e.g. `/v1/models` or `/props?verbose=1`.
    pub path_and_query: String,
    /// Request headers (hop-by-hop/length/host are dropped by the forwarder).
    pub headers: HeaderMap,
    /// Request body (already request-transformed for chat bodies).
    pub body: Vec<u8>,
}

/// An upstream response with status, headers, and a streamed body.
pub struct ForwardResponse {
    /// HTTP status code from the upstream.
    pub status: u16,
    /// Response headers from the upstream.
    pub headers: HeaderMap,
    /// Streamed response body.
    pub body: ByteStream,
}

/// The upstream the proxy forwards to. Abstracted for testability.
pub trait Upstream: Send + Sync + 'static {
    /// Forward a request upstream and return the streamed response.
    fn forward(
        &self,
        req: ForwardRequest,
    ) -> impl std::future::Future<Output = Result<ForwardResponse, ProxyError>> + Send;
}

/// Proxy configuration.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Upstream host reported by `/health`.
    pub target_host: String,
    /// Upstream port reported by `/health`.
    pub target_port: u16,
    /// Fold mid-conversation system messages into the top-level `system`.
    pub merge_system: bool,
    /// When set, every non-`/health` request must carry this bearer token —
    /// the LAN-gateway posture (a public bind without a key is refused by
    /// the launcher's serve guard before the proxy ever starts).
    pub api_key: Option<String>,
}

/// Shared proxy state.
#[derive(Debug)]
pub struct ProxyState<U: Upstream> {
    /// The injected upstream.
    pub upstream: U,
    /// Proxy config.
    pub config: ProxyConfig,
}

/// Build the proxy router: `GET /health` plus a catch-all forwarding fallback.
pub fn router<U: Upstream>(state: Arc<ProxyState<U>>) -> Router {
    Router::new()
        .route("/health", get(health::<U>))
        .fallback(proxy_handler::<U>)
        .with_state(state)
}

async fn health<U: Upstream>(State(state): State<Arc<ProxyState<U>>>) -> Json<Value> {
    Json(json!({
        "target_host": state.config.target_host,
        "target_port": state.config.target_port,
        "status": "ok",
    }))
}

async fn proxy_handler<U: Upstream>(
    State(state): State<Arc<ProxyState<U>>>,
    req: axum::extract::Request,
) -> Response {
    if let Some(expected) = &state.config.api_key {
        let authorized = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|token| token == expected)
            || req
                .headers()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|token| token == expected);
        if !authorized {
            return status_body(StatusCode::UNAUTHORIZED, "missing or wrong api key");
        }
    }
    let method = req.method().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let headers = req.headers().clone();

    let bytes = match axum::body::to_bytes(req.into_body(), MAX_BODY_BYTES).await {
        Ok(b) => b,
        Err(_) => return status_body(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"),
    };

    // Request-side transforms apply only to a chat body (a JSON object with
    // `messages`); GET probes and other endpoints forward their body unchanged.
    let (out_body, is_chat) = match serde_json::from_slice::<Value>(&bytes) {
        Ok(mut v) => {
            let chat = v.get("messages").is_some();
            if chat {
                transform_request(&mut v, state.config.merge_system);
                (
                    serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec()),
                    true,
                )
            } else {
                (bytes.to_vec(), false)
            }
        }
        Err(_) => (bytes.to_vec(), false),
    };

    let fwd = ForwardRequest {
        method,
        path_and_query,
        headers,
        body: out_body,
    };
    match state.upstream.forward(fwd).await {
        Ok(resp) => build_response(resp, is_chat).await,
        Err(e) => status_body(StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

/// Is this an SSE (streaming) response?
fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"))
}

/// Build the client-facing response from the upstream one: copy status + safe
/// headers, then strip `<think>` per content type (SSE per-delta; non-streaming
/// chat JSON in place; everything else untouched).
async fn build_response(resp: ForwardResponse, is_chat: bool) -> Response {
    let status = StatusCode::from_u16(resp.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let sse = is_event_stream(&resp.headers);

    let mut builder = Response::builder().status(status);
    if let Some(hs) = builder.headers_mut() {
        copy_headers(&resp.headers, hs);
    }

    if sse {
        // Stream the upstream through the framing-safe SSE think filter.
        let body = Body::from_stream(sse_think_stream(resp.body));
        return builder.body(body).unwrap_or_else(|_| {
            status_body(StatusCode::INTERNAL_SERVER_ERROR, "response build failed")
        });
    }

    // Non-streaming: buffer, then strip only a chat JSON body (pass /models,
    // /props, and other non-chat bodies through unchanged).
    let buffered = match collect_stream(resp.body).await {
        Ok(b) => b,
        Err(e) => return status_body(StatusCode::BAD_GATEWAY, &e.to_string()),
    };
    let out = if is_chat {
        strip_think_json_response(&buffered)
    } else {
        buffered
    };
    builder
        .body(Body::from(out))
        .unwrap_or_else(|_| status_body(StatusCode::INTERNAL_SERVER_ERROR, "response build failed"))
}

/// Drain a byte stream into a buffer (non-streaming responses are small — model
/// lists, `/props`, a single completion object).
async fn collect_stream(mut body: ByteStream) -> Result<Vec<u8>, ProxyError> {
    let mut buf = Vec::new();
    while let Some(chunk) = body.next().await {
        buf.extend_from_slice(&chunk?);
    }
    Ok(buf)
}

/// Wrap an upstream SSE byte stream, rewriting each event's delta text through a
/// single [`SseThinkFilter`] and flushing its held-back tail at end of stream.
///
/// **Cross-repo contract (StreamTruncated).** A mid-stream upstream error is
/// surfaced as a terminal `Err` *after* the held-back tail is flushed. Into
/// [`Body::from_stream`] that `Err` aborts the client response body, so a
/// consumer sees a truncated (retryable) stream — never a clean EOF. Consumers
/// (e.g. LocalPilot's `ProviderError::StreamTruncated` retry) depend on this: a
/// mid-stream drop MUST NOT be turned into a graceful end. This is pinned by
/// `build_response_aborts_the_client_body_on_a_mid_stream_error` at the HTTP
/// boundary, not only by the combinator test. Note the flushed tail from
/// [`SseThinkFilter::finish`] may be an incomplete trailing SSE frame when the
/// drop lands mid-event — harmless because the stream is already being torn down
/// as truncated.
fn sse_think_stream(inner: ByteStream) -> impl Stream<Item = Result<Bytes, ProxyError>> + Send {
    // Drives the fold: normal streaming, a one-shot error emission (after the
    // held-back tail has been flushed), or terminated.
    enum Phase {
        Streaming,
        Erroring(ProxyError),
        Done,
    }
    futures::stream::unfold(
        (inner, SseThinkFilter::new(), Phase::Streaming),
        |(mut inner, mut filter, phase)| async move {
            match phase {
                Phase::Done => None,
                // The upstream error, emitted on its own poll so the flushed tail
                // (yielded on the previous poll) reaches the client first.
                Phase::Erroring(e) => Some((Err(e), (inner, filter, Phase::Done))),
                Phase::Streaming => match inner.next().await {
                    Some(Ok(chunk)) => {
                        let out = filter.push(&chunk);
                        Some((Ok(Bytes::from(out)), (inner, filter, Phase::Streaming)))
                    }
                    // Upstream dropped mid-stream. Flush the stripper's held-back
                    // tail (up to `HOLDBACK` visible chars withheld for split-tag
                    // safety) *before* surfacing the error — otherwise an abort
                    // silently truncates the last characters of text already
                    // generated, on top of losing the rest. The clean-EOF (`None`)
                    // and terminator paths already flush; this is the error path.
                    Some(Err(e)) => {
                        let tail = filter.finish();
                        if tail.is_empty() {
                            Some((Err(e), (inner, filter, Phase::Done)))
                        } else {
                            Some((Ok(Bytes::from(tail)), (inner, filter, Phase::Erroring(e))))
                        }
                    }
                    None => {
                        let tail = filter.finish();
                        Some((Ok(Bytes::from(tail)), (inner, filter, Phase::Done)))
                    }
                },
            }
        },
    )
}

fn status_body(status: StatusCode, msg: &str) -> Response {
    let mut r = Response::new(Body::from(msg.to_string()));
    *r.status_mut() = status;
    r
}

/// Bind loopback `listen_port` and run the no-think proxy until the process
/// exits, forwarding to the upstream named in `config`.
///
/// # Errors
/// [`ProxyError::Listen`] when the socket cannot be bound or serving fails.
pub async fn serve_proxy(listen_port: u16, config: ProxyConfig) -> Result<(), ProxyError> {
    serve_proxy_on("127.0.0.1", listen_port, config).await
}

/// [`serve_proxy`] with an explicit listen host (`0.0.0.0` for a LAN
/// gateway; pair a public bind with `api_key`).
///
/// # Errors
/// [`ProxyError::Listen`] when the socket cannot be bound or serving fails.
pub async fn serve_proxy_on(
    listen_host: &str,
    listen_port: u16,
    config: ProxyConfig,
) -> Result<(), ProxyError> {
    let upstream = ReqwestUpstream::new(format!(
        "http://{}:{}",
        config.target_host, config.target_port
    ));
    let state = Arc::new(ProxyState { upstream, config });
    let listener = tokio::net::TcpListener::bind((listen_host, listen_port))
        .await
        .map_err(|e| ProxyError::Listen(format!("{listen_host}:{listen_port}: {e}")))?;
    axum::serve(listener, router(state))
        .await
        .map_err(|e| ProxyError::Listen(e.to_string()))
}

/// Production upstream over reqwest (loopback llama-server).
#[derive(Debug, Clone)]
pub struct ReqwestUpstream {
    base_url: String,
    client: reqwest::Client,
}

impl ReqwestUpstream {
    /// A new upstream targeting `base_url` (e.g. `http://127.0.0.1:8080`).
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
        }
    }
}

impl Upstream for ReqwestUpstream {
    async fn forward(&self, req: ForwardRequest) -> Result<ForwardResponse, ProxyError> {
        let url = format!("{}{}", self.base_url, req.path_and_query);
        // Copy the method and headers (minus hop-by-hop/length/host) so GET
        // probes, query strings, and content negotiation reach the upstream.
        let mut fwd_headers = HeaderMap::new();
        copy_headers(&req.headers, &mut fwd_headers);
        let resp = self
            .client
            .request(req.method, &url)
            .headers(fwd_headers)
            .body(req.body)
            .send()
            .await
            .map_err(|e| ProxyError::Upstream(e.to_string()))?;
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let body: ByteStream = Box::pin(
            resp.bytes_stream()
                .map(|r| r.map_err(|e| ProxyError::Upstream(e.to_string()))),
        );
        Ok(ForwardResponse {
            status,
            headers,
            body,
        })
    }
}

/// Build a single-chunk [`ByteStream`] from a buffer (test helper).
#[cfg(test)]
fn once_stream(bytes: Vec<u8>) -> ByteStream {
    Box::pin(futures::stream::once(async move { Ok(Bytes::from(bytes)) }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::http::Request;
    use std::sync::Mutex;
    use tower::ServiceExt;

    /// A mock upstream that records the forwarded request and returns a
    /// configurable response (status, content-type, body).
    struct MockUpstream {
        last_body: Mutex<Vec<u8>>,
        last_method: Mutex<String>,
        last_path: Mutex<String>,
        last_headers: Mutex<HeaderMap>,
        resp_status: u16,
        resp_content_type: Option<String>,
        resp_body: Vec<u8>,
    }

    impl MockUpstream {
        fn new() -> Self {
            Self {
                last_body: Mutex::new(Vec::new()),
                last_method: Mutex::new(String::new()),
                last_path: Mutex::new(String::new()),
                last_headers: Mutex::new(HeaderMap::new()),
                resp_status: 200,
                resp_content_type: None,
                resp_body: b"answer<think>hidden</think> done".to_vec(),
            }
        }
        fn responding(status: u16, content_type: Option<&str>, body: &[u8]) -> Self {
            Self {
                resp_status: status,
                resp_content_type: content_type.map(str::to_string),
                resp_body: body.to_vec(),
                ..Self::new()
            }
        }
    }

    impl Upstream for MockUpstream {
        async fn forward(&self, req: ForwardRequest) -> Result<ForwardResponse, ProxyError> {
            *self.last_body.lock().unwrap() = req.body;
            *self.last_method.lock().unwrap() = req.method.to_string();
            *self.last_path.lock().unwrap() = req.path_and_query;
            *self.last_headers.lock().unwrap() = req.headers;
            let mut headers = HeaderMap::new();
            if let Some(ct) = &self.resp_content_type {
                headers.insert(http::header::CONTENT_TYPE, ct.parse().unwrap());
            }
            Ok(ForwardResponse {
                status: self.resp_status,
                headers,
                body: once_stream(self.resp_body.clone()),
            })
        }
    }

    fn state_with(mock: MockUpstream, api_key: Option<&str>) -> Arc<ProxyState<MockUpstream>> {
        Arc::new(ProxyState {
            upstream: mock,
            config: ProxyConfig {
                target_host: "127.0.0.1".into(),
                target_port: 8080,
                merge_system: true,
                api_key: api_key.map(str::to_string),
            },
        })
    }

    fn state() -> Arc<ProxyState<MockUpstream>> {
        state_with(MockUpstream::new(), None)
    }

    fn keyed_state(key: &str) -> Arc<ProxyState<MockUpstream>> {
        state_with(MockUpstream::new(), Some(key))
    }

    #[tokio::test]
    async fn strips_think_from_response_and_transforms_request() {
        let st = state();
        let app = router(st.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .body(Body::from(
                r#"{"thinking":{"type":"enabled"},"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), MAX_BODY_BYTES)
            .await
            .unwrap();
        assert_eq!(&body[..], b"answer done"); // <think> stripped from response

        // request-side transforms applied before forwarding
        let sent = String::from_utf8(st.upstream.last_body.lock().unwrap().clone()).unwrap();
        assert!(!sent.contains("thinking")); // root thinking key removed
        assert!(sent.contains("\"system\":\"sys\"")); // system message merged up
    }

    #[tokio::test]
    async fn get_probe_preserves_method_query_and_passes_body_through() {
        // A GET /props?verbose=1 must reach upstream as GET with the query, and
        // its (non-chat) response body must pass through untouched.
        let st = state_with(
            MockUpstream::responding(
                200,
                Some("application/json"),
                br#"{"modalities":{"vision":true}}"#,
            ),
            None,
        );
        let app = router(st.clone());
        let req = Request::builder()
            .method("GET")
            .uri("/props?verbose=1")
            .header("x-custom", "abc")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), MAX_BODY_BYTES)
            .await
            .unwrap();
        assert_eq!(&body[..], br#"{"modalities":{"vision":true}}"#); // untouched
        assert_eq!(*st.upstream.last_method.lock().unwrap(), "GET");
        assert_eq!(*st.upstream.last_path.lock().unwrap(), "/props?verbose=1");
        // Client headers reach upstream; hop-by-hop/host are dropped.
        let sent = st.upstream.last_headers.lock().unwrap().clone();
        assert_eq!(sent.get("x-custom").unwrap(), "abc");
        assert!(sent.get("host").is_none());
    }

    #[tokio::test]
    async fn get_models_list_is_not_think_stripped() {
        // A model list is not a chat body — even if it somehow contained the
        // token, we must not rewrite a non-chat JSON response.
        let st = state_with(
            MockUpstream::responding(200, Some("application/json"), br#"{"data":[{"id":"m"}]}"#),
            None,
        );
        let app = router(st);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), MAX_BODY_BYTES)
            .await
            .unwrap();
        assert_eq!(&body[..], br#"{"data":[{"id":"m"}]}"#);
    }

    #[tokio::test]
    async fn streaming_sse_response_is_stripped_per_delta_without_corrupting_framing() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"keep <think>\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"secret</think> done\"}}]}\n\n\
                   data: [DONE]\n\n";
        let st = state_with(
            MockUpstream::responding(200, Some("text/event-stream"), sse.as_bytes()),
            None,
        );
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .body(Body::from(
                r#"{"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(http::header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let body = axum::body::to_bytes(resp.into_body(), MAX_BODY_BYTES)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(!text.contains("secret"), "leaked think content: {text}");
        assert!(
            !text.contains("<think>") && !text.contains("</think>"),
            "leaked tag: {text}"
        );
        assert!(text.contains("[DONE]"));
        assert_eq!(text.matches("data: ").count(), text.matches("\n\n").count());
    }

    #[tokio::test]
    async fn upstream_error_mid_stream_flushes_the_held_back_tail_before_erroring() {
        // A content delta whose trailing chars are held back for split-tag
        // safety, then an upstream drop. The abort must still deliver the
        // held-back tail (before surfacing the error) — otherwise the last
        // characters of already-generated text are silently truncated on top of
        // losing the rest (the "cut off mid-word" bug on an unstable local
        // server). The clean-EOF path already flushes; this covers the error path.
        let delta = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello world\"}}]}\n\n";
        let upstream: ByteStream = Box::pin(futures::stream::iter(vec![
            Ok(Bytes::from(delta)),
            Err(ProxyError::Upstream("connection reset".to_string())),
        ]));
        let mut out = Box::pin(sse_think_stream(upstream));
        let mut text = String::new();
        let mut saw_error = false;
        while let Some(item) = out.next().await {
            match item {
                Ok(bytes) => text.push_str(&String::from_utf8_lossy(&bytes)),
                Err(_) => {
                    saw_error = true;
                    break;
                }
            }
        }
        assert!(
            saw_error,
            "the upstream error must still surface to the client"
        );
        // The full visible text survived the abort (the ≤HOLDBACK tail flushed).
        let visible: String = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|p| serde_json::from_str::<Value>(p).ok())
            .filter_map(|v| {
                v.pointer("/choices/0/delta/content")
                    .and_then(|c| c.as_str().map(String::from))
            })
            .collect();
        assert_eq!(
            visible, "Hello world",
            "held-back tail lost on abort: {text}"
        );
    }

    #[tokio::test]
    async fn build_response_aborts_the_client_body_on_a_mid_stream_error() {
        // The cross-repo StreamTruncated contract, asserted at the HTTP body
        // boundary (not just the combinator): when the upstream drops mid-stream,
        // the client-facing response body must ABORT — surface an error frame —
        // never end cleanly. That abort is exactly what makes a consumer (e.g.
        // LocalPilot's `ProviderError::StreamTruncated` retry) treat the reply as
        // truncated-and-retryable instead of accepting a silently cut-off answer.
        // A refactor of `sse_think_stream` to a "graceful end" would keep the
        // combinator test green while breaking this — so this drives `build_response`.
        let delta = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
        let body: ByteStream = Box::pin(futures::stream::iter(vec![
            Ok(Bytes::from(delta)),
            Err(ProxyError::Upstream("connection reset".to_string())),
        ]));
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        let fwd = ForwardResponse {
            status: 200,
            headers,
            body,
        };
        let resp = build_response(fwd, true).await;
        assert_eq!(resp.status(), StatusCode::OK);
        // Collecting the client body MUST fail: a clean EOF here would mean the
        // consumer sees a complete (but truncated) reply and never retries.
        let collected = axum::body::to_bytes(resp.into_body(), MAX_BODY_BYTES).await;
        assert!(
            collected.is_err(),
            "a mid-stream upstream error must abort the client body, not end it cleanly"
        );
    }

    #[tokio::test]
    async fn health_reports_the_target() {
        let app = router(state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), MAX_BODY_BYTES)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["target_host"], json!("127.0.0.1"));
        assert_eq!(v["target_port"], json!(8080));
        assert_eq!(v["status"], json!("ok"));
    }

    #[tokio::test]
    async fn serve_proxy_binds_and_answers_health_over_a_real_socket() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let server = tokio::spawn(serve_proxy(
            port,
            ProxyConfig {
                target_host: "127.0.0.1".into(),
                target_port: 8080,
                merge_system: true,
                api_key: None,
            },
        ));

        let url = format!("http://127.0.0.1:{port}/health");
        let mut last_err = String::new();
        let mut answered = None;
        for _ in 0..50 {
            match reqwest::get(&url).await {
                Ok(resp) => {
                    answered = Some(resp.json::<Value>().await.unwrap());
                    break;
                }
                Err(e) => {
                    last_err = e.to_string();
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
            }
        }
        let v = answered.unwrap_or_else(|| panic!("proxy never answered /health: {last_err}"));
        assert_eq!(v["target_port"], json!(8080));
        server.abort();
    }

    #[tokio::test]
    async fn a_keyed_proxy_refuses_unauthorized_forwards_but_not_health() {
        let app = router(keyed_state("secret"));
        // No key: refused before any upstream contact.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .body(Body::from("{}"))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Bearer and x-api-key spellings both pass.
        for (name, value) in [("authorization", "Bearer secret"), ("x-api-key", "secret")] {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(name, value)
                .body(Body::from("{}"))
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        // Health stays open so target checks keep working.
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn serve_proxy_reports_a_bind_conflict() {
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = holder.local_addr().unwrap().port();
        let err = serve_proxy(
            port,
            ProxyConfig {
                target_host: "127.0.0.1".into(),
                target_port: 8080,
                merge_system: true,
                api_key: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ProxyError::Listen(_)));
    }
}

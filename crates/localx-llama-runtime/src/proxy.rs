//! The in-process no-think proxy (axum) — the replacement for the python sidecar.
//!
//! Composes the tested [`crate::nothink`] transforms into a real forwarding
//! proxy. The upstream is injected via the [`Upstream`] trait so the forward path
//! is unit-testable with a mock (no live server needed); [`ReqwestUpstream`] is
//! the production impl.
//!
//! This first cut buffers and strips non-streaming responses. True SSE streaming
//! (per-chunk [`crate::nothink::ThinkStripper`]) and the per-turn `[no output]`
//! fallback within a parsed response are follow-ups; the streaming stripper and
//! the fallback are already unit-tested in `nothink`.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{Json, Response},
    routing::get,
    Router,
};
use serde_json::{json, Value};

use crate::nothink::{strip_think, transform_request};

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

/// A buffered upstream response.
#[derive(Debug, Clone)]
pub struct ForwardResponse {
    /// HTTP status code from the upstream.
    pub status: u16,
    /// Response body bytes.
    pub body: Vec<u8>,
}

/// The upstream the proxy forwards to. Abstracted for testability.
pub trait Upstream: Send + Sync + 'static {
    /// Forward a request body to `path` and return the buffered response.
    fn forward(
        &self,
        path: String,
        body: Vec<u8>,
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
    let path = req.uri().path().to_string();

    let bytes = match axum::body::to_bytes(req.into_body(), MAX_BODY_BYTES).await {
        Ok(b) => b,
        Err(_) => return status_body(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"),
    };

    // Request-side transforms: strip thinking keys at the root, merge system msgs.
    let out_body = match serde_json::from_slice::<Value>(&bytes) {
        Ok(mut v) => {
            transform_request(&mut v, state.config.merge_system);
            serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec())
        }
        Err(_) => bytes.to_vec(),
    };

    match state.upstream.forward(path, out_body).await {
        Ok(resp) => {
            let stripped = strip_think(&String::from_utf8_lossy(&resp.body));
            let mut r = Response::new(Body::from(stripped));
            *r.status_mut() = StatusCode::from_u16(resp.status).unwrap_or(StatusCode::BAD_GATEWAY);
            r
        }
        Err(e) => status_body(StatusCode::BAD_GATEWAY, &e.to_string()),
    }
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
    let upstream = ReqwestUpstream::new(format!(
        "http://{}:{}",
        config.target_host, config.target_port
    ));
    let state = Arc::new(ProxyState { upstream, config });
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", listen_port))
        .await
        .map_err(|e| ProxyError::Listen(format!("127.0.0.1:{listen_port}: {e}")))?;
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
    async fn forward(&self, path: String, body: Vec<u8>) -> Result<ForwardResponse, ProxyError> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .client
            .post(url)
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| ProxyError::Upstream(e.to_string()))?;
        let status = resp.status().as_u16();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProxyError::Upstream(e.to_string()))?;
        Ok(ForwardResponse {
            status,
            body: bytes.to_vec(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::http::Request;
    use std::sync::Mutex;
    use tower::ServiceExt;

    struct MockUpstream {
        last_body: Mutex<Vec<u8>>,
    }

    impl Upstream for MockUpstream {
        async fn forward(
            &self,
            _path: String,
            body: Vec<u8>,
        ) -> Result<ForwardResponse, ProxyError> {
            *self.last_body.lock().unwrap() = body;
            Ok(ForwardResponse {
                status: 200,
                body: b"answer<think>hidden</think> done".to_vec(),
            })
        }
    }

    fn state() -> Arc<ProxyState<MockUpstream>> {
        Arc::new(ProxyState {
            upstream: MockUpstream {
                last_body: Mutex::new(Vec::new()),
            },
            config: ProxyConfig {
                target_host: "127.0.0.1".into(),
                target_port: 8080,
                merge_system: true,
            },
        })
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
    async fn serve_proxy_reports_a_bind_conflict() {
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = holder.local_addr().unwrap().port();
        let err = serve_proxy(
            port,
            ProxyConfig {
                target_host: "127.0.0.1".into(),
                target_port: 8080,
                merge_system: true,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ProxyError::Listen(_)));
    }
}

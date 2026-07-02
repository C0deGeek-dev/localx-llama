//! Server-lifecycle helpers: embed-serve args, binary resolution, health-wait.
//!
//! The detached spawn itself is in [`crate::spawn`]; the port primitives are in
//! [`crate::net`]. Source-build orchestration (Windows vcvars/cmake) and
//! socket→PID reaping are OS-specific and handled at the app layer (see plan
//! decision), on top of the tested `plan_proxy_action` reap logic.

use std::path::{Path, PathBuf};

use tokio::time::{sleep, Duration, Instant};

use crate::error::RuntimeError;
use crate::net::is_port_listening;

/// The llama-server executable name for this platform.
pub fn server_exe_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// Build the CPU-only embed-serve arguments.
///
/// `-ngl 0` is load-bearing: a GPU-resident embed model would steal VRAM from a
/// chat model running alongside it, so embeddings stay on the CPU and the chat
/// model is byte-identical whether or not embeddings are running. `--pooling last`
/// is required by Qwen3-Embedding.
pub fn embed_server_args(model_path: &str, port: u16) -> Vec<String> {
    vec![
        "-m".into(),
        model_path.into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--embedding".into(),
        "--pooling".into(),
        "last".into(),
        "-ngl".into(),
        "0".into(),
    ]
}

/// Resolve a llama-server binary: configured path, then a PATH search, else the
/// bring-your-own error (O3: off-Windows we don't auto-build, the user provides one).
pub fn resolve_server_binary(
    configured: Option<&Path>,
    path_dirs: &[PathBuf],
    exe_name: &str,
) -> Result<PathBuf, RuntimeError> {
    if let Some(p) = configured {
        if p.exists() {
            return Ok(p.to_path_buf());
        }
    }
    for dir in path_dirs {
        let candidate = dir.join(exe_name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(RuntimeError::NoServerBinary)
}

/// Poll a loopback port until something is listening or the timeout elapses.
pub async fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    loop {
        if is_port_listening(port) {
            return true;
        }
        if start.elapsed() >= timeout {
            return false;
        }
        sleep(Duration::from_millis(100)).await;
    }
}

/// The HTTP status `GET /health` on a loopback port answers right now.
fn health_status(port: u16) -> Option<u16> {
    use std::io::{Read, Write};
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream =
        std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(500)).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok()?;
    write!(
        stream,
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut buf = [0_u8; 64];
    let read = stream.read(&mut buf).ok()?;
    String::from_utf8_lossy(&buf[..read])
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Block until the server's `/health` answers HTTP 200, or the timeout
/// elapses. A llama-server binds its port immediately and answers 503 while
/// the model is still loading, so a listening socket is NOT readiness — a
/// request sent then gets an error body instead of an answer.
#[must_use]
pub fn wait_for_ready(port: u16, timeout: std::time::Duration) -> bool {
    let start = std::time::Instant::now();
    loop {
        if health_status(port) == Some(200) {
            return true;
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, TcpListener};

    #[test]
    fn embed_args_force_cpu_and_last_pooling() {
        let args = embed_server_args("/models/embed.gguf", 8090);
        let j = args.join(" ");
        assert!(j.contains("-ngl 0")); // load-bearing: zero VRAM
        assert!(j.contains("--embedding"));
        assert!(j.contains("--pooling last"));
        assert!(j.contains("--port 8090"));
    }

    #[test]
    fn resolve_binary_prefers_configured_then_errors() {
        // A file known to exist: this crate's own manifest.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        assert_eq!(
            resolve_server_binary(Some(&manifest), &[], "x").unwrap(),
            manifest
        );
        // Nothing configured, empty PATH -> bring-your-own error.
        assert!(matches!(
            resolve_server_binary(None, &[], "nope").unwrap_err(),
            RuntimeError::NoServerBinary
        ));
    }

    #[tokio::test]
    async fn waits_for_a_listening_port() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(wait_for_port(port, Duration::from_secs(1)).await);
    }

    #[tokio::test]
    async fn times_out_on_a_closed_port() {
        let port = crate::net::free_port().unwrap(); // released, nothing listening
        assert!(!wait_for_port(port, Duration::from_millis(300)).await);
    }

    /// A mock server answering `/health` with a scripted status sequence
    /// (the last status repeats forever).
    fn spawn_health_server(statuses: &'static [u16]) -> u16 {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for (served, stream) in listener.incoming().enumerate() {
                let Ok(mut stream) = stream else { break };
                let status = statuses[served.min(statuses.len() - 1)];
                let mut buf = [0_u8; 512];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                let reason = if status == 200 {
                    "OK"
                } else {
                    "Service Unavailable"
                };
                let _ = std::io::Write::write_all(
                    &mut stream,
                    format!(
                        "HTTP/1.1 {status} {reason}\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
                    )
                    .as_bytes(),
                );
            }
        });
        port
    }

    #[test]
    fn a_listening_but_still_loading_server_is_not_ready() {
        // 503 while the model loads, then 200: readiness waits for the 200.
        let port = spawn_health_server(&[503, 503, 200]);
        assert!(wait_for_ready(port, std::time::Duration::from_secs(10)));

        // A server that never becomes healthy times out even though the
        // port is listening the whole time.
        let stuck = spawn_health_server(&[503]);
        assert!(is_port_listening(stuck));
        assert!(!wait_for_ready(
            stuck,
            std::time::Duration::from_millis(600)
        ));
    }

    #[test]
    fn readiness_times_out_when_nothing_listens() {
        let port = crate::net::free_port().unwrap();
        assert!(!wait_for_ready(port, std::time::Duration::from_millis(300)));
    }
}

//! Loopback port utilities for server lifecycle.
//!
//! Bind to `127.0.0.1` literally (never `localhost`) — the Rust client does not
//! fall back from a `localhost`→`::1` resolution, so IPv4 loopback is fixed.

use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

/// Whether a loopback port can be bound (i.e. nothing is using it).
pub fn is_port_free(port: u16) -> bool {
    TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_ok()
}

/// Ask the OS for a free loopback port.
pub fn free_port() -> Option<u16> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).ok()?;
    listener.local_addr().ok().map(|a| a.port())
}

/// Whether something is listening on a loopback port (upstream liveness).
///
/// This is the `not is_port_free` signal the launcher uses to infer the server
/// is up before probing the proxy target.
pub fn is_port_listening(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn free_port_is_bindable() {
        let port = free_port().unwrap();
        // Just released by free_port -> bindable again.
        assert!(is_port_free(port));
    }

    #[test]
    fn bound_port_is_listening_and_not_free() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(is_port_listening(port)); // OS accepts into the backlog
        assert!(!is_port_free(port)); // in use
        drop(listener);
        assert!(is_port_free(port)); // released
    }
}

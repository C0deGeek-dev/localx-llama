//! No-think proxy target classification + server health state machine (pure).
//!
//! The tri-state target check and the reap-before-probe / repoint-on-mismatch
//! ordering are the control flow, not a bool — a dead upstream orphan still
//! answers `/health`, so it must be reaped before the target test or every
//! request strands behind a bare 502.

/// Result of checking what a running proxy points at (the launcher's tri-state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyTarget {
    /// Proxy up and pointed at the wanted upstream.
    Matching,
    /// Proxy up but pointed at a different upstream.
    Mismatch,
    /// Proxy absent, or up but its target is unverifiable (`/health` unreadable).
    Absent,
}

/// Classify a proxy's target. `reported` is the `{target_host, target_port}` it
/// returns from `/health`, or `None` when unreachable/unverifiable.
pub fn classify_proxy_target(
    proxy_up: bool,
    reported: Option<(&str, u16)>,
    wanted: (&str, u16),
) -> ProxyTarget {
    if !proxy_up {
        return ProxyTarget::Absent;
    }
    match reported {
        None => ProxyTarget::Absent,
        Some(t) if t == wanted => ProxyTarget::Matching,
        Some(_) => ProxyTarget::Mismatch,
    }
}

/// The action to take before routing an agent through the proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyAction {
    /// Proxy already up and matching — use it as-is.
    Reuse,
    /// Proxy up but pointed elsewhere — tear down and restart at the wanted target.
    Repoint,
    /// Proxy up but the upstream is dead (or target unverifiable) — reap the
    /// orphan first, then start fresh. This ordering is load-bearing.
    ReapThenStart,
    /// No proxy — start one.
    Start,
}

/// Decide the proxy action. Reap-before-probe: a proxy over a dead upstream is
/// reaped before any target comparison.
pub fn plan_proxy_action(proxy_up: bool, upstream_up: bool, target: ProxyTarget) -> ProxyAction {
    if !proxy_up {
        return ProxyAction::Start;
    }
    if !upstream_up {
        // dead-upstream orphan still answers /health -> reap before probing target
        return ProxyAction::ReapThenStart;
    }
    match target {
        ProxyTarget::Matching => ProxyAction::Reuse,
        ProxyTarget::Mismatch => ProxyAction::Repoint,
        ProxyTarget::Absent => ProxyAction::ReapThenStart,
    }
}

/// Overall serve health as surfaced to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthState {
    /// Proxy and upstream both up.
    Ok,
    /// Proxy up but upstream down — requests get a bare 502 (stale proxy).
    StaleProxy,
    /// Upstream up but the proxy is not running.
    ProxyDown,
    /// Nothing serving.
    Down,
}

/// Classify serve health from proxy/upstream liveness.
pub fn classify_health(proxy_up: bool, upstream_up: bool) -> HealthState {
    match (proxy_up, upstream_up) {
        (true, true) => HealthState::Ok,
        (true, false) => HealthState::StaleProxy,
        (false, true) => HealthState::ProxyDown,
        (false, false) => HealthState::Down,
    }
}

/// A copy-paste remediation for a non-Ok health state.
pub fn remediation(state: HealthState) -> &'static str {
    match state {
        HealthState::Ok => "",
        HealthState::Down => "llmdefaultserve",
        HealthState::StaleProxy | HealthState::ProxyDown => "llmstop; llmdefaultserve",
    }
}

/// The operator-facing description of a non-Ok health state, with the ports.
pub fn health_description(state: HealthState, proxy_port: u16, upstream_port: u16) -> String {
    match state {
        HealthState::Ok => String::new(),
        HealthState::StaleProxy => format!(
            "The no-think proxy is up on {proxy_port} but the upstream model server on              {upstream_port} is down, so requests return a bare 502."
        ),
        HealthState::ProxyDown => format!(
            "The model server on {upstream_port} is up but the no-think proxy on              {proxy_port} is not running."
        ),
        HealthState::Down => {
            "Neither the no-think proxy nor the model server is running.".to_string()
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn proxy_target_is_tristate() {
        let wanted = ("127.0.0.1", 8080);
        assert_eq!(
            classify_proxy_target(true, Some(("127.0.0.1", 8080)), wanted),
            ProxyTarget::Matching
        );
        assert_eq!(
            classify_proxy_target(true, Some(("127.0.0.1", 9090)), wanted),
            ProxyTarget::Mismatch
        );
        assert_eq!(
            classify_proxy_target(true, None, wanted),
            ProxyTarget::Absent
        );
        assert_eq!(
            classify_proxy_target(false, Some(("127.0.0.1", 8080)), wanted),
            ProxyTarget::Absent
        );
    }

    #[test]
    fn reap_before_probe_when_upstream_dead() {
        // proxy up, upstream down -> reap regardless of what the target looks like.
        assert_eq!(
            plan_proxy_action(true, false, ProxyTarget::Matching),
            ProxyAction::ReapThenStart
        );
        assert_eq!(
            plan_proxy_action(true, false, ProxyTarget::Mismatch),
            ProxyAction::ReapThenStart
        );
    }

    #[test]
    fn repoint_on_mismatch_reuse_on_match() {
        assert_eq!(
            plan_proxy_action(true, true, ProxyTarget::Matching),
            ProxyAction::Reuse
        );
        assert_eq!(
            plan_proxy_action(true, true, ProxyTarget::Mismatch),
            ProxyAction::Repoint
        );
        assert_eq!(
            plan_proxy_action(false, false, ProxyTarget::Absent),
            ProxyAction::Start
        );
    }

    #[test]
    fn health_matrix_and_remediation() {
        assert_eq!(classify_health(true, true), HealthState::Ok);
        assert_eq!(classify_health(true, false), HealthState::StaleProxy);
        assert_eq!(classify_health(false, false), HealthState::Down);
        assert_eq!(remediation(HealthState::Ok), "");
        assert_eq!(
            remediation(HealthState::StaleProxy),
            "llmstop; llmdefaultserve"
        );
        assert_eq!(remediation(HealthState::Down), "llmdefaultserve");
    }

    #[test]
    fn health_distinguishes_proxy_down_from_fully_down() {
        assert_eq!(classify_health(false, true), HealthState::ProxyDown);
        assert_eq!(classify_health(false, false), HealthState::Down);
        assert_eq!(
            remediation(HealthState::ProxyDown),
            "llmstop; llmdefaultserve"
        );
        assert_eq!(remediation(HealthState::Down), "llmdefaultserve");
        let text = health_description(HealthState::ProxyDown, 11435, 8080);
        assert!(text.contains("8080 is up"));
        assert!(text.contains("11435 is not running"));
    }
}

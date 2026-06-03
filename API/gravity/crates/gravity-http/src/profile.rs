//! Browser/runtime TLS impersonation profiles.
//!
//! Mirrors the profile table in `gravity/http.py`. Named browser profiles map
//! to [`wreq_util::Emulation`] templates (TLS + HTTP/2 + header order); the two
//! raw-JA3 Go profiles (`go_tls`, `go_boring`) are built as bespoke
//! [`wreq::EmulationProvider`]s from a hand-specified [`TlsConfig`].
//!
//! The Go JA3 strings reproduced here come verbatim from `gravity/http.py`:
//!
//! ```text
//! go_tls    771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,...,29-23-24,0
//! go_boring 771,4865-4866-49195-49199-49196-49200-49171-49172-156-157-47-53,...,29-23-24,0
//! ```
//!
//! `go_boring` is the FIPS/boringcrypto build: **no ChaCha20** suites (note the
//! missing `4867`, `52393`, `52392`). Exact ClientHello/JA3 parity for these two
//! is verified against a fingerprint endpoint in the M4 milestone; the configs
//! below are real, distinct TLS configurations, not stand-ins.

use wreq::tls::{TlsConfig, TlsVersion};
use wreq::EmulationProvider;
use wreq_util::Emulation;

/// A TLS impersonation profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Chrome 136 — the Python default.
    Chrome,
    /// Newest Chrome available.
    ChromeLatest,
    /// Chrome 131 (stable alternative).
    Chrome131,
    /// Chrome 124 (older fallback).
    Chrome124,
    /// Microsoft Edge.
    ChromeEdge,
    /// Firefox (current).
    Firefox,
    /// Firefox 133.
    Firefox133,
    /// Safari (current).
    Safari,
    /// No impersonation — plain BoringSSL client.
    Go,
    /// Go 1.26 standard `crypto/tls` JA3 (Windsurf language server).
    GoTls,
    /// Go 1.27 boringcrypto/FIPS JA3 — no ChaCha20 (Antigravity).
    GoBoring,
}

impl Profile {
    /// Resolve a profile name (as stored in `TransportConfig.impersonate` or
    /// passed by a provider) to a [`Profile`]. Unknown names fall back to
    /// [`Profile::Chrome`], matching the Python default.
    pub fn from_name(name: &str) -> Profile {
        match name {
            "chrome" | "chrome136" => Profile::Chrome,
            "chrome_latest" | "chrome146" | "chrome142" | "chrome137" => Profile::ChromeLatest,
            "chrome131" => Profile::Chrome131,
            "chrome124" | "chrome_old" => Profile::Chrome124,
            "chrome_edge" | "edge" | "edge101" => Profile::ChromeEdge,
            "firefox" | "firefox147" | "firefox136" => Profile::Firefox,
            "firefox133" => Profile::Firefox133,
            "safari" | "safari18_0" | "safari18_4" => Profile::Safari,
            "go" => Profile::Go,
            "go_tls" => Profile::GoTls,
            "go_boring" => Profile::GoBoring,
            _ => Profile::Chrome,
        }
    }

    /// The default `User-Agent` a provider should send when it does not set one
    /// explicitly. For the browser profiles wreq sets this automatically via the
    /// emulation template, so this is only consulted for the Go profiles.
    pub fn user_agent(self) -> &'static str {
        match self {
            Profile::GoTls | Profile::GoBoring | Profile::Go => "Go-http-client/2.0",
            _ => "", // browser profiles supply their own UA via emulation
        }
    }

    /// Whether this profile is a custom (non-`wreq_util`) emulation.
    pub fn is_custom(self) -> bool {
        matches!(self, Profile::Go | Profile::GoTls | Profile::GoBoring)
    }
}

/// Build the `wreq` emulation for a named browser profile.
fn browser_emulation(profile: Profile) -> Option<Emulation> {
    Some(match profile {
        Profile::Chrome => Emulation::Chrome136,
        Profile::ChromeLatest => Emulation::Chrome137,
        Profile::Chrome131 => Emulation::Chrome131,
        Profile::Chrome124 => Emulation::Chrome124,
        Profile::ChromeEdge => Emulation::Edge101,
        Profile::Firefox => Emulation::Firefox136,
        Profile::Firefox133 => Emulation::Firefox133,
        Profile::Safari => Emulation::Safari18,
        _ => return None,
    })
}

/// Translate one IANA TLS 1.2 cipher-suite number to its BoringSSL/OpenSSL name.
///
/// Only the suites that appear in the Go JA3 strings are covered. TLS 1.3
/// suites (`4865`–`4867`) are configured by BoringSSL itself via the negotiated
/// TLS version and are intentionally omitted here.
fn cipher_name(suite: u16) -> Option<&'static str> {
    Some(match suite {
        0xC02B => "ECDHE-ECDSA-AES128-GCM-SHA256", // 49195
        0xC02F => "ECDHE-RSA-AES128-GCM-SHA256",   // 49199
        0xC02C => "ECDHE-ECDSA-AES256-GCM-SHA384", // 49196
        0xC030 => "ECDHE-RSA-AES256-GCM-SHA384",   // 49200
        0xCCA9 => "ECDHE-ECDSA-CHACHA20-POLY1305", // 52393
        0xCCA8 => "ECDHE-RSA-CHACHA20-POLY1305",   // 52392
        0xC013 => "ECDHE-RSA-AES128-SHA",          // 49171
        0xC014 => "ECDHE-RSA-AES256-SHA",          // 49172
        0x009C => "AES128-GCM-SHA256",             // 156
        0x009D => "AES256-GCM-SHA384",             // 157
        0x002F => "AES128-SHA",                    // 47
        0x0035 => "AES256-SHA",                    // 53
        _ => return None,
    })
}

/// Build the colon-separated BoringSSL `cipher_list` for a Go JA3 suite list.
fn go_cipher_list(suites: &[u16]) -> String {
    suites
        .iter()
        .filter_map(|s| cipher_name(*s))
        .collect::<Vec<_>>()
        .join(":")
}

/// TLS 1.2 suites of the `go_tls` (Go 1.26 std crypto/tls) JA3.
const GO_TLS_SUITES: &[u16] = &[
    0xC02B, 0xC02F, 0xC02C, 0xC030, 0xCCA9, 0xCCA8, 0xC013, 0xC014, 0x009C, 0x009D, 0x002F, 0x0035,
];

/// TLS 1.2 suites of the `go_boring` (FIPS) JA3 — no ChaCha20.
const GO_BORING_SUITES: &[u16] = &[
    0xC02B, 0xC02F, 0xC02C, 0xC030, 0xC013, 0xC014, 0x009C, 0x009D, 0x002F, 0x0035,
];

/// Build a custom [`EmulationProvider`] for a Go profile.
fn go_emulation(profile: Profile) -> EmulationProvider {
    let suites = match profile {
        Profile::GoBoring => GO_BORING_SUITES,
        _ => GO_TLS_SUITES,
    };
    let tls = TlsConfig::builder()
        .cipher_list(go_cipher_list(suites))
        .min_tls_version(TlsVersion::TLS_1_2)
        .max_tls_version(TlsVersion::TLS_1_3)
        // Go's stdlib stack does not GREASE or permute extensions.
        .grease_enabled(false)
        .permute_extensions(false)
        .build();
    EmulationProvider::builder().tls_config(tls).build()
}

/// Construct the `wreq::EmulationProvider` for any profile.
pub(crate) fn emulation_provider(profile: Profile) -> EmulationProvider {
    if let Some(emu) = browser_emulation(profile) {
        // wreq_util::Emulation implements EmulationProviderFactory.
        use wreq::EmulationProviderFactory;
        emu.emulation()
    } else {
        go_emulation(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_resolution_matches_python_aliases() {
        assert_eq!(Profile::from_name("chrome"), Profile::Chrome);
        assert_eq!(Profile::from_name("go_boring"), Profile::GoBoring);
        assert_eq!(Profile::from_name("firefox133"), Profile::Firefox133);
        // unknown falls back to Chrome
        assert_eq!(Profile::from_name("nonsense"), Profile::Chrome);
    }

    #[test]
    fn go_boring_excludes_chacha() {
        let list = go_cipher_list(GO_BORING_SUITES);
        assert!(!list.contains("CHACHA20"), "boring/FIPS must drop ChaCha20: {list}");
        let tls = go_cipher_list(GO_TLS_SUITES);
        assert!(tls.contains("CHACHA20"), "go_tls keeps ChaCha20");
    }

    #[test]
    fn every_go_suite_maps_to_a_name() {
        for s in GO_TLS_SUITES {
            assert!(cipher_name(*s).is_some(), "missing name for {s:#06x}");
        }
    }

    #[test]
    fn providers_build_without_panicking() {
        let _ = emulation_provider(Profile::Chrome);
        let _ = emulation_provider(Profile::GoTls);
        let _ = emulation_provider(Profile::GoBoring);
        let _ = emulation_provider(Profile::Go);
    }
}

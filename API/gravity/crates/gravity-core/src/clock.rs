//! Injectable time and entropy.
//!
//! Provider request signing (zlm/glm HMAC time buckets, ChatGPT PoW seeds,
//! request UUIDs) depends on the wall clock and randomness. To make those code
//! paths byte-for-byte reproducible under golden-trace tests, every such source
//! is funneled through [`Clock`] and [`Entropy`] instead of calling
//! `SystemTime::now` / the OS RNG directly.
//!
//! Production code uses [`SystemClock`] and [`OsEntropy`]; tests use
//! [`FixedClock`] and [`SeededEntropy`].

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// A source of wall-clock time.
pub trait Clock: Send + Sync {
    /// Milliseconds since the Unix epoch.
    fn now_millis(&self) -> u64;

    /// Seconds since the Unix epoch.
    fn now_secs(&self) -> u64 {
        self.now_millis() / 1000
    }
}

/// A source of randomness.
///
/// Kept deliberately small: enough to fill a buffer and draw a `u64`. Providers
/// build UUIDs / nonces / PoW seeds on top.
pub trait Entropy: Send + Sync {
    /// Fill `dest` with random bytes.
    fn fill(&self, dest: &mut [u8]);

    /// Draw a random `u64`.
    fn next_u64(&self) -> u64 {
        let mut b = [0u8; 8];
        self.fill(&mut b);
        u64::from_le_bytes(b)
    }
}

/// Real wall clock backed by [`SystemTime`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_millis(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// OS-backed entropy.
///
/// Uses a per-call seed mixed from [`SystemClock`] and the address of a stack
/// allocation, then a SplitMix64 expansion. This avoids a hard dependency on a
/// specific RNG crate in `gravity-core`; binding/provider crates that need
/// cryptographic randomness inject their own [`Entropy`].
#[derive(Debug, Clone, Copy, Default)]
pub struct OsEntropy;

impl Entropy for OsEntropy {
    fn fill(&self, dest: &mut [u8]) {
        let mut state = SystemClock.now_millis() ^ (dest.as_ptr() as u64).rotate_left(17);
        for chunk in dest.chunks_mut(8) {
            // SplitMix64
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let bytes = z.to_le_bytes();
            for (d, s) in chunk.iter_mut().zip(bytes) {
                *d = s;
            }
        }
    }
}

/// A fixed clock that always returns the same instant — for tests.
#[derive(Debug, Clone, Copy)]
pub struct FixedClock(pub u64);

impl Clock for FixedClock {
    fn now_millis(&self) -> u64 {
        self.0
    }
}

/// A deterministic SplitMix64 RNG seeded from a fixed value — for tests.
#[derive(Debug)]
pub struct SeededEntropy {
    state: std::sync::Mutex<u64>,
}

impl SeededEntropy {
    /// Create a seeded entropy source.
    pub fn new(seed: u64) -> Self {
        SeededEntropy {
            state: std::sync::Mutex::new(seed),
        }
    }
}

impl Entropy for SeededEntropy {
    fn fill(&self, dest: &mut [u8]) {
        let mut state = self.state.lock().expect("entropy mutex poisoned");
        for chunk in dest.chunks_mut(8) {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let bytes = z.to_le_bytes();
            for (d, s) in chunk.iter_mut().zip(bytes) {
                *d = s;
            }
        }
    }
}

/// Bundle of injected time + entropy passed through request contexts.
#[derive(Clone)]
pub struct Env {
    /// Wall clock.
    pub clock: Arc<dyn Clock>,
    /// Randomness.
    pub entropy: Arc<dyn Entropy>,
}

impl Env {
    /// The production environment: real clock + OS entropy.
    pub fn system() -> Self {
        Env {
            clock: Arc::new(SystemClock),
            entropy: Arc::new(OsEntropy),
        }
    }

    /// A deterministic environment for tests.
    pub fn fixed(millis: u64, seed: u64) -> Self {
        Env {
            clock: Arc::new(FixedClock(millis)),
            entropy: Arc::new(SeededEntropy::new(seed)),
        }
    }
}

impl Default for Env {
    fn default() -> Self {
        Env::system()
    }
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env")
            .field("now_millis", &self.clock.now_millis())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_clock_is_stable() {
        let c = FixedClock(1_700_000_000_000);
        assert_eq!(c.now_millis(), 1_700_000_000_000);
        assert_eq!(c.now_secs(), 1_700_000_000);
    }

    #[test]
    fn seeded_entropy_is_reproducible() {
        let a = SeededEntropy::new(42);
        let b = SeededEntropy::new(42);
        let mut ba = [0u8; 16];
        let mut bb = [0u8; 16];
        a.fill(&mut ba);
        b.fill(&mut bb);
        assert_eq!(ba, bb);
    }
}
